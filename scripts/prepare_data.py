"""Publication adapter: obtain pinned data and rehydrate the original splits.
This utility is new packaging code, not an original experiment script.
No dataset data is bundled. All downloads and full captions stay under ignored work/.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.split_normalization import normalize_caption
ROOT = Path(__file__).resolve().parents[1]
SPEC = json.loads((ROOT / "provenance/dataset.json").read_text())


def prepare(dataset_root, metadata_only=False):
    destination = ROOT / "work"
    destination.mkdir(exist_ok=True)
    metadata = dataset_root / "dataset.csv"
    assert hashlib.sha256(metadata.read_bytes()).hexdigest() == SPEC["metadata_sha256"], "Dataset metadata checksum mismatch"
    data = pd.read_csv(metadata).set_index("id", verify_integrity=True)
    shutil.copyfile(metadata, destination / "dataset.csv")
    split_out = destination / "splits"
    split_out.mkdir(exist_ok=True)
    columns = ["id", "text_caption", "image_path", "platform", "label", "normalized_caption", "image_sha256", "leakage_group_id", "fold", "split"]
    for name, expected in SPEC["original_split_sha256"].items():
        ids = pd.read_csv(ROOT / "splits" / name)
        selected = data.loc[ids["id"]].reset_index()
        for key in ["image_path", "platform", "label"]:
            assert selected[key].equals(ids[key]), f"Dataset/split mismatch: {name}:{key}"
        ids["text_caption"] = selected["text_caption"]
        ids["normalized_caption"] = ids["text_caption"].apply(normalize_caption)
        output = split_out / name
        ids[columns].to_csv(output, index=False)
        assert hashlib.sha256(output.read_bytes()).hexdigest() == expected, f"Original split checksum not reproduced: {name}"
    if metadata_only:
        print("Rehydrated and checksum-verified original split metadata only; images were not copied.")
        return
    # Check all image byte hashes against the exact frozen ID metadata.
    for row in pd.read_csv(ROOT / "splits/split_manifest.csv").itertuples(index=False):
        relative = Path(row.image_path)
        assert not relative.is_absolute() and ".." not in relative.parts
        source = dataset_root / relative
        assert hashlib.sha256(source.read_bytes()).hexdigest() == row.image_sha256, f"Image checksum mismatch: ID {row.id}"
    image_out = destination / "image"
    image_out.mkdir(exist_ok=True)
    for row in pd.read_csv(ROOT / "splits/split_manifest.csv").itertuples(index=False):
        shutil.copyfile(dataset_root / row.image_path, destination / row.image_path)
    print("Prepared pinned dataset and byte-identical original splits under work/.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, help="Existing downloaded dataset with dataset.csv and image/. Read-only source.")
    parser.add_argument("--download", action="store_true", help="Download the pinned Hugging Face dataset into work/download/.")
    parser.add_argument("--metadata-only", action="store_true", help="Rehydrate and verify split metadata without hashing or copying images.")
    args = parser.parse_args()
    if bool(args.dataset_root) == args.download:
        parser.error("Choose exactly one of --dataset-root or --download")
    if args.download:
        from huggingface_hub import snapshot_download
        source = Path(snapshot_download(repo_id=SPEC["repository"], repo_type="dataset", revision=SPEC["revision"], allow_patterns=["dataset.csv", "image/*.jpg"], local_dir=ROOT / "work/download"))
    else:
        source = args.dataset_root.resolve()
    assert source != (ROOT / "work").resolve(), "Use a separate downloaded source, not work/ itself"
    prepare(source, metadata_only=args.metadata_only)

if __name__ == "__main__":
    main()
