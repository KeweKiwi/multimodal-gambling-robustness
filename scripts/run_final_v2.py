"""Export of the sanitized FINAL V2 notebook; computationally expensive.
Run from repository root, after scripts/prepare_data.py and clean embedding extraction.
All generated outputs are isolated under work/. See provenance/source_cells.json.
"""
from IPython.display import display

# Publication notebook cell pub-002
from pathlib import Path
import json
import platform
import random
import re
import sys
import time
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
import gc
import torch.nn.functional as F
import copy

from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from transformers import AutoModel, AutoProcessor
from sentence_transformers import SentenceTransformer
from PIL import Image, ImageFilter

# Publication notebook cell pub-003
# Reproducibility
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# Menentukan device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


# Notebook berada di folder notebooks/
current_dir = Path.cwd().resolve()

if current_dir.name in {"notebooks", "scripts"}:
    repository_root = current_dir.parent
else:
    repository_root = current_dir

# Publication adaptation: isolate all regenerated outputs from reference artifacts.
project_root = repository_root / "work"


# Folder input bersama
split_dir = project_root / "splits"
clean_embedding_dir = project_root / "embeddings" / "clean"
raw_image_dir = project_root / "image"


# Folder eksperimen final
experiment_version = "final_v2"

embedding_dir = (
    project_root
    / "embeddings"
    / experiment_version
)

text_embedding_dir = (
    embedding_dir
    / "text"
)

image_embedding_dir = (
    embedding_dir
    / "image"
)

model_dir = (
    project_root
    / "models"
    / experiment_version
)

perturbation_dir = (
    project_root
    / "perturbations"
    / experiment_version
)

perturbation_image_dir = (
    perturbation_dir
    / "images"
)

perturbation_manifest_dir = (
    perturbation_dir
    / "manifests"
)

result_dir = (
    project_root
    / "results"
    / experiment_version
)

config_dir = result_dir / "configs"
prediction_dir = result_dir / "predictions"
summary_dir = result_dir / "summaries"
per_platform_dir = result_dir / "per_platform"
bootstrap_dir = result_dir / "bootstrap"
diagnostic_dir = result_dir / "diagnostics"


# Pastikan seluruh folder tersedia
required_directories = [
    split_dir,
    clean_embedding_dir,
    raw_image_dir,
    text_embedding_dir,
    image_embedding_dir,
    model_dir,
    perturbation_image_dir,
    perturbation_manifest_dir,
    config_dir,
    prediction_dir,
    summary_dir,
    per_platform_dir,
    bootstrap_dir,
    diagnostic_dir,
]

for directory in required_directories:
    directory.mkdir(
        parents=True,
        exist_ok=True
    )


# Validasi input utama
required_input_paths = [
    split_dir / "train.csv",
    split_dir / "validation.csv",
    split_dir / "test.csv",
    split_dir / "split_manifest.csv",
    clean_embedding_dir
    / "train_multilingual_e5_large.npy",
    clean_embedding_dir
    / "validation_multilingual_e5_large.npy",
    clean_embedding_dir
    / "test_multilingual_e5_large.npy",
    clean_embedding_dir
    / "train_siglip2_so400m_patch14_384.npy",
    clean_embedding_dir
    / "validation_siglip2_so400m_patch14_384.npy",
    clean_embedding_dir
    / "test_siglip2_so400m_patch14_384.npy",
]

missing_inputs = [
    str(path)
    for path in required_input_paths
    if not path.exists()
]

if missing_inputs:
    raise FileNotFoundError(
        "Input berikut tidak ditemukan:\n"
        + "\n".join(missing_inputs)
    )


# Simpan informasi environment
runtime_info = {
    "experiment_version": experiment_version,
    "seed": SEED,
    "project_root": "work",
    "python_version": sys.version,
    "platform": platform.platform(),
    "torch_version": torch.__version__,
    "device": str(device),
    "mps_available": (
        torch.backends.mps.is_available()
    ),
    "cuda_available": (
        torch.cuda.is_available()
    ),
}

runtime_path = (
    config_dir
    / "runtime_environment.json"
)

with runtime_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        runtime_info,
        file,
        indent=2,
        ensure_ascii=False
    )


print("=== FINAL V2 ENVIRONMENT SETUP ===")
print("Project root :", project_root)
print("Experiment   :", experiment_version)
print("Device       :", device)
print("Split dir    :", split_dir)
print("Clean embed  :", clean_embedding_dir)
print("Final embed  :", embedding_dir)
print("Model dir    :", model_dir)
print("Perturbation :", perturbation_dir)
print("Result dir   :", result_dir)
print("\nAll required inputs found.")
print("Runtime saved to:", runtime_path)

# Publication notebook cell pub-005
import hashlib


# Load split
train_df = pd.read_csv(
    split_dir / "train.csv"
)

validation_df = pd.read_csv(
    split_dir / "validation.csv"
)

test_df = pd.read_csv(
    split_dir / "test.csv"
)


# Ukuran yang telah dibekukan
expected_split_sizes = {
    "train": 3524,
    "validation": 753,
    "test": 751,
}

split_frames = {
    "train": train_df,
    "validation": validation_df,
    "test": test_df,
}


# Kolom wajib
required_columns = {
    "id",
    "text_caption",
    "image_path",
    "platform",
    "label",
    "leakage_group_id",
}


# Validasi setiap split
for split_name, split_df in split_frames.items():

    missing_columns = (
        required_columns
        - set(split_df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"{split_name} kehilangan kolom: "
            f"{sorted(missing_columns)}"
        )

    expected_size = expected_split_sizes[
        split_name
    ]

    if len(split_df) != expected_size:
        raise ValueError(
            f"Ukuran {split_name} berubah: "
            f"{len(split_df)} != {expected_size}"
        )

    if split_df["id"].duplicated().any():
        duplicate_ids = (
            split_df.loc[
                split_df["id"].duplicated(
                    keep=False
                ),
                "id"
            ]
            .tolist()
        )

        raise ValueError(
            f"Duplicate ID pada {split_name}: "
            f"{duplicate_ids[:10]}"
        )

    if split_df["text_caption"].isna().any():
        raise ValueError(
            f"Ada caption kosong pada {split_name}"
        )

    if split_df["label"].isna().any():
        raise ValueError(
            f"Ada label kosong pada {split_name}"
        )

    invalid_labels = (
        set(split_df["label"].unique())
        - {0, 1, 0.0, 1.0}
    )

    if invalid_labels:
        raise ValueError(
            f"Label tidak valid pada {split_name}: "
            f"{invalid_labels}"
        )


# Gabungkan untuk audit lintas split
combined_split_df = pd.concat(
    [
        train_df.assign(
            evaluated_split="train"
        ),
        validation_df.assign(
            evaluated_split="validation"
        ),
        test_df.assign(
            evaluated_split="test"
        ),
    ],
    ignore_index=True
)


# Validasi ID tidak menyeberang split
id_split_counts = (
    combined_split_df
    .groupby("id")["evaluated_split"]
    .nunique()
)

cross_split_id_count = int(
    (id_split_counts > 1).sum()
)

if cross_split_id_count != 0:
    raise ValueError(
        f"Ada {cross_split_id_count} ID "
        "yang menyeberang split."
    )


# Validasi leakage group tidak menyeberang
group_split_counts = (
    combined_split_df
    .groupby(
        "leakage_group_id"
    )["evaluated_split"]
    .nunique()
)

cross_split_group_count = int(
    (group_split_counts > 1).sum()
)

if cross_split_group_count != 0:
    raise ValueError(
        f"Ada {cross_split_group_count} "
        "leakage group yang menyeberang split."
    )


# Validasi seluruh gambar tersedia
combined_split_df["resolved_image_path"] = (
    combined_split_df["image_path"]
    .apply(
        lambda relative_path:
        project_root / str(relative_path)
    )
)

combined_split_df["image_exists"] = (
    combined_split_df[
        "resolved_image_path"
    ]
    .apply(lambda path: path.exists())
)

missing_image_count = int(
    (~combined_split_df["image_exists"]).sum()
)

if missing_image_count != 0:
    missing_examples = (
        combined_split_df.loc[
            ~combined_split_df["image_exists"],
            [
                "id",
                "image_path",
                "evaluated_split",
            ]
        ]
        .head(10)
    )

    display(missing_examples)

    raise FileNotFoundError(
        f"Ada {missing_image_count} "
        "gambar yang tidak ditemukan."
    )


# Hitung checksum split manifest
split_manifest_path = (
    split_dir / "split_manifest.csv"
)

with split_manifest_path.open("rb") as file:
    split_manifest_sha256 = (
        hashlib.sha256(
            file.read()
        )
        .hexdigest()
    )

expected_split_sha256 = (
    "767d106d78a12fa02860336ccaba3b262"
    "1fbf4640b02b01035cf3252c3422888"
)

checksum_matches = (
    split_manifest_sha256
    == expected_split_sha256
)

if not checksum_matches:
    raise ValueError(
        "Checksum split manifest berubah.\n"
        f"Expected: {expected_split_sha256}\n"
        f"Actual:   {split_manifest_sha256}"
    )


# Split summary
split_summary_df = pd.DataFrame([
    {
        "split": split_name,
        "rows": len(split_df),
        "percentage": round(
            len(split_df)
            / len(combined_split_df)
            * 100,
            2
        ),
        "unique_ids": int(
            split_df["id"].nunique()
        ),
        "unique_leakage_groups": int(
            split_df[
                "leakage_group_id"
            ].nunique()
        ),
    }
    for split_name, split_df
    in split_frames.items()
])


# Label distribution
label_distribution_df = (
    combined_split_df
    .groupby(
        [
            "evaluated_split",
            "label",
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
    .rename(
        columns={
            "evaluated_split": "split"
        }
    )
)


# Platform × label distribution
platform_label_distribution_df = (
    combined_split_df
    .groupby(
        [
            "evaluated_split",
            "platform",
            "label",
        ]
    )
    .size()
    .reset_index(
        name="count"
    )
    .rename(
        columns={
            "evaluated_split": "split"
        }
    )
)


# Simpan hasil audit
split_summary_path = (
    summary_dir
    / "split_summary.csv"
)

label_distribution_path = (
    summary_dir
    / "split_label_distribution.csv"
)

platform_distribution_path = (
    summary_dir
    / "split_platform_label_distribution.csv"
)

split_validation_path = (
    config_dir
    / "split_validation.json"
)

split_summary_df.to_csv(
    split_summary_path,
    index=False
)

label_distribution_df.to_csv(
    label_distribution_path,
    index=False
)

platform_label_distribution_df.to_csv(
    platform_distribution_path,
    index=False
)

split_validation_report = {
    "total_rows": int(
        len(combined_split_df)
    ),
    "train_rows": int(
        len(train_df)
    ),
    "validation_rows": int(
        len(validation_df)
    ),
    "test_rows": int(
        len(test_df)
    ),
    "cross_split_ids": (
        cross_split_id_count
    ),
    "cross_split_leakage_groups": (
        cross_split_group_count
    ),
    "missing_images": (
        missing_image_count
    ),
    "split_manifest_sha256": (
        split_manifest_sha256
    ),
    "checksum_matches": (
        checksum_matches
    ),
}

with split_validation_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        split_validation_report,
        file,
        indent=2,
        ensure_ascii=False
    )


print(
    "=== FINAL V2 SPLIT VALIDATION ==="
)

display(split_summary_df)

print("\n=== LABEL DISTRIBUTION ===")
display(label_distribution_df)

print(
    "\n=== PLATFORM × LABEL DISTRIBUTION ==="
)
display(platform_label_distribution_df)

print(
    "\nCross-split IDs:",
    cross_split_id_count
)

print(
    "Cross-split leakage groups:",
    cross_split_group_count
)

print(
    "Missing images:",
    missing_image_count
)

print(
    "Split manifest SHA-256:",
    split_manifest_sha256
)

print(
    "Checksum matches:",
    checksum_matches
)

print("\nSaved to:")
print(split_summary_path)
print(label_distribution_path)
print(platform_distribution_path)
print(split_validation_path)

# Publication notebook cell pub-007
clean_embedding_paths = {
    "train_text": (
        clean_embedding_dir
        / "train_multilingual_e5_large.npy"
    ),
    "validation_text": (
        clean_embedding_dir
        / "validation_multilingual_e5_large.npy"
    ),
    "test_text": (
        clean_embedding_dir
        / "test_multilingual_e5_large.npy"
    ),
    "train_image": (
        clean_embedding_dir
        / "train_siglip2_so400m_patch14_384.npy"
    ),
    "validation_image": (
        clean_embedding_dir
        / "validation_siglip2_so400m_patch14_384.npy"
    ),
    "test_image": (
        clean_embedding_dir
        / "test_siglip2_so400m_patch14_384.npy"
    ),
}


expected_embedding_specs = {
    "train_text": {
        "rows": len(train_df),
        "dimensions": 1024,
        "modality": "text",
        "split": "train",
    },
    "validation_text": {
        "rows": len(validation_df),
        "dimensions": 1024,
        "modality": "text",
        "split": "validation",
    },
    "test_text": {
        "rows": len(test_df),
        "dimensions": 1024,
        "modality": "text",
        "split": "test",
    },
    "train_image": {
        "rows": len(train_df),
        "dimensions": 1152,
        "modality": "image",
        "split": "train",
    },
    "validation_image": {
        "rows": len(validation_df),
        "dimensions": 1152,
        "modality": "image",
        "split": "validation",
    },
    "test_image": {
        "rows": len(test_df),
        "dimensions": 1152,
        "modality": "image",
        "split": "test",
    },
}


def calculate_file_sha256(
    file_path,
    chunk_size=1024 * 1024
):
    sha256_hash = hashlib.sha256()

    with file_path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)

            if not chunk:
                break

            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


clean_embeddings = {}
embedding_audit_rows = []

for embedding_name, embedding_path in (
    clean_embedding_paths.items()
):
    if not embedding_path.exists():
        raise FileNotFoundError(
            f"Embedding tidak ditemukan: "
            f"{embedding_path}"
        )

    embedding_array = np.load(
        embedding_path,
        mmap_mode="r"
    )

    expected_spec = expected_embedding_specs[
        embedding_name
    ]

    expected_shape = (
        expected_spec["rows"],
        expected_spec["dimensions"],
    )

    if embedding_array.shape != expected_shape:
        raise ValueError(
            f"Shape {embedding_name} salah: "
            f"{embedding_array.shape} "
            f"!= {expected_shape}"
        )

    finite_values = bool(
        np.isfinite(embedding_array).all()
    )

    if not finite_values:
        raise ValueError(
            f"{embedding_name} mengandung "
            "NaN atau infinity."
        )

    embedding_norms = np.linalg.norm(
        embedding_array.astype(np.float32),
        axis=1
    )

    mean_norm = float(
        embedding_norms.mean()
    )

    min_norm = float(
        embedding_norms.min()
    )

    max_norm = float(
        embedding_norms.max()
    )

    if not 0.98 <= mean_norm <= 1.02:
        raise ValueError(
            f"Mean norm {embedding_name} "
            f"tidak mendekati 1: {mean_norm}"
        )

    file_sha256 = calculate_file_sha256(
        embedding_path
    )

    clean_embeddings[
        embedding_name
    ] = embedding_array

    embedding_audit_rows.append({
        "embedding_name": embedding_name,
        "split": expected_spec["split"],
        "modality": expected_spec["modality"],
        "rows": int(
            embedding_array.shape[0]
        ),
        "dimensions": int(
            embedding_array.shape[1]
        ),
        "dtype": str(
            embedding_array.dtype
        ),
        "finite_values": finite_values,
        "mean_l2_norm": mean_norm,
        "min_l2_norm": min_norm,
        "max_l2_norm": max_norm,
        "file_size_mb": round(
            embedding_path.stat().st_size
            / (1024 ** 2),
            2
        ),
        "sha256": file_sha256,
        "file_path": str(
            embedding_path
        ),
    })


clean_embedding_audit_df = pd.DataFrame(
    embedding_audit_rows
)


clean_embedding_audit_path = (
    summary_dir
    / "clean_embedding_audit.csv"
)

clean_embedding_validation_path = (
    config_dir
    / "clean_embedding_validation.json"
)


clean_embedding_audit_df.to_csv(
    clean_embedding_audit_path,
    index=False
)


clean_embedding_validation_report = {
    "total_embeddings": int(
        len(clean_embedding_audit_df)
    ),
    "all_shapes_valid": True,
    "all_values_finite": bool(
        clean_embedding_audit_df[
            "finite_values"
        ].all()
    ),
    "all_mean_norms_near_one": bool(
        clean_embedding_audit_df[
            "mean_l2_norm"
        ]
        .between(
            0.98,
            1.02
        )
        .all()
    ),
    "text_embedding_dimension": 1024,
    "image_embedding_dimension": 1152,
    "encoder_status": "frozen",
    "embeddings": (
        clean_embedding_audit_df[
            [
                "embedding_name",
                "sha256",
            ]
        ]
        .set_index("embedding_name")[
            "sha256"
        ]
        .to_dict()
    ),
}


with clean_embedding_validation_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        clean_embedding_validation_report,
        file,
        indent=2,
        ensure_ascii=False
    )


print(
    "=== FINAL V2 CLEAN EMBEDDING VALIDATION ==="
)

display(
    clean_embedding_audit_df[
        [
            "embedding_name",
            "split",
            "modality",
            "rows",
            "dimensions",
            "dtype",
            "finite_values",
            "mean_l2_norm",
            "file_size_mb",
        ]
    ]
)

print(
    "\nAll shapes valid:",
    True
)

print(
    "All values finite:",
    clean_embedding_validation_report[
        "all_values_finite"
    ]
)

print(
    "All mean norms near one:",
    clean_embedding_validation_report[
        "all_mean_norms_near_one"
    ]
)

print("\nSaved to:")
print(clean_embedding_audit_path)
print(clean_embedding_validation_path)

# Publication notebook cell pub-009
final_lexicon = [
    "slot",
    "gacor",
    "maxwin",
    "jackpot",
    "scatter",
    "mahjong",
    "pragmatic",
    "togel",
    "casino",
    "freebet",
    "deposit",
    "depo",
    "withdraw",
    "bonus",
    "rtp",
    "bet",
    "jp",
]

development_df = pd.concat(
    [
        train_df.assign(
            development_split="train"
        ),
        validation_df.assign(
            development_split="validation"
        ),
    ],
    ignore_index=True
)

development_captions = (
    development_df["text_caption"]
    .fillna("")
    .astype(str)
)

lexicon_audit_rows = []
compound_candidate_rows = []

for term in final_lexicon:

    # Controlled match:
    # - #slot cocok
    # - slot88 cocok
    # - depo10k cocok
    # - Deportes tidak cocok dengan depo
    # - better tidak cocok dengan bet
    exact_pattern = re.compile(
        rf"(?<![A-Za-z])"
        rf"{re.escape(term)}"
        rf"(?![A-Za-z])",
        flags=re.IGNORECASE
    )

    exact_mask = (
        development_captions
        .str.contains(
            exact_pattern,
            regex=True
        )
    )

    # Broad match hanya untuk menemukan
    # kemungkinan compound hashtag atau false substring.
    broad_mask = (
        development_captions
        .str.contains(
            term,
            case=False,
            regex=False
        )
    )

    exact_df = development_df[
        exact_mask
    ]

    broad_df = development_df[
        broad_mask
    ]

    exact_count = int(
        len(exact_df)
    )

    exact_positive_count = int(
        (exact_df["label"] == 1).sum()
    )

    exact_negative_count = int(
        (exact_df["label"] == 0).sum()
    )

    exact_positive_rate = (
        exact_positive_count / exact_count
        if exact_count > 0
        else 0
    )

    broad_count = int(
        len(broad_df)
    )

    broad_positive_count = int(
        (broad_df["label"] == 1).sum()
    )

    broad_positive_rate = (
        broad_positive_count / broad_count
        if broad_count > 0
        else 0
    )

    possible_compound_count = int(
        (broad_mask & ~exact_mask).sum()
    )

    term_role = (
        "ambiguous_term"
        if term == "jackpot"
        else "core_promotional_term"
    )

    lexicon_audit_rows.append({
        "term": term,
        "term_role": term_role,
        "exact_caption_count": exact_count,
        "exact_positive_count": (
            exact_positive_count
        ),
        "exact_negative_count": (
            exact_negative_count
        ),
        "exact_positive_rate": round(
            exact_positive_rate,
            4
        ),
        "broad_caption_count": broad_count,
        "broad_positive_count": (
            broad_positive_count
        ),
        "broad_positive_rate": round(
            broad_positive_rate,
            4
        ),
        "possible_compound_count": (
            possible_compound_count
        ),
        "lexicon_status": "locked",
    })

    # Ambil bentuk token yang hanya ditemukan
    # melalui broad substring matching.
    possible_compound_df = (
        development_df[
            broad_mask & ~exact_mask
        ]
    )

    for _, row in possible_compound_df.iterrows():

        caption = str(
            row["text_caption"]
        )

        caption_tokens = re.findall(
            r"[A-Za-z0-9_]+",
            caption.lower()
        )

        matching_tokens = sorted({
            token
            for token in caption_tokens
            if term in token
        })

        for token in matching_tokens:
            compound_candidate_rows.append({
                "term": term,
                "token": token,
                "id": row["id"],
                "development_split": (
                    row["development_split"]
                ),
                "platform": row["platform"],
                "label": int(row["label"]),
                "text_caption": caption,
            })


lexicon_audit_df = (
    pd.DataFrame(
        lexicon_audit_rows
    )
    .sort_values(
        [
            "exact_caption_count",
            "exact_positive_rate",
        ],
        ascending=[
            False,
            False,
        ]
    )
    .reset_index(drop=True)
)


compound_candidate_df = pd.DataFrame(
    compound_candidate_rows
)

if not compound_candidate_df.empty:

    compound_form_summary_df = (
        compound_candidate_df
        .groupby(
            [
                "term",
                "token",
            ],
            as_index=False
        )
        .agg(
            caption_count=(
                "id",
                "nunique"
            ),
            positive_count=(
                "label",
                "sum"
            ),
        )
    )

    compound_form_summary_df[
        "negative_count"
    ] = (
        compound_form_summary_df[
            "caption_count"
        ]
        - compound_form_summary_df[
            "positive_count"
        ]
    )

    compound_form_summary_df[
        "positive_rate"
    ] = (
        compound_form_summary_df[
            "positive_count"
        ]
        / compound_form_summary_df[
            "caption_count"
        ]
    )

    compound_form_summary_df = (
        compound_form_summary_df
        .sort_values(
            [
                "term",
                "caption_count",
            ],
            ascending=[
                True,
                False,
            ]
        )
        .reset_index(drop=True)
    )

else:
    compound_form_summary_df = pd.DataFrame(
        columns=[
            "term",
            "token",
            "caption_count",
            "positive_count",
            "negative_count",
            "positive_rate",
        ]
    )


lexicon_audit_path = (
    summary_dir
    / "text_obfuscation_final_lexicon_audit.csv"
)

compound_summary_path = (
    diagnostic_dir
    / "text_obfuscation_compound_candidate_forms.csv"
)

compound_samples_path = (
    diagnostic_dir
    / "text_obfuscation_compound_candidate_samples.csv"
)

lexicon_config_path = (
    config_dir
    / "text_obfuscation_lexicon.json"
)


lexicon_audit_df.to_csv(
    lexicon_audit_path,
    index=False
)

compound_form_summary_df.to_csv(
    compound_summary_path,
    index=False
)

compound_candidate_df.to_csv(
    compound_samples_path,
    index=False
)


lexicon_config = {
    "lexicon_version": "final_v2",
    "selection_source": (
        "train_and_validation_only"
    ),
    "terms": final_lexicon,
    "number_of_terms": len(
        final_lexicon
    ),
    "core_terms": [
        term
        for term in final_lexicon
        if term != "jackpot"
    ],
    "ambiguous_terms": [
        "jackpot"
    ],
    "excluded_terms": {
        "judi": (
            "predominantly negative/contextual"
        ),
        "judol": (
            "predominantly negative/contextual"
        ),
        "sportsbook": (
            "insufficient development support"
        ),
        "bigwin": (
            "development support below 20"
        ),
        "betting": (
            "insufficient development support"
        ),
        "buybonus": (
            "insufficient development support"
        ),
    },
    "lexicon_status": "locked",
}


with lexicon_config_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        lexicon_config,
        file,
        indent=2,
        ensure_ascii=False
    )


print(
    "=== FINAL V2 LEXICON AUDIT ==="
)

display(lexicon_audit_df)

print(
    "\n=== TOP COMPOUND CANDIDATE FORMS ==="
)

display(
    compound_form_summary_df
    .groupby(
        "term",
        group_keys=False
    )
    .head(10)
    .reset_index(drop=True)
)

print(
    "\nLocked terms:",
    len(final_lexicon)
)

print(
    "Development rows:",
    len(development_df)
)

print("\nSaved to:")
print(lexicon_audit_path)
print(compound_summary_path)
print(compound_samples_path)
print(lexicon_config_path)

# Publication notebook cell pub-011
compound_min_support = 5
compound_min_positive_rate = 0.95

exact_only_terms = {
    "jackpot",
    "jp",
    "depo",
}


approved_compound_forms_df = (
    compound_form_summary_df[
        (
            compound_form_summary_df[
                "caption_count"
            ]
            >= compound_min_support
        )
        &
        (
            compound_form_summary_df[
                "positive_rate"
            ]
            >= compound_min_positive_rate
        )
        &
        (
            ~compound_form_summary_df[
                "term"
            ].isin(exact_only_terms)
        )
        &
        (
            ~compound_form_summary_df[
                "token"
            ].isin(final_lexicon)
        )
    ]
    .copy()
    .sort_values(
        [
            "term",
            "caption_count",
            "positive_rate",
            "token",
        ],
        ascending=[
            True,
            False,
            False,
            True,
        ]
    )
    .reset_index(drop=True)
)


approved_compound_tokens = sorted(
    set(
        approved_compound_forms_df[
            "token"
        ].astype(str)
    )
)


approved_compound_summary_df = (
    approved_compound_forms_df
    .groupby(
        "term",
        as_index=False
    )
    .agg(
        approved_compound_count=(
            "token",
            "nunique"
        ),
        covered_caption_count=(
            "caption_count",
            "sum"
        ),
        minimum_positive_rate=(
            "positive_rate",
            "min"
        ),
    )
)


# Term yang tidak memiliki compound approved
missing_compound_terms = sorted(
    set(final_lexicon)
    - set(
        approved_compound_summary_df[
            "term"
        ]
    )
)


matcher_config = {
    "matcher_version": "final_v2",
    "selection_source": (
        "train_and_validation_only"
    ),
    "exact_terms": final_lexicon,
    "exact_only_terms": sorted(
        exact_only_terms
    ),
    "compound_selection": {
        "minimum_caption_support": (
            compound_min_support
        ),
        "minimum_positive_rate": (
            compound_min_positive_rate
        ),
    },
    "approved_compound_tokens": (
        approved_compound_tokens
    ),
    "approved_compound_token_count": (
        len(approved_compound_tokens)
    ),
    "matching_policy": {
        "exact_matching": (
            "letter-boundary controlled matching; "
            "numeric suffixes remain allowed"
        ),
        "compound_matching": (
            "only development-approved full tokens"
        ),
        "case_sensitive": False,
        "test_used_for_rule_selection": False,
    },
    "matcher_status": "locked",
}


approved_compound_path = (
    summary_dir
    / "text_obfuscation_approved_compound_forms.csv"
)

approved_compound_summary_path = (
    summary_dir
    / "text_obfuscation_approved_compound_summary.csv"
)

matcher_config_path = (
    config_dir
    / "text_obfuscation_matcher_config.json"
)


approved_compound_forms_df.to_csv(
    approved_compound_path,
    index=False
)

approved_compound_summary_df.to_csv(
    approved_compound_summary_path,
    index=False
)

with matcher_config_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        matcher_config,
        file,
        indent=2,
        ensure_ascii=False
    )


print(
    "=== FINAL V2 APPROVED COMPOUND MATCHER ==="
)

display(
    approved_compound_summary_df
)

print(
    "\n=== APPROVED COMPOUND FORM EXAMPLES ==="
)

display(
    approved_compound_forms_df
    .groupby(
        "term",
        group_keys=False
    )
    .head(10)
    .reset_index(drop=True)
)


print(
    "\nExact terms:",
    len(final_lexicon)
)

print(
    "Exact-only terms:",
    sorted(exact_only_terms)
)

print(
    "Approved compound tokens:",
    len(approved_compound_tokens)
)

print(
    "Terms without approved compounds:",
    missing_compound_terms
)

print(
    "Test used for matcher selection:",
    False
)

print("\nSaved to:")
print(approved_compound_path)
print(approved_compound_summary_path)
print(matcher_config_path)

# Publication notebook cell pub-013
approved_compound_map = {
    term: set(
        group["token"]
        .astype(str)
        .str.lower()
    )
    for term, group
    in approved_compound_forms_df.groupby("term")
}


def resolve_overlapping_matches(
    candidates
):
    """
    Memilih match tanpa overlap.

    Prioritas:
    1. term yang lebih panjang;
    2. posisi lebih awal;
    3. nama term.
    """

    sorted_candidates = sorted(
        candidates,
        key=lambda item: (
            -(
                item["end"]
                - item["start"]
            ),
            item["start"],
            item["term"],
        )
    )

    selected_matches = []
    occupied_positions = set()

    for candidate in sorted_candidates:
        candidate_positions = set(
            range(
                candidate["start"],
                candidate["end"]
            )
        )

        if (
            occupied_positions
            & candidate_positions
        ):
            continue

        selected_matches.append(
            candidate
        )

        occupied_positions.update(
            candidate_positions
        )

    return sorted(
        selected_matches,
        key=lambda item: item["start"]
    )


def find_final_text_matches(text):
    """
    Menemukan exact term dan approved compound term.
    """

    text = (
        ""
        if pd.isna(text)
        else str(text)
    )

    candidates = []

    # Exact matching:
    # slot88 cocok, Deportes tidak dianggap depo.
    for term in final_lexicon:
        exact_pattern = re.compile(
            rf"(?<![A-Za-z])"
            rf"{re.escape(term)}"
            rf"(?![A-Za-z])",
            flags=re.IGNORECASE
        )

        for match in exact_pattern.finditer(text):
            candidates.append({
                "term": term,
                "match_type": "exact",
                "matched_text": match.group(0),
                "start": match.start(),
                "end": match.end(),
            })

    # Compound matching dilakukan per token utuh.
    token_matches = list(
        re.finditer(
            r"[A-Za-z0-9_]+",
            text
        )
    )

    for token_match in token_matches:
        token_text = token_match.group(0)
        token_lower = token_text.lower()

        for term, approved_tokens in (
            approved_compound_map.items()
        ):
            if token_lower not in approved_tokens:
                continue

            term_pattern = re.compile(
                re.escape(term),
                flags=re.IGNORECASE
            )

            for local_match in (
                term_pattern.finditer(
                    token_text
                )
            ):
                candidates.append({
                    "term": term,
                    "match_type": (
                        "approved_compound"
                    ),
                    "matched_text": (
                        local_match.group(0)
                    ),
                    "start": (
                        token_match.start()
                        + local_match.start()
                    ),
                    "end": (
                        token_match.start()
                        + local_match.end()
                    ),
                    "compound_token": (
                        token_text
                    ),
                })

    return resolve_overlapping_matches(
        candidates
    )


def apply_full_character_period_obfuscation(
    text
):
    """
    Contoh:
    slot   -> s.l.o.t
    gacor  -> g.a.c.o.r
    """

    text = (
        ""
        if pd.isna(text)
        else str(text)
    )

    matches = find_final_text_matches(
        text
    )

    obfuscated_text = text

    # Ganti dari kanan ke kiri agar indeks tidak bergeser.
    for match in reversed(matches):
        original_segment = (
            obfuscated_text[
                match["start"]:
                match["end"]
            ]
        )

        transformed_segment = ".".join(
            original_segment
        )

        obfuscated_text = (
            obfuscated_text[
                :match["start"]
            ]
            + transformed_segment
            + obfuscated_text[
                match["end"]:
            ]
        )

    return {
        "original_text": text,
        "obfuscated_text": obfuscated_text,
        "text_changed": (
            obfuscated_text != text
        ),
        "matched_terms": [
            match["term"]
            for match in matches
        ],
        "unique_matched_terms": sorted({
            match["term"]
            for match in matches
        }),
        "match_types": [
            match["match_type"]
            for match in matches
        ],
        "target_occurrence_count": len(
            matches
        ),
        "matches": matches,
    }


# Unit tests untuk false substring.
matcher_unit_tests = {
    "DeportesEnTikTok": [],
    "better performance": [],
    "alphabet": [],
    "#slot88 gacor": [
        "slot",
        "gacor",
    ],
    "main jackpot sekarang": [
        "jackpot",
    ],
    "freebet tersedia": [
        "freebet",
    ],
}


unit_test_rows = []

for test_text, expected_terms in (
    matcher_unit_tests.items()
):
    test_result = (
        apply_full_character_period_obfuscation(
            test_text
        )
    )

    actual_terms = (
        test_result[
            "unique_matched_terms"
        ]
    )

    passed = (
        actual_terms
        == sorted(expected_terms)
    )

    unit_test_rows.append({
        "input_text": test_text,
        "expected_terms": expected_terms,
        "actual_terms": actual_terms,
        "obfuscated_text": (
            test_result[
                "obfuscated_text"
            ]
        ),
        "passed": passed,
    })


matcher_unit_test_df = pd.DataFrame(
    unit_test_rows
)

if not matcher_unit_test_df[
    "passed"
].all():
    display(matcher_unit_test_df)

    raise ValueError(
        "Final matcher gagal pada unit test."
    )


# Dry run pada train + validation.
development_match_results = (
    development_df["text_caption"]
    .apply(
        apply_full_character_period_obfuscation
    )
)


development_match_audit_df = (
    development_df[
        [
            "id",
            "development_split",
            "platform",
            "label",
            "text_caption",
        ]
    ]
    .copy()
)


development_match_audit_df[
    "obfuscated_text"
] = development_match_results.apply(
    lambda result:
    result["obfuscated_text"]
)

development_match_audit_df[
    "text_changed"
] = development_match_results.apply(
    lambda result:
    result["text_changed"]
)

development_match_audit_df[
    "matched_terms"
] = development_match_results.apply(
    lambda result:
    "|".join(
        result["unique_matched_terms"]
    )
)

development_match_audit_df[
    "target_occurrence_count"
] = development_match_results.apply(
    lambda result:
    result["target_occurrence_count"]
)


# Summary per term menggunakan caption unik,
# bukan penjumlahan token-level count.
development_term_rows = []

for term in final_lexicon:
    term_mask = (
        development_match_results
        .apply(
            lambda result:
            term
            in result[
                "unique_matched_terms"
            ]
        )
    )

    matched_term_df = (
        development_df[
            term_mask
        ]
    )

    caption_count = int(
        len(matched_term_df)
    )

    positive_count = int(
        (
            matched_term_df["label"]
            == 1
        ).sum()
    )

    negative_count = int(
        (
            matched_term_df["label"]
            == 0
        ).sum()
    )

    occurrence_count = int(
        development_match_results[
            term_mask
        ]
        .apply(
            lambda result:
            result[
                "matched_terms"
            ].count(term)
        )
        .sum()
    )

    development_term_rows.append({
        "term": term,
        "unique_caption_count": (
            caption_count
        ),
        "occurrence_count": (
            occurrence_count
        ),
        "positive_count": (
            positive_count
        ),
        "negative_count": (
            negative_count
        ),
        "positive_rate": (
            round(
                positive_count
                / caption_count,
                4
            )
            if caption_count > 0
            else 0
        ),
    })


development_match_summary_df = (
    pd.DataFrame(
        development_term_rows
    )
    .sort_values(
        [
            "unique_caption_count",
            "positive_rate",
        ],
        ascending=[
            False,
            False,
        ]
    )
    .reset_index(drop=True)
)


changed_development_df = (
    development_match_audit_df[
        development_match_audit_df[
            "text_changed"
        ]
    ]
    .copy()
)


matcher_map_path = (
    config_dir
    / "text_obfuscation_matcher_map.json"
)

matcher_unit_test_path = (
    diagnostic_dir
    / "text_obfuscation_matcher_unit_tests.csv"
)

development_match_summary_path = (
    summary_dir
    / "text_obfuscation_development_match_summary.csv"
)

development_match_samples_path = (
    diagnostic_dir
    / "text_obfuscation_development_match_samples.csv"
)


matcher_map_config = {
    "matcher_version": "final_v2",
    "exact_terms": final_lexicon,
    "exact_only_terms": sorted(
        exact_only_terms
    ),
    "approved_compound_map": {
        term: sorted(tokens)
        for term, tokens
        in approved_compound_map.items()
    },
    "overlap_resolution": (
        "longest_match_first"
    ),
    "text_transformation": (
        "full_character_period_separator"
    ),
    "separator": ".",
    "test_used": False,
    "status": "validated_and_locked",
}


with matcher_map_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        matcher_map_config,
        file,
        indent=2,
        ensure_ascii=False
    )


matcher_unit_test_df.to_csv(
    matcher_unit_test_path,
    index=False
)

development_match_summary_df.to_csv(
    development_match_summary_path,
    index=False
)

changed_development_df.to_csv(
    development_match_samples_path,
    index=False
)


print(
    "=== FINAL V2 MATCHER UNIT TESTS ==="
)

display(matcher_unit_test_df)

print(
    "\n=== DEVELOPMENT MATCH SUMMARY ==="
)

display(
    development_match_summary_df
)

print(
    "\nDevelopment captions:",
    len(development_df)
)

print(
    "Changed development captions:",
    len(changed_development_df)
)

print(
    "Development coverage:",
    round(
        len(changed_development_df)
        / len(development_df)
        * 100,
        2
    ),
    "%"
)

print(
    "All unit tests passed:",
    bool(
        matcher_unit_test_df[
            "passed"
        ].all()
    )
)

print(
    "Test used:",
    False
)

print("\nSaved to:")
print(matcher_map_path)
print(matcher_unit_test_path)
print(development_match_summary_path)
print(development_match_samples_path)

# Publication notebook cell pub-015
# Folder manifest teks final
text_manifest_dir = (
    perturbation_manifest_dir
    / "text"
)

text_manifest_dir.mkdir(
    parents=True,
    exist_ok=True
)


# Terapkan matcher final pada test caption
test_obfuscation_results = (
    test_df["text_caption"]
    .apply(
        apply_full_character_period_obfuscation
    )
)


# Manifest utama
test_text_obfuscated_df = (
    test_df[
        [
            "id",
            "platform",
            "label",
            "image_path",
            "text_caption",
        ]
    ]
    .copy()
    .rename(
        columns={
            "text_caption": "original_caption"
        }
    )
)


test_text_obfuscated_df[
    "obfuscated_caption"
] = test_obfuscation_results.apply(
    lambda result:
    result["obfuscated_text"]
)

test_text_obfuscated_df[
    "text_changed"
] = test_obfuscation_results.apply(
    lambda result:
    result["text_changed"]
)

test_text_obfuscated_df[
    "matched_terms"
] = test_obfuscation_results.apply(
    lambda result:
    "|".join(
        result["unique_matched_terms"]
    )
)

test_text_obfuscated_df[
    "matched_terms_json"
] = test_obfuscation_results.apply(
    lambda result:
    json.dumps(
        result["unique_matched_terms"],
        ensure_ascii=False
    )
)

test_text_obfuscated_df[
    "match_types"
] = test_obfuscation_results.apply(
    lambda result:
    "|".join(
        sorted(
            set(
                result["match_types"]
            )
        )
    )
)

test_text_obfuscated_df[
    "target_occurrence_count"
] = test_obfuscation_results.apply(
    lambda result:
    result["target_occurrence_count"]
)

test_text_obfuscated_df[
    "matches_json"
] = test_obfuscation_results.apply(
    lambda result:
    json.dumps(
        result["matches"],
        ensure_ascii=False
    )
)


# Validasi urutan dan identitas baris
if not test_text_obfuscated_df[
    "id"
].reset_index(drop=True).equals(
    test_df["id"].reset_index(drop=True)
):
    raise ValueError(
        "Urutan ID manifest obfuscation berubah."
    )

if len(test_text_obfuscated_df) != len(test_df):
    raise ValueError(
        "Jumlah baris manifest tidak sama "
        "dengan test set."
    )


# Caption yang berubah harus memiliki target
invalid_changed_rows = (
    test_text_obfuscated_df[
        test_text_obfuscated_df[
            "text_changed"
        ]
        &
        (
            test_text_obfuscated_df[
                "target_occurrence_count"
            ]
            == 0
        )
    ]
)

if not invalid_changed_rows.empty:
    raise ValueError(
        "Ada caption berubah tanpa target term."
    )


# Caption yang tidak berubah tidak boleh memiliki target
invalid_unchanged_rows = (
    test_text_obfuscated_df[
        ~test_text_obfuscated_df[
            "text_changed"
        ]
        &
        (
            test_text_obfuscated_df[
                "target_occurrence_count"
            ]
            > 0
        )
    ]
)

if not invalid_unchanged_rows.empty:
    raise ValueError(
        "Ada target term yang terdeteksi "
        "tetapi caption tidak berubah."
    )


# Pastikan target yang telah diserang
# tidak lagi terdeteksi oleh matcher final
remaining_match_results = (
    test_text_obfuscated_df[
        "obfuscated_caption"
    ]
    .apply(
        find_final_text_matches
    )
)

test_text_obfuscated_df[
    "remaining_target_count"
] = (
    remaining_match_results
    .apply(len)
)

remaining_target_total = int(
    test_text_obfuscated_df[
        "remaining_target_count"
    ].sum()
)

if remaining_target_total != 0:
    remaining_examples = (
        test_text_obfuscated_df[
            test_text_obfuscated_df[
                "remaining_target_count"
            ]
            > 0
        ]
        .head(10)
    )

    display(remaining_examples)

    raise ValueError(
        f"Masih ada {remaining_target_total} "
        "target yang belum di-obfuscate."
    )


# Ringkasan coverage per label
text_obfuscation_coverage_df = (
    test_text_obfuscated_df
    .groupby(
        "label",
        as_index=False
    )
    .agg(
        total_captions=(
            "id",
            "count"
        ),
        changed_captions=(
            "text_changed",
            "sum"
        ),
        target_occurrences=(
            "target_occurrence_count",
            "sum"
        ),
    )
)

text_obfuscation_coverage_df[
    "coverage_percentage"
] = (
    text_obfuscation_coverage_df[
        "changed_captions"
    ]
    / text_obfuscation_coverage_df[
        "total_captions"
    ]
    * 100
).round(2)


# Frekuensi per term pada test
test_term_frequency_rows = []

for term in final_lexicon:
    term_caption_mask = (
        test_obfuscation_results
        .apply(
            lambda result:
            term
            in result[
                "unique_matched_terms"
            ]
        )
    )

    term_caption_count = int(
        term_caption_mask.sum()
    )

    term_occurrence_count = int(
        test_obfuscation_results
        .apply(
            lambda result:
            result[
                "matched_terms"
            ].count(term)
        )
        .sum()
    )

    matched_labels = (
        test_df.loc[
            term_caption_mask,
            "label"
        ]
    )

    positive_count = int(
        (matched_labels == 1).sum()
    )

    negative_count = int(
        (matched_labels == 0).sum()
    )

    test_term_frequency_rows.append({
        "term": term,
        "caption_count": (
            term_caption_count
        ),
        "occurrence_count": (
            term_occurrence_count
        ),
        "positive_count": (
            positive_count
        ),
        "negative_count": (
            negative_count
        ),
        "positive_rate": (
            round(
                positive_count
                / term_caption_count,
                4
            )
            if term_caption_count > 0
            else 0
        ),
    })


test_term_frequency_df = (
    pd.DataFrame(
        test_term_frequency_rows
    )
    .sort_values(
        [
            "caption_count",
            "positive_rate",
        ],
        ascending=[
            False,
            False,
        ]
    )
    .reset_index(drop=True)
)


# Statistik keseluruhan
changed_test_count = int(
    test_text_obfuscated_df[
        "text_changed"
    ].sum()
)

total_target_occurrences = int(
    test_text_obfuscated_df[
        "target_occurrence_count"
    ].sum()
)

test_coverage_percentage = round(
    changed_test_count
    / len(test_text_obfuscated_df)
    * 100,
    2
)


# Simpan manifest dan ringkasan
text_manifest_path = (
    text_manifest_dir
    / "test_text_obfuscated_fullchar_period.csv"
)

coverage_summary_path = (
    summary_dir
    / "text_obfuscation_test_coverage.csv"
)

term_frequency_path = (
    summary_dir
    / "text_obfuscation_test_term_frequency.csv"
)

changed_samples_path = (
    diagnostic_dir
    / "text_obfuscation_test_changed_samples.csv"
)

test_report_path = (
    config_dir
    / "text_obfuscation_test_report.json"
)


test_text_obfuscated_df.to_csv(
    text_manifest_path,
    index=False
)

text_obfuscation_coverage_df.to_csv(
    coverage_summary_path,
    index=False
)

test_term_frequency_df.to_csv(
    term_frequency_path,
    index=False
)

test_text_obfuscated_df[
    test_text_obfuscated_df[
        "text_changed"
    ]
].to_csv(
    changed_samples_path,
    index=False
)


# Checksum manifest setelah file disimpan
text_manifest_sha256 = (
    calculate_file_sha256(
        text_manifest_path
    )
)


test_obfuscation_report = {
    "condition_name": (
        "full_character_period_separator"
    ),
    "test_rows": int(
        len(test_text_obfuscated_df)
    ),
    "changed_captions": (
        changed_test_count
    ),
    "coverage_percentage": (
        test_coverage_percentage
    ),
    "target_occurrences_changed": (
        total_target_occurrences
    ),
    "remaining_target_occurrences": (
        remaining_target_total
    ),
    "lexicon_size": len(
        final_lexicon
    ),
    "separator": ".",
    "matcher_status": (
        "validated_and_locked"
    ),
    "manifest_sha256": (
        text_manifest_sha256
    ),
    "test_used_for_rule_selection": False,
}

with test_report_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        test_obfuscation_report,
        file,
        indent=2,
        ensure_ascii=False
    )


print(
    "=== FINAL V2 TEXT OBFUSCATION TEST ==="
)

display(
    text_obfuscation_coverage_df
)

print(
    "\n=== TEST TERM FREQUENCY ==="
)

display(
    test_term_frequency_df
)

print(
    "\n=== CHANGED SAMPLE EXAMPLES ==="
)

display(
    test_text_obfuscated_df[
        test_text_obfuscated_df[
            "text_changed"
        ]
    ][
        [
            "id",
            "label",
            "platform",
            "original_caption",
            "obfuscated_caption",
            "matched_terms",
            "target_occurrence_count",
        ]
    ]
    .head(15)
)

print(
    "\nTest rows:",
    len(test_text_obfuscated_df)
)

print(
    "Changed captions:",
    changed_test_count
)

print(
    "Coverage:",
    test_coverage_percentage,
    "%"
)

print(
    "Target occurrences changed:",
    total_target_occurrences
)

print(
    "Remaining target occurrences:",
    remaining_target_total
)

print(
    "Manifest SHA-256:",
    text_manifest_sha256
)

print("\nSaved to:")
print(text_manifest_path)
print(coverage_summary_path)
print(term_frequency_path)
print(changed_samples_path)
print(test_report_path)

# Publication notebook cell pub-017
image_corruption_config = {
    "version": "final_v2",
    "selection_source": "validation_only",
    "gaussian_blur": {
        "mild_radius": 2,
        "strong_radius": 5,
    },
    "pixelation": {
        "p50_scale": 0.50,
        "p25_scale": 0.25,
        "p12_5_scale": 0.125,
        "downscale_resampling": "BOX",
        "upscale_resampling": "NEAREST",
    },
    "primary_joint_conditions": {
        "J1": {
            "text": "full_character_period_separator",
            "image": "gaussian_blur_r5",
        },
        "J2": {
            "text": "full_character_period_separator",
            "image": "pixelation_p25",
        },
        "J3": {
            "text": "full_character_period_separator",
            "image": "blur_r5_then_pixelation_p25",
        },
    },
    "compound_transform_order": [
        "gaussian_blur_r5",
        "pixelation_p25",
    ],
    "output_format": "PNG",
    "test_used_for_severity_selection": False,
}


def apply_gaussian_blur(
    image,
    radius
):
    return image.filter(
        ImageFilter.GaussianBlur(
            radius=radius
        )
    )


def apply_pixelation(
    image,
    scale
):
    if not 0 < scale < 1:
        raise ValueError(
            "Pixelation scale harus "
            "berada antara 0 dan 1."
        )

    original_width, original_height = (
        image.size
    )

    reduced_width = max(
        1,
        round(
            original_width * scale
        )
    )

    reduced_height = max(
        1,
        round(
            original_height * scale
        )
    )

    reduced_image = image.resize(
        (
            reduced_width,
            reduced_height,
        ),
        resample=Image.Resampling.BOX
    )

    pixelated_image = reduced_image.resize(
        (
            original_width,
            original_height,
        ),
        resample=Image.Resampling.NEAREST
    )

    return pixelated_image


def apply_blur_then_pixelation(
    image,
    blur_radius=5,
    pixelation_scale=0.25
):
    blurred_image = apply_gaussian_blur(
        image=image,
        radius=blur_radius
    )

    return apply_pixelation(
        image=blurred_image,
        scale=pixelation_scale
    )


# Pilih satu validation sample per platform.
# Label dibuat bergantian agar preview tidak
# hanya berasal dari satu kelas.
preview_rows = []

platform_names = sorted(
    validation_df["platform"].unique()
)

for platform_index, platform_name in enumerate(
    platform_names
):
    desired_label = platform_index % 2

    candidate_df = validation_df[
        (
            validation_df["platform"]
            == platform_name
        )
        &
        (
            validation_df["label"]
            == desired_label
        )
    ]

    if candidate_df.empty:
        candidate_df = validation_df[
            validation_df["platform"]
            == platform_name
        ]

    selected_row = candidate_df.sample(
        n=1,
        random_state=(
            SEED + platform_index
        )
    ).iloc[0]

    preview_rows.append(
        selected_row
    )


image_corruption_preview_df = pd.DataFrame(
    preview_rows
).reset_index(drop=True)


preview_conditions = [
    "Original",
    "Blur r2",
    "Blur r5",
    "Pixel 50%",
    "Pixel 25%",
    "Pixel 12.5%",
    "Blur r5 + Pixel 25%",
]


figure, axes = plt.subplots(
    nrows=len(
        image_corruption_preview_df
    ),
    ncols=len(
        preview_conditions
    ),
    figsize=(21, 10)
)


for row_index, row in (
    image_corruption_preview_df.iterrows()
):
    original_path = (
        project_root
        / str(row["image_path"])
    )

    if not original_path.exists():
        raise FileNotFoundError(
            f"Gambar preview tidak ada: "
            f"{original_path}"
        )

    with Image.open(
        original_path
    ) as loaded_image:
        original_image = (
            loaded_image
            .convert("RGB")
        )

    transformed_images = [
        original_image,
        apply_gaussian_blur(
            original_image,
            radius=2
        ),
        apply_gaussian_blur(
            original_image,
            radius=5
        ),
        apply_pixelation(
            original_image,
            scale=0.50
        ),
        apply_pixelation(
            original_image,
            scale=0.25
        ),
        apply_pixelation(
            original_image,
            scale=0.125
        ),
        apply_blur_then_pixelation(
            original_image,
            blur_radius=5,
            pixelation_scale=0.25
        ),
    ]

    for column_index, (
        condition_name,
        transformed_image
    ) in enumerate(
        zip(
            preview_conditions,
            transformed_images
        )
    ):
        if (
            transformed_image.size
            != original_image.size
        ):
            raise ValueError(
                "Ukuran output corruption "
                "berubah."
            )

        axes[
            row_index,
            column_index
        ].imshow(
            transformed_image
        )

        axes[
            row_index,
            column_index
        ].axis("off")

        if row_index == 0:
            axes[
                row_index,
                column_index
            ].set_title(
                condition_name,
                fontsize=11
            )

        if column_index == 0:
            axes[
                row_index,
                column_index
            ].set_ylabel(
                (
                    f"{row['platform']}\n"
                    f"ID {int(row['id'])}\n"
                    f"Label {int(row['label'])}"
                ),
                fontsize=10
            )


figure.suptitle(
    (
        "Final v2 Image-Corruption "
        "Severity Preview — Validation Set"
    ),
    fontsize=14
)

figure.tight_layout()


preview_figure_path = (
    diagnostic_dir
    / "image_corruption_severity_preview.png"
)

figure.savefig(
    preview_figure_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()


preview_samples_path = (
    diagnostic_dir
    / "image_corruption_preview_samples.csv"
)

image_corruption_preview_df[
    [
        "id",
        "platform",
        "label",
        "image_path",
        "text_caption",
    ]
].to_csv(
    preview_samples_path,
    index=False
)


image_corruption_config_path = (
    config_dir
    / "image_corruption_config.json"
)

with image_corruption_config_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        image_corruption_config,
        file,
        indent=2,
        ensure_ascii=False
    )


print(
    "=== FINAL V2 IMAGE CORRUPTION PREVIEW ==="
)

display(
    image_corruption_preview_df[
        [
            "id",
            "platform",
            "label",
            "image_path",
        ]
    ]
)

print(
    "\nValidation samples used:",
    len(
        image_corruption_preview_df
    )
)

print(
    "Test used for severity selection:",
    False
)

print(
    "Primary pixelation scale:",
    image_corruption_config[
        "pixelation"
    ][
        "p25_scale"
    ]
)

print(
    "Compound order:",
    " -> ".join(
        image_corruption_config[
            "compound_transform_order"
        ]
    )
)

print("\nSaved to:")
print(preview_figure_path)
print(preview_samples_path)
print(image_corruption_config_path)

# Publication notebook cell pub-018
# Revisi primary pixelation berdasarkan
# visual audit validation set

PRIMARY_PIXELATION_SCALE = 0.125

image_corruption_config[
    "pixelation"
][
    "primary_joint_scale"
] = PRIMARY_PIXELATION_SCALE

image_corruption_config[
    "primary_joint_conditions"
][
    "J2"
][
    "image"
] = "pixelation_p12_5"

image_corruption_config[
    "primary_joint_conditions"
][
    "J3"
][
    "image"
] = (
    "blur_r5_then_pixelation_p12_5"
)

image_corruption_config[
    "compound_transform_order"
] = [
    "gaussian_blur_r5",
    "pixelation_p12_5",
]

image_corruption_config[
    "severity_selection_reason"
] = (
    "Validation preview showed that "
    "pixelation scales 0.50 and 0.25 were "
    "too mild, while 0.125 visibly degraded "
    "fine details without removing the main "
    "semantic content."
)

with image_corruption_config_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        image_corruption_config,
        file,
        indent=2,
        ensure_ascii=False
    )

print(
    "Primary pixelation scale:",
    PRIMARY_PIXELATION_SCALE
)

print(
    "J2:",
    image_corruption_config[
        "primary_joint_conditions"
    ]["J2"]
)

print(
    "J3:",
    image_corruption_config[
        "primary_joint_conditions"
    ]["J3"]
)

print(
    "Config updated:",
    image_corruption_config_path
)

# Publication notebook cell pub-020
# Folder manifest image
image_manifest_dir = (
    perturbation_manifest_dir
    / "image"
)

image_manifest_dir.mkdir(
    parents=True,
    exist_ok=True
)


# Folder kondisi compound baru.
# Folder compound p25 lama tidak dipakai lagi.
compound_p12_5_dir = (
    perturbation_image_dir
    / "blur_r5_pixelation_p12_5"
)

compound_p12_5_dir.mkdir(
    parents=True,
    exist_ok=True
)


# False berarti:
# - file valid yang sudah ada akan digunakan kembali;
# - proses dapat dilanjutkan setelah berhenti;
# - hasil tidak ditimpa tanpa sengaja.
OVERWRITE_IMAGE_CORRUPTIONS = False


image_corruption_conditions = {
    "gaussian_blur_r2": {
        "output_dir": (
            perturbation_image_dir
            / "gaussian_blur_r2"
        ),
        "blur_radius": 2,
        "pixelation_scale": None,
        "transform_order": [
            "gaussian_blur_r2"
        ],
    },
    "gaussian_blur_r5": {
        "output_dir": (
            perturbation_image_dir
            / "gaussian_blur_r5"
        ),
        "blur_radius": 5,
        "pixelation_scale": None,
        "transform_order": [
            "gaussian_blur_r5"
        ],
    },
    "pixelation_p50": {
        "output_dir": (
            perturbation_image_dir
            / "pixelation_p50"
        ),
        "blur_radius": None,
        "pixelation_scale": 0.50,
        "transform_order": [
            "pixelation_p50"
        ],
    },
    "pixelation_p25": {
        "output_dir": (
            perturbation_image_dir
            / "pixelation_p25"
        ),
        "blur_radius": None,
        "pixelation_scale": 0.25,
        "transform_order": [
            "pixelation_p25"
        ],
    },
    "pixelation_p12_5": {
        "output_dir": (
            perturbation_image_dir
            / "pixelation_p12_5"
        ),
        "blur_radius": None,
        "pixelation_scale": 0.125,
        "transform_order": [
            "pixelation_p12_5"
        ],
    },
    "blur_r5_pixelation_p12_5": {
        "output_dir": (
            compound_p12_5_dir
        ),
        "blur_radius": 5,
        "pixelation_scale": 0.125,
        "transform_order": [
            "gaussian_blur_r5",
            "pixelation_p12_5",
        ],
    },
}


for condition_config in (
    image_corruption_conditions.values()
):
    condition_config[
        "output_dir"
    ].mkdir(
        parents=True,
        exist_ok=True
    )


def format_image_id(sample_id):
    """
    Menghindari nama seperti image_10.0.png.
    """

    try:
        numeric_id = float(sample_id)

        if numeric_id.is_integer():
            return str(int(numeric_id))

    except (
        TypeError,
        ValueError,
    ):
        pass

    return str(sample_id)


def transform_image_for_condition(
    image,
    condition_name
):
    """
    Semua kondisi selalu dimulai dari clean image.
    """

    if condition_name == "gaussian_blur_r2":
        return apply_gaussian_blur(
            image=image,
            radius=2
        )

    if condition_name == "gaussian_blur_r5":
        return apply_gaussian_blur(
            image=image,
            radius=5
        )

    if condition_name == "pixelation_p50":
        return apply_pixelation(
            image=image,
            scale=0.50
        )

    if condition_name == "pixelation_p25":
        return apply_pixelation(
            image=image,
            scale=0.25
        )

    if condition_name == "pixelation_p12_5":
        return apply_pixelation(
            image=image,
            scale=0.125
        )

    if (
        condition_name
        == "blur_r5_pixelation_p12_5"
    ):
        return apply_blur_then_pixelation(
            image=image,
            blur_radius=5,
            pixelation_scale=0.125
        )

    raise ValueError(
        f"Condition tidak dikenal: "
        f"{condition_name}"
    )


generation_start_time = time.time()

condition_manifest_frames = []
generation_summary_rows = []


for condition_index, (
    condition_name,
    condition_config
) in enumerate(
    image_corruption_conditions.items(),
    start=1
):
    print(
        f"\n[{condition_index}/"
        f"{len(image_corruption_conditions)}] "
        f"Generating: {condition_name}"
    )

    condition_start_time = time.time()

    output_dir = condition_config[
        "output_dir"
    ]

    manifest_rows = []

    generated_count = 0
    reused_count = 0
    total_output_bytes = 0

    for row_number, (
        _,
        row
    ) in enumerate(
        test_df.iterrows(),
        start=1
    ):
        sample_id = format_image_id(
            row["id"]
        )

        original_image_path = (
            project_root
            / str(row["image_path"])
        )

        output_image_path = (
            output_dir
            / f"image_{sample_id}.png"
        )

        if not original_image_path.exists():
            raise FileNotFoundError(
                "Original image tidak ditemukan: "
                f"{original_image_path}"
            )

        generated_now = False

        if (
            output_image_path.exists()
            and not OVERWRITE_IMAGE_CORRUPTIONS
        ):
            with Image.open(
                original_image_path
            ) as original_loaded:
                original_size = (
                    original_loaded.size
                )

            with Image.open(
                output_image_path
            ) as output_loaded:
                output_size = (
                    output_loaded.size
                )

            if output_size != original_size:
                raise ValueError(
                    "Ukuran file existing tidak valid: "
                    f"{output_image_path}"
                )

            reused_count += 1

        else:
            with Image.open(
                original_image_path
            ) as loaded_image:
                original_image = (
                    loaded_image
                    .convert("RGB")
                )

            original_size = (
                original_image.size
            )

            transformed_image = (
                transform_image_for_condition(
                    image=original_image,
                    condition_name=condition_name
                )
            )

            if (
                transformed_image.size
                != original_size
            ):
                raise ValueError(
                    "Ukuran gambar berubah pada "
                    f"ID {row['id']} "
                    f"condition {condition_name}"
                )

            transformed_image.save(
                output_image_path,
                format="PNG",
                compress_level=1
            )

            generated_count += 1
            generated_now = True

        if not output_image_path.exists():
            raise FileNotFoundError(
                "Output gagal dibuat: "
                f"{output_image_path}"
            )

        with Image.open(
            output_image_path
        ) as validated_output:
            validated_mode = (
                validated_output.mode
            )

            validated_width, (
                validated_height
            ) = validated_output.size

        output_sha256 = (
            calculate_file_sha256(
                output_image_path
            )
        )

        output_file_size = (
            output_image_path.stat().st_size
        )

        total_output_bytes += (
            output_file_size
        )

        manifest_rows.append({
            "id": row["id"],
            "platform": row["platform"],
            "label": int(row["label"]),
            "condition": condition_name,
            "original_image_path": str(
                row["image_path"]
            ),
            "perturbed_image_path": str(
                output_image_path.relative_to(
                    project_root
                )
            ),
            "width": int(
                validated_width
            ),
            "height": int(
                validated_height
            ),
            "mode": validated_mode,
            "blur_radius": (
                condition_config[
                    "blur_radius"
                ]
            ),
            "pixelation_scale": (
                condition_config[
                    "pixelation_scale"
                ]
            ),
            "transform_order": (
                " -> ".join(
                    condition_config[
                        "transform_order"
                    ]
                )
            ),
            "output_format": "PNG",
            "output_file_size_bytes": int(
                output_file_size
            ),
            "output_sha256": output_sha256,
            "generated_now": generated_now,
        })

        if (
            row_number % 100 == 0
            or row_number == len(test_df)
        ):
            print(
                f"  {row_number}/"
                f"{len(test_df)}"
            )

    condition_manifest_df = (
        pd.DataFrame(
            manifest_rows
        )
    )

    if len(condition_manifest_df) != len(test_df):
        raise ValueError(
            f"Jumlah manifest {condition_name} "
            "tidak sama dengan test set."
        )

    if (
        condition_manifest_df[
            "id"
        ]
        .reset_index(drop=True)
        .equals(
            test_df["id"]
            .reset_index(drop=True)
        )
        is False
    ):
        raise ValueError(
            f"Urutan ID berubah pada "
            f"{condition_name}."
        )

    duplicate_output_paths = int(
        condition_manifest_df[
            "perturbed_image_path"
        ]
        .duplicated()
        .sum()
    )

    if duplicate_output_paths != 0:
        raise ValueError(
            f"Ada duplicate output path pada "
            f"{condition_name}."
        )

    condition_manifest_path = (
        image_manifest_dir
        / f"test_{condition_name}.csv"
    )

    condition_manifest_df.to_csv(
        condition_manifest_path,
        index=False
    )

    manifest_sha256 = (
        calculate_file_sha256(
            condition_manifest_path
        )
    )

    condition_duration_seconds = (
        time.time()
        - condition_start_time
    )

    generation_summary_rows.append({
        "condition": condition_name,
        "rows": int(
            len(condition_manifest_df)
        ),
        "generated_now": int(
            generated_count
        ),
        "reused_existing": int(
            reused_count
        ),
        "missing_outputs": 0,
        "blur_radius": (
            condition_config[
                "blur_radius"
            ]
        ),
        "pixelation_scale": (
            condition_config[
                "pixelation_scale"
            ]
        ),
        "transform_order": (
            " -> ".join(
                condition_config[
                    "transform_order"
                ]
            )
        ),
        "total_output_size_mb": round(
            total_output_bytes
            / (1024 ** 2),
            2
        ),
        "duration_seconds": round(
            condition_duration_seconds,
            2
        ),
        "manifest_sha256": (
            manifest_sha256
        ),
        "manifest_path": str(
            condition_manifest_path
        ),
    })

    condition_manifest_frames.append(
        condition_manifest_df
    )


image_corruption_generation_summary_df = (
    pd.DataFrame(
        generation_summary_rows
    )
)


combined_image_manifest_df = pd.concat(
    condition_manifest_frames,
    ignore_index=True
)


expected_combined_rows = (
    len(test_df)
    * len(
        image_corruption_conditions
    )
)

if (
    len(combined_image_manifest_df)
    != expected_combined_rows
):
    raise ValueError(
        "Jumlah combined manifest tidak valid: "
        f"{len(combined_image_manifest_df)} "
        f"!= {expected_combined_rows}"
    )


combined_manifest_path = (
    image_manifest_dir
    / "test_all_image_corruptions.csv"
)

generation_summary_path = (
    summary_dir
    / "image_corruption_generation_summary.csv"
)

generation_report_path = (
    config_dir
    / "image_corruption_generation_report.json"
)


combined_image_manifest_df.to_csv(
    combined_manifest_path,
    index=False
)

image_corruption_generation_summary_df.to_csv(
    generation_summary_path,
    index=False
)


total_generation_seconds = (
    time.time()
    - generation_start_time
)


generation_report = {
    "experiment_version": (
        experiment_version
    ),
    "test_rows_per_condition": int(
        len(test_df)
    ),
    "number_of_conditions": int(
        len(image_corruption_conditions)
    ),
    "expected_output_images": int(
        expected_combined_rows
    ),
    "combined_manifest_rows": int(
        len(combined_image_manifest_df)
    ),
    "output_format": "PNG",
    "overwrite_existing": bool(
        OVERWRITE_IMAGE_CORRUPTIONS
    ),
    "total_duration_seconds": round(
        total_generation_seconds,
        2
    ),
    "conditions": (
        image_corruption_generation_summary_df
        .set_index("condition")
        .to_dict(orient="index")
    ),
}


with generation_report_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        generation_report,
        file,
        indent=2,
        ensure_ascii=False
    )


print(
    "\n=== FINAL V2 IMAGE CORRUPTION GENERATION ==="
)

display(
    image_corruption_generation_summary_df[
        [
            "condition",
            "rows",
            "generated_now",
            "reused_existing",
            "missing_outputs",
            "blur_radius",
            "pixelation_scale",
            "total_output_size_mb",
            "duration_seconds",
        ]
    ]
)

print(
    "\nConditions:",
    len(image_corruption_conditions)
)

print(
    "Images per condition:",
    len(test_df)
)

print(
    "Combined manifest rows:",
    len(combined_image_manifest_df)
)

print(
    "Total duration:",
    round(
        total_generation_seconds,
        2
    ),
    "seconds"
)

print("\nSaved to:")
print(combined_manifest_path)
print(generation_summary_path)
print(generation_report_path)

# Publication notebook cell pub-022
qc_condition_files = {
    "gaussian_blur_r5": (
        image_manifest_dir
        / "test_gaussian_blur_r5.csv"
    ),
    "pixelation_p12_5": (
        image_manifest_dir
        / "test_pixelation_p12_5.csv"
    ),
    "blur_r5_pixelation_p12_5": (
        image_manifest_dir
        / "test_blur_r5_pixelation_p12_5.csv"
    ),
}


# Load manifest utama
qc_manifest_frames = {}

for condition_name, manifest_path in (
    qc_condition_files.items()
):
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest tidak ditemukan: {manifest_path}"
        )

    condition_manifest_df = pd.read_csv(
        manifest_path
    )

    if len(condition_manifest_df) != len(test_df):
        raise ValueError(
            f"Jumlah baris {condition_name} tidak valid."
        )

    qc_manifest_frames[
        condition_name
    ] = condition_manifest_df


# Gabungkan path ketiga kondisi
qc_full_df = test_df[
    [
        "id",
        "platform",
        "label",
        "image_path",
        "text_caption",
    ]
].copy()


for condition_name, condition_manifest_df in (
    qc_manifest_frames.items()
):
    condition_path_df = (
        condition_manifest_df[
            [
                "id",
                "perturbed_image_path",
            ]
        ]
        .rename(
            columns={
                "perturbed_image_path": (
                    f"{condition_name}_path"
                )
            }
        )
    )

    qc_full_df = qc_full_df.merge(
        condition_path_df,
        on="id",
        how="left",
        validate="one_to_one"
    )


# Ambil satu sampel untuk setiap platform × label
qc_sample_rows = []

for platform_name in sorted(
    test_df["platform"].unique()
):
    for label_value in [0, 1]:
        candidate_df = qc_full_df[
            (
                qc_full_df["platform"]
                == platform_name
            )
            &
            (
                qc_full_df["label"]
                == label_value
            )
        ]

        if candidate_df.empty:
            continue

        selected_row = candidate_df.sample(
            n=1,
            random_state=(
                SEED
                + label_value
                + len(qc_sample_rows)
            )
        ).iloc[0]

        qc_sample_rows.append(
            selected_row
        )


image_corruption_qc_samples_df = (
    pd.DataFrame(qc_sample_rows)
    .reset_index(drop=True)
)


def load_rgb_image(relative_path):
    image_path = (
        project_root
        / str(relative_path)
    )

    if not image_path.exists():
        raise FileNotFoundError(
            f"Gambar tidak ditemukan: {image_path}"
        )

    with Image.open(image_path) as loaded_image:
        rgb_image = loaded_image.convert("RGB")

    return rgb_image


def resize_for_comparison(
    image,
    size=(384, 384)
):
    return image.resize(
        size,
        resample=Image.Resampling.BICUBIC
    )


def calculate_image_mae(
    first_image,
    second_image
):
    first_array = np.asarray(
        resize_for_comparison(
            first_image
        ),
        dtype=np.float32
    )

    second_array = np.asarray(
        resize_for_comparison(
            second_image
        ),
        dtype=np.float32
    )

    return float(
        np.mean(
            np.abs(
                first_array
                - second_array
            )
        )
    )


qc_metric_rows = []

preview_condition_columns = {
    "Original": "image_path",
    "Blur r5": "gaussian_blur_r5_path",
    "Pixel 12.5%": "pixelation_p12_5_path",
    "Blur r5 → Pixel 12.5%": (
        "blur_r5_pixelation_p12_5_path"
    ),
}


figure, axes = plt.subplots(
    nrows=len(
        image_corruption_qc_samples_df
    ),
    ncols=len(
        preview_condition_columns
    ),
    figsize=(16, 24)
)


for row_index, row in (
    image_corruption_qc_samples_df.iterrows()
):
    original_image = load_rgb_image(
        row["image_path"]
    )

    blur_image = load_rgb_image(
        row["gaussian_blur_r5_path"]
    )

    pixel_image = load_rgb_image(
        row["pixelation_p12_5_path"]
    )

    compound_image = load_rgb_image(
        row[
            "blur_r5_pixelation_p12_5_path"
        ]
    )

    image_map = {
        "Original": original_image,
        "Blur r5": blur_image,
        "Pixel 12.5%": pixel_image,
        "Blur r5 → Pixel 12.5%": (
            compound_image
        ),
    }

    original_size = original_image.size

    for condition_name, condition_image in (
        image_map.items()
    ):
        if condition_image.size != original_size:
            raise ValueError(
                f"Ukuran berubah pada ID {row['id']} "
                f"condition {condition_name}."
            )

    blur_mae = calculate_image_mae(
        original_image,
        blur_image
    )

    pixel_mae = calculate_image_mae(
        original_image,
        pixel_image
    )

    compound_mae = calculate_image_mae(
        original_image,
        compound_image
    )

    compound_vs_blur_mae = (
        calculate_image_mae(
            compound_image,
            blur_image
        )
    )

    compound_vs_pixel_mae = (
        calculate_image_mae(
            compound_image,
            pixel_image
        )
    )

    qc_metric_rows.append({
        "id": row["id"],
        "platform": row["platform"],
        "label": int(row["label"]),
        "original_width": int(
            original_size[0]
        ),
        "original_height": int(
            original_size[1]
        ),
        "blur_r5_mae": blur_mae,
        "pixelation_p12_5_mae": (
            pixel_mae
        ),
        "compound_mae": compound_mae,
        "compound_vs_blur_mae": (
            compound_vs_blur_mae
        ),
        "compound_vs_pixel_mae": (
            compound_vs_pixel_mae
        ),
        "automatic_integrity_passed": bool(
            blur_mae > 0
            and pixel_mae > 0
            and compound_mae > 0
            and compound_vs_blur_mae > 0
            and compound_vs_pixel_mae > 0
        ),
    })

    for column_index, (
        condition_name,
        condition_image
    ) in enumerate(
        image_map.items()
    ):
        axes[
            row_index,
            column_index
        ].imshow(
            condition_image
        )

        axes[
            row_index,
            column_index
        ].axis("off")

        if row_index == 0:
            axes[
                row_index,
                column_index
            ].set_title(
                condition_name,
                fontsize=12
            )

        if column_index == 0:
            axes[
                row_index,
                column_index
            ].set_ylabel(
                (
                    f"{row['platform']}\n"
                    f"ID {int(row['id'])}\n"
                    f"Label {int(row['label'])}"
                ),
                fontsize=10
            )


figure.suptitle(
    (
        "Final v2 Image-Corruption Quality Control\n"
        "Balanced by Platform and Label"
    ),
    fontsize=15
)

figure.tight_layout()


qc_figure_path = (
    diagnostic_dir
    / "image_corruption_final_qc_preview.png"
)

figure.savefig(
    qc_figure_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()


image_corruption_qc_metrics_df = (
    pd.DataFrame(
        qc_metric_rows
    )
)


if not image_corruption_qc_metrics_df[
    "automatic_integrity_passed"
].all():
    display(
        image_corruption_qc_metrics_df
    )

    raise ValueError(
        "Ada image corruption yang gagal "
        "automatic integrity check."
    )


qc_summary_df = pd.DataFrame([
    {
        "condition": "gaussian_blur_r5",
        "mean_mae_from_clean": (
            image_corruption_qc_metrics_df[
                "blur_r5_mae"
            ].mean()
        ),
        "minimum_mae_from_clean": (
            image_corruption_qc_metrics_df[
                "blur_r5_mae"
            ].min()
        ),
    },
    {
        "condition": "pixelation_p12_5",
        "mean_mae_from_clean": (
            image_corruption_qc_metrics_df[
                "pixelation_p12_5_mae"
            ].mean()
        ),
        "minimum_mae_from_clean": (
            image_corruption_qc_metrics_df[
                "pixelation_p12_5_mae"
            ].min()
        ),
    },
    {
        "condition": (
            "blur_r5_pixelation_p12_5"
        ),
        "mean_mae_from_clean": (
            image_corruption_qc_metrics_df[
                "compound_mae"
            ].mean()
        ),
        "minimum_mae_from_clean": (
            image_corruption_qc_metrics_df[
                "compound_mae"
            ].min()
        ),
    },
])


qc_samples_path = (
    diagnostic_dir
    / "image_corruption_final_qc_samples.csv"
)

qc_metrics_path = (
    diagnostic_dir
    / "image_corruption_final_qc_metrics.csv"
)

qc_summary_path = (
    summary_dir
    / "image_corruption_final_qc_summary.csv"
)


image_corruption_qc_samples_df.to_csv(
    qc_samples_path,
    index=False
)

image_corruption_qc_metrics_df.to_csv(
    qc_metrics_path,
    index=False
)

qc_summary_df.to_csv(
    qc_summary_path,
    index=False
)


print(
    "=== FINAL V2 IMAGE CORRUPTION QC ==="
)

display(
    image_corruption_qc_metrics_df
)

print(
    "\n=== QC CONDITION SUMMARY ==="
)

display(
    qc_summary_df
)

print(
    "\nAutomatic integrity passed:",
    bool(
        image_corruption_qc_metrics_df[
            "automatic_integrity_passed"
        ].all()
    )
)

print(
    "Visual samples:",
    len(
        image_corruption_qc_samples_df
    )
)

print("\nSaved to:")
print(qc_figure_path)
print(qc_samples_path)
print(qc_metrics_path)
print(qc_summary_path)

# Publication notebook cell pub-024
TEXT_ENCODER_NAME = (
    "intfloat/multilingual-e5-large"
)

text_obfuscated_embedding_path = (
    text_embedding_dir
    / (
        "test_text_obfuscated_fullchar_period_"
        "multilingual_e5_large.npy"
    )
)

OVERWRITE_TEXT_OBFUSCATED_EMBEDDING = False


# Validasi urutan manifest terhadap test set
if not (
    test_text_obfuscated_df["id"]
    .reset_index(drop=True)
    .equals(
        test_df["id"]
        .reset_index(drop=True)
    )
):
    raise ValueError(
        "Urutan manifest text obfuscation "
        "berbeda dari test set."
    )


# Caption input E5
obfuscated_text_inputs = (
    test_text_obfuscated_df[
        "obfuscated_caption"
    ]
    .fillna("")
    .astype(str)
    .apply(
        lambda caption:
        f"query: {caption}"
    )
    .tolist()
)


if (
    text_obfuscated_embedding_path.exists()
    and not OVERWRITE_TEXT_OBFUSCATED_EMBEDDING
):
    print(
        "Menggunakan embedding existing:"
    )
    print(
        text_obfuscated_embedding_path
    )

    test_text_obfuscated_embeddings = (
        np.load(
            text_obfuscated_embedding_path
        )
    )

    embedding_generated_now = False

else:
    text_embedding_start_time = (
        time.time()
    )

    text_encoder = SentenceTransformer(
        TEXT_ENCODER_NAME,
        device=str(device)
    )

    test_text_obfuscated_embeddings = (
        text_encoder.encode(
            obfuscated_text_inputs,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        .astype(np.float32)
    )

    np.save(
        text_obfuscated_embedding_path,
        test_text_obfuscated_embeddings
    )

    embedding_duration_seconds = (
        time.time()
        - text_embedding_start_time
    )

    embedding_generated_now = True

    del text_encoder

    gc.collect()

    if device.type == "mps":
        torch.mps.empty_cache()

    elif device.type == "cuda":
        torch.cuda.empty_cache()


# Validasi shape
expected_text_embedding_shape = (
    len(test_df),
    1024,
)

if (
    test_text_obfuscated_embeddings.shape
    != expected_text_embedding_shape
):
    raise ValueError(
        "Shape text-obfuscated embedding salah: "
        f"{test_text_obfuscated_embeddings.shape} "
        f"!= {expected_text_embedding_shape}"
    )


# Validasi nilai
all_values_finite = bool(
    np.isfinite(
        test_text_obfuscated_embeddings
    ).all()
)

if not all_values_finite:
    raise ValueError(
        "Text-obfuscated embedding "
        "mengandung NaN atau infinity."
    )


# Validasi normalisasi
text_obfuscated_norms = np.linalg.norm(
    test_text_obfuscated_embeddings,
    axis=1
)

mean_text_obfuscated_norm = float(
    text_obfuscated_norms.mean()
)

minimum_text_obfuscated_norm = float(
    text_obfuscated_norms.min()
)

maximum_text_obfuscated_norm = float(
    text_obfuscated_norms.max()
)

if not (
    0.98
    <= mean_text_obfuscated_norm
    <= 1.02
):
    raise ValueError(
        "Mean norm text-obfuscated embedding "
        "tidak mendekati 1."
    )


# Bandingkan dengan clean test embeddings
clean_test_text_embeddings = np.asarray(
    clean_embeddings["test_text"],
    dtype=np.float32
)

row_cosine_similarity = np.sum(
    clean_test_text_embeddings
    * test_text_obfuscated_embeddings,
    axis=1
)

changed_mask = (
    test_text_obfuscated_df[
        "text_changed"
    ]
    .to_numpy(dtype=bool)
)

unchanged_mask = ~changed_mask


changed_mean_cosine = float(
    row_cosine_similarity[
        changed_mask
    ].mean()
)

unchanged_mean_cosine = float(
    row_cosine_similarity[
        unchanged_mask
    ].mean()
)

changed_minimum_cosine = float(
    row_cosine_similarity[
        changed_mask
    ].min()
)

unchanged_minimum_cosine = float(
    row_cosine_similarity[
        unchanged_mask
    ].min()
)


# Caption yang tidak diubah seharusnya
# menghasilkan embedding hampir identik.
if unchanged_mean_cosine < 0.999:
    raise ValueError(
        "Embedding caption unchanged tidak "
        "cukup konsisten dengan clean embedding: "
        f"{unchanged_mean_cosine}"
    )


text_embedding_sha256 = (
    calculate_file_sha256(
        text_obfuscated_embedding_path
    )
)


text_embedding_audit_df = pd.DataFrame([
    {
        "embedding_name": (
            "test_text_obfuscated_"
            "fullchar_period"
        ),
        "rows": int(
            test_text_obfuscated_embeddings
            .shape[0]
        ),
        "dimensions": int(
            test_text_obfuscated_embeddings
            .shape[1]
        ),
        "dtype": str(
            test_text_obfuscated_embeddings
            .dtype
        ),
        "finite_values": (
            all_values_finite
        ),
        "mean_l2_norm": (
            mean_text_obfuscated_norm
        ),
        "minimum_l2_norm": (
            minimum_text_obfuscated_norm
        ),
        "maximum_l2_norm": (
            maximum_text_obfuscated_norm
        ),
        "changed_caption_count": int(
            changed_mask.sum()
        ),
        "unchanged_caption_count": int(
            unchanged_mask.sum()
        ),
        "changed_mean_cosine_vs_clean": (
            changed_mean_cosine
        ),
        "changed_minimum_cosine_vs_clean": (
            changed_minimum_cosine
        ),
        "unchanged_mean_cosine_vs_clean": (
            unchanged_mean_cosine
        ),
        "unchanged_minimum_cosine_vs_clean": (
            unchanged_minimum_cosine
        ),
        "generated_now": (
            embedding_generated_now
        ),
        "file_size_mb": round(
            text_obfuscated_embedding_path
            .stat()
            .st_size
            / (1024 ** 2),
            2
        ),
        "sha256": (
            text_embedding_sha256
        ),
        "file_path": str(
            text_obfuscated_embedding_path
        ),
    }
])


text_embedding_audit_path = (
    summary_dir
    / (
        "text_obfuscated_embedding_"
        "audit.csv"
    )
)

text_embedding_report_path = (
    config_dir
    / (
        "text_obfuscated_embedding_"
        "report.json"
    )
)


text_embedding_audit_df.to_csv(
    text_embedding_audit_path,
    index=False
)


text_embedding_report = {
    "encoder": TEXT_ENCODER_NAME,
    "encoder_status": "frozen",
    "prefix": "query: ",
    "normalize_embeddings": True,
    "batch_size": 32,
    "shape": list(
        test_text_obfuscated_embeddings
        .shape
    ),
    "changed_caption_count": int(
        changed_mask.sum()
    ),
    "unchanged_caption_count": int(
        unchanged_mask.sum()
    ),
    "mean_l2_norm": (
        mean_text_obfuscated_norm
    ),
    "changed_mean_cosine_vs_clean": (
        changed_mean_cosine
    ),
    "unchanged_mean_cosine_vs_clean": (
        unchanged_mean_cosine
    ),
    "generated_now": (
        embedding_generated_now
    ),
    "sha256": (
        text_embedding_sha256
    ),
}


with text_embedding_report_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        text_embedding_report,
        file,
        indent=2,
        ensure_ascii=False
    )


print(
    "=== FINAL V2 TEXT-OBFUSCATED "
    "EMBEDDING ==="
)

display(
    text_embedding_audit_df[
        [
            "rows",
            "dimensions",
            "dtype",
            "finite_values",
            "mean_l2_norm",
            "changed_caption_count",
            "unchanged_caption_count",
            "changed_mean_cosine_vs_clean",
            "unchanged_mean_cosine_vs_clean",
            "generated_now",
            "file_size_mb",
        ]
    ]
)

print(
    "\nShape:",
    test_text_obfuscated_embeddings.shape
)

print(
    "Mean norm:",
    mean_text_obfuscated_norm
)

print(
    "Changed mean cosine vs clean:",
    changed_mean_cosine
)

print(
    "Unchanged mean cosine vs clean:",
    unchanged_mean_cosine
)

print(
    "SHA-256:",
    text_embedding_sha256
)

print("\nSaved to:")
print(text_obfuscated_embedding_path)
print(text_embedding_audit_path)
print(text_embedding_report_path)

# Publication notebook cell pub-026
IMAGE_ENCODER_NAME = (
    "google/siglip2-so400m-patch14-384"
)

if device.type in {
    "mps",
    "cuda",
}:
    IMAGE_MODEL_DTYPE = torch.float16
else:
    IMAGE_MODEL_DTYPE = torch.float32


# Load processor
siglip_processor = (
    AutoProcessor.from_pretrained(
        IMAGE_ENCODER_NAME
    )
)


# Kompatibel dengan beberapa versi Transformers
try:
    siglip_model = (
        AutoModel.from_pretrained(
            IMAGE_ENCODER_NAME,
            dtype=IMAGE_MODEL_DTYPE
        )
    )

except TypeError:
    siglip_model = (
        AutoModel.from_pretrained(
            IMAGE_ENCODER_NAME,
            torch_dtype=IMAGE_MODEL_DTYPE
        )
    )


siglip_model = (
    siglip_model
    .to(device)
    .eval()
)


def move_siglip_inputs_to_device(
    processor_inputs
):
    moved_inputs = {}

    for input_name, input_tensor in (
        processor_inputs.items()
    ):
        if torch.is_floating_point(
            input_tensor
        ):
            moved_inputs[input_name] = (
                input_tensor.to(
                    device=device,
                    dtype=IMAGE_MODEL_DTYPE
                )
            )

        else:
            moved_inputs[input_name] = (
                input_tensor.to(
                    device=device
                )
            )

    return moved_inputs


def extract_siglip_feature_tensor(
    model_output
):
    """
    Menangani perbedaan bentuk output
    pada beberapa versi Transformers.
    """

    if isinstance(
        model_output,
        torch.Tensor
    ):
        return model_output

    if hasattr(
        model_output,
        "pooler_output"
    ):
        return model_output.pooler_output

    if hasattr(
        model_output,
        "image_embeds"
    ):
        return model_output.image_embeds

    raise TypeError(
        "Format output get_image_features "
        "tidak dikenali."
    )


def encode_images_with_siglip(
    image_paths
):
    images = []

    for relative_path in image_paths:
        absolute_path = (
            project_root
            / str(relative_path)
        )

        if not absolute_path.exists():
            raise FileNotFoundError(
                f"Gambar tidak ditemukan: "
                f"{absolute_path}"
            )

        with Image.open(
            absolute_path
        ) as loaded_image:
            images.append(
                loaded_image
                .convert("RGB")
                .copy()
            )

    processor_inputs = (
        siglip_processor(
            images=images,
            return_tensors="pt"
        )
    )

    model_inputs = (
        move_siglip_inputs_to_device(
            processor_inputs
        )
    )

    with torch.inference_mode():
        model_output = (
            siglip_model
            .get_image_features(
                **model_inputs
            )
        )

        feature_tensor = (
            extract_siglip_feature_tensor(
                model_output
            )
        )

        normalized_features = (
            F.normalize(
                feature_tensor.float(),
                p=2,
                dim=1
            )
        )

    return (
        normalized_features
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# Satu sampel untuk setiap platform × label
siglip_validation_rows = []

for platform_name in sorted(
    test_df["platform"].unique()
):
    for label_value in [0, 1]:

        candidate_df = test_df[
            (
                test_df["platform"]
                == platform_name
            )
            &
            (
                test_df["label"]
                == label_value
            )
        ]

        if candidate_df.empty:
            continue

        selected_row = (
            candidate_df
            .sample(
                n=1,
                random_state=(
                    SEED
                    + len(
                        siglip_validation_rows
                    )
                )
            )
            .iloc[0]
        )

        siglip_validation_rows.append(
            selected_row
        )


siglip_validation_samples_df = (
    pd.DataFrame(
        siglip_validation_rows
    )
    .reset_index(drop=True)
)


# Ekstrak ulang clean features
reproduced_clean_embeddings = (
    encode_images_with_siglip(
        siglip_validation_samples_df[
            "image_path"
        ].tolist()
    )
)


expected_validation_shape = (
    len(
        siglip_validation_samples_df
    ),
    1152,
)

if (
    reproduced_clean_embeddings.shape
    != expected_validation_shape
):
    raise ValueError(
        "Shape reproduced SigLIP embedding "
        "tidak valid: "
        f"{reproduced_clean_embeddings.shape}"
    )


# Ambil posisi tiap ID pada test set
test_id_to_position = {
    sample_id: position
    for position, sample_id
    in enumerate(
        test_df["id"].tolist()
    )
}

sample_positions = [
    test_id_to_position[
        sample_id
    ]
    for sample_id
    in siglip_validation_samples_df[
        "id"
    ].tolist()
]


clean_reference_embeddings = (
    np.asarray(
        clean_embeddings[
            "test_image"
        ][
            sample_positions
        ],
        dtype=np.float32
    )
)


# Normalisasi ulang sebagai pengaman
clean_reference_embeddings = (
    clean_reference_embeddings
    / np.linalg.norm(
        clean_reference_embeddings,
        axis=1,
        keepdims=True
    )
)


reproduction_cosine = np.sum(
    clean_reference_embeddings
    * reproduced_clean_embeddings,
    axis=1
)


reproduced_norms = np.linalg.norm(
    reproduced_clean_embeddings,
    axis=1
)


siglip_reproduction_audit_df = (
    siglip_validation_samples_df[
        [
            "id",
            "platform",
            "label",
            "image_path",
        ]
    ]
    .copy()
)

siglip_reproduction_audit_df[
    "reproduced_l2_norm"
] = reproduced_norms

siglip_reproduction_audit_df[
    "cosine_vs_saved_clean"
] = reproduction_cosine


mean_reproduction_cosine = float(
    reproduction_cosine.mean()
)

minimum_reproduction_cosine = float(
    reproduction_cosine.min()
)

mean_reproduced_norm = float(
    reproduced_norms.mean()
)


# Toleransi kecil untuk inference float16
if mean_reproduction_cosine < 0.995:
    raise ValueError(
        "Mean cosine reproduksi SigLIP "
        "terlalu rendah: "
        f"{mean_reproduction_cosine}"
    )

if minimum_reproduction_cosine < 0.990:
    raise ValueError(
        "Minimum cosine reproduksi SigLIP "
        "terlalu rendah: "
        f"{minimum_reproduction_cosine}"
    )

if not (
    0.98
    <= mean_reproduced_norm
    <= 1.02
):
    raise ValueError(
        "Norm reproduced SigLIP embedding "
        "tidak mendekati 1."
    )


siglip_reproduction_audit_path = (
    diagnostic_dir
    / "siglip2_clean_reproduction_audit.csv"
)

siglip_reproduction_report_path = (
    config_dir
    / "siglip2_clean_reproduction_report.json"
)


siglip_reproduction_audit_df.to_csv(
    siglip_reproduction_audit_path,
    index=False
)


siglip_reproduction_report = {
    "encoder": IMAGE_ENCODER_NAME,
    "encoder_status": "frozen",
    "device": str(device),
    "model_dtype": str(
        IMAGE_MODEL_DTYPE
    ),
    "sample_count": int(
        len(
            siglip_reproduction_audit_df
        )
    ),
    "embedding_dimension": 1152,
    "mean_reproduced_l2_norm": (
        mean_reproduced_norm
    ),
    "mean_cosine_vs_saved_clean": (
        mean_reproduction_cosine
    ),
    "minimum_cosine_vs_saved_clean": (
        minimum_reproduction_cosine
    ),
    "validation_passed": True,
}


with siglip_reproduction_report_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        siglip_reproduction_report,
        file,
        indent=2,
        ensure_ascii=False
    )


print(
    "=== FINAL V2 SIGLIP 2 "
    "CLEAN REPRODUCTION ==="
)

display(
    siglip_reproduction_audit_df
)

print(
    "\nModel dtype:",
    IMAGE_MODEL_DTYPE
)

print(
    "Reproduced shape:",
    reproduced_clean_embeddings.shape
)

print(
    "Mean reproduced norm:",
    mean_reproduced_norm
)

print(
    "Mean cosine vs saved clean:",
    mean_reproduction_cosine
)

print(
    "Minimum cosine vs saved clean:",
    minimum_reproduction_cosine
)

print(
    "Validation passed:",
    True
)

print("\nSaved to:")
print(siglip_reproduction_audit_path)
print(siglip_reproduction_report_path)

# Publication notebook cell pub-028
IMAGE_EMBEDDING_BATCH_SIZE = 16
OVERWRITE_IMAGE_EMBEDDINGS = False


image_embedding_conditions = {
    "gaussian_blur_r2": {
        "manifest_path": (
            image_manifest_dir
            / "test_gaussian_blur_r2.csv"
        ),
        "embedding_path": (
            image_embedding_dir
            / "test_siglip2_gaussian_blur_r2.npy"
        ),
    },
    "gaussian_blur_r5": {
        "manifest_path": (
            image_manifest_dir
            / "test_gaussian_blur_r5.csv"
        ),
        "embedding_path": (
            image_embedding_dir
            / "test_siglip2_gaussian_blur_r5.npy"
        ),
    },
    "pixelation_p50": {
        "manifest_path": (
            image_manifest_dir
            / "test_pixelation_p50.csv"
        ),
        "embedding_path": (
            image_embedding_dir
            / "test_siglip2_pixelation_p50.npy"
        ),
    },
    "pixelation_p25": {
        "manifest_path": (
            image_manifest_dir
            / "test_pixelation_p25.csv"
        ),
        "embedding_path": (
            image_embedding_dir
            / "test_siglip2_pixelation_p25.npy"
        ),
    },
    "pixelation_p12_5": {
        "manifest_path": (
            image_manifest_dir
            / "test_pixelation_p12_5.csv"
        ),
        "embedding_path": (
            image_embedding_dir
            / "test_siglip2_pixelation_p12_5.npy"
        ),
    },
    "blur_r5_pixelation_p12_5": {
        "manifest_path": (
            image_manifest_dir
            / "test_blur_r5_pixelation_p12_5.csv"
        ),
        "embedding_path": (
            image_embedding_dir
            / (
                "test_siglip2_"
                "blur_r5_pixelation_p12_5.npy"
            )
        ),
    },
}


def encode_siglip_paths_in_batches(
    relative_image_paths,
    batch_size=16
):
    embedding_batches = []

    total_images = len(
        relative_image_paths
    )

    for batch_start in range(
        0,
        total_images,
        batch_size
    ):
        batch_end = min(
            batch_start + batch_size,
            total_images
        )

        batch_paths = (
            relative_image_paths[
                batch_start:batch_end
            ]
        )

        batch_embeddings = (
            encode_images_with_siglip(
                batch_paths
            )
        )

        embedding_batches.append(
            batch_embeddings
        )

        print(
            f"  {batch_end}/{total_images}",
            end="\r"
        )

    print(
        f"  {total_images}/{total_images}"
    )

    return np.concatenate(
        embedding_batches,
        axis=0
    ).astype(np.float32)


clean_test_image_embeddings = np.asarray(
    clean_embeddings["test_image"],
    dtype=np.float32
)

clean_test_image_embeddings = (
    clean_test_image_embeddings
    / np.linalg.norm(
        clean_test_image_embeddings,
        axis=1,
        keepdims=True
    )
)


image_embedding_audit_rows = []
perturbed_image_embeddings = {}

all_embedding_start_time = time.time()


for condition_index, (
    condition_name,
    condition_config
) in enumerate(
    image_embedding_conditions.items(),
    start=1
):
    print(
        f"\n[{condition_index}/"
        f"{len(image_embedding_conditions)}] "
        f"{condition_name}"
    )

    condition_start_time = time.time()

    manifest_path = condition_config[
        "manifest_path"
    ]

    embedding_path = condition_config[
        "embedding_path"
    ]

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest tidak ditemukan: "
            f"{manifest_path}"
        )

    condition_manifest_df = pd.read_csv(
        manifest_path
    )

    if len(condition_manifest_df) != len(
        test_df
    ):
        raise ValueError(
            f"Jumlah baris manifest "
            f"{condition_name} tidak valid."
        )

    manifest_ids = (
        condition_manifest_df["id"]
        .apply(format_image_id)
        .tolist()
    )

    test_ids = (
        test_df["id"]
        .apply(format_image_id)
        .tolist()
    )

    if manifest_ids != test_ids:
        raise ValueError(
            f"Urutan ID berubah pada "
            f"{condition_name}."
        )

    relative_image_paths = (
        condition_manifest_df[
            "perturbed_image_path"
        ]
        .astype(str)
        .tolist()
    )

    missing_paths = [
        relative_path
        for relative_path
        in relative_image_paths
        if not (
            project_root
            / relative_path
        ).exists()
    ]

    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} gambar "
            f"{condition_name} tidak ditemukan."
        )

    if (
        embedding_path.exists()
        and not OVERWRITE_IMAGE_EMBEDDINGS
    ):
        print(
            "  Menggunakan embedding existing."
        )

        condition_embeddings = np.load(
            embedding_path
        )

        generated_now = False

    else:
        condition_embeddings = (
            encode_siglip_paths_in_batches(
                relative_image_paths,
                batch_size=(
                    IMAGE_EMBEDDING_BATCH_SIZE
                )
            )
        )

        np.save(
            embedding_path,
            condition_embeddings
        )

        generated_now = True

    expected_shape = (
        len(test_df),
        1152,
    )

    if (
        condition_embeddings.shape
        != expected_shape
    ):
        raise ValueError(
            f"Shape {condition_name} salah: "
            f"{condition_embeddings.shape} "
            f"!= {expected_shape}"
        )

    finite_values = bool(
        np.isfinite(
            condition_embeddings
        ).all()
    )

    if not finite_values:
        raise ValueError(
            f"{condition_name} mengandung "
            "NaN atau infinity."
        )

    condition_norms = np.linalg.norm(
        condition_embeddings,
        axis=1
    )

    mean_norm = float(
        condition_norms.mean()
    )

    minimum_norm = float(
        condition_norms.min()
    )

    maximum_norm = float(
        condition_norms.max()
    )

    if not 0.98 <= mean_norm <= 1.02:
        raise ValueError(
            f"Mean norm {condition_name} "
            f"tidak mendekati 1."
        )

    cosine_vs_clean = np.sum(
        clean_test_image_embeddings
        * condition_embeddings,
        axis=1
    )

    mean_cosine_vs_clean = float(
        cosine_vs_clean.mean()
    )

    minimum_cosine_vs_clean = float(
        cosine_vs_clean.min()
    )

    maximum_cosine_vs_clean = float(
        cosine_vs_clean.max()
    )

    embedding_sha256 = (
        calculate_file_sha256(
            embedding_path
        )
    )

    condition_duration = (
        time.time()
        - condition_start_time
    )

    perturbed_image_embeddings[
        condition_name
    ] = condition_embeddings

    image_embedding_audit_rows.append({
        "condition": condition_name,
        "rows": int(
            condition_embeddings.shape[0]
        ),
        "dimensions": int(
            condition_embeddings.shape[1]
        ),
        "dtype": str(
            condition_embeddings.dtype
        ),
        "finite_values": finite_values,
        "mean_l2_norm": mean_norm,
        "minimum_l2_norm": minimum_norm,
        "maximum_l2_norm": maximum_norm,
        "mean_cosine_vs_clean": (
            mean_cosine_vs_clean
        ),
        "minimum_cosine_vs_clean": (
            minimum_cosine_vs_clean
        ),
        "maximum_cosine_vs_clean": (
            maximum_cosine_vs_clean
        ),
        "generated_now": generated_now,
        "batch_size": (
            IMAGE_EMBEDDING_BATCH_SIZE
        ),
        "duration_seconds": round(
            condition_duration,
            2
        ),
        "file_size_mb": round(
            embedding_path.stat().st_size
            / (1024 ** 2),
            2
        ),
        "sha256": embedding_sha256,
        "file_path": str(
            embedding_path
        ),
    })

    gc.collect()

    if device.type == "mps":
        torch.mps.empty_cache()

    elif device.type == "cuda":
        torch.cuda.empty_cache()


image_embedding_audit_df = (
    pd.DataFrame(
        image_embedding_audit_rows
    )
)


total_embedding_duration = (
    time.time()
    - all_embedding_start_time
)


image_embedding_audit_path = (
    summary_dir
    / "perturbed_image_embedding_audit.csv"
)

image_embedding_report_path = (
    config_dir
    / "perturbed_image_embedding_report.json"
)


image_embedding_audit_df.to_csv(
    image_embedding_audit_path,
    index=False
)


image_embedding_report = {
    "encoder": IMAGE_ENCODER_NAME,
    "encoder_status": "frozen",
    "device": str(device),
    "model_dtype": str(
        IMAGE_MODEL_DTYPE
    ),
    "batch_size": (
        IMAGE_EMBEDDING_BATCH_SIZE
    ),
    "number_of_conditions": int(
        len(image_embedding_audit_df)
    ),
    "rows_per_condition": int(
        len(test_df)
    ),
    "embedding_dimension": 1152,
    "all_values_finite": bool(
        image_embedding_audit_df[
            "finite_values"
        ].all()
    ),
    "all_mean_norms_near_one": bool(
        image_embedding_audit_df[
            "mean_l2_norm"
        ]
        .between(
            0.98,
            1.02
        )
        .all()
    ),
    "total_duration_seconds": round(
        total_embedding_duration,
        2
    ),
    "conditions": (
        image_embedding_audit_df
        .set_index("condition")
        .to_dict(orient="index")
    ),
}


with image_embedding_report_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        image_embedding_report,
        file,
        indent=2,
        ensure_ascii=False
    )


print(
    "\n=== FINAL V2 PERTURBED "
    "IMAGE EMBEDDINGS ==="
)

display(
    image_embedding_audit_df[
        [
            "condition",
            "rows",
            "dimensions",
            "dtype",
            "finite_values",
            "mean_l2_norm",
            "mean_cosine_vs_clean",
            "minimum_cosine_vs_clean",
            "generated_now",
            "duration_seconds",
            "file_size_mb",
        ]
    ]
)

print(
    "\nAll values finite:",
    image_embedding_report[
        "all_values_finite"
    ]
)

print(
    "All mean norms near one:",
    image_embedding_report[
        "all_mean_norms_near_one"
    ]
)

print(
    "Total duration:",
    round(
        total_embedding_duration,
        2
    ),
    "seconds"
)

print("\nSaved to:")
print(image_embedding_audit_path)
print(image_embedding_report_path)

# Publication notebook cell pub-030
# Lepaskan encoder SigLIP dari memory karena
# seluruh image embeddings sudah selesai dibuat.
if "siglip_model" in globals():
    del siglip_model

if "siglip_processor" in globals():
    del siglip_processor

gc.collect()

if device.type == "mps":
    torch.mps.empty_cache()

elif device.type == "cuda":
    torch.cuda.empty_cache()


# Konfigurasi training final
TRAINING_SEEDS = [
    42,
    52,
    62,
    72,
    82,
]

BATCH_SIZE = 64
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5

LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

HIDDEN_DIM = 256
PROJECTION_DIM = 512
DROPOUT_RATE = 0.30

THRESHOLD_CANDIDATES = np.round(
    np.arange(
        0.05,
        0.951,
        0.01
    ),
    2
)


FINAL_MODEL_NAMES = [
    "text_only_mlp",
    "image_only_mlp",
    "average_logit_fusion",
    "learnable_weighted_logit_fusion",
    "raw_concatenation_fusion",
    "projection_fusion",
    "contrastive_similarity_fusion",
]


training_protocol = {
    "experiment_version": experiment_version,
    "training_data": (
        "clean_train_embeddings_only"
    ),
    "validation_data": (
        "clean_validation_embeddings"
    ),
    "test_data": (
        "clean_and_frozen_perturbed_test_sets"
    ),
    "encoder_status": "frozen",
    "text_encoder": (
        "intfloat/multilingual-e5-large"
    ),
    "image_encoder": (
        "google/siglip2-so400m-patch14-384"
    ),
    "text_embedding_dimension": 1024,
    "image_embedding_dimension": 1152,
    "random_seeds": TRAINING_SEEDS,
    "batch_size": BATCH_SIZE,
    "maximum_epochs": MAX_EPOCHS,
    "early_stopping": {
        "monitor": "validation_loss",
        "patience": (
            EARLY_STOPPING_PATIENCE
        ),
        "restore_best_weights": True,
    },
    "optimizer": "AdamW",
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "loss": "BCEWithLogitsLoss",
    "class_weighting": None,
    "hidden_dimension": HIDDEN_DIM,
    "projection_dimension": (
        PROJECTION_DIM
    ),
    "dropout_rate": DROPOUT_RATE,
    "threshold_selection": {
        "source": (
            "clean_validation_only"
        ),
        "objective": (
            "maximum_positive_class_f1"
        ),
        "minimum_threshold": float(
            THRESHOLD_CANDIDATES.min()
        ),
        "maximum_threshold": float(
            THRESHOLD_CANDIDATES.max()
        ),
        "step": 0.01,
    },
    "primary_metric": (
        "positive_class_f1"
    ),
    "secondary_metrics": [
        "accuracy",
        "precision",
        "recall",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "confusion_matrix",
    ],
    "models": FINAL_MODEL_NAMES,
    "test_time_retraining": False,
    "perturbation_specific_threshold": False,
    "protocol_status": "locked",
}


training_protocol_path = (
    config_dir
    / "final_training_protocol.json"
)

with training_protocol_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        training_protocol,
        file,
        indent=2,
        ensure_ascii=False
    )


# Label arrays
y_train = (
    train_df["label"]
    .astype(np.float32)
    .to_numpy()
)

y_validation = (
    validation_df["label"]
    .astype(np.float32)
    .to_numpy()
)

y_test = (
    test_df["label"]
    .astype(np.float32)
    .to_numpy()
)


# Clean embedding arrays
X_train_text = np.asarray(
    clean_embeddings["train_text"],
    dtype=np.float32
)

X_validation_text = np.asarray(
    clean_embeddings[
        "validation_text"
    ],
    dtype=np.float32
)

X_test_text = np.asarray(
    clean_embeddings["test_text"],
    dtype=np.float32
)

X_train_image = np.asarray(
    clean_embeddings["train_image"],
    dtype=np.float32
)

X_validation_image = np.asarray(
    clean_embeddings[
        "validation_image"
    ],
    dtype=np.float32
)

X_test_image = np.asarray(
    clean_embeddings["test_image"],
    dtype=np.float32
)


# Validasi jumlah baris
array_shape_rows = [
    {
        "array_name": "X_train_text",
        "rows": X_train_text.shape[0],
        "dimensions": X_train_text.shape[1],
        "expected_rows": len(y_train),
    },
    {
        "array_name": (
            "X_validation_text"
        ),
        "rows": (
            X_validation_text.shape[0]
        ),
        "dimensions": (
            X_validation_text.shape[1]
        ),
        "expected_rows": len(
            y_validation
        ),
    },
    {
        "array_name": "X_test_text",
        "rows": X_test_text.shape[0],
        "dimensions": X_test_text.shape[1],
        "expected_rows": len(y_test),
    },
    {
        "array_name": "X_train_image",
        "rows": X_train_image.shape[0],
        "dimensions": (
            X_train_image.shape[1]
        ),
        "expected_rows": len(y_train),
    },
    {
        "array_name": (
            "X_validation_image"
        ),
        "rows": (
            X_validation_image.shape[0]
        ),
        "dimensions": (
            X_validation_image.shape[1]
        ),
        "expected_rows": len(
            y_validation
        ),
    },
    {
        "array_name": "X_test_image",
        "rows": X_test_image.shape[0],
        "dimensions": (
            X_test_image.shape[1]
        ),
        "expected_rows": len(y_test),
    },
]


training_input_audit_df = pd.DataFrame(
    array_shape_rows
)

training_input_audit_df[
    "row_count_valid"
] = (
    training_input_audit_df["rows"]
    == training_input_audit_df[
        "expected_rows"
    ]
)


if not training_input_audit_df[
    "row_count_valid"
].all():
    display(training_input_audit_df)

    raise ValueError(
        "Jumlah baris training input "
        "tidak konsisten."
    )


training_input_audit_path = (
    summary_dir
    / "final_training_input_audit.csv"
)

training_input_audit_df.to_csv(
    training_input_audit_path,
    index=False
)


print(
    "=== FINAL V2 TRAINING PROTOCOL ==="
)

display(training_input_audit_df)

print(
    "\nTraining seeds:",
    TRAINING_SEEDS
)

print(
    "Models:",
    len(FINAL_MODEL_NAMES)
)

print(
    "Batch size:",
    BATCH_SIZE
)

print(
    "Maximum epochs:",
    MAX_EPOCHS
)

print(
    "Early stopping monitor:",
    training_protocol[
        "early_stopping"
    ]["monitor"]
)

print(
    "Threshold source:",
    training_protocol[
        "threshold_selection"
    ]["source"]
)

print(
    "Protocol status:",
    training_protocol[
        "protocol_status"
    ]
)

print("\nSaved to:")
print(training_protocol_path)
print(training_input_audit_path)

# Publication notebook cell pub-032
def set_training_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_tensor_loader(
    feature_arrays,
    labels,
    batch_size,
    shuffle,
    seed,
):
    feature_tensors = [
        torch.from_numpy(
            np.asarray(
                feature_array,
                dtype=np.float32
            )
        )
        for feature_array in feature_arrays
    ]

    label_tensor = torch.from_numpy(
        np.asarray(
            labels,
            dtype=np.float32
        )
    )

    dataset = TensorDataset(
        *feature_tensors,
        label_tensor
    )

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=(
            generator
            if shuffle
            else None
        ),
        num_workers=0,
        pin_memory=(
            device.type == "cuda"
        ),
        drop_last=False,
    )


def move_batch_to_device(batch):
    *feature_tensors, label_tensor = batch

    feature_tensors = [
        feature_tensor.to(
            device=device,
            dtype=torch.float32
        )
        for feature_tensor in feature_tensors
    ]

    label_tensor = label_tensor.to(
        device=device,
        dtype=torch.float32
    )

    return feature_tensors, label_tensor


def count_trainable_parameters(model):
    return int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )


def train_binary_model(
    model,
    train_loader,
    validation_loader,
    learning_rate,
    weight_decay,
    maximum_epochs,
    patience,
):
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    loss_function = (
        torch.nn.BCEWithLogitsLoss()
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_state_dict = None

    history_rows = []

    for epoch in range(
        1,
        maximum_epochs + 1
    ):
        model.train()

        total_train_loss = 0.0
        total_train_samples = 0

        for batch in train_loader:
            features, labels = (
                move_batch_to_device(batch)
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                *features
            ).reshape(-1)

            loss = loss_function(
                logits,
                labels
            )

            loss.backward()
            optimizer.step()

            batch_size = int(
                labels.shape[0]
            )

            total_train_loss += (
                float(loss.item())
                * batch_size
            )

            total_train_samples += (
                batch_size
            )

        mean_train_loss = (
            total_train_loss
            / total_train_samples
        )

        model.eval()

        total_validation_loss = 0.0
        total_validation_samples = 0

        with torch.inference_mode():
            for batch in validation_loader:
                features, labels = (
                    move_batch_to_device(
                        batch
                    )
                )

                logits = model(
                    *features
                ).reshape(-1)

                loss = loss_function(
                    logits,
                    labels
                )

                batch_size = int(
                    labels.shape[0]
                )

                total_validation_loss += (
                    float(loss.item())
                    * batch_size
                )

                total_validation_samples += (
                    batch_size
                )

        mean_validation_loss = (
            total_validation_loss
            / total_validation_samples
        )

        improved = (
            mean_validation_loss
            < best_validation_loss
            - 1e-6
        )

        if improved:
            best_validation_loss = (
                mean_validation_loss
            )

            best_epoch = epoch

            best_state_dict = copy.deepcopy(
                {
                    key: value
                    .detach()
                    .cpu()
                    .clone()
                    for key, value
                    in model.state_dict().items()
                }
            )

            epochs_without_improvement = 0

        else:
            epochs_without_improvement += 1

        history_rows.append({
            "epoch": epoch,
            "train_loss": (
                mean_train_loss
            ),
            "validation_loss": (
                mean_validation_loss
            ),
            "is_best_epoch": improved,
        })

        if (
            epochs_without_improvement
            >= patience
        ):
            break

    if best_state_dict is None:
        raise RuntimeError(
            "Training tidak menghasilkan "
            "best model state."
        )

    model.load_state_dict(
        best_state_dict
    )

    model = model.to(device)
    model.eval()

    training_history_df = pd.DataFrame(
        history_rows
    )

    training_summary = {
        "best_epoch": int(
            best_epoch
        ),
        "best_validation_loss": float(
            best_validation_loss
        ),
        "epochs_completed": int(
            len(training_history_df)
        ),
        "early_stopped": bool(
            len(training_history_df)
            < maximum_epochs
        ),
    }

    return (
        model,
        training_history_df,
        training_summary,
    )


def predict_binary_logits(
    model,
    data_loader,
):
    model.eval()

    collected_logits = []
    collected_labels = []

    with torch.inference_mode():
        for batch in data_loader:
            features, labels = (
                move_batch_to_device(batch)
            )

            logits = model(
                *features
            ).reshape(-1)

            collected_logits.append(
                logits.cpu().numpy()
            )

            collected_labels.append(
                labels.cpu().numpy()
            )

    logits_array = np.concatenate(
        collected_logits
    ).astype(np.float32)

    labels_array = np.concatenate(
        collected_labels
    ).astype(np.float32)

    return logits_array, labels_array


def sigmoid_numpy(logits):
    logits = np.asarray(
        logits,
        dtype=np.float64
    )

    return 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                logits,
                -50,
                50
            )
        )
    )


def select_validation_threshold(
    labels,
    probabilities,
    threshold_candidates,
):
    threshold_rows = []

    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    for threshold in threshold_candidates:
        predictions = (
            probabilities >= threshold
        ).astype(np.int64)

        threshold_rows.append({
            "threshold": float(
                threshold
            ),
            "f1": float(
                f1_score(
                    labels,
                    predictions,
                    zero_division=0
                )
            ),
            "precision": float(
                precision_score(
                    labels,
                    predictions,
                    zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    labels,
                    predictions,
                    zero_division=0
                )
            ),
            "accuracy": float(
                accuracy_score(
                    labels,
                    predictions
                )
            ),
        })

    threshold_search_df = pd.DataFrame(
        threshold_rows
    )

    best_row = (
        threshold_search_df
        .sort_values(
            [
                "f1",
                "precision",
                "recall",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
                True,
            ]
        )
        .iloc[0]
    )

    best_threshold = float(
        best_row["threshold"]
    )

    return (
        best_threshold,
        threshold_search_df,
    )


def calculate_binary_metrics(
    labels,
    probabilities,
    threshold,
):
    labels = np.asarray(
        labels,
        dtype=np.int64
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    confusion = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    )

    true_negative = int(
        confusion[0, 0]
    )

    false_positive = int(
        confusion[0, 1]
    )

    false_negative = int(
        confusion[1, 0]
    )

    true_positive = int(
        confusion[1, 1]
    )

    return {
        "threshold": float(
            threshold
        ),
        "accuracy": float(
            accuracy_score(
                labels,
                predictions
            )
        ),
        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0
            )
        ),
        "f1": float(
            f1_score(
                labels,
                predictions,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities
            )
        ),
        "pr_auc": float(
            average_precision_score(
                labels,
                probabilities
            )
        ),
        "brier_score": float(
            brier_score_loss(
                labels,
                probabilities
            )
        ),
        "tn": true_negative,
        "fp": false_positive,
        "fn": false_negative,
        "tp": true_positive,
    }


print(
    "=== FINAL V2 SHARED TRAINING UTILITIES ==="
)

print(
    "Device:",
    device
)

print(
    "Threshold candidates:",
    len(THRESHOLD_CANDIDATES)
)

print(
    "Minimum threshold:",
    THRESHOLD_CANDIDATES.min()
)

print(
    "Maximum threshold:",
    THRESHOLD_CANDIDATES.max()
)

print(
    "Shared utilities ready:",
    True
)

# Publication notebook cell pub-034
class TextOnlyMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim=1024,
        hidden_dim=256,
        dropout=0.30,
    ):
        super().__init__()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(
                input_dim,
                hidden_dim
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                dropout
            ),
            torch.nn.Linear(
                hidden_dim,
                1
            ),
        )

    def forward(self, text_embeddings):
        return self.classifier(
            text_embeddings
        ).squeeze(-1)


def load_torch_checkpoint(
    checkpoint_path
):
    try:
        return torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False
        )

    except TypeError:
        return torch.load(
            checkpoint_path,
            map_location="cpu"
        )


TEXT_ONLY_MODEL_CONFIG = {
    "model_name": "text_only_mlp",
    "text_encoder": (
        "intfloat/multilingual-e5-large"
    ),
    "encoder_status": "frozen",
    "input_dimension": 1024,
    "hidden_dimension": HIDDEN_DIM,
    "dropout_rate": DROPOUT_RATE,
    "output_dimension": 1,
    "activation": "ReLU",
    "loss": "BCEWithLogitsLoss",
}


text_only_config_path = (
    config_dir
    / "text_only_mlp_architecture.json"
)

with text_only_config_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        TEXT_ONLY_MODEL_CONFIG,
        file,
        indent=2,
        ensure_ascii=False
    )


# False:
# checkpoint yang sudah valid akan digunakan kembali.
RETRAIN_TEXT_ONLY_MODELS = False


text_only_training_rows = []
text_only_model_paths = {}
text_only_validation_outputs = {}


for seed_index, seed in enumerate(
    TRAINING_SEEDS,
    start=1
):
    print(
        f"\n[{seed_index}/"
        f"{len(TRAINING_SEEDS)}] "
        f"Text-only MLP — seed {seed}"
    )

    set_training_seed(seed)

    checkpoint_path = (
        model_dir
        / f"text_only_mlp_seed_{seed}.pt"
    )

    training_history_path = (
        diagnostic_dir
        / (
            "training_history_"
            f"text_only_mlp_seed_{seed}.csv"
        )
    )

    threshold_search_path = (
        diagnostic_dir
        / (
            "threshold_search_"
            f"text_only_mlp_seed_{seed}.csv"
        )
    )

    validation_prediction_path = (
        prediction_dir
        / (
            "validation_"
            f"text_only_mlp_seed_{seed}.csv"
        )
    )


    # Loader dibuat ulang untuk setiap seed
    # agar shuffle train reproducible.
    train_loader = create_tensor_loader(
        feature_arrays=[
            X_train_text
        ],
        labels=y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed,
    )

    validation_loader = (
        create_tensor_loader(
            feature_arrays=[
                X_validation_text
            ],
            labels=y_validation,
            batch_size=BATCH_SIZE,
            shuffle=False,
            seed=seed,
        )
    )


    model = TextOnlyMLP(
        input_dim=1024,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT_RATE,
    )

    trainable_parameters = (
        count_trainable_parameters(
            model
        )
    )

    expected_parameter_count = 262657

    if (
        trainable_parameters
        != expected_parameter_count
    ):
        raise ValueError(
            "Jumlah parameter Text-only MLP "
            "tidak sesuai: "
            f"{trainable_parameters} "
            f"!= {expected_parameter_count}"
        )


    generated_now = True

    if (
        checkpoint_path.exists()
        and not RETRAIN_TEXT_ONLY_MODELS
    ):
        print(
            "  Menggunakan checkpoint existing."
        )

        checkpoint = (
            load_torch_checkpoint(
                checkpoint_path
            )
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        model = (
            model
            .to(device)
            .eval()
        )

        training_summary = checkpoint[
            "training_summary"
        ]

        generated_now = False

    else:
        training_start_time = time.time()

        (
            model,
            training_history_df,
            training_summary,
        ) = train_binary_model(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            learning_rate=(
                LEARNING_RATE
            ),
            weight_decay=(
                WEIGHT_DECAY
            ),
            maximum_epochs=(
                MAX_EPOCHS
            ),
            patience=(
                EARLY_STOPPING_PATIENCE
            ),
        )

        training_duration_seconds = (
            time.time()
            - training_start_time
        )

        training_history_df[
            "seed"
        ] = seed

        training_history_df[
            "model_name"
        ] = "text_only_mlp"

        training_history_df.to_csv(
            training_history_path,
            index=False
        )

        training_summary[
            "training_duration_seconds"
        ] = round(
            training_duration_seconds,
            2
        )


    # Validation inference
    validation_logits, (
        validation_labels
    ) = predict_binary_logits(
        model=model,
        data_loader=validation_loader,
    )

    if not np.array_equal(
        validation_labels.astype(
            np.int64
        ),
        y_validation.astype(
            np.int64
        )
    ):
        raise ValueError(
            "Urutan validation label berubah "
            f"pada seed {seed}."
        )

    validation_probabilities = (
        sigmoid_numpy(
            validation_logits
        )
    )


    # Threshold dipilih dari clean validation
    (
        selected_threshold,
        threshold_search_df,
    ) = select_validation_threshold(
        labels=y_validation,
        probabilities=(
            validation_probabilities
        ),
        threshold_candidates=(
            THRESHOLD_CANDIDATES
        ),
    )

    validation_metrics = (
        calculate_binary_metrics(
            labels=y_validation,
            probabilities=(
                validation_probabilities
            ),
            threshold=(
                selected_threshold
            ),
        )
    )


    threshold_search_df[
        "seed"
    ] = seed

    threshold_search_df[
        "model_name"
    ] = "text_only_mlp"

    threshold_search_df.to_csv(
        threshold_search_path,
        index=False
    )


    validation_predictions = (
        validation_probabilities
        >= selected_threshold
    ).astype(np.int64)


    validation_prediction_df = (
        validation_df[
            [
                "id",
                "platform",
                "label",
            ]
        ]
        .copy()
    )

    validation_prediction_df[
        "model_name"
    ] = "text_only_mlp"

    validation_prediction_df[
        "seed"
    ] = seed

    validation_prediction_df[
        "logit"
    ] = validation_logits

    validation_prediction_df[
        "probability"
    ] = validation_probabilities

    validation_prediction_df[
        "threshold"
    ] = selected_threshold

    validation_prediction_df[
        "prediction"
    ] = validation_predictions

    validation_prediction_df[
        "is_correct"
    ] = (
        validation_predictions
        == y_validation.astype(
            np.int64
        )
    )

    validation_prediction_df.to_csv(
        validation_prediction_path,
        index=False
    )


    # Bila checkpoint baru, simpan setelah
    # threshold validation telah ditentukan.
    if generated_now:
        checkpoint_payload = {
            "model_name": (
                "text_only_mlp"
            ),
            "seed": seed,
            "model_state_dict": {
                key: value
                .detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            },
            "model_config": (
                TEXT_ONLY_MODEL_CONFIG
            ),
            "training_protocol": {
                "batch_size": BATCH_SIZE,
                "learning_rate": (
                    LEARNING_RATE
                ),
                "weight_decay": (
                    WEIGHT_DECAY
                ),
                "maximum_epochs": (
                    MAX_EPOCHS
                ),
                "early_stopping_patience": (
                    EARLY_STOPPING_PATIENCE
                ),
                "early_stopping_monitor": (
                    "validation_loss"
                ),
            },
            "training_summary": (
                training_summary
            ),
            "selected_threshold": (
                selected_threshold
            ),
            "validation_metrics": (
                validation_metrics
            ),
            "trainable_parameters": (
                trainable_parameters
            ),
        }

        torch.save(
            checkpoint_payload,
            checkpoint_path
        )

    else:
        checkpoint_threshold = float(
            checkpoint[
                "selected_threshold"
            ]
        )

        if not np.isclose(
            checkpoint_threshold,
            selected_threshold,
            atol=1e-8
        ):
            raise ValueError(
                "Threshold checkpoint berbeda "
                "dari hasil validation saat ini: "
                f"{checkpoint_threshold} "
                f"!= {selected_threshold}"
            )


    text_only_model_paths[
        seed
    ] = checkpoint_path

    text_only_validation_outputs[
        seed
    ] = {
        "logits": validation_logits,
        "probabilities": (
            validation_probabilities
        ),
        "threshold": (
            selected_threshold
        ),
        "metrics": (
            validation_metrics
        ),
    }


    text_only_training_rows.append({
        "model_name": "text_only_mlp",
        "seed": seed,
        "trainable_parameters": (
            trainable_parameters
        ),
        "best_epoch": int(
            training_summary[
                "best_epoch"
            ]
        ),
        "epochs_completed": int(
            training_summary[
                "epochs_completed"
            ]
        ),
        "early_stopped": bool(
            training_summary[
                "early_stopped"
            ]
        ),
        "best_validation_loss": float(
            training_summary[
                "best_validation_loss"
            ]
        ),
        "selected_threshold": float(
            selected_threshold
        ),
        "validation_accuracy": (
            validation_metrics[
                "accuracy"
            ]
        ),
        "validation_precision": (
            validation_metrics[
                "precision"
            ]
        ),
        "validation_recall": (
            validation_metrics[
                "recall"
            ]
        ),
        "validation_f1": (
            validation_metrics[
                "f1"
            ]
        ),
        "validation_roc_auc": (
            validation_metrics[
                "roc_auc"
            ]
        ),
        "validation_pr_auc": (
            validation_metrics[
                "pr_auc"
            ]
        ),
        "validation_brier_score": (
            validation_metrics[
                "brier_score"
            ]
        ),
        "validation_tn": (
            validation_metrics["tn"]
        ),
        "validation_fp": (
            validation_metrics["fp"]
        ),
        "validation_fn": (
            validation_metrics["fn"]
        ),
        "validation_tp": (
            validation_metrics["tp"]
        ),
        "generated_now": (
            generated_now
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
    })


    print(
        "  Best epoch:",
        training_summary["best_epoch"]
    )

    print(
        "  Best validation loss:",
        round(
            training_summary[
                "best_validation_loss"
            ],
            6
        )
    )

    print(
        "  Threshold:",
        selected_threshold
    )

    print(
        "  Validation F1:",
        round(
            validation_metrics["f1"],
            6
        )
    )

    print(
        "  Validation PR-AUC:",
        round(
            validation_metrics["pr_auc"],
            6
        )
    )

    print(
        "  Saved:",
        checkpoint_path
    )


    del model
    del train_loader
    del validation_loader

    gc.collect()

    if device.type == "mps":
        torch.mps.empty_cache()

    elif device.type == "cuda":
        torch.cuda.empty_cache()


text_only_training_summary_df = (
    pd.DataFrame(
        text_only_training_rows
    )
    .sort_values("seed")
    .reset_index(drop=True)
)


text_only_aggregate_df = pd.DataFrame([
    {
        "model_name": (
            "text_only_mlp"
        ),
        "number_of_seeds": int(
            len(
                text_only_training_summary_df
            )
        ),
        "mean_best_epoch": float(
            text_only_training_summary_df[
                "best_epoch"
            ].mean()
        ),
        "mean_selected_threshold": float(
            text_only_training_summary_df[
                "selected_threshold"
            ].mean()
        ),
        "std_selected_threshold": float(
            text_only_training_summary_df[
                "selected_threshold"
            ].std(ddof=1)
        ),
        "mean_validation_f1": float(
            text_only_training_summary_df[
                "validation_f1"
            ].mean()
        ),
        "std_validation_f1": float(
            text_only_training_summary_df[
                "validation_f1"
            ].std(ddof=1)
        ),
        "minimum_validation_f1": float(
            text_only_training_summary_df[
                "validation_f1"
            ].min()
        ),
        "maximum_validation_f1": float(
            text_only_training_summary_df[
                "validation_f1"
            ].max()
        ),
        "mean_validation_pr_auc": float(
            text_only_training_summary_df[
                "validation_pr_auc"
            ].mean()
        ),
        "mean_validation_brier_score": float(
            text_only_training_summary_df[
                "validation_brier_score"
            ].mean()
        ),
    }
])


text_only_training_summary_path = (
    summary_dir
    / "text_only_mlp_training_summary.csv"
)

text_only_aggregate_path = (
    summary_dir
    / "text_only_mlp_validation_aggregate.csv"
)


text_only_training_summary_df.to_csv(
    text_only_training_summary_path,
    index=False
)

text_only_aggregate_df.to_csv(
    text_only_aggregate_path,
    index=False
)


print(
    "\n=== FINAL V2 TEXT-ONLY MLP "
    "TRAINING SUMMARY ==="
)

display(
    text_only_training_summary_df[
        [
            "seed",
            "best_epoch",
            "epochs_completed",
            "best_validation_loss",
            "selected_threshold",
            "validation_precision",
            "validation_recall",
            "validation_f1",
            "validation_pr_auc",
            "validation_brier_score",
            "generated_now",
        ]
    ]
)

print(
    "\n=== TEXT-ONLY VALIDATION "
    "AGGREGATE ==="
)

display(
    text_only_aggregate_df
)

print(
    "\nCheckpoints:",
    len(text_only_model_paths)
)

print(
    "Trainable parameters:",
    expected_parameter_count
)

print("\nSaved to:")
print(text_only_training_summary_path)
print(text_only_aggregate_path)

# Publication notebook cell pub-036
class ImageOnlyMLP(torch.nn.Module):
    def __init__(
        self,
        input_dim=1152,
        hidden_dim=256,
        dropout=0.30,
    ):
        super().__init__()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(
                input_dim,
                hidden_dim
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(
                dropout
            ),
            torch.nn.Linear(
                hidden_dim,
                1
            ),
        )

    def forward(self, image_embeddings):
        return self.classifier(
            image_embeddings
        ).squeeze(-1)


IMAGE_ONLY_MODEL_CONFIG = {
    "model_name": "image_only_mlp",
    "image_encoder": (
        "google/siglip2-so400m-patch14-384"
    ),
    "encoder_status": "frozen",
    "input_dimension": 1152,
    "hidden_dimension": HIDDEN_DIM,
    "dropout_rate": DROPOUT_RATE,
    "output_dimension": 1,
    "activation": "ReLU",
    "loss": "BCEWithLogitsLoss",
}


image_only_config_path = (
    config_dir
    / "image_only_mlp_architecture.json"
)

with image_only_config_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        IMAGE_ONLY_MODEL_CONFIG,
        file,
        indent=2,
        ensure_ascii=False
    )


RETRAIN_IMAGE_ONLY_MODELS = False

image_only_training_rows = []
image_only_model_paths = {}
image_only_validation_outputs = {}


for seed_index, seed in enumerate(
    TRAINING_SEEDS,
    start=1
):
    print(
        f"\n[{seed_index}/"
        f"{len(TRAINING_SEEDS)}] "
        f"Image-only MLP — seed {seed}"
    )

    set_training_seed(seed)

    checkpoint_path = (
        model_dir
        / f"image_only_mlp_seed_{seed}.pt"
    )

    training_history_path = (
        diagnostic_dir
        / (
            "training_history_"
            f"image_only_mlp_seed_{seed}.csv"
        )
    )

    threshold_search_path = (
        diagnostic_dir
        / (
            "threshold_search_"
            f"image_only_mlp_seed_{seed}.csv"
        )
    )

    validation_prediction_path = (
        prediction_dir
        / (
            "validation_"
            f"image_only_mlp_seed_{seed}.csv"
        )
    )


    train_loader = create_tensor_loader(
        feature_arrays=[
            X_train_image
        ],
        labels=y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed,
    )

    validation_loader = (
        create_tensor_loader(
            feature_arrays=[
                X_validation_image
            ],
            labels=y_validation,
            batch_size=BATCH_SIZE,
            shuffle=False,
            seed=seed,
        )
    )


    model = ImageOnlyMLP(
        input_dim=1152,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT_RATE,
    )

    trainable_parameters = (
        count_trainable_parameters(
            model
        )
    )

    expected_image_parameter_count = 295425

    if (
        trainable_parameters
        != expected_image_parameter_count
    ):
        raise ValueError(
            "Jumlah parameter Image-only MLP "
            "tidak sesuai: "
            f"{trainable_parameters} "
            f"!= {expected_image_parameter_count}"
        )


    generated_now = True

    if (
        checkpoint_path.exists()
        and not RETRAIN_IMAGE_ONLY_MODELS
    ):
        print(
            "  Menggunakan checkpoint existing."
        )

        checkpoint = load_torch_checkpoint(
            checkpoint_path
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        model = (
            model
            .to(device)
            .eval()
        )

        training_summary = checkpoint[
            "training_summary"
        ]

        generated_now = False

    else:
        training_start_time = time.time()

        (
            model,
            training_history_df,
            training_summary,
        ) = train_binary_model(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            learning_rate=(
                LEARNING_RATE
            ),
            weight_decay=(
                WEIGHT_DECAY
            ),
            maximum_epochs=(
                MAX_EPOCHS
            ),
            patience=(
                EARLY_STOPPING_PATIENCE
            ),
        )

        training_duration_seconds = (
            time.time()
            - training_start_time
        )

        training_history_df[
            "seed"
        ] = seed

        training_history_df[
            "model_name"
        ] = "image_only_mlp"

        training_history_df.to_csv(
            training_history_path,
            index=False
        )

        training_summary[
            "training_duration_seconds"
        ] = round(
            training_duration_seconds,
            2
        )


    validation_logits, validation_labels = (
        predict_binary_logits(
            model=model,
            data_loader=validation_loader,
        )
    )

    if not np.array_equal(
        validation_labels.astype(
            np.int64
        ),
        y_validation.astype(
            np.int64
        )
    ):
        raise ValueError(
            "Urutan validation label berubah "
            f"pada seed {seed}."
        )


    validation_probabilities = (
        sigmoid_numpy(
            validation_logits
        )
    )


    (
        selected_threshold,
        threshold_search_df,
    ) = select_validation_threshold(
        labels=y_validation,
        probabilities=(
            validation_probabilities
        ),
        threshold_candidates=(
            THRESHOLD_CANDIDATES
        ),
    )


    validation_metrics = (
        calculate_binary_metrics(
            labels=y_validation,
            probabilities=(
                validation_probabilities
            ),
            threshold=(
                selected_threshold
            ),
        )
    )


    threshold_search_df[
        "seed"
    ] = seed

    threshold_search_df[
        "model_name"
    ] = "image_only_mlp"

    threshold_search_df.to_csv(
        threshold_search_path,
        index=False
    )


    validation_predictions = (
        validation_probabilities
        >= selected_threshold
    ).astype(np.int64)


    validation_prediction_df = (
        validation_df[
            [
                "id",
                "platform",
                "label",
            ]
        ]
        .copy()
    )

    validation_prediction_df[
        "model_name"
    ] = "image_only_mlp"

    validation_prediction_df[
        "seed"
    ] = seed

    validation_prediction_df[
        "logit"
    ] = validation_logits

    validation_prediction_df[
        "probability"
    ] = validation_probabilities

    validation_prediction_df[
        "threshold"
    ] = selected_threshold

    validation_prediction_df[
        "prediction"
    ] = validation_predictions

    validation_prediction_df[
        "is_correct"
    ] = (
        validation_predictions
        == y_validation.astype(
            np.int64
        )
    )

    validation_prediction_df.to_csv(
        validation_prediction_path,
        index=False
    )


    if generated_now:
        checkpoint_payload = {
            "model_name": (
                "image_only_mlp"
            ),
            "seed": seed,
            "model_state_dict": {
                key: value
                .detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            },
            "model_config": (
                IMAGE_ONLY_MODEL_CONFIG
            ),
            "training_protocol": {
                "batch_size": BATCH_SIZE,
                "learning_rate": (
                    LEARNING_RATE
                ),
                "weight_decay": (
                    WEIGHT_DECAY
                ),
                "maximum_epochs": (
                    MAX_EPOCHS
                ),
                "early_stopping_patience": (
                    EARLY_STOPPING_PATIENCE
                ),
                "early_stopping_monitor": (
                    "validation_loss"
                ),
            },
            "training_summary": (
                training_summary
            ),
            "selected_threshold": (
                selected_threshold
            ),
            "validation_metrics": (
                validation_metrics
            ),
            "trainable_parameters": (
                trainable_parameters
            ),
        }

        torch.save(
            checkpoint_payload,
            checkpoint_path
        )

    else:
        checkpoint_threshold = float(
            checkpoint[
                "selected_threshold"
            ]
        )

        if not np.isclose(
            checkpoint_threshold,
            selected_threshold,
            atol=1e-8
        ):
            raise ValueError(
                "Threshold checkpoint berbeda "
                "dari hasil validation saat ini: "
                f"{checkpoint_threshold} "
                f"!= {selected_threshold}"
            )


    image_only_model_paths[
        seed
    ] = checkpoint_path

    image_only_validation_outputs[
        seed
    ] = {
        "logits": validation_logits,
        "probabilities": (
            validation_probabilities
        ),
        "threshold": (
            selected_threshold
        ),
        "metrics": (
            validation_metrics
        ),
    }


    image_only_training_rows.append({
        "model_name": "image_only_mlp",
        "seed": seed,
        "trainable_parameters": (
            trainable_parameters
        ),
        "best_epoch": int(
            training_summary[
                "best_epoch"
            ]
        ),
        "epochs_completed": int(
            training_summary[
                "epochs_completed"
            ]
        ),
        "early_stopped": bool(
            training_summary[
                "early_stopped"
            ]
        ),
        "best_validation_loss": float(
            training_summary[
                "best_validation_loss"
            ]
        ),
        "selected_threshold": float(
            selected_threshold
        ),
        "validation_accuracy": (
            validation_metrics[
                "accuracy"
            ]
        ),
        "validation_precision": (
            validation_metrics[
                "precision"
            ]
        ),
        "validation_recall": (
            validation_metrics[
                "recall"
            ]
        ),
        "validation_f1": (
            validation_metrics[
                "f1"
            ]
        ),
        "validation_roc_auc": (
            validation_metrics[
                "roc_auc"
            ]
        ),
        "validation_pr_auc": (
            validation_metrics[
                "pr_auc"
            ]
        ),
        "validation_brier_score": (
            validation_metrics[
                "brier_score"
            ]
        ),
        "validation_tn": (
            validation_metrics["tn"]
        ),
        "validation_fp": (
            validation_metrics["fp"]
        ),
        "validation_fn": (
            validation_metrics["fn"]
        ),
        "validation_tp": (
            validation_metrics["tp"]
        ),
        "generated_now": (
            generated_now
        ),
        "checkpoint_path": str(
            checkpoint_path
        ),
    })


    print(
        "  Best epoch:",
        training_summary["best_epoch"]
    )

    print(
        "  Best validation loss:",
        round(
            training_summary[
                "best_validation_loss"
            ],
            6
        )
    )

    print(
        "  Threshold:",
        selected_threshold
    )

    print(
        "  Validation F1:",
        round(
            validation_metrics["f1"],
            6
        )
    )

    print(
        "  Validation PR-AUC:",
        round(
            validation_metrics["pr_auc"],
            6
        )
    )

    print(
        "  Saved:",
        checkpoint_path
    )


    del model
    del train_loader
    del validation_loader

    gc.collect()

    if device.type == "mps":
        torch.mps.empty_cache()

    elif device.type == "cuda":
        torch.cuda.empty_cache()


image_only_training_summary_df = (
    pd.DataFrame(
        image_only_training_rows
    )
    .sort_values("seed")
    .reset_index(drop=True)
)


image_only_aggregate_df = pd.DataFrame([
    {
        "model_name": (
            "image_only_mlp"
        ),
        "number_of_seeds": int(
            len(
                image_only_training_summary_df
            )
        ),
        "mean_best_epoch": float(
            image_only_training_summary_df[
                "best_epoch"
            ].mean()
        ),
        "mean_selected_threshold": float(
            image_only_training_summary_df[
                "selected_threshold"
            ].mean()
        ),
        "std_selected_threshold": float(
            image_only_training_summary_df[
                "selected_threshold"
            ].std(ddof=1)
        ),
        "mean_validation_f1": float(
            image_only_training_summary_df[
                "validation_f1"
            ].mean()
        ),
        "std_validation_f1": float(
            image_only_training_summary_df[
                "validation_f1"
            ].std(ddof=1)
        ),
        "minimum_validation_f1": float(
            image_only_training_summary_df[
                "validation_f1"
            ].min()
        ),
        "maximum_validation_f1": float(
            image_only_training_summary_df[
                "validation_f1"
            ].max()
        ),
        "mean_validation_pr_auc": float(
            image_only_training_summary_df[
                "validation_pr_auc"
            ].mean()
        ),
        "mean_validation_brier_score": float(
            image_only_training_summary_df[
                "validation_brier_score"
            ].mean()
        ),
    }
])


image_only_training_summary_path = (
    summary_dir
    / "image_only_mlp_training_summary.csv"
)

image_only_aggregate_path = (
    summary_dir
    / "image_only_mlp_validation_aggregate.csv"
)


image_only_training_summary_df.to_csv(
    image_only_training_summary_path,
    index=False
)

image_only_aggregate_df.to_csv(
    image_only_aggregate_path,
    index=False
)


print(
    "\n=== FINAL V2 IMAGE-ONLY MLP "
    "TRAINING SUMMARY ==="
)

display(
    image_only_training_summary_df[
        [
            "seed",
            "best_epoch",
            "epochs_completed",
            "best_validation_loss",
            "selected_threshold",
            "validation_precision",
            "validation_recall",
            "validation_f1",
            "validation_pr_auc",
            "validation_brier_score",
            "generated_now",
        ]
    ]
)

print(
    "\n=== IMAGE-ONLY VALIDATION "
    "AGGREGATE ==="
)

display(
    image_only_aggregate_df
)

print(
    "\nCheckpoints:",
    len(image_only_model_paths)
)

print(
    "Trainable parameters:",
    expected_image_parameter_count
)

print("\nSaved to:")
print(image_only_training_summary_path)
print(image_only_aggregate_path)

# Publication notebook cell pub-038
AVERAGE_TEXT_WEIGHT = 0.50
AVERAGE_IMAGE_WEIGHT = 0.50

if not np.isclose(
    AVERAGE_TEXT_WEIGHT
    + AVERAGE_IMAGE_WEIGHT,
    1.0
):
    raise ValueError(
        "Bobot Average Fusion harus "
        "berjumlah 1."
    )


average_fusion_config = {
    "model_name": (
        "average_logit_fusion"
    ),
    "fusion_type": (
        "late_logit_fusion"
    ),
    "text_component": (
        "text_only_mlp"
    ),
    "image_component": (
        "image_only_mlp"
    ),
    "text_weight": (
        AVERAGE_TEXT_WEIGHT
    ),
    "image_weight": (
        AVERAGE_IMAGE_WEIGHT
    ),
    "fusion_stage": (
        "before_sigmoid"
    ),
    "trainable_fusion_parameters": 0,
    "threshold_source": (
        "clean_validation_only"
    ),
}


average_fusion_config_path = (
    config_dir
    / "average_logit_fusion_config.json"
)

with average_fusion_config_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        average_fusion_config,
        file,
        indent=2,
        ensure_ascii=False
    )


average_fusion_rows = []
average_fusion_validation_outputs = {}
average_fusion_descriptor_paths = {}


for seed_index, seed in enumerate(
    TRAINING_SEEDS,
    start=1
):
    print(
        f"\n[{seed_index}/"
        f"{len(TRAINING_SEEDS)}] "
        f"Average Logit Fusion — seed {seed}"
    )

    text_prediction_path = (
        prediction_dir
        / (
            "validation_"
            f"text_only_mlp_seed_{seed}.csv"
        )
    )

    image_prediction_path = (
        prediction_dir
        / (
            "validation_"
            f"image_only_mlp_seed_{seed}.csv"
        )
    )

    if not text_prediction_path.exists():
        raise FileNotFoundError(
            "Text validation prediction "
            "tidak ditemukan: "
            f"{text_prediction_path}"
        )

    if not image_prediction_path.exists():
        raise FileNotFoundError(
            "Image validation prediction "
            "tidak ditemukan: "
            f"{image_prediction_path}"
        )


    text_prediction_df = pd.read_csv(
        text_prediction_path
    )

    image_prediction_df = pd.read_csv(
        image_prediction_path
    )


    required_prediction_columns = {
        "id",
        "platform",
        "label",
        "logit",
    }

    if not required_prediction_columns.issubset(
        text_prediction_df.columns
    ):
        raise ValueError(
            "Kolom prediction text "
            "tidak lengkap."
        )

    if not required_prediction_columns.issubset(
        image_prediction_df.columns
    ):
        raise ValueError(
            "Kolom prediction image "
            "tidak lengkap."
        )


    text_component_df = (
        text_prediction_df[
            [
                "id",
                "platform",
                "label",
                "logit",
            ]
        ]
        .rename(
            columns={
                "logit": "text_logit"
            }
        )
    )

    image_component_df = (
        image_prediction_df[
            [
                "id",
                "platform",
                "label",
                "logit",
            ]
        ]
        .rename(
            columns={
                "logit": "image_logit"
            }
        )
    )


    fusion_validation_df = (
        text_component_df.merge(
            image_component_df,
            on=[
                "id",
                "platform",
                "label",
            ],
            how="inner",
            validate="one_to_one"
        )
    )


    if (
        len(fusion_validation_df)
        != len(validation_df)
    ):
        raise ValueError(
            "Jumlah baris Average Fusion "
            f"seed {seed} tidak valid."
        )


    expected_validation_ids = (
        validation_df["id"]
        .tolist()
    )

    actual_validation_ids = (
        fusion_validation_df["id"]
        .tolist()
    )

    if (
        actual_validation_ids
        != expected_validation_ids
    ):
        raise ValueError(
            "Urutan ID validation berubah "
            f"pada seed {seed}."
        )


    fusion_logits = (
        AVERAGE_TEXT_WEIGHT
        * fusion_validation_df[
            "text_logit"
        ].to_numpy(
            dtype=np.float32
        )
        +
        AVERAGE_IMAGE_WEIGHT
        * fusion_validation_df[
            "image_logit"
        ].to_numpy(
            dtype=np.float32
        )
    )


    fusion_probabilities = (
        sigmoid_numpy(
            fusion_logits
        )
    )


    (
        selected_threshold,
        threshold_search_df,
    ) = select_validation_threshold(
        labels=y_validation,
        probabilities=(
            fusion_probabilities
        ),
        threshold_candidates=(
            THRESHOLD_CANDIDATES
        ),
    )


    validation_metrics = (
        calculate_binary_metrics(
            labels=y_validation,
            probabilities=(
                fusion_probabilities
            ),
            threshold=(
                selected_threshold
            ),
        )
    )


    threshold_search_df[
        "seed"
    ] = seed

    threshold_search_df[
        "model_name"
    ] = "average_logit_fusion"


    threshold_search_path = (
        diagnostic_dir
        / (
            "threshold_search_"
            "average_logit_fusion_"
            f"seed_{seed}.csv"
        )
    )

    threshold_search_df.to_csv(
        threshold_search_path,
        index=False
    )


    fusion_predictions = (
        fusion_probabilities
        >= selected_threshold
    ).astype(np.int64)


    fusion_validation_df[
        "model_name"
    ] = "average_logit_fusion"

    fusion_validation_df[
        "seed"
    ] = seed

    fusion_validation_df[
        "text_weight"
    ] = AVERAGE_TEXT_WEIGHT

    fusion_validation_df[
        "image_weight"
    ] = AVERAGE_IMAGE_WEIGHT

    fusion_validation_df[
        "fusion_logit"
    ] = fusion_logits

    fusion_validation_df[
        "probability"
    ] = fusion_probabilities

    fusion_validation_df[
        "threshold"
    ] = selected_threshold

    fusion_validation_df[
        "prediction"
    ] = fusion_predictions

    fusion_validation_df[
        "is_correct"
    ] = (
        fusion_predictions
        == y_validation.astype(
            np.int64
        )
    )


    validation_prediction_path = (
        prediction_dir
        / (
            "validation_"
            "average_logit_fusion_"
            f"seed_{seed}.csv"
        )
    )

    fusion_validation_df.to_csv(
        validation_prediction_path,
        index=False
    )


    descriptor = {
        "model_name": (
            "average_logit_fusion"
        ),
        "seed": seed,
        "text_component_checkpoint": str(
            model_dir
            / (
                "text_only_mlp_"
                f"seed_{seed}.pt"
            )
        ),
        "image_component_checkpoint": str(
            model_dir
            / (
                "image_only_mlp_"
                f"seed_{seed}.pt"
            )
        ),
        "text_weight": (
            AVERAGE_TEXT_WEIGHT
        ),
        "image_weight": (
            AVERAGE_IMAGE_WEIGHT
        ),
        "selected_threshold": float(
            selected_threshold
        ),
        "validation_metrics": (
            validation_metrics
        ),
        "trainable_fusion_parameters": 0,
    }


    descriptor_path = (
        model_dir
        / (
            "average_logit_fusion_"
            f"seed_{seed}.json"
        )
    )

    with descriptor_path.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            descriptor,
            file,
            indent=2,
            ensure_ascii=False
        )


    average_fusion_descriptor_paths[
        seed
    ] = descriptor_path

    average_fusion_validation_outputs[
        seed
    ] = {
        "text_logits": (
            fusion_validation_df[
                "text_logit"
            ].to_numpy(
                dtype=np.float32
            )
        ),
        "image_logits": (
            fusion_validation_df[
                "image_logit"
            ].to_numpy(
                dtype=np.float32
            )
        ),
        "logits": fusion_logits,
        "probabilities": (
            fusion_probabilities
        ),
        "threshold": (
            selected_threshold
        ),
        "metrics": (
            validation_metrics
        ),
    }


    average_fusion_rows.append({
        "model_name": (
            "average_logit_fusion"
        ),
        "seed": seed,
        "text_weight": (
            AVERAGE_TEXT_WEIGHT
        ),
        "image_weight": (
            AVERAGE_IMAGE_WEIGHT
        ),
        "trainable_fusion_parameters": 0,
        "selected_threshold": float(
            selected_threshold
        ),
        "validation_accuracy": (
            validation_metrics[
                "accuracy"
            ]
        ),
        "validation_precision": (
            validation_metrics[
                "precision"
            ]
        ),
        "validation_recall": (
            validation_metrics[
                "recall"
            ]
        ),
        "validation_f1": (
            validation_metrics[
                "f1"
            ]
        ),
        "validation_roc_auc": (
            validation_metrics[
                "roc_auc"
            ]
        ),
        "validation_pr_auc": (
            validation_metrics[
                "pr_auc"
            ]
        ),
        "validation_brier_score": (
            validation_metrics[
                "brier_score"
            ]
        ),
        "validation_tn": (
            validation_metrics["tn"]
        ),
        "validation_fp": (
            validation_metrics["fp"]
        ),
        "validation_fn": (
            validation_metrics["fn"]
        ),
        "validation_tp": (
            validation_metrics["tp"]
        ),
        "descriptor_path": str(
            descriptor_path
        ),
    })


    print(
        "  Threshold:",
        selected_threshold
    )

    print(
        "  Validation F1:",
        round(
            validation_metrics["f1"],
            6
        )
    )

    print(
        "  Validation PR-AUC:",
        round(
            validation_metrics["pr_auc"],
            6
        )
    )

    print(
        "  Brier score:",
        round(
            validation_metrics[
                "brier_score"
            ],
            6
        )
    )


average_fusion_summary_df = (
    pd.DataFrame(
        average_fusion_rows
    )
    .sort_values("seed")
    .reset_index(drop=True)
)


average_fusion_aggregate_df = pd.DataFrame([
    {
        "model_name": (
            "average_logit_fusion"
        ),
        "number_of_seeds": int(
            len(
                average_fusion_summary_df
            )
        ),
        "text_weight": (
            AVERAGE_TEXT_WEIGHT
        ),
        "image_weight": (
            AVERAGE_IMAGE_WEIGHT
        ),
        "mean_selected_threshold": float(
            average_fusion_summary_df[
                "selected_threshold"
            ].mean()
        ),
        "std_selected_threshold": float(
            average_fusion_summary_df[
                "selected_threshold"
            ].std(ddof=1)
        ),
        "mean_validation_f1": float(
            average_fusion_summary_df[
                "validation_f1"
            ].mean()
        ),
        "std_validation_f1": float(
            average_fusion_summary_df[
                "validation_f1"
            ].std(ddof=1)
        ),
        "minimum_validation_f1": float(
            average_fusion_summary_df[
                "validation_f1"
            ].min()
        ),
        "maximum_validation_f1": float(
            average_fusion_summary_df[
                "validation_f1"
            ].max()
        ),
        "mean_validation_pr_auc": float(
            average_fusion_summary_df[
                "validation_pr_auc"
            ].mean()
        ),
        "mean_validation_brier_score": float(
            average_fusion_summary_df[
                "validation_brier_score"
            ].mean()
        ),
    }
])


average_fusion_summary_path = (
    summary_dir
    / (
        "average_logit_fusion_"
        "validation_summary.csv"
    )
)

average_fusion_aggregate_path = (
    summary_dir
    / (
        "average_logit_fusion_"
        "validation_aggregate.csv"
    )
)


average_fusion_summary_df.to_csv(
    average_fusion_summary_path,
    index=False
)

average_fusion_aggregate_df.to_csv(
    average_fusion_aggregate_path,
    index=False
)


print(
    "\n=== FINAL V2 AVERAGE LOGIT "
    "FUSION SUMMARY ==="
)

display(
    average_fusion_summary_df[
        [
            "seed",
            "text_weight",
            "image_weight",
            "selected_threshold",
            "validation_precision",
            "validation_recall",
            "validation_f1",
            "validation_pr_auc",
            "validation_brier_score",
        ]
    ]
)

print(
    "\n=== AVERAGE LOGIT FUSION "
    "AGGREGATE ==="
)

display(
    average_fusion_aggregate_df
)

print(
    "\nFusion descriptors:",
    len(
        average_fusion_descriptor_paths
    )
)

print(
    "Trainable fusion parameters:",
    0
)

print("\nSaved to:")
print(average_fusion_summary_path)
print(average_fusion_aggregate_path)

# Publication notebook cell pub-040
class LearnableWeightedLogitFusion(
    torch.nn.Module
):
    def __init__(self):
        super().__init__()

        # softmax([0, 0]) = [0.5, 0.5]
        self.raw_weights = torch.nn.Parameter(
            torch.zeros(2)
        )

    def normalized_weights(self):
        return torch.softmax(
            self.raw_weights,
            dim=0
        )

    def forward(
        self,
        text_logits,
        image_logits,
    ):
        weights = self.normalized_weights()

        return (
            weights[0]
            * text_logits.reshape(-1)
            +
            weights[1]
            * image_logits.reshape(-1)
        )


def load_component_model(
    model_type,
    checkpoint_path,
):
    checkpoint = load_torch_checkpoint(
        checkpoint_path
    )

    if model_type == "text":
        model = TextOnlyMLP(
            input_dim=1024,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT_RATE,
        )

    elif model_type == "image":
        model = ImageOnlyMLP(
            input_dim=1152,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT_RATE,
        )

    else:
        raise ValueError(
            f"Model type tidak dikenal: "
            f"{model_type}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device).eval()

    for parameter in model.parameters():
        parameter.requires_grad = False

    return model


def get_component_logits(
    model,
    embeddings,
    labels,
    seed,
):
    loader = create_tensor_loader(
        feature_arrays=[embeddings],
        labels=labels,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=seed,
    )

    logits, returned_labels = (
        predict_binary_logits(
            model,
            loader
        )
    )

    if not np.array_equal(
        returned_labels.astype(np.int64),
        np.asarray(
            labels,
            dtype=np.int64
        )
    ):
        raise ValueError(
            "Urutan label berubah."
        )

    return logits.astype(np.float32)


RETRAIN_WEIGHTED_FUSION = False

weighted_fusion_rows = []
weighted_fusion_paths = {}


for seed in TRAINING_SEEDS:
    print(
        f"\nLearnable Weighted Fusion "
        f"— seed {seed}"
    )

    set_training_seed(seed)

    text_model = load_component_model(
        model_type="text",
        checkpoint_path=(
            model_dir
            / f"text_only_mlp_seed_{seed}.pt"
        ),
    )

    image_model = load_component_model(
        model_type="image",
        checkpoint_path=(
            model_dir
            / f"image_only_mlp_seed_{seed}.pt"
        ),
    )


    # Component logits: clean train
    train_text_logits = get_component_logits(
        text_model,
        X_train_text,
        y_train,
        seed,
    )

    train_image_logits = get_component_logits(
        image_model,
        X_train_image,
        y_train,
        seed,
    )


    # Component logits: clean validation
    validation_text_logits = (
        get_component_logits(
            text_model,
            X_validation_text,
            y_validation,
            seed,
        )
    )

    validation_image_logits = (
        get_component_logits(
            image_model,
            X_validation_image,
            y_validation,
            seed,
        )
    )


    fusion_train_loader = (
        create_tensor_loader(
            feature_arrays=[
                train_text_logits.reshape(-1, 1),
                train_image_logits.reshape(-1, 1),
            ],
            labels=y_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=seed,
        )
    )

    fusion_validation_loader = (
        create_tensor_loader(
            feature_arrays=[
                validation_text_logits.reshape(
                    -1,
                    1
                ),
                validation_image_logits.reshape(
                    -1,
                    1
                ),
            ],
            labels=y_validation,
            batch_size=BATCH_SIZE,
            shuffle=False,
            seed=seed,
        )
    )


    checkpoint_path = (
        model_dir
        / (
            "learnable_weighted_logit_"
            f"fusion_seed_{seed}.pt"
        )
    )

    history_path = (
        diagnostic_dir
        / (
            "training_history_"
            "learnable_weighted_logit_"
            f"fusion_seed_{seed}.csv"
        )
    )


    fusion_model = (
        LearnableWeightedLogitFusion()
    )

    generated_now = True

    if (
        checkpoint_path.exists()
        and not RETRAIN_WEIGHTED_FUSION
    ):
        checkpoint = load_torch_checkpoint(
            checkpoint_path
        )

        fusion_model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        fusion_model = (
            fusion_model
            .to(device)
            .eval()
        )

        training_summary = checkpoint[
            "training_summary"
        ]

        generated_now = False

    else:
        (
            fusion_model,
            history_df,
            training_summary,
        ) = train_binary_model(
            model=fusion_model,
            train_loader=(
                fusion_train_loader
            ),
            validation_loader=(
                fusion_validation_loader
            ),
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            maximum_epochs=MAX_EPOCHS,
            patience=(
                EARLY_STOPPING_PATIENCE
            ),
        )

        history_df["seed"] = seed
        history_df.to_csv(
            history_path,
            index=False
        )


    learned_weights = (
        fusion_model
        .normalized_weights()
        .detach()
        .cpu()
        .numpy()
    )

    text_weight = float(
        learned_weights[0]
    )

    image_weight = float(
        learned_weights[1]
    )


    validation_logits, _ = (
        predict_binary_logits(
            fusion_model,
            fusion_validation_loader
        )
    )

    validation_probabilities = (
        sigmoid_numpy(
            validation_logits
        )
    )


    selected_threshold, _ = (
        select_validation_threshold(
            labels=y_validation,
            probabilities=(
                validation_probabilities
            ),
            threshold_candidates=(
                THRESHOLD_CANDIDATES
            ),
        )
    )


    validation_metrics = (
        calculate_binary_metrics(
            labels=y_validation,
            probabilities=(
                validation_probabilities
            ),
            threshold=selected_threshold,
        )
    )


    if generated_now:
        torch.save(
            {
                "model_name": (
                    "learnable_weighted_"
                    "logit_fusion"
                ),
                "seed": seed,
                "model_state_dict": {
                    key: value
                    .detach()
                    .cpu()
                    for key, value
                    in fusion_model
                    .state_dict()
                    .items()
                },
                "text_component_checkpoint": str(
                    model_dir
                    / (
                        "text_only_mlp_"
                        f"seed_{seed}.pt"
                    )
                ),
                "image_component_checkpoint": str(
                    model_dir
                    / (
                        "image_only_mlp_"
                        f"seed_{seed}.pt"
                    )
                ),
                "learned_text_weight": (
                    text_weight
                ),
                "learned_image_weight": (
                    image_weight
                ),
                "selected_threshold": (
                    selected_threshold
                ),
                "training_summary": (
                    training_summary
                ),
                "validation_metrics": (
                    validation_metrics
                ),
            },
            checkpoint_path
        )


    weighted_fusion_paths[
        seed
    ] = checkpoint_path


    weighted_fusion_rows.append({
        "model_name": (
            "learnable_weighted_"
            "logit_fusion"
        ),
        "seed": seed,
        "best_epoch": int(
            training_summary[
                "best_epoch"
            ]
        ),
        "epochs_completed": int(
            training_summary[
                "epochs_completed"
            ]
        ),
        "best_validation_loss": float(
            training_summary[
                "best_validation_loss"
            ]
        ),
        "learned_text_weight": (
            text_weight
        ),
        "learned_image_weight": (
            image_weight
        ),
        "selected_threshold": (
            selected_threshold
        ),
        "validation_precision": (
            validation_metrics[
                "precision"
            ]
        ),
        "validation_recall": (
            validation_metrics[
                "recall"
            ]
        ),
        "validation_f1": (
            validation_metrics[
                "f1"
            ]
        ),
        "validation_pr_auc": (
            validation_metrics[
                "pr_auc"
            ]
        ),
        "validation_brier_score": (
            validation_metrics[
                "brier_score"
            ]
        ),
        "generated_now": (
            generated_now
        ),
    })


    print(
        "  Text weight:",
        round(text_weight, 4)
    )

    print(
        "  Image weight:",
        round(image_weight, 4)
    )

    print(
        "  Threshold:",
        selected_threshold
    )

    print(
        "  Validation F1:",
        round(
            validation_metrics["f1"],
            6
        )
    )


    del text_model
    del image_model
    del fusion_model

    gc.collect()

    if device.type == "mps":
        torch.mps.empty_cache()


weighted_fusion_summary_df = (
    pd.DataFrame(
        weighted_fusion_rows
    )
    .sort_values("seed")
    .reset_index(drop=True)
)


weighted_fusion_aggregate_df = pd.DataFrame([
    {
        "model_name": (
            "learnable_weighted_"
            "logit_fusion"
        ),
        "number_of_seeds": len(
            weighted_fusion_summary_df
        ),
        "mean_text_weight": (
            weighted_fusion_summary_df[
                "learned_text_weight"
            ].mean()
        ),
        "std_text_weight": (
            weighted_fusion_summary_df[
                "learned_text_weight"
            ].std(ddof=1)
        ),
        "mean_image_weight": (
            weighted_fusion_summary_df[
                "learned_image_weight"
            ].mean()
        ),
        "std_image_weight": (
            weighted_fusion_summary_df[
                "learned_image_weight"
            ].std(ddof=1)
        ),
        "mean_selected_threshold": (
            weighted_fusion_summary_df[
                "selected_threshold"
            ].mean()
        ),
        "mean_validation_f1": (
            weighted_fusion_summary_df[
                "validation_f1"
            ].mean()
        ),
        "std_validation_f1": (
            weighted_fusion_summary_df[
                "validation_f1"
            ].std(ddof=1)
        ),
        "mean_validation_pr_auc": (
            weighted_fusion_summary_df[
                "validation_pr_auc"
            ].mean()
        ),
        "mean_validation_brier_score": (
            weighted_fusion_summary_df[
                "validation_brier_score"
            ].mean()
        ),
    }
])


weighted_fusion_summary_path = (
    summary_dir
    / (
        "learnable_weighted_logit_"
        "fusion_validation_summary.csv"
    )
)

weighted_fusion_aggregate_path = (
    summary_dir
    / (
        "learnable_weighted_logit_"
        "fusion_validation_aggregate.csv"
    )
)


weighted_fusion_summary_df.to_csv(
    weighted_fusion_summary_path,
    index=False
)

weighted_fusion_aggregate_df.to_csv(
    weighted_fusion_aggregate_path,
    index=False
)


print(
    "\n=== FINAL V2 LEARNABLE WEIGHTED "
    "LOGIT FUSION SUMMARY ==="
)

display(
    weighted_fusion_summary_df
)

print(
    "\n=== LEARNABLE WEIGHTED LOGIT "
    "FUSION AGGREGATE ==="
)

display(
    weighted_fusion_aggregate_df
)

# Publication notebook cell pub-042
class RawConcatenationFusion(torch.nn.Module):
    def __init__(
        self,
        text_dim=1024,
        image_dim=1152,
        hidden_dim=256,
        dropout=0.30,
    ):
        super().__init__()

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(
                text_dim + image_dim,
                hidden_dim
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(
                hidden_dim,
                1
            ),
        )

    def forward(
        self,
        text_embeddings,
        image_embeddings,
    ):
        combined_embeddings = torch.cat(
            [
                text_embeddings,
                image_embeddings,
            ],
            dim=1
        )

        return self.classifier(
            combined_embeddings
        ).squeeze(-1)


RAW_CONCAT_MODEL_NAME = (
    "raw_concatenation_fusion"
)

RAW_CONCAT_PARAMETER_COUNT = 557569
RETRAIN_RAW_CONCAT = False

raw_concat_rows = []
raw_concat_model_paths = {}


for seed in TRAINING_SEEDS:
    print(
        f"\nRaw Concatenation Fusion "
        f"— seed {seed}"
    )

    set_training_seed(seed)

    train_loader = create_tensor_loader(
        feature_arrays=[
            X_train_text,
            X_train_image,
        ],
        labels=y_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=seed,
    )

    validation_loader = create_tensor_loader(
        feature_arrays=[
            X_validation_text,
            X_validation_image,
        ],
        labels=y_validation,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=seed,
    )

    model = RawConcatenationFusion(
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT_RATE,
    )

    parameter_count = (
        count_trainable_parameters(model)
    )

    if (
        parameter_count
        != RAW_CONCAT_PARAMETER_COUNT
    ):
        raise ValueError(
            "Jumlah parameter Raw Concatenation "
            f"salah: {parameter_count}"
        )

    checkpoint_path = (
        model_dir
        / (
            "raw_concatenation_fusion_"
            f"seed_{seed}.pt"
        )
    )

    history_path = (
        diagnostic_dir
        / (
            "training_history_"
            "raw_concatenation_fusion_"
            f"seed_{seed}.csv"
        )
    )

    generated_now = True

    if (
        checkpoint_path.exists()
        and not RETRAIN_RAW_CONCAT
    ):
        checkpoint = load_torch_checkpoint(
            checkpoint_path
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        model = model.to(device).eval()

        training_summary = checkpoint[
            "training_summary"
        ]

        generated_now = False

    else:
        (
            model,
            history_df,
            training_summary,
        ) = train_binary_model(
            model=model,
            train_loader=train_loader,
            validation_loader=(
                validation_loader
            ),
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            maximum_epochs=MAX_EPOCHS,
            patience=(
                EARLY_STOPPING_PATIENCE
            ),
        )

        history_df["seed"] = seed
        history_df["model_name"] = (
            RAW_CONCAT_MODEL_NAME
        )

        history_df.to_csv(
            history_path,
            index=False
        )

    validation_logits, labels = (
        predict_binary_logits(
            model,
            validation_loader
        )
    )

    validation_probabilities = (
        sigmoid_numpy(
            validation_logits
        )
    )

    selected_threshold, _ = (
        select_validation_threshold(
            labels=labels,
            probabilities=(
                validation_probabilities
            ),
            threshold_candidates=(
                THRESHOLD_CANDIDATES
            ),
        )
    )

    validation_metrics = (
        calculate_binary_metrics(
            labels=labels,
            probabilities=(
                validation_probabilities
            ),
            threshold=selected_threshold,
        )
    )

    if generated_now:
        torch.save(
            {
                "model_name": (
                    RAW_CONCAT_MODEL_NAME
                ),
                "seed": seed,
                "model_state_dict": {
                    key: value.detach().cpu()
                    for key, value
                    in model.state_dict().items()
                },
                "training_summary": (
                    training_summary
                ),
                "selected_threshold": (
                    selected_threshold
                ),
                "validation_metrics": (
                    validation_metrics
                ),
                "trainable_parameters": (
                    parameter_count
                ),
            },
            checkpoint_path
        )

    raw_concat_model_paths[
        seed
    ] = checkpoint_path

    raw_concat_rows.append({
        "model_name": (
            RAW_CONCAT_MODEL_NAME
        ),
        "seed": seed,
        "trainable_parameters": (
            parameter_count
        ),
        "best_epoch": (
            training_summary["best_epoch"]
        ),
        "epochs_completed": (
            training_summary[
                "epochs_completed"
            ]
        ),
        "best_validation_loss": (
            training_summary[
                "best_validation_loss"
            ]
        ),
        "selected_threshold": (
            selected_threshold
        ),
        "validation_precision": (
            validation_metrics["precision"]
        ),
        "validation_recall": (
            validation_metrics["recall"]
        ),
        "validation_f1": (
            validation_metrics["f1"]
        ),
        "validation_pr_auc": (
            validation_metrics["pr_auc"]
        ),
        "validation_brier_score": (
            validation_metrics[
                "brier_score"
            ]
        ),
        "generated_now": generated_now,
    })

    print(
        "  Best epoch:",
        training_summary["best_epoch"]
    )

    print(
        "  Threshold:",
        selected_threshold
    )

    print(
        "  Validation F1:",
        round(
            validation_metrics["f1"],
            6
        )
    )

    del model
    del train_loader
    del validation_loader

    gc.collect()

    if device.type == "mps":
        torch.mps.empty_cache()


raw_concat_summary_df = (
    pd.DataFrame(raw_concat_rows)
    .sort_values("seed")
    .reset_index(drop=True)
)


raw_concat_aggregate_df = pd.DataFrame([
    {
        "model_name": (
            RAW_CONCAT_MODEL_NAME
        ),
        "number_of_seeds": len(
            raw_concat_summary_df
        ),
        "mean_best_epoch": (
            raw_concat_summary_df[
                "best_epoch"
            ].mean()
        ),
        "mean_selected_threshold": (
            raw_concat_summary_df[
                "selected_threshold"
            ].mean()
        ),
        "mean_validation_f1": (
            raw_concat_summary_df[
                "validation_f1"
            ].mean()
        ),
        "std_validation_f1": (
            raw_concat_summary_df[
                "validation_f1"
            ].std(ddof=1)
        ),
        "minimum_validation_f1": (
            raw_concat_summary_df[
                "validation_f1"
            ].min()
        ),
        "maximum_validation_f1": (
            raw_concat_summary_df[
                "validation_f1"
            ].max()
        ),
        "mean_validation_pr_auc": (
            raw_concat_summary_df[
                "validation_pr_auc"
            ].mean()
        ),
        "mean_validation_brier_score": (
            raw_concat_summary_df[
                "validation_brier_score"
            ].mean()
        ),
    }
])


raw_concat_summary_path = (
    summary_dir
    / (
        "raw_concatenation_fusion_"
        "validation_summary.csv"
    )
)

raw_concat_aggregate_path = (
    summary_dir
    / (
        "raw_concatenation_fusion_"
        "validation_aggregate.csv"
    )
)


raw_concat_summary_df.to_csv(
    raw_concat_summary_path,
    index=False
)

raw_concat_aggregate_df.to_csv(
    raw_concat_aggregate_path,
    index=False
)


print(
    "\n=== FINAL V2 RAW CONCATENATION "
    "FUSION SUMMARY ==="
)

display(raw_concat_summary_df)

print(
    "\n=== RAW CONCATENATION FUSION "
    "AGGREGATE ==="
)

display(raw_concat_aggregate_df)

# Publication notebook cell pub-044
class ProjectionFusion(torch.nn.Module):
    def __init__(
        self,
        text_dim=1024,
        image_dim=1152,
        projection_dim=512,
        hidden_dim=256,
        dropout=0.30,
    ):
        super().__init__()

        self.text_projection = torch.nn.Sequential(
            torch.nn.Linear(
                text_dim,
                projection_dim
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

        self.image_projection = torch.nn.Sequential(
            torch.nn.Linear(
                image_dim,
                projection_dim
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(
                projection_dim * 2,
                hidden_dim
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(
                hidden_dim,
                1
            ),
        )

    def forward(
        self,
        text_embeddings,
        image_embeddings,
    ):
        text_features = self.text_projection(
            text_embeddings
        )

        image_features = self.image_projection(
            image_embeddings
        )

        fused_features = torch.cat(
            [
                text_features,
                image_features,
            ],
            dim=1
        )

        return self.classifier(
            fused_features
        ).squeeze(-1)

# Publication notebook cell pub-045
def run_two_input_experiment(
    model_name,
    model_factory,
    expected_parameters,
    retrain=False,
):
    result_rows = []
    checkpoint_paths = {}

    for seed in TRAINING_SEEDS:
        print(
            f"\n{model_name} — seed {seed}"
        )

        set_training_seed(seed)

        train_loader = create_tensor_loader(
            feature_arrays=[
                X_train_text,
                X_train_image,
            ],
            labels=y_train,
            batch_size=BATCH_SIZE,
            shuffle=True,
            seed=seed,
        )

        validation_loader = create_tensor_loader(
            feature_arrays=[
                X_validation_text,
                X_validation_image,
            ],
            labels=y_validation,
            batch_size=BATCH_SIZE,
            shuffle=False,
            seed=seed,
        )

        model = model_factory()

        parameter_count = (
            count_trainable_parameters(model)
        )

        if parameter_count != expected_parameters:
            raise ValueError(
                f"Parameter {model_name} salah: "
                f"{parameter_count} "
                f"!= {expected_parameters}"
            )

        checkpoint_path = (
            model_dir
            / f"{model_name}_seed_{seed}.pt"
        )

        history_path = (
            diagnostic_dir
            / (
                f"training_history_"
                f"{model_name}_seed_{seed}.csv"
            )
        )

        generated_now = True

        if (
            checkpoint_path.exists()
            and not retrain
        ):
            checkpoint = load_torch_checkpoint(
                checkpoint_path
            )

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            model = model.to(device).eval()

            training_summary = checkpoint[
                "training_summary"
            ]

            generated_now = False

        else:
            (
                model,
                history_df,
                training_summary,
            ) = train_binary_model(
                model=model,
                train_loader=train_loader,
                validation_loader=(
                    validation_loader
                ),
                learning_rate=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                maximum_epochs=MAX_EPOCHS,
                patience=(
                    EARLY_STOPPING_PATIENCE
                ),
            )

            history_df["seed"] = seed
            history_df["model_name"] = (
                model_name
            )

            history_df.to_csv(
                history_path,
                index=False
            )

        validation_logits, labels = (
            predict_binary_logits(
                model,
                validation_loader
            )
        )

        if not np.array_equal(
            labels.astype(np.int64),
            y_validation.astype(np.int64)
        ):
            raise ValueError(
                "Urutan validation label berubah."
            )

        validation_probabilities = (
            sigmoid_numpy(
                validation_logits
            )
        )

        selected_threshold, _ = (
            select_validation_threshold(
                labels=labels,
                probabilities=(
                    validation_probabilities
                ),
                threshold_candidates=(
                    THRESHOLD_CANDIDATES
                ),
            )
        )

        validation_metrics = (
            calculate_binary_metrics(
                labels=labels,
                probabilities=(
                    validation_probabilities
                ),
                threshold=(
                    selected_threshold
                ),
            )
        )

        if generated_now:
            torch.save(
                {
                    "model_name": model_name,
                    "seed": seed,
                    "model_state_dict": {
                        key: value
                        .detach()
                        .cpu()
                        for key, value
                        in model.state_dict().items()
                    },
                    "training_summary": (
                        training_summary
                    ),
                    "selected_threshold": (
                        selected_threshold
                    ),
                    "validation_metrics": (
                        validation_metrics
                    ),
                    "trainable_parameters": (
                        parameter_count
                    ),
                },
                checkpoint_path
            )

        checkpoint_paths[seed] = (
            checkpoint_path
        )

        result_rows.append({
            "model_name": model_name,
            "seed": seed,
            "trainable_parameters": (
                parameter_count
            ),
            "best_epoch": (
                training_summary[
                    "best_epoch"
                ]
            ),
            "epochs_completed": (
                training_summary[
                    "epochs_completed"
                ]
            ),
            "best_validation_loss": (
                training_summary[
                    "best_validation_loss"
                ]
            ),
            "selected_threshold": (
                selected_threshold
            ),
            "validation_precision": (
                validation_metrics[
                    "precision"
                ]
            ),
            "validation_recall": (
                validation_metrics[
                    "recall"
                ]
            ),
            "validation_f1": (
                validation_metrics["f1"]
            ),
            "validation_pr_auc": (
                validation_metrics[
                    "pr_auc"
                ]
            ),
            "validation_brier_score": (
                validation_metrics[
                    "brier_score"
                ]
            ),
            "generated_now": generated_now,
        })

        print(
            "  Best epoch:",
            training_summary["best_epoch"]
        )

        print(
            "  Threshold:",
            selected_threshold
        )

        print(
            "  Validation F1:",
            round(
                validation_metrics["f1"],
                6
            )
        )

        del model
        del train_loader
        del validation_loader

        gc.collect()

        if device.type == "mps":
            torch.mps.empty_cache()

    summary_df = (
        pd.DataFrame(result_rows)
        .sort_values("seed")
        .reset_index(drop=True)
    )

    aggregate_df = pd.DataFrame([
        {
            "model_name": model_name,
            "number_of_seeds": len(
                summary_df
            ),
            "mean_best_epoch": (
                summary_df[
                    "best_epoch"
                ].mean()
            ),
            "mean_selected_threshold": (
                summary_df[
                    "selected_threshold"
                ].mean()
            ),
            "mean_validation_f1": (
                summary_df[
                    "validation_f1"
                ].mean()
            ),
            "std_validation_f1": (
                summary_df[
                    "validation_f1"
                ].std(ddof=1)
            ),
            "minimum_validation_f1": (
                summary_df[
                    "validation_f1"
                ].min()
            ),
            "maximum_validation_f1": (
                summary_df[
                    "validation_f1"
                ].max()
            ),
            "mean_validation_pr_auc": (
                summary_df[
                    "validation_pr_auc"
                ].mean()
            ),
            "mean_validation_brier_score": (
                summary_df[
                    "validation_brier_score"
                ].mean()
            ),
        }
    ])

    summary_path = (
        summary_dir
        / f"{model_name}_validation_summary.csv"
    )

    aggregate_path = (
        summary_dir
        / f"{model_name}_validation_aggregate.csv"
    )

    summary_df.to_csv(
        summary_path,
        index=False
    )

    aggregate_df.to_csv(
        aggregate_path,
        index=False
    )

    return (
        summary_df,
        aggregate_df,
        checkpoint_paths,
    )

# Publication notebook cell pub-046
PROJECTION_FUSION_PARAMETER_COUNT = 1377793

(
    projection_fusion_summary_df,
    projection_fusion_aggregate_df,
    projection_fusion_model_paths,
) = run_two_input_experiment(
    model_name="projection_fusion",
    model_factory=lambda: ProjectionFusion(
        text_dim=1024,
        image_dim=1152,
        projection_dim=PROJECTION_DIM,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT_RATE,
    ),
    expected_parameters=(
        PROJECTION_FUSION_PARAMETER_COUNT
    ),
    retrain=False,
)


print(
    "\n=== FINAL V2 PROJECTION "
    "FUSION SUMMARY ==="
)

display(
    projection_fusion_summary_df
)

print(
    "\n=== PROJECTION FUSION "
    "AGGREGATE ==="
)

display(
    projection_fusion_aggregate_df
)

# Publication notebook cell pub-047
# Code Cell

class ContrastiveSimilarityFusion(
    torch.nn.Module
):
    def __init__(
        self,
        text_dim=1024,
        image_dim=1152,
        projection_dim=512,
    ):
        super().__init__()

        self.text_projection = torch.nn.Linear(
            text_dim,
            projection_dim
        )

        self.image_projection = torch.nn.Linear(
            image_dim,
            projection_dim
        )

        # exp(log(1 / 0.07)) = 14.2857
        self.logit_scale = torch.nn.Parameter(
            torch.tensor(
                np.log(1 / 0.07),
                dtype=torch.float32
            )
        )

    def forward(
        self,
        text_embeddings,
        image_embeddings,
    ):
        text_features = F.normalize(
            self.text_projection(
                text_embeddings
            ),
            p=2,
            dim=1
        )

        image_features = F.normalize(
            self.image_projection(
                image_embeddings
            ),
            p=2,
            dim=1
        )

        cosine_similarity = torch.sum(
            text_features * image_features,
            dim=1
        )

        # Pembatasan mengikuti praktik umum
        # agar scale tidak membesar tanpa batas.
        scale = (
            self.logit_scale
            .exp()
            .clamp(max=100.0)
        )

        return scale * cosine_similarity

# Publication notebook cell pub-048
contrastive_config = {
    "model_name": (
        "contrastive_similarity_fusion"
    ),
    "text_dimension": 1024,
    "image_dimension": 1152,
    "projection_dimension": 512,
    "similarity": "cosine",
    "normalization": "L2",
    "initial_temperature": 0.07,
    "initial_logit_scale": float(
        1 / 0.07
    ),
    "maximum_logit_scale": 100.0,
    "loss": "BCEWithLogitsLoss",
    "encoder_status": "frozen",
}

contrastive_config_path = (
    config_dir
    / (
        "contrastive_similarity_"
        "fusion_config.json"
    )
)

with contrastive_config_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        contrastive_config,
        file,
        indent=2
    )

# Publication notebook cell pub-049
CONTRASTIVE_PARAMETER_COUNT = 1115137

(
    contrastive_summary_df,
    contrastive_aggregate_df,
    contrastive_model_paths,
) = run_two_input_experiment(
    model_name=(
        "contrastive_similarity_fusion"
    ),
    model_factory=lambda: (
        ContrastiveSimilarityFusion(
            text_dim=1024,
            image_dim=1152,
            projection_dim=PROJECTION_DIM,
        )
    ),
    expected_parameters=(
        CONTRASTIVE_PARAMETER_COUNT
    ),
    retrain=False,
)

# Publication notebook cell pub-050
temperature_rows = []

for seed in TRAINING_SEEDS:
    checkpoint = load_torch_checkpoint(
        contrastive_model_paths[seed]
    )

    raw_logit_scale = float(
        checkpoint[
            "model_state_dict"
        ]["logit_scale"].item()
    )

    learned_logit_scale = min(
        float(np.exp(raw_logit_scale)),
        100.0
    )

    learned_temperature = (
        1.0 / learned_logit_scale
    )

    temperature_rows.append({
        "seed": seed,
        "learned_logit_scale": (
            learned_logit_scale
        ),
        "learned_temperature": (
            learned_temperature
        ),
    })


temperature_df = pd.DataFrame(
    temperature_rows
)


contrastive_summary_df = (
    contrastive_summary_df.merge(
        temperature_df,
        on="seed",
        how="left",
        validate="one_to_one"
    )
)


contrastive_aggregate_df[
    "mean_logit_scale"
] = (
    contrastive_summary_df[
        "learned_logit_scale"
    ].mean()
)

contrastive_aggregate_df[
    "std_logit_scale"
] = (
    contrastive_summary_df[
        "learned_logit_scale"
    ].std(ddof=1)
)

contrastive_aggregate_df[
    "mean_temperature"
] = (
    contrastive_summary_df[
        "learned_temperature"
    ].mean()
)


contrastive_summary_path = (
    summary_dir
    / (
        "contrastive_similarity_fusion_"
        "validation_summary.csv"
    )
)

contrastive_aggregate_path = (
    summary_dir
    / (
        "contrastive_similarity_fusion_"
        "validation_aggregate.csv"
    )
)


contrastive_summary_df.to_csv(
    contrastive_summary_path,
    index=False
)

contrastive_aggregate_df.to_csv(
    contrastive_aggregate_path,
    index=False
)


print(
    "\n=== FINAL V2 CONTRASTIVE "
    "SIMILARITY FUSION SUMMARY ==="
)

display(
    contrastive_summary_df
)

print(
    "\n=== CONTRASTIVE SIMILARITY "
    "FUSION AGGREGATE ==="
)

display(
    contrastive_aggregate_df
)

# Publication notebook cell pub-052
validation_aggregate_frames = [
    text_only_aggregate_df,
    image_only_aggregate_df,
    average_fusion_aggregate_df,
    weighted_fusion_aggregate_df,
    raw_concat_aggregate_df,
    projection_fusion_aggregate_df,
    contrastive_aggregate_df,
]

validation_comparison_df = pd.concat(
    validation_aggregate_frames,
    ignore_index=True,
    sort=False
)

comparison_columns = [
    "model_name",
    "number_of_seeds",
    "mean_validation_f1",
    "std_validation_f1",
    "mean_validation_pr_auc",
    "mean_validation_brier_score",
    "mean_selected_threshold",
]

validation_comparison_df = (
    validation_comparison_df[
        comparison_columns
    ]
    .sort_values(
        "mean_validation_f1",
        ascending=False
    )
    .reset_index(drop=True)
)

validation_comparison_df[
    "f1_rank"
] = (
    validation_comparison_df[
        "mean_validation_f1"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

validation_comparison_df[
    "pr_auc_rank"
] = (
    validation_comparison_df[
        "mean_validation_pr_auc"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)

validation_comparison_df[
    "brier_rank"
] = (
    validation_comparison_df[
        "mean_validation_brier_score"
    ]
    .rank(
        ascending=True,
        method="min"
    )
    .astype(int)
)

validation_comparison_path = (
    summary_dir
    / "final_clean_validation_model_comparison.csv"
)

validation_comparison_df.to_csv(
    validation_comparison_path,
    index=False
)

if len(validation_comparison_df) != 7:
    raise ValueError(
        "Jumlah model pada validation comparison "
        "harus sama dengan 7."
    )

if not (
    validation_comparison_df[
        "number_of_seeds"
    ] == 5
).all():
    raise ValueError(
        "Tidak semua model memiliki lima seed."
    )

print(
    "=== FINAL V2 CLEAN VALIDATION "
    "MODEL COMPARISON ==="
)

display(validation_comparison_df)

print(
    "\nModels:",
    len(validation_comparison_df)
)

print(
    "All models have five seeds:",
    bool(
        (
            validation_comparison_df[
                "number_of_seeds"
            ] == 5
        ).all()
    )
)

print(
    "Best validation F1:",
    validation_comparison_df.iloc[0][
        "model_name"
    ]
)

print(
    "\nSaved to:"
)

print(validation_comparison_path)

# Publication notebook cell pub-054
def infer_model_logits(
    model,
    feature_arrays,
    labels,
    seed,
):
    loader = create_tensor_loader(
        feature_arrays=feature_arrays,
        labels=labels,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=seed,
    )

    logits, returned_labels = (
        predict_binary_logits(
            model,
            loader
        )
    )

    if not np.array_equal(
        returned_labels.astype(np.int64),
        np.asarray(labels, dtype=np.int64)
    ):
        raise ValueError(
            "Urutan label test berubah."
        )

    return logits.astype(np.float32)


def load_multimodal_checkpoint_model(
    model_name,
    seed,
):
    checkpoint_path = (
        model_dir
        / f"{model_name}_seed_{seed}.pt"
    )

    checkpoint = load_torch_checkpoint(
        checkpoint_path
    )

    if model_name == "raw_concatenation_fusion":
        model = RawConcatenationFusion(
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT_RATE,
        )

    elif model_name == "projection_fusion":
        model = ProjectionFusion(
            text_dim=1024,
            image_dim=1152,
            projection_dim=PROJECTION_DIM,
            hidden_dim=HIDDEN_DIM,
            dropout=DROPOUT_RATE,
        )

    elif (
        model_name
        == "contrastive_similarity_fusion"
    ):
        model = ContrastiveSimilarityFusion(
            text_dim=1024,
            image_dim=1152,
            projection_dim=PROJECTION_DIM,
        )

    else:
        raise ValueError(
            f"Model tidak dikenal: {model_name}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return (
        model.to(device).eval(),
        checkpoint,
    )


clean_test_metric_rows = []
clean_test_prediction_frames = []
clean_test_logits_cache = {}


def record_clean_test_result(
    model_name,
    seed,
    logits,
    threshold,
):
    probabilities = sigmoid_numpy(logits)

    metrics = calculate_binary_metrics(
        labels=y_test,
        probabilities=probabilities,
        threshold=threshold,
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    clean_test_metric_rows.append({
        "model_name": model_name,
        "seed": seed,
        "threshold": threshold,
        "test_accuracy": metrics["accuracy"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_f1": metrics["f1"],
        "test_roc_auc": metrics["roc_auc"],
        "test_pr_auc": metrics["pr_auc"],
        "test_brier_score": (
            metrics["brier_score"]
        ),
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
    })

    prediction_df = test_df[
        [
            "id",
            "platform",
            "label",
        ]
    ].copy()

    prediction_df["model_name"] = model_name
    prediction_df["seed"] = seed
    prediction_df["condition"] = "clean"
    prediction_df["logit"] = logits
    prediction_df["probability"] = (
        probabilities
    )
    prediction_df["threshold"] = threshold
    prediction_df["prediction"] = predictions
    prediction_df["is_correct"] = (
        predictions
        == y_test.astype(np.int64)
    )

    clean_test_prediction_frames.append(
        prediction_df
    )

    clean_test_logits_cache[
        (model_name, seed)
    ] = logits


for seed in TRAINING_SEEDS:
    print(f"\nClean test — seed {seed}")

    # Text-only
    text_checkpoint = load_torch_checkpoint(
        model_dir
        / f"text_only_mlp_seed_{seed}.pt"
    )

    text_model = TextOnlyMLP(
        input_dim=1024,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT_RATE,
    )

    text_model.load_state_dict(
        text_checkpoint["model_state_dict"]
    )

    text_model = text_model.to(device).eval()

    text_logits = infer_model_logits(
        model=text_model,
        feature_arrays=[X_test_text],
        labels=y_test,
        seed=seed,
    )

    record_clean_test_result(
        model_name="text_only_mlp",
        seed=seed,
        logits=text_logits,
        threshold=float(
            text_checkpoint[
                "selected_threshold"
            ]
        ),
    )


    # Image-only
    image_checkpoint = load_torch_checkpoint(
        model_dir
        / f"image_only_mlp_seed_{seed}.pt"
    )

    image_model = ImageOnlyMLP(
        input_dim=1152,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT_RATE,
    )

    image_model.load_state_dict(
        image_checkpoint["model_state_dict"]
    )

    image_model = image_model.to(device).eval()

    image_logits = infer_model_logits(
        model=image_model,
        feature_arrays=[X_test_image],
        labels=y_test,
        seed=seed,
    )

    record_clean_test_result(
        model_name="image_only_mlp",
        seed=seed,
        logits=image_logits,
        threshold=float(
            image_checkpoint[
                "selected_threshold"
            ]
        ),
    )


    # Average Logit Fusion
    average_descriptor_path = (
        model_dir
        / (
            "average_logit_fusion_"
            f"seed_{seed}.json"
        )
    )

    with average_descriptor_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        average_descriptor = json.load(file)

    average_logits = (
        0.5 * text_logits
        + 0.5 * image_logits
    )

    record_clean_test_result(
        model_name="average_logit_fusion",
        seed=seed,
        logits=average_logits,
        threshold=float(
            average_descriptor[
                "selected_threshold"
            ]
        ),
    )


    # Learnable Weighted Fusion
    weighted_checkpoint = (
        load_torch_checkpoint(
            model_dir
            / (
                "learnable_weighted_logit_"
                f"fusion_seed_{seed}.pt"
            )
        )
    )

    text_weight = float(
        weighted_checkpoint[
            "learned_text_weight"
        ]
    )

    image_weight = float(
        weighted_checkpoint[
            "learned_image_weight"
        ]
    )

    weighted_logits = (
        text_weight * text_logits
        + image_weight * image_logits
    )

    record_clean_test_result(
        model_name=(
            "learnable_weighted_logit_fusion"
        ),
        seed=seed,
        logits=weighted_logits,
        threshold=float(
            weighted_checkpoint[
                "selected_threshold"
            ]
        ),
    )


    # Trainable multimodal models
    for model_name in [
        "raw_concatenation_fusion",
        "projection_fusion",
        "contrastive_similarity_fusion",
    ]:
        model, checkpoint = (
            load_multimodal_checkpoint_model(
                model_name,
                seed,
            )
        )

        logits = infer_model_logits(
            model=model,
            feature_arrays=[
                X_test_text,
                X_test_image,
            ],
            labels=y_test,
            seed=seed,
        )

        record_clean_test_result(
            model_name=model_name,
            seed=seed,
            logits=logits,
            threshold=float(
                checkpoint[
                    "selected_threshold"
                ]
            ),
        )

        del model


    print("  Models evaluated: 7")

    del text_model
    del image_model

    gc.collect()

    if device.type == "mps":
        torch.mps.empty_cache()


clean_test_summary_df = (
    pd.DataFrame(clean_test_metric_rows)
    .sort_values(
        [
            "model_name",
            "seed",
        ]
    )
    .reset_index(drop=True)
)


clean_test_aggregate_df = (
    clean_test_summary_df
    .groupby(
        "model_name",
        as_index=False
    )
    .agg(
        number_of_seeds=(
            "seed",
            "nunique"
        ),
        mean_test_f1=(
            "test_f1",
            "mean"
        ),
        std_test_f1=(
            "test_f1",
            "std"
        ),
        mean_test_precision=(
            "test_precision",
            "mean"
        ),
        mean_test_recall=(
            "test_recall",
            "mean"
        ),
        mean_test_pr_auc=(
            "test_pr_auc",
            "mean"
        ),
        mean_test_roc_auc=(
            "test_roc_auc",
            "mean"
        ),
        mean_test_brier_score=(
            "test_brier_score",
            "mean"
        ),
    )
    .sort_values(
        "mean_test_f1",
        ascending=False
    )
    .reset_index(drop=True)
)


clean_test_aggregate_df[
    "clean_test_rank"
] = (
    np.arange(
        1,
        len(clean_test_aggregate_df) + 1
    )
)


clean_test_predictions_df = pd.concat(
    clean_test_prediction_frames,
    ignore_index=True
)


clean_test_summary_path = (
    summary_dir
    / "clean_test_per_seed_summary.csv"
)

clean_test_aggregate_path = (
    summary_dir
    / "clean_test_model_aggregate.csv"
)

clean_test_predictions_path = (
    prediction_dir
    / "clean_test_all_model_predictions.csv"
)


clean_test_summary_df.to_csv(
    clean_test_summary_path,
    index=False
)

clean_test_aggregate_df.to_csv(
    clean_test_aggregate_path,
    index=False
)

clean_test_predictions_df.to_csv(
    clean_test_predictions_path,
    index=False
)


if len(clean_test_summary_df) != 35:
    raise ValueError(
        "Hasil clean test harus berisi "
        "7 model × 5 seed = 35 baris."
    )


print(
    "\n=== FINAL V2 CLEAN TEST "
    "PER-SEED SUMMARY ==="
)

display(clean_test_summary_df)

print(
    "\n=== FINAL V2 CLEAN TEST "
    "MODEL AGGREGATE ==="
)

display(clean_test_aggregate_df)

print(
    "\nTotal evaluations:",
    len(clean_test_summary_df)
)

print(
    "Prediction rows:",
    len(clean_test_predictions_df)
)

print("\nSaved to:")
print(clean_test_summary_path)
print(clean_test_aggregate_path)
print(clean_test_predictions_path)

# Publication notebook cell pub-056
# Pastikan embedding hasil perturbation tersedia
X_test_text_obfuscated = np.asarray(
    test_text_obfuscated_embeddings,
    dtype=np.float32
)

X_test_image_corrupted = {
    condition: np.asarray(
        embeddings,
        dtype=np.float32
    )
    for condition, embeddings
    in perturbed_image_embeddings.items()
}


robustness_conditions = {
    "clean": {
        "text": X_test_text,
        "image": X_test_image,
        "category": "clean",
    },

    "text_obfuscated": {
        "text": X_test_text_obfuscated,
        "image": X_test_image,
        "category": "text_only_perturbation",
    },

    "gaussian_blur_r2": {
        "text": X_test_text,
        "image": X_test_image_corrupted[
            "gaussian_blur_r2"
        ],
        "category": "image_only_perturbation",
    },

    "gaussian_blur_r5": {
        "text": X_test_text,
        "image": X_test_image_corrupted[
            "gaussian_blur_r5"
        ],
        "category": "image_only_perturbation",
    },

    "pixelation_p50": {
        "text": X_test_text,
        "image": X_test_image_corrupted[
            "pixelation_p50"
        ],
        "category": "image_only_perturbation",
    },

    "pixelation_p25": {
        "text": X_test_text,
        "image": X_test_image_corrupted[
            "pixelation_p25"
        ],
        "category": "image_only_perturbation",
    },

    "pixelation_p12_5": {
        "text": X_test_text,
        "image": X_test_image_corrupted[
            "pixelation_p12_5"
        ],
        "category": "image_only_perturbation",
    },

    "blur_r5_pixelation_p12_5": {
        "text": X_test_text,
        "image": X_test_image_corrupted[
            "blur_r5_pixelation_p12_5"
        ],
        "category": "compound_image_perturbation",
    },

    "joint_j1_text_obfuscated_blur_r5": {
        "text": X_test_text_obfuscated,
        "image": X_test_image_corrupted[
            "gaussian_blur_r5"
        ],
        "category": "joint_perturbation",
    },

    "joint_j2_text_obfuscated_pixelation_p12_5": {
        "text": X_test_text_obfuscated,
        "image": X_test_image_corrupted[
            "pixelation_p12_5"
        ],
        "category": "joint_perturbation",
    },

    "joint_j3_text_obfuscated_blur_r5_pixelation_p12_5": {
        "text": X_test_text_obfuscated,
        "image": X_test_image_corrupted[
            "blur_r5_pixelation_p12_5"
        ],
        "category": "joint_perturbation",
    },
}


condition_audit_rows = []

for condition_name, condition_data in (
    robustness_conditions.items()
):
    text_array = condition_data["text"]
    image_array = condition_data["image"]

    text_shape_valid = (
        text_array.shape
        == (len(test_df), 1024)
    )

    image_shape_valid = (
        image_array.shape
        == (len(test_df), 1152)
    )

    finite_values = bool(
        np.isfinite(text_array).all()
        and np.isfinite(image_array).all()
    )

    condition_audit_rows.append({
        "condition": condition_name,
        "category": condition_data["category"],
        "text_rows": text_array.shape[0],
        "text_dimensions": text_array.shape[1],
        "image_rows": image_array.shape[0],
        "image_dimensions": image_array.shape[1],
        "text_shape_valid": text_shape_valid,
        "image_shape_valid": image_shape_valid,
        "finite_values": finite_values,
    })


robustness_condition_audit_df = pd.DataFrame(
    condition_audit_rows
)


if not (
    robustness_condition_audit_df[
        [
            "text_shape_valid",
            "image_shape_valid",
            "finite_values",
        ]
    ].all().all()
):
    display(
        robustness_condition_audit_df
    )

    raise ValueError(
        "Ada kondisi robustness yang tidak valid."
    )


robustness_condition_audit_path = (
    summary_dir
    / "robustness_condition_input_audit.csv"
)

robustness_condition_audit_df.to_csv(
    robustness_condition_audit_path,
    index=False
)


robustness_condition_config = {
    condition_name: {
        "category": condition_data["category"],
        "text_shape": list(
            condition_data["text"].shape
        ),
        "image_shape": list(
            condition_data["image"].shape
        ),
    }
    for condition_name, condition_data
    in robustness_conditions.items()
}


robustness_condition_config_path = (
    config_dir
    / "robustness_test_conditions.json"
)

with robustness_condition_config_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        robustness_condition_config,
        file,
        indent=2,
        ensure_ascii=False
    )


print(
    "=== FINAL V2 ROBUSTNESS "
    "CONDITION AUDIT ==="
)

display(
    robustness_condition_audit_df
)

print(
    "\nTotal conditions:",
    len(robustness_conditions)
)

print(
    "Joint conditions:",
    int(
        (
            robustness_condition_audit_df[
                "category"
            ]
            == "joint_perturbation"
        ).sum()
    )
)

print(
    "All inputs valid:",
    bool(
        robustness_condition_audit_df[
            [
                "text_shape_valid",
                "image_shape_valid",
                "finite_values",
            ]
        ].all().all()
    )
)

print("\nSaved to:")
print(robustness_condition_audit_path)
print(robustness_condition_config_path)

# Publication notebook cell pub-058
robustness_metric_rows = []
robustness_prediction_frames = []


def record_robustness_result(
    model_name,
    seed,
    condition_name,
    category,
    logits,
    threshold,
):
    probabilities = sigmoid_numpy(logits)

    metrics = calculate_binary_metrics(
        labels=y_test,
        probabilities=probabilities,
        threshold=threshold,
    )

    predictions = (
        probabilities >= threshold
    ).astype(np.int64)

    robustness_metric_rows.append({
        "model_name": model_name,
        "seed": seed,
        "condition": condition_name,
        "category": category,
        "threshold": threshold,
        "test_accuracy": metrics["accuracy"],
        "test_precision": metrics["precision"],
        "test_recall": metrics["recall"],
        "test_f1": metrics["f1"],
        "test_roc_auc": metrics["roc_auc"],
        "test_pr_auc": metrics["pr_auc"],
        "test_brier_score": metrics[
            "brier_score"
        ],
        "tn": metrics["tn"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "tp": metrics["tp"],
    })

    prediction_df = test_df[
        [
            "id",
            "platform",
            "label",
        ]
    ].copy()

    prediction_df["model_name"] = model_name
    prediction_df["seed"] = seed
    prediction_df["condition"] = (
        condition_name
    )
    prediction_df["category"] = category
    prediction_df["logit"] = logits
    prediction_df["probability"] = (
        probabilities
    )
    prediction_df["threshold"] = threshold
    prediction_df["prediction"] = predictions
    prediction_df["is_correct"] = (
        predictions
        == y_test.astype(np.int64)
    )

    robustness_prediction_frames.append(
        prediction_df
    )


for seed_index, seed in enumerate(
    TRAINING_SEEDS,
    start=1
):
    print(
        f"\n[{seed_index}/"
        f"{len(TRAINING_SEEDS)}] "
        f"Robustness evaluation — seed {seed}"
    )

    # Unimodal models
    text_checkpoint = load_torch_checkpoint(
        model_dir
        / f"text_only_mlp_seed_{seed}.pt"
    )

    image_checkpoint = load_torch_checkpoint(
        model_dir
        / f"image_only_mlp_seed_{seed}.pt"
    )

    text_model = TextOnlyMLP(
        input_dim=1024,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT_RATE,
    )

    image_model = ImageOnlyMLP(
        input_dim=1152,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT_RATE,
    )

    text_model.load_state_dict(
        text_checkpoint["model_state_dict"]
    )

    image_model.load_state_dict(
        image_checkpoint["model_state_dict"]
    )

    text_model = text_model.to(device).eval()
    image_model = image_model.to(device).eval()


    # Average Fusion configuration
    average_descriptor_path = (
        model_dir
        / (
            "average_logit_fusion_"
            f"seed_{seed}.json"
        )
    )

    with average_descriptor_path.open(
        "r",
        encoding="utf-8"
    ) as file:
        average_descriptor = json.load(file)


    # Learnable Weighted Fusion configuration
    weighted_checkpoint = (
        load_torch_checkpoint(
            model_dir
            / (
                "learnable_weighted_logit_"
                f"fusion_seed_{seed}.pt"
            )
        )
    )

    weighted_text_weight = float(
        weighted_checkpoint[
            "learned_text_weight"
        ]
    )

    weighted_image_weight = float(
        weighted_checkpoint[
            "learned_image_weight"
        ]
    )


    # Trainable multimodal models
    multimodal_models = {}
    multimodal_checkpoints = {}

    for model_name in [
        "raw_concatenation_fusion",
        "projection_fusion",
        "contrastive_similarity_fusion",
    ]:
        model, checkpoint = (
            load_multimodal_checkpoint_model(
                model_name,
                seed,
            )
        )

        multimodal_models[
            model_name
        ] = model

        multimodal_checkpoints[
            model_name
        ] = checkpoint


    for condition_name, condition_data in (
        robustness_conditions.items()
    ):
        print(
            f"  {condition_name}",
            end="\r"
        )

        condition_text = (
            condition_data["text"]
        )

        condition_image = (
            condition_data["image"]
        )

        category = condition_data[
            "category"
        ]


        # Text-only logits
        text_logits = infer_model_logits(
            model=text_model,
            feature_arrays=[
                condition_text
            ],
            labels=y_test,
            seed=seed,
        )

        record_robustness_result(
            model_name="text_only_mlp",
            seed=seed,
            condition_name=condition_name,
            category=category,
            logits=text_logits,
            threshold=float(
                text_checkpoint[
                    "selected_threshold"
                ]
            ),
        )


        # Image-only logits
        image_logits = infer_model_logits(
            model=image_model,
            feature_arrays=[
                condition_image
            ],
            labels=y_test,
            seed=seed,
        )

        record_robustness_result(
            model_name="image_only_mlp",
            seed=seed,
            condition_name=condition_name,
            category=category,
            logits=image_logits,
            threshold=float(
                image_checkpoint[
                    "selected_threshold"
                ]
            ),
        )


        # Average Logit Fusion
        average_logits = (
            0.5 * text_logits
            + 0.5 * image_logits
        )

        record_robustness_result(
            model_name=(
                "average_logit_fusion"
            ),
            seed=seed,
            condition_name=condition_name,
            category=category,
            logits=average_logits,
            threshold=float(
                average_descriptor[
                    "selected_threshold"
                ]
            ),
        )


        # Learnable Weighted Logit Fusion
        weighted_logits = (
            weighted_text_weight
            * text_logits
            +
            weighted_image_weight
            * image_logits
        )

        record_robustness_result(
            model_name=(
                "learnable_weighted_"
                "logit_fusion"
            ),
            seed=seed,
            condition_name=condition_name,
            category=category,
            logits=weighted_logits,
            threshold=float(
                weighted_checkpoint[
                    "selected_threshold"
                ]
            ),
        )


        # Raw Concatenation, Projection,
        # dan Contrastive Similarity
        for model_name, model in (
            multimodal_models.items()
        ):
            logits = infer_model_logits(
                model=model,
                feature_arrays=[
                    condition_text,
                    condition_image,
                ],
                labels=y_test,
                seed=seed,
            )

            record_robustness_result(
                model_name=model_name,
                seed=seed,
                condition_name=condition_name,
                category=category,
                logits=logits,
                threshold=float(
                    multimodal_checkpoints[
                        model_name
                    ][
                        "selected_threshold"
                    ]
                ),
            )

    print(
        f"  Completed "
        f"{len(robustness_conditions)} "
        "conditions."
    )


    del text_model
    del image_model

    for model in multimodal_models.values():
        del model

    gc.collect()

    if device.type == "mps":
        torch.mps.empty_cache()

    elif device.type == "cuda":
        torch.cuda.empty_cache()


# Per-seed metrics
robustness_per_seed_df = (
    pd.DataFrame(
        robustness_metric_rows
    )
)


expected_metric_rows = (
    7
    * len(TRAINING_SEEDS)
    * len(robustness_conditions)
)

if (
    len(robustness_per_seed_df)
    != expected_metric_rows
):
    raise ValueError(
        "Jumlah robustness evaluation salah: "
        f"{len(robustness_per_seed_df)} "
        f"!= {expected_metric_rows}"
    )


# Tambahkan clean reference untuk setiap model × seed
clean_reference_df = (
    robustness_per_seed_df[
        robustness_per_seed_df[
            "condition"
        ] == "clean"
    ][
        [
            "model_name",
            "seed",
            "test_f1",
            "test_pr_auc",
            "test_brier_score",
        ]
    ]
    .rename(
        columns={
            "test_f1": "clean_f1",
            "test_pr_auc": (
                "clean_pr_auc"
            ),
            "test_brier_score": (
                "clean_brier_score"
            ),
        }
    )
)


robustness_per_seed_df = (
    robustness_per_seed_df.merge(
        clean_reference_df,
        on=[
            "model_name",
            "seed",
        ],
        how="left",
        validate="many_to_one"
    )
)


robustness_per_seed_df[
    "f1_drop"
] = (
    robustness_per_seed_df["clean_f1"]
    - robustness_per_seed_df["test_f1"]
)

robustness_per_seed_df[
    "relative_f1_drop_pct"
] = (
    robustness_per_seed_df["f1_drop"]
    / robustness_per_seed_df["clean_f1"]
    * 100
)

robustness_per_seed_df[
    "pr_auc_drop"
] = (
    robustness_per_seed_df[
        "clean_pr_auc"
    ]
    - robustness_per_seed_df[
        "test_pr_auc"
    ]
)

robustness_per_seed_df[
    "brier_increase"
] = (
    robustness_per_seed_df[
        "test_brier_score"
    ]
    - robustness_per_seed_df[
        "clean_brier_score"
    ]
)


# Aggregate lima seed
robustness_aggregate_df = (
    robustness_per_seed_df
    .groupby(
        [
            "model_name",
            "condition",
            "category",
        ],
        as_index=False
    )
    .agg(
        number_of_seeds=(
            "seed",
            "nunique"
        ),
        mean_test_f1=(
            "test_f1",
            "mean"
        ),
        std_test_f1=(
            "test_f1",
            "std"
        ),
        mean_test_precision=(
            "test_precision",
            "mean"
        ),
        mean_test_recall=(
            "test_recall",
            "mean"
        ),
        mean_test_pr_auc=(
            "test_pr_auc",
            "mean"
        ),
        mean_test_brier_score=(
            "test_brier_score",
            "mean"
        ),
        mean_f1_drop=(
            "f1_drop",
            "mean"
        ),
        mean_relative_f1_drop_pct=(
            "relative_f1_drop_pct",
            "mean"
        ),
        mean_pr_auc_drop=(
            "pr_auc_drop",
            "mean"
        ),
        mean_brier_increase=(
            "brier_increase",
            "mean"
        ),
    )
)


# Joint-condition summary
joint_robustness_df = (
    robustness_aggregate_df[
        robustness_aggregate_df[
            "category"
        ] == "joint_perturbation"
    ]
    .sort_values(
        [
            "condition",
            "mean_test_f1",
        ],
        ascending=[
            True,
            False,
        ]
    )
    .reset_index(drop=True)
)


# Worst-case condition setiap model
perturbed_aggregate_df = (
    robustness_aggregate_df[
        robustness_aggregate_df[
            "condition"
        ] != "clean"
    ]
)


worst_case_indices = (
    perturbed_aggregate_df
    .groupby("model_name")[
        "mean_test_f1"
    ]
    .idxmin()
)


worst_case_df = (
    perturbed_aggregate_df
    .loc[
        worst_case_indices,
        [
            "model_name",
            "condition",
            "category",
            "mean_test_f1",
            "std_test_f1",
            "mean_f1_drop",
            "mean_relative_f1_drop_pct",
            "mean_test_pr_auc",
            "mean_brier_increase",
        ]
    ]
    .rename(
        columns={
            "condition": (
                "worst_condition"
            ),
            "category": (
                "worst_category"
            ),
            "mean_test_f1": (
                "worst_case_f1"
            ),
            "std_test_f1": (
                "worst_case_f1_std"
            ),
        }
    )
    .sort_values(
        "worst_case_f1",
        ascending=False
    )
    .reset_index(drop=True)
)


# Semua predictions
robustness_predictions_df = pd.concat(
    robustness_prediction_frames,
    ignore_index=True
)


expected_prediction_rows = (
    expected_metric_rows
    * len(test_df)
)

if (
    len(robustness_predictions_df)
    != expected_prediction_rows
):
    raise ValueError(
        "Jumlah prediction rows salah."
    )


# Save outputs
robustness_per_seed_path = (
    summary_dir
    / "robustness_per_seed_summary.csv"
)

robustness_aggregate_path = (
    summary_dir
    / "robustness_model_condition_aggregate.csv"
)

joint_robustness_path = (
    summary_dir
    / "joint_robustness_aggregate.csv"
)

worst_case_path = (
    summary_dir
    / "robustness_worst_case_summary.csv"
)

robustness_predictions_path = (
    prediction_dir
    / "robustness_all_predictions.csv"
)


robustness_per_seed_df.to_csv(
    robustness_per_seed_path,
    index=False
)

robustness_aggregate_df.to_csv(
    robustness_aggregate_path,
    index=False
)

joint_robustness_df.to_csv(
    joint_robustness_path,
    index=False
)

worst_case_df.to_csv(
    worst_case_path,
    index=False
)

robustness_predictions_df.to_csv(
    robustness_predictions_path,
    index=False
)


print(
    "\n=== FINAL V2 JOINT "
    "ROBUSTNESS AGGREGATE ==="
)

display(
    joint_robustness_df[
        [
            "condition",
            "model_name",
            "mean_test_f1",
            "std_test_f1",
            "mean_f1_drop",
            "mean_relative_f1_drop_pct",
            "mean_test_pr_auc",
            "mean_test_brier_score",
        ]
    ]
)


print(
    "\n=== FINAL V2 ROBUSTNESS "
    "WORST-CASE SUMMARY ==="
)

display(worst_case_df)


print(
    "\nMetric evaluations:",
    len(robustness_per_seed_df)
)

print(
    "Prediction rows:",
    len(robustness_predictions_df)
)

print(
    "All thresholds frozen:",
    True
)

print("\nSaved to:")
print(robustness_per_seed_path)
print(robustness_aggregate_path)
print(joint_robustness_path)
print(worst_case_path)
print(robustness_predictions_path)

# Publication notebook cell pub-060
BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_SEED = 42

primary_conditions = [
    "clean",
    "joint_j1_text_obfuscated_blur_r5",
    "joint_j2_text_obfuscated_pixelation_p12_5",
    (
        "joint_j3_text_obfuscated_"
        "blur_r5_pixelation_p12_5"
    ),
]

top_model_pairs = [
    (
        "learnable_weighted_logit_fusion",
        "average_logit_fusion",
    ),
    (
        "learnable_weighted_logit_fusion",
        "raw_concatenation_fusion",
    ),
    (
        "raw_concatenation_fusion",
        "average_logit_fusion",
    ),
]


def get_prediction_matrix(
    predictions_df,
    model_name,
    condition,
):
    filtered_df = predictions_df[
        (
            predictions_df["model_name"]
            == model_name
        )
        &
        (
            predictions_df["condition"]
            == condition
        )
    ]

    prediction_matrix_df = (
        filtered_df.pivot(
            index="id",
            columns="seed",
            values="prediction",
        )
        .reindex(test_df["id"])
    )

    if prediction_matrix_df.isna().any().any():
        raise ValueError(
            f"Prediction tidak lengkap: "
            f"{model_name}, {condition}"
        )

    if prediction_matrix_df.shape != (
        len(test_df),
        len(TRAINING_SEEDS),
    ):
        raise ValueError(
            f"Shape prediction salah: "
            f"{prediction_matrix_df.shape}"
        )

    return prediction_matrix_df.to_numpy(
        dtype=np.int64
    )


def calculate_mean_seed_f1(
    labels,
    prediction_matrix,
    sample_indices,
):
    sampled_labels = (
        labels[sample_indices]
        .reshape(-1, 1)
    )

    sampled_predictions = (
        prediction_matrix[
            sample_indices
        ]
    )

    true_positive = (
        (
            sampled_labels == 1
        )
        &
        (
            sampled_predictions == 1
        )
    ).sum(axis=0)

    false_positive = (
        (
            sampled_labels == 0
        )
        &
        (
            sampled_predictions == 1
        )
    ).sum(axis=0)

    false_negative = (
        (
            sampled_labels == 1
        )
        &
        (
            sampled_predictions == 0
        )
    ).sum(axis=0)

    denominator = (
        2 * true_positive
        + false_positive
        + false_negative
    )

    seed_f1 = np.divide(
        2 * true_positive,
        denominator,
        out=np.zeros_like(
            denominator,
            dtype=np.float64
        ),
        where=denominator != 0,
    )

    return float(seed_f1.mean())


bootstrap_rng = np.random.default_rng(
    BOOTSTRAP_SEED
)

bootstrap_rows = []

labels_array = y_test.astype(np.int64)
full_indices = np.arange(len(labels_array))


for condition in primary_conditions:
    print(f"\nBootstrap: {condition}")

    model_matrices = {
        model_name: get_prediction_matrix(
            predictions_df=(
                robustness_predictions_df
            ),
            model_name=model_name,
            condition=condition,
        )
        for model_name in {
            model
            for pair in top_model_pairs
            for model in pair
        }
    }

    for model_a, model_b in top_model_pairs:
        matrix_a = model_matrices[model_a]
        matrix_b = model_matrices[model_b]

        observed_f1_a = (
            calculate_mean_seed_f1(
                labels_array,
                matrix_a,
                full_indices,
            )
        )

        observed_f1_b = (
            calculate_mean_seed_f1(
                labels_array,
                matrix_b,
                full_indices,
            )
        )

        observed_difference = (
            observed_f1_a
            - observed_f1_b
        )

        bootstrap_differences = np.empty(
            BOOTSTRAP_ITERATIONS,
            dtype=np.float64,
        )

        for iteration in range(
            BOOTSTRAP_ITERATIONS
        ):
            sampled_indices = (
                bootstrap_rng.integers(
                    low=0,
                    high=len(labels_array),
                    size=len(labels_array),
                )
            )

            bootstrap_f1_a = (
                calculate_mean_seed_f1(
                    labels_array,
                    matrix_a,
                    sampled_indices,
                )
            )

            bootstrap_f1_b = (
                calculate_mean_seed_f1(
                    labels_array,
                    matrix_b,
                    sampled_indices,
                )
            )

            bootstrap_differences[
                iteration
            ] = (
                bootstrap_f1_a
                - bootstrap_f1_b
            )

        ci_lower, ci_upper = np.percentile(
            bootstrap_differences,
            [2.5, 97.5],
        )

        probability_difference_le_zero = (
            bootstrap_differences <= 0
        ).mean()

        probability_difference_ge_zero = (
            bootstrap_differences >= 0
        ).mean()

        approximate_p_value = min(
            1.0,
            2
            * min(
                probability_difference_le_zero,
                probability_difference_ge_zero,
            ),
        )

        significant = bool(
            ci_lower > 0
            or ci_upper < 0
        )

        if observed_difference > 0:
            favored_model = model_a
        elif observed_difference < 0:
            favored_model = model_b
        else:
            favored_model = "tie"

        bootstrap_rows.append({
            "condition": condition,
            "model_a": model_a,
            "model_b": model_b,
            "observed_mean_f1_a": (
                observed_f1_a
            ),
            "observed_mean_f1_b": (
                observed_f1_b
            ),
            "observed_f1_difference_a_minus_b": (
                observed_difference
            ),
            "ci_95_lower": float(ci_lower),
            "ci_95_upper": float(ci_upper),
            "approximate_p_value": float(
                approximate_p_value
            ),
            "significant_95_percent": (
                significant
            ),
            "favored_model": favored_model,
            "bootstrap_iterations": (
                BOOTSTRAP_ITERATIONS
            ),
        })

        print(
            f"  {model_a} vs {model_b}: "
            f"diff={observed_difference:.6f}, "
            f"CI=[{ci_lower:.6f}, "
            f"{ci_upper:.6f}]"
        )


paired_bootstrap_top3_df = pd.DataFrame(
    bootstrap_rows
)


paired_bootstrap_path = (
    bootstrap_dir
    / (
        "paired_bootstrap_top3_"
        "primary_conditions.csv"
    )
)

paired_bootstrap_top3_df.to_csv(
    paired_bootstrap_path,
    index=False
)


print(
    "\n=== FINAL V2 PAIRED BOOTSTRAP "
    "TOP-3 MODELS ==="
)

display(
    paired_bootstrap_top3_df[
        [
            "condition",
            "model_a",
            "model_b",
            "observed_f1_difference_a_minus_b",
            "ci_95_lower",
            "ci_95_upper",
            "approximate_p_value",
            "significant_95_percent",
            "favored_model",
        ]
    ]
)

print(
    "\nComparisons:",
    len(paired_bootstrap_top3_df)
)

print(
    "Bootstrap iterations:",
    BOOTSTRAP_ITERATIONS
)

print(
    "Saved to:",
    paired_bootstrap_path
)

# Publication notebook cell pub-062
platform_metric_rows = []

for (
    model_name,
    seed,
    condition,
    platform_name
), group_df in robustness_predictions_df.groupby(
    [
        "model_name",
        "seed",
        "condition",
        "platform",
    ]
):
    labels = (
        group_df["label"]
        .astype(np.int64)
        .to_numpy()
    )

    probabilities = (
        group_df["probability"]
        .to_numpy(dtype=np.float64)
    )

    threshold = float(
        group_df["threshold"].iloc[0]
    )

    metrics = calculate_binary_metrics(
        labels=labels,
        probabilities=probabilities,
        threshold=threshold,
    )

    platform_metric_rows.append({
        "model_name": model_name,
        "seed": seed,
        "condition": condition,
        "platform": platform_name,
        "samples": len(group_df),
        "positive_samples": int(
            labels.sum()
        ),
        "f1": metrics["f1"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "pr_auc": metrics["pr_auc"],
        "brier_score": metrics[
            "brier_score"
        ],
    })


platform_per_seed_df = pd.DataFrame(
    platform_metric_rows
)


platform_aggregate_df = (
    platform_per_seed_df
    .groupby(
        [
            "model_name",
            "condition",
            "platform",
        ],
        as_index=False
    )
    .agg(
        samples=("samples", "first"),
        positive_samples=(
            "positive_samples",
            "first"
        ),
        mean_f1=("f1", "mean"),
        std_f1=("f1", "std"),
        mean_precision=(
            "precision",
            "mean"
        ),
        mean_recall=("recall", "mean"),
        mean_pr_auc=("pr_auc", "mean"),
        mean_brier_score=(
            "brier_score",
            "mean"
        ),
    )
)


primary_platform_conditions = [
    "clean",
    "joint_j1_text_obfuscated_blur_r5",
    (
        "joint_j2_text_obfuscated_"
        "pixelation_p12_5"
    ),
    (
        "joint_j3_text_obfuscated_"
        "blur_r5_pixelation_p12_5"
    ),
]


primary_platform_df = (
    platform_aggregate_df[
        platform_aggregate_df[
            "condition"
        ].isin(
            primary_platform_conditions
        )
    ]
    .sort_values(
        [
            "condition",
            "platform",
            "mean_f1",
        ],
        ascending=[
            True,
            True,
            False,
        ]
    )
    .reset_index(drop=True)
)


platform_per_seed_path = (
    per_platform_dir
    / "robustness_platform_per_seed.csv"
)

platform_aggregate_path = (
    per_platform_dir
    / "robustness_platform_aggregate.csv"
)

primary_platform_path = (
    per_platform_dir
    / "primary_conditions_platform_summary.csv"
)


platform_per_seed_df.to_csv(
    platform_per_seed_path,
    index=False
)

platform_aggregate_df.to_csv(
    platform_aggregate_path,
    index=False
)

primary_platform_df.to_csv(
    primary_platform_path,
    index=False
)


print(
    "=== FINAL V2 PRIMARY CONDITIONS "
    "BY PLATFORM ==="
)

display(
    primary_platform_df[
        [
            "condition",
            "platform",
            "model_name",
            "samples",
            "positive_samples",
            "mean_f1",
            "std_f1",
            "mean_precision",
            "mean_recall",
            "mean_pr_auc",
        ]
    ]
)

print(
    "\nPlatforms:",
    sorted(
        primary_platform_df[
            "platform"
        ].unique()
    )
)

print(
    "Rows:",
    len(primary_platform_df)
)

print("\nSaved to:")
print(platform_per_seed_path)
print(platform_aggregate_path)
print(primary_platform_path)

# Publication notebook cell pub-064
clean_platform_reference_df = (
    platform_aggregate_df[
        platform_aggregate_df[
            "condition"
        ] == "clean"
    ][
        [
            "model_name",
            "platform",
            "mean_f1",
        ]
    ]
    .rename(
        columns={
            "mean_f1": "clean_f1"
        }
    )
)


joint_platform_drop_df = (
    platform_aggregate_df[
        platform_aggregate_df[
            "condition"
        ].isin(
            [
                "joint_j1_text_obfuscated_blur_r5",
                (
                    "joint_j2_text_obfuscated_"
                    "pixelation_p12_5"
                ),
                (
                    "joint_j3_text_obfuscated_"
                    "blur_r5_pixelation_p12_5"
                ),
            ]
        )
    ]
    .merge(
        clean_platform_reference_df,
        on=[
            "model_name",
            "platform",
        ],
        how="left",
        validate="many_to_one",
    )
)


joint_platform_drop_df[
    "f1_drop_from_clean"
] = (
    joint_platform_drop_df[
        "clean_f1"
    ]
    -
    joint_platform_drop_df[
        "mean_f1"
    ]
)


joint_platform_drop_df[
    "relative_f1_drop_pct"
] = (
    joint_platform_drop_df[
        "f1_drop_from_clean"
    ]
    /
    joint_platform_drop_df[
        "clean_f1"
    ]
    * 100
)


condition_short_names = {
    "joint_j1_text_obfuscated_blur_r5": "J1",
    (
        "joint_j2_text_obfuscated_"
        "pixelation_p12_5"
    ): "J2",
    (
        "joint_j3_text_obfuscated_"
        "blur_r5_pixelation_p12_5"
    ): "J3",
}


joint_platform_drop_df[
    "joint_condition"
] = (
    joint_platform_drop_df[
        "condition"
    ]
    .map(condition_short_names)
)


platform_robustness_summary_df = (
    joint_platform_drop_df[
        [
            "joint_condition",
            "platform",
            "model_name",
            "samples",
            "positive_samples",
            "clean_f1",
            "mean_f1",
            "std_f1",
            "f1_drop_from_clean",
            "relative_f1_drop_pct",
        ]
    ]
    .sort_values(
        [
            "joint_condition",
            "platform",
            "mean_f1",
        ],
        ascending=[
            True,
            True,
            False,
        ]
    )
    .reset_index(drop=True)
)


# Model dengan F1 tertinggi untuk
# setiap condition × platform
best_platform_indices = (
    platform_robustness_summary_df
    .groupby(
        [
            "joint_condition",
            "platform",
        ]
    )[
        "mean_f1"
    ]
    .idxmax()
)


best_model_per_platform_df = (
    platform_robustness_summary_df
    .loc[
        best_platform_indices
    ]
    .sort_values(
        [
            "joint_condition",
            "platform",
        ]
    )
    .reset_index(drop=True)
)


platform_robustness_summary_path = (
    per_platform_dir
    / "joint_platform_robustness_drop.csv"
)

best_model_per_platform_path = (
    per_platform_dir
    / "best_model_per_joint_platform.csv"
)


platform_robustness_summary_df.to_csv(
    platform_robustness_summary_path,
    index=False
)

best_model_per_platform_df.to_csv(
    best_model_per_platform_path,
    index=False
)


print(
    "=== FINAL V2 BEST MODEL "
    "PER JOINT CONDITION AND PLATFORM ==="
)

display(
    best_model_per_platform_df[
        [
            "joint_condition",
            "platform",
            "model_name",
            "samples",
            "positive_samples",
            "clean_f1",
            "mean_f1",
            "f1_drop_from_clean",
            "relative_f1_drop_pct",
        ]
    ]
)


print(
    "\n=== PLATFORM ROBUSTNESS "
    "SUMMARY — TOP THREE FUSIONS ==="
)

display(
    platform_robustness_summary_df[
        platform_robustness_summary_df[
            "model_name"
        ].isin(
            [
                "average_logit_fusion",
                (
                    "learnable_weighted_"
                    "logit_fusion"
                ),
                "raw_concatenation_fusion",
            ]
        )
    ]
)


print("\nSaved to:")
print(platform_robustness_summary_path)
print(best_model_per_platform_path)

# Publication notebook cell pub-066
tiktok_conditions = {
    "clean": "Clean",
    "joint_j1_text_obfuscated_blur_r5": "J1",
    "joint_j2_text_obfuscated_pixelation_p12_5": "J2",
    (
        "joint_j3_text_obfuscated_"
        "blur_r5_pixelation_p12_5"
    ): "J3",
}

tiktok_models = [
    "average_logit_fusion",
    "learnable_weighted_logit_fusion",
    "raw_concatenation_fusion",
]


tiktok_df = robustness_predictions_df[
    (
        robustness_predictions_df["platform"]
        == "TikTok"
    )
    &
    (
        robustness_predictions_df["model_name"]
        .isin(tiktok_models)
    )
    &
    (
        robustness_predictions_df["condition"]
        .isin(tiktok_conditions)
    )
].copy()


confusion_rows = []

for (
    model_name,
    seed,
    condition
), group_df in tiktok_df.groupby(
    [
        "model_name",
        "seed",
        "condition",
    ]
):
    labels = group_df[
        "label"
    ].to_numpy(dtype=np.int64)

    predictions = group_df[
        "prediction"
    ].to_numpy(dtype=np.int64)

    tn, fp, fn, tp = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    ).ravel()

    confusion_rows.append({
        "model_name": model_name,
        "seed": seed,
        "condition": (
            tiktok_conditions[condition]
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": precision_score(
            labels,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            labels,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            labels,
            predictions,
            zero_division=0
        ),
        "false_negative_rate": (
            fn / (fn + tp)
            if (fn + tp) > 0
            else 0
        ),
        "false_positive_rate": (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0
        ),
    })


tiktok_confusion_per_seed_df = pd.DataFrame(
    confusion_rows
)


tiktok_confusion_summary_df = (
    tiktok_confusion_per_seed_df
    .groupby(
        [
            "condition",
            "model_name",
        ],
        as_index=False
    )
    .agg(
        mean_tn=("tn", "mean"),
        mean_fp=("fp", "mean"),
        mean_fn=("fn", "mean"),
        mean_tp=("tp", "mean"),
        mean_precision=(
            "precision",
            "mean"
        ),
        mean_recall=("recall", "mean"),
        mean_f1=("f1", "mean"),
        mean_false_negative_rate=(
            "false_negative_rate",
            "mean"
        ),
        mean_false_positive_rate=(
            "false_positive_rate",
            "mean"
        ),
    )
    .sort_values(
        [
            "condition",
            "mean_f1",
        ],
        ascending=[
            True,
            False,
        ]
    )
    .reset_index(drop=True)
)


tiktok_confusion_path = (
    diagnostic_dir
    / "tiktok_top3_confusion_summary.csv"
)

tiktok_confusion_summary_df.to_csv(
    tiktok_confusion_path,
    index=False
)


print(
    "=== FINAL V2 TIKTOK "
    "CONFUSION DIAGNOSIS ==="
)

display(
    tiktok_confusion_summary_df
)

print(
    "\nTikTok samples:",
    tiktok_df["id"].nunique()
)

print(
    "Positive samples:",
    int(
        test_df[
            test_df["platform"]
            == "TikTok"
        ]["label"].sum()
    )
)

print(
    "Saved to:",
    tiktok_confusion_path
)

# Publication notebook cell pub-068
tiktok_positive_predictions_df = (
    robustness_predictions_df[
        (
            robustness_predictions_df["platform"]
            == "TikTok"
        )
        &
        (
            robustness_predictions_df["label"]
            == 1
        )
        &
        (
            robustness_predictions_df["model_name"]
            .isin(tiktok_models)
        )
        &
        (
            robustness_predictions_df["condition"]
            .isin(tiktok_conditions)
        )
    ]
    .copy()
)


persistent_fn_df = (
    tiktok_positive_predictions_df
    .groupby(
        [
            "id",
            "model_name",
            "condition",
        ],
        as_index=False
    )
    .agg(
        missed_seed_count=(
            "prediction",
            lambda values:
            int((values == 0).sum())
        ),
        correct_seed_count=(
            "prediction",
            lambda values:
            int((values == 1).sum())
        ),
        mean_probability=(
            "probability",
            "mean"
        ),
        minimum_probability=(
            "probability",
            "min"
        ),
        maximum_probability=(
            "probability",
            "max"
        ),
        mean_threshold=(
            "threshold",
            "mean"
        ),
    )
)


persistent_fn_df[
    "condition_short"
] = (
    persistent_fn_df["condition"]
    .map(tiktok_conditions)
)


persistent_fn_df = persistent_fn_df[
    persistent_fn_df[
        "missed_seed_count"
    ] >= 4
].copy()


sample_context_df = test_df[
    [
        "id",
        "text_caption",
        "image_path",
    ]
].copy()


obfuscated_context_df = (
    test_text_obfuscated_df[
        [
            "id",
            "obfuscated_caption",
        ]
    ]
)


persistent_fn_df = (
    persistent_fn_df
    .merge(
        sample_context_df,
        on="id",
        how="left",
        validate="many_to_one"
    )
    .merge(
        obfuscated_context_df,
        on="id",
        how="left",
        validate="many_to_one"
    )
    .sort_values(
        [
            "missed_seed_count",
            "mean_probability",
        ],
        ascending=[
            False,
            True,
        ]
    )
    .reset_index(drop=True)
)


persistent_fn_summary_df = (
    persistent_fn_df
    .groupby(
        [
            "condition_short",
            "model_name",
        ],
        as_index=False
    )
    .agg(
        persistent_fn_samples=(
            "id",
            "nunique"
        ),
        missed_by_all_five_seeds=(
            "missed_seed_count",
            lambda values:
            int((values == 5).sum())
        ),
        mean_probability=(
            "mean_probability",
            "mean"
        ),
    )
    .sort_values(
        [
            "condition_short",
            "persistent_fn_samples",
        ],
        ascending=[
            True,
            False,
        ]
    )
)


persistent_fn_path = (
    diagnostic_dir
    / "tiktok_persistent_false_negatives.csv"
)

persistent_fn_summary_path = (
    diagnostic_dir
    / "tiktok_persistent_false_negative_summary.csv"
)


persistent_fn_df.to_csv(
    persistent_fn_path,
    index=False
)

persistent_fn_summary_df.to_csv(
    persistent_fn_summary_path,
    index=False
)


print(
    "=== FINAL V2 TIKTOK PERSISTENT "
    "FALSE-NEGATIVE SUMMARY ==="
)

display(
    persistent_fn_summary_df
)


print(
    "\n=== MOST PERSISTENT FALSE "
    "NEGATIVE SAMPLES ==="
)

display(
    persistent_fn_df[
        [
            "id",
            "condition_short",
            "model_name",
            "missed_seed_count",
            "mean_probability",
            "mean_threshold",
            "text_caption",
            "obfuscated_caption",
            "image_path",
        ]
    ].head(20)
)


print(
    "\nUnique persistent TikTok positives:",
    persistent_fn_df["id"].nunique()
)

print("\nSaved to:")
print(persistent_fn_path)
print(persistent_fn_summary_path)

# Publication notebook cell pub-070
# Pilih enam ID yang paling konsisten gagal
persistent_visual_ids = (
    persistent_fn_df
    .groupby("id", as_index=False)
    .agg(
        maximum_missed_seeds=(
            "missed_seed_count",
            "max"
        ),
        minimum_mean_probability=(
            "mean_probability",
            "min"
        ),
    )
    .sort_values(
        [
            "maximum_missed_seeds",
            "minimum_mean_probability",
        ],
        ascending=[False, True]
    )
    .head(6)["id"]
    .tolist()
)


review_df = test_df[
    [
        "id",
        "platform",
        "label",
        "text_caption",
        "image_path",
    ]
].copy()


review_df = review_df.merge(
    test_text_obfuscated_df[
        [
            "id",
            "obfuscated_caption",
        ]
    ],
    on="id",
    how="left",
)


review_manifests = {
    "blur_r5_path": (
        image_manifest_dir
        / "test_gaussian_blur_r5.csv"
    ),
    "pixelation_p12_5_path": (
        image_manifest_dir
        / "test_pixelation_p12_5.csv"
    ),
    "compound_path": (
        image_manifest_dir
        / (
            "test_blur_r5_"
            "pixelation_p12_5.csv"
        )
    ),
}


for column_name, manifest_path in (
    review_manifests.items()
):
    path_df = pd.read_csv(
        manifest_path
    )[
        [
            "id",
            "perturbed_image_path",
        ]
    ].rename(
        columns={
            "perturbed_image_path":
            column_name
        }
    )

    review_df = review_df.merge(
        path_df,
        on="id",
        how="left",
        validate="one_to_one",
    )


review_df = (
    review_df[
        review_df["id"].isin(
            persistent_visual_ids
        )
    ]
    .set_index("id")
    .loc[persistent_visual_ids]
    .reset_index()
)


image_columns = {
    "Clean": "image_path",
    "Blur r5": "blur_r5_path",
    "Pixelation 12.5%": (
        "pixelation_p12_5_path"
    ),
    "Blur r5 → Pixelation 12.5%": (
        "compound_path"
    ),
}


figure, axes = plt.subplots(
    nrows=len(review_df),
    ncols=4,
    figsize=(14, 4 * len(review_df)),
)

axes = np.atleast_2d(axes)


for row_index, row in review_df.iterrows():
    for column_index, (
        title,
        path_column
    ) in enumerate(
        image_columns.items()
    ):
        image = load_rgb_image(
            row[path_column]
        )

        axes[
            row_index,
            column_index
        ].imshow(image)

        axes[
            row_index,
            column_index
        ].axis("off")

        if row_index == 0:
            axes[
                row_index,
                column_index
            ].set_title(title)

        if column_index == 0:
            axes[
                row_index,
                column_index
            ].set_ylabel(
                f"ID {int(row['id'])}",
                fontsize=11
            )


figure.suptitle(
    "Persistent TikTok False Negatives",
    fontsize=15
)

figure.tight_layout()


persistent_fn_figure_path = (
    diagnostic_dir
    / (
        "tiktok_persistent_false_"
        "negative_visual_review.png"
    )
)

figure.savefig(
    persistent_fn_figure_path,
    dpi=150,
    bbox_inches="tight"
)

plt.show()


print(
    "=== PERSISTENT FALSE-NEGATIVE "
    "CAPTIONS ==="
)

display(
    review_df[
        [
            "id",
            "text_caption",
            "obfuscated_caption",
            "image_path",
        ]
    ]
)

print(
    "\nSaved to:",
    persistent_fn_figure_path
)

# Publication notebook cell pub-072
audit_categories = [
    "explicit_promotion",
    "implicit_gambling_context",
    "discussion_or_complaint",
    "unrelated_content",
    "ambiguous",
]

persistent_audit_ids = [
    2881,
    2875,
    2987,
    2967,
    2911,
    3012,
]

label_quality_audit_df = (
    test_df[
        test_df["id"].isin(
            persistent_audit_ids
        )
    ][
        [
            "id",
            "platform",
            "label",
            "text_caption",
            "image_path",
        ]
    ]
    .copy()
)

label_quality_audit_df[
    "audit_category"
] = ""

label_quality_audit_df[
    "explicit_promotion_evidence"
] = ""

label_quality_audit_df[
    "gambling_context_evidence"
] = ""

label_quality_audit_df[
    "possible_label_issue"
] = ""

label_quality_audit_df[
    "reviewer_notes"
] = ""


label_quality_audit_path = (
    diagnostic_dir
    / "tiktok_persistent_fn_label_audit.csv"
)

label_quality_audit_df.to_csv(
    label_quality_audit_path,
    index=False
)


print(
    "=== TIKTOK PERSISTENT FN "
    "LABEL-AUDIT TEMPLATE ==="
)

display(label_quality_audit_df)

print("\nAllowed categories:")
print(audit_categories)

print("\nSaved to:")
print(label_quality_audit_path)

# Publication notebook cell pub-073
manual_label_audit = {
    2875: {
        "audit_category": "unrelated_content",
        "explicit_promotion_evidence": (
            "Tidak ditemukan ajakan, bonus, tautan, "
            "nama situs, atau promosi perjudian."
        ),
        "gambling_context_evidence": (
            "Tidak ada konteks perjudian yang terlihat "
            "pada caption maupun gambar."
        ),
        "possible_label_issue": "yes",
        "reviewer_notes": (
            "Gambar menunjukkan aktivitas panen kelapa "
            "sawit dan caption hanya berisi mention akun. "
            "Label positif tampak tidak didukung oleh "
            "konten yang tersedia."
        ),
    },

    2881: {
        "audit_category": (
            "implicit_gambling_context"
        ),
        "explicit_promotion_evidence": (
            "Tidak ditemukan CTA, bonus, tautan, "
            "deposit, atau nama situs."
        ),
        "gambling_context_evidence": (
            "Terdapat teks visual 'Notif ringtone "
            "mahjong'. Mahjong dapat berkaitan dengan "
            "tema permainan judi, tetapi konteks promosi "
            "tidak terlihat."
        ),
        "possible_label_issue": "uncertain",
        "reviewer_notes": (
            "Ada referensi Mahjong, tetapi kontennya "
            "lebih menyerupai hiburan atau ringtone "
            "daripada promosi perjudian eksplisit."
        ),
    },

    2911: {
        "audit_category": (
            "implicit_gambling_context"
        ),
        "explicit_promotion_evidence": (
            "Tidak terlihat CTA, tautan, bonus, "
            "atau informasi situs."
        ),
        "gambling_context_evidence": (
            "Caption menggunakan istilah 'Jackpot' "
            "dan gambar menampilkan sejumlah uang."
        ),
        "possible_label_issue": "uncertain",
        "reviewer_notes": (
            "Sinyal perjudian lebih kuat daripada "
            "sampel lain, tetapi bukti bahwa konten "
            "tersebut merupakan promosi masih terbatas."
        ),
    },

    2967: {
        "audit_category": (
            "discussion_or_complaint"
        ),
        "explicit_promotion_evidence": (
            "Tidak ditemukan ajakan berjudi, bonus, "
            "tautan, atau nama situs."
        ),
        "gambling_context_evidence": (
            "Caption menyebut 'admin' dan 'all in', "
            "yang dapat merujuk pada pengalaman berjudi."
        ),
        "possible_label_issue": "yes",
        "reviewer_notes": (
            "Konten tampak membahas pengalaman atau "
            "kekecewaan, bukan mempromosikan layanan "
            "perjudian."
        ),
    },

    2987: {
        "audit_category": (
            "discussion_or_complaint"
        ),
        "explicit_promotion_evidence": (
            "Tidak ditemukan CTA, tautan, bonus, "
            "atau nama situs."
        ),
        "gambling_context_evidence": (
            "Teks visual menyebut 'depo' dan "
            "'situs lagi gacor', sehingga konteks "
            "perjudian cukup jelas."
        ),
        "possible_label_issue": "yes",
        "reviewer_notes": (
            "Konten menyampaikan cerita atau keluhan "
            "tentang perjudian. Konteks judi ada, tetapi "
            "indikasi promosi tidak terlihat."
        ),
    },

    3012: {
        "audit_category": "ambiguous",
        "explicit_promotion_evidence": (
            "Tidak ditemukan CTA, bonus, tautan, "
            "deposit, atau nama situs."
        ),
        "gambling_context_evidence": (
            "Caption menyebut akun bernama Mahjong, "
            "tetapi gambar hanya menunjukkan makanan."
        ),
        "possible_label_issue": "yes",
        "reviewer_notes": (
            "Keterkaitan perjudian hanya berasal dari "
            "nama akun atau mention. Modalitas gambar "
            "tidak mendukung label promosi perjudian."
        ),
    },
}


for sample_id, audit_values in (
    manual_label_audit.items()
):
    row_mask = (
        label_quality_audit_df["id"]
        == sample_id
    )

    for column_name, value in (
        audit_values.items()
    ):
        label_quality_audit_df.loc[
            row_mask,
            column_name
        ] = value


label_quality_audit_df.to_csv(
    label_quality_audit_path,
    index=False
)


print(
    "=== COMPLETED TIKTOK "
    "PERSISTENT FN LABEL AUDIT ==="
)

display(
    label_quality_audit_df[
        [
            "id",
            "audit_category",
            "explicit_promotion_evidence",
            "gambling_context_evidence",
            "possible_label_issue",
            "reviewer_notes",
        ]
    ]
)

print(
    "\nCompleted rows:",
    int(
        (
            label_quality_audit_df[
                "audit_category"
            ] != ""
        ).sum()
    ),
    "/",
    len(label_quality_audit_df)
)

print(
    "Primary test labels changed:",
    False
)

print(
    "Saved to:",
    label_quality_audit_path
)

# Publication notebook cell pub-075
audit_category_summary_df = (
    label_quality_audit_df[
        "audit_category"
    ]
    .value_counts()
    .rename_axis("audit_category")
    .reset_index(name="sample_count")
)

audit_issue_summary_df = (
    label_quality_audit_df[
        "possible_label_issue"
    ]
    .value_counts()
    .rename_axis("possible_label_issue")
    .reset_index(name="sample_count")
)

audit_overview_df = pd.DataFrame([
    {
        "audited_samples": int(
            len(label_quality_audit_df)
        ),
        "possible_label_issue_yes": int(
            (
                label_quality_audit_df[
                    "possible_label_issue"
                ] == "yes"
            ).sum()
        ),
        "possible_label_issue_uncertain": int(
            (
                label_quality_audit_df[
                    "possible_label_issue"
                ] == "uncertain"
            ).sum()
        ),
        "primary_test_labels_changed": False,
        "audit_usage": (
            "exploratory_error_analysis_only"
        ),
    }
])

audit_category_summary_path = (
    diagnostic_dir
    / "tiktok_label_audit_category_summary.csv"
)

audit_issue_summary_path = (
    diagnostic_dir
    / "tiktok_label_audit_issue_summary.csv"
)

audit_overview_path = (
    diagnostic_dir
    / "tiktok_label_audit_overview.csv"
)

audit_category_summary_df.to_csv(
    audit_category_summary_path,
    index=False
)

audit_issue_summary_df.to_csv(
    audit_issue_summary_path,
    index=False
)

audit_overview_df.to_csv(
    audit_overview_path,
    index=False
)

print(
    "=== FINAL V2 TIKTOK "
    "LABEL-AUDIT SUMMARY ==="
)

display(audit_category_summary_df)
display(audit_issue_summary_df)
display(audit_overview_df)

print(
    "\nPrimary test labels changed:",
    False
)

# Publication notebook cell pub-077
main_condition_names = {
    "clean": "clean_f1",
    "joint_j1_text_obfuscated_blur_r5": "j1_f1",
    (
        "joint_j2_text_obfuscated_"
        "pixelation_p12_5"
    ): "j2_f1",
    (
        "joint_j3_text_obfuscated_"
        "blur_r5_pixelation_p12_5"
    ): "j3_f1",
}


main_robustness_df = (
    robustness_aggregate_df[
        robustness_aggregate_df[
            "condition"
        ].isin(main_condition_names)
    ]
    .assign(
        result_column=lambda df:
        df["condition"].map(
            main_condition_names
        )
    )
    .pivot(
        index="model_name",
        columns="result_column",
        values="mean_test_f1",
    )
    .reset_index()
)


required_result_columns = [
    "clean_f1",
    "j1_f1",
    "j2_f1",
    "j3_f1",
]

if main_robustness_df[
    required_result_columns
].isna().any().any():
    raise ValueError(
        "Ada hasil utama yang belum lengkap."
    )


main_robustness_df[
    "mean_joint_f1"
] = (
    main_robustness_df[
        [
            "j1_f1",
            "j2_f1",
            "j3_f1",
        ]
    ].mean(axis=1)
)


main_robustness_df[
    "worst_joint_f1"
] = (
    main_robustness_df[
        [
            "j1_f1",
            "j2_f1",
            "j3_f1",
        ]
    ].min(axis=1)
)


main_robustness_df[
    "worst_joint_drop"
] = (
    main_robustness_df["clean_f1"]
    -
    main_robustness_df[
        "worst_joint_f1"
    ]
)


main_robustness_df[
    "worst_joint_relative_drop_pct"
] = (
    main_robustness_df[
        "worst_joint_drop"
    ]
    /
    main_robustness_df["clean_f1"]
    * 100
)


main_robustness_df[
    "clean_rank"
] = (
    main_robustness_df[
        "clean_f1"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


main_robustness_df[
    "robustness_rank"
] = (
    main_robustness_df[
        "worst_joint_f1"
    ]
    .rank(
        ascending=False,
        method="min"
    )
    .astype(int)
)


main_robustness_df = (
    main_robustness_df[
        [
            "model_name",
            "clean_f1",
            "j1_f1",
            "j2_f1",
            "j3_f1",
            "mean_joint_f1",
            "worst_joint_f1",
            "worst_joint_drop",
            "worst_joint_relative_drop_pct",
            "clean_rank",
            "robustness_rank",
        ]
    ]
    .sort_values(
        "robustness_rank"
    )
    .reset_index(drop=True)
)


main_robustness_path = (
    summary_dir
    / "final_main_robustness_results.csv"
)

main_robustness_df.to_csv(
    main_robustness_path,
    index=False
)


print(
    "=== FINAL V2 MAIN "
    "ROBUSTNESS RESULTS ==="
)

display(main_robustness_df)

print(
    "\nBest clean model:",
    main_robustness_df
    .sort_values(
        "clean_rank"
    )
    .iloc[0]["model_name"]
)

print(
    "Best worst-case model:",
    main_robustness_df
    .sort_values(
        "robustness_rank"
    )
    .iloc[0]["model_name"]
)

print(
    "\nSaved to:",
    main_robustness_path
)

# Publication notebook cell pub-079
severity_order = {
    "clean": 0,
    "text_obfuscated": 1,
    "gaussian_blur_r2": 2,
    "gaussian_blur_r5": 3,
    "pixelation_p50": 4,
    "pixelation_p25": 5,
    "pixelation_p12_5": 6,
    "blur_r5_pixelation_p12_5": 7,
}

single_corruption_df = (
    robustness_aggregate_df[
        robustness_aggregate_df[
            "condition"
        ].isin(severity_order)
    ]
    .copy()
)

single_corruption_df[
    "display_order"
] = (
    single_corruption_df["condition"]
    .map(severity_order)
)

single_corruption_summary_df = (
    single_corruption_df[
        [
            "model_name",
            "condition",
            "category",
            "mean_test_f1",
            "std_test_f1",
            "mean_f1_drop",
            "mean_relative_f1_drop_pct",
            "mean_test_pr_auc",
            "mean_brier_increase",
            "display_order",
        ]
    ]
    .sort_values(
        [
            "model_name",
            "display_order",
        ]
    )
    .reset_index(drop=True)
)


# Kondisi single corruption paling merusak
perturbed_single_df = (
    single_corruption_summary_df[
        single_corruption_summary_df[
            "condition"
        ] != "clean"
    ]
)

worst_single_indices = (
    perturbed_single_df
    .groupby("model_name")[
        "mean_test_f1"
    ]
    .idxmin()
)

worst_single_corruption_df = (
    perturbed_single_df
    .loc[
        worst_single_indices,
        [
            "model_name",
            "condition",
            "mean_test_f1",
            "mean_f1_drop",
            "mean_relative_f1_drop_pct",
            "mean_test_pr_auc",
            "mean_brier_increase",
        ]
    ]
    .rename(
        columns={
            "condition":
            "most_damaging_single_condition",
            "mean_test_f1":
            "worst_single_f1",
        }
    )
    .sort_values(
        "worst_single_f1",
        ascending=False
    )
    .reset_index(drop=True)
)


# Tabel severity khusus blur dan pixelation
image_severity_df = (
    single_corruption_summary_df[
        single_corruption_summary_df[
            "condition"
        ].isin(
            [
                "gaussian_blur_r2",
                "gaussian_blur_r5",
                "pixelation_p50",
                "pixelation_p25",
                "pixelation_p12_5",
                "blur_r5_pixelation_p12_5",
            ]
        )
    ]
    .copy()
)


single_corruption_path = (
    summary_dir
    / "single_corruption_severity_summary.csv"
)

worst_single_path = (
    summary_dir
    / "most_damaging_single_corruption.csv"
)

image_severity_path = (
    summary_dir
    / "image_corruption_severity_results.csv"
)


single_corruption_summary_df.to_csv(
    single_corruption_path,
    index=False
)

worst_single_corruption_df.to_csv(
    worst_single_path,
    index=False
)

image_severity_df.to_csv(
    image_severity_path,
    index=False
)


print(
    "=== FINAL V2 MOST DAMAGING "
    "SINGLE CORRUPTION ==="
)

display(worst_single_corruption_df)


print(
    "\n=== IMAGE CORRUPTION "
    "SEVERITY — TOP THREE FUSIONS ==="
)

display(
    image_severity_df[
        image_severity_df[
            "model_name"
        ].isin(
            [
                "learnable_weighted_logit_fusion",
                "raw_concatenation_fusion",
                "average_logit_fusion",
            ]
        )
    ][
        [
            "model_name",
            "condition",
            "mean_test_f1",
            "mean_f1_drop",
            "mean_relative_f1_drop_pct",
            "mean_test_pr_auc",
        ]
    ]
)

# Publication notebook cell pub-081
# Nama singkat agar grafik mudah dibaca
model_labels = {
    "learnable_weighted_logit_fusion": "Weighted",
    "raw_concatenation_fusion": "Raw Concat",
    "average_logit_fusion": "Average",
    "projection_fusion": "Projection",
    "contrastive_similarity_fusion": "Contrastive",
    "text_only_mlp": "Text-only",
    "image_only_mlp": "Image-only",
}

plot_df = main_robustness_df.copy()
plot_df["model"] = (
    plot_df["model_name"]
    .map(model_labels)
)

joint_plot_df = (
    plot_df[
        [
            "model",
            "clean_f1",
            "j1_f1",
            "j2_f1",
            "j3_f1",
        ]
    ]
    .melt(
        id_vars="model",
        var_name="condition",
        value_name="f1",
    )
)

condition_labels = {
    "clean_f1": "Clean",
    "j1_f1": "J1",
    "j2_f1": "J2",
    "j3_f1": "J3",
}

joint_plot_df["condition"] = (
    joint_plot_df["condition"]
    .map(condition_labels)
)

condition_order = [
    "Clean",
    "J1",
    "J2",
    "J3",
]

plt.figure(figsize=(11, 6))

for model_name in plot_df["model"]:
    model_data = joint_plot_df[
        joint_plot_df["model"]
        == model_name
    ].set_index("condition").loc[
        condition_order
    ]

    plt.plot(
        condition_order,
        model_data["f1"],
        marker="o",
        label=model_name,
    )

plt.xlabel("Test condition")
plt.ylabel("Mean F1")
plt.title(
    "Clean and Joint-Perturbation Performance"
)
plt.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left"
)
plt.grid(alpha=0.25)
plt.tight_layout()

main_robustness_figure_path = (
    diagnostic_dir
    / "main_joint_robustness_comparison.png"
)

plt.savefig(
    main_robustness_figure_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# Publication notebook cell pub-082
top_three_models = [
    "learnable_weighted_logit_fusion",
    "raw_concatenation_fusion",
    "average_logit_fusion",
]

severity_conditions = [
    "gaussian_blur_r2",
    "gaussian_blur_r5",
    "pixelation_p50",
    "pixelation_p25",
    "pixelation_p12_5",
    "blur_r5_pixelation_p12_5",
]

severity_labels = {
    "gaussian_blur_r2": "Blur r2",
    "gaussian_blur_r5": "Blur r5",
    "pixelation_p50": "Pixel 50%",
    "pixelation_p25": "Pixel 25%",
    "pixelation_p12_5": "Pixel 12.5%",
    "blur_r5_pixelation_p12_5": (
        "Blur r5 → Pixel 12.5%"
    ),
}

severity_plot_df = image_severity_df[
    image_severity_df["model_name"].isin(
        top_three_models
    )
].copy()

severity_plot_df["model"] = (
    severity_plot_df["model_name"]
    .map(model_labels)
)

severity_plot_df["condition_label"] = (
    severity_plot_df["condition"]
    .map(severity_labels)
)

plt.figure(figsize=(12, 6))

for model_name in [
    "Weighted",
    "Raw Concat",
    "Average",
]:
    model_data = (
        severity_plot_df[
            severity_plot_df["model"]
            == model_name
        ]
        .set_index("condition")
        .loc[severity_conditions]
    )

    plt.plot(
        [
            severity_labels[condition]
            for condition
            in severity_conditions
        ],
        model_data["mean_test_f1"],
        marker="o",
        label=model_name,
    )

plt.xlabel("Image corruption")
plt.ylabel("Mean F1")
plt.title(
    "Image-Corruption Severity on Top Fusion Models"
)
plt.xticks(rotation=20)
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()

severity_figure_path = (
    diagnostic_dir
    / "top3_image_corruption_severity.png"
)

plt.savefig(
    severity_figure_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("Saved to:")
print(main_robustness_figure_path)
print(severity_figure_path)

# Publication notebook cell pub-084
final_model_selection_df = (
    main_robustness_df[
        main_robustness_df["model_name"].isin(
            [
                "learnable_weighted_logit_fusion",
                "raw_concatenation_fusion",
                "average_logit_fusion",
            ]
        )
    ]
    .copy()
)


model_roles = {
    "learnable_weighted_logit_fusion": (
        "Primary descriptive model: "
        "highest clean and worst-case joint F1"
    ),
    "raw_concatenation_fusion": (
        "Robustness comparator: highest mean joint F1 "
        "and strongest TikTok performance"
    ),
    "average_logit_fusion": (
        "Simple non-trainable fusion baseline with "
        "competitive robustness"
    ),
}


final_model_selection_df[
    "interpretation"
] = (
    final_model_selection_df[
        "model_name"
    ].map(model_roles)
)


final_model_selection_df[
    "significantly_better_than_other_top_models"
] = False


final_model_selection_df = (
    final_model_selection_df[
        [
            "model_name",
            "clean_f1",
            "j1_f1",
            "j2_f1",
            "j3_f1",
            "mean_joint_f1",
            "worst_joint_f1",
            "worst_joint_relative_drop_pct",
            "clean_rank",
            "robustness_rank",
            "significantly_better_than_other_top_models",
            "interpretation",
        ]
    ]
    .sort_values(
        "robustness_rank"
    )
    .reset_index(drop=True)
)


final_findings = {
    "primary_descriptive_model": (
        "learnable_weighted_logit_fusion"
    ),
    "primary_model_clean_f1": float(
        final_model_selection_df.loc[
            final_model_selection_df[
                "model_name"
            ]
            == "learnable_weighted_logit_fusion",
            "clean_f1",
        ].iloc[0]
    ),
    "primary_model_worst_joint_f1": float(
        final_model_selection_df.loc[
            final_model_selection_df[
                "model_name"
            ]
            == "learnable_weighted_logit_fusion",
            "worst_joint_f1",
        ].iloc[0]
    ),
    "highest_mean_joint_model": (
        "raw_concatenation_fusion"
    ),
    "highest_mean_joint_f1": float(
        final_model_selection_df.loc[
            final_model_selection_df[
                "model_name"
            ]
            == "raw_concatenation_fusion",
            "mean_joint_f1",
        ].iloc[0]
    ),
    "most_damaging_joint_condition": (
        "J3: text obfuscation + blur r5 "
        "followed by pixelation 12.5%"
    ),
    "most_difficult_platform": "TikTok",
    "main_tiktok_failure_mode": (
        "False negatives and low recall "
        "on positive gambling samples"
    ),
    "weighted_mean_text_weight": float(
        weighted_fusion_aggregate_df[
            "mean_text_weight"
        ].iloc[0]
    ),
    "weighted_mean_image_weight": float(
        weighted_fusion_aggregate_df[
            "mean_image_weight"
        ].iloc[0]
    ),
    "top_three_bootstrap_significant": False,
    "primary_labels_modified": False,
    "final_interpretation": (
        "Weighted, Raw Concatenation, and Average "
        "Fusion form the top performance tier. "
        "Their differences were not statistically "
        "significant."
    ),
}


final_model_selection_path = (
    summary_dir
    / "final_model_selection_summary.csv"
)

final_findings_path = (
    config_dir
    / "final_experimental_findings.json"
)


final_model_selection_df.to_csv(
    final_model_selection_path,
    index=False
)

with final_findings_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        final_findings,
        file,
        indent=2,
        ensure_ascii=False
    )


print(
    "=== FINAL V2 MODEL SELECTION SUMMARY ==="
)

display(final_model_selection_df)


print(
    "\nPrimary descriptive model:",
    final_findings[
        "primary_descriptive_model"
    ]
)

print(
    "Highest mean joint model:",
    final_findings[
        "highest_mean_joint_model"
    ]
)

print(
    "Most damaging joint condition:",
    final_findings[
        "most_damaging_joint_condition"
    ]
)

print(
    "Most difficult platform:",
    final_findings[
        "most_difficult_platform"
    ]
)

print(
    "Top-three differences significant:",
    final_findings[
        "top_three_bootstrap_significant"
    ]
)

print("\nSaved to:")
print(final_model_selection_path)
print(final_findings_path)

# Publication notebook cell pub-086
final_model_selection_df.columns.name = None

paper_model_names = {
    "learnable_weighted_logit_fusion": (
        "Learnable Weighted Fusion"
    ),
    "raw_concatenation_fusion": (
        "Raw Concatenation Fusion"
    ),
    "average_logit_fusion": (
        "Average Logit Fusion"
    ),
}

paper_main_results_df = (
    final_model_selection_df
    .copy()
)

paper_main_results_df[
    "Model"
] = (
    paper_main_results_df[
        "model_name"
    ].map(paper_model_names)
)

percentage_columns = {
    "clean_f1": "Clean F1 (%)",
    "j1_f1": "J1 F1 (%)",
    "j2_f1": "J2 F1 (%)",
    "j3_f1": "J3 F1 (%)",
    "mean_joint_f1": "Mean Joint F1 (%)",
    "worst_joint_f1": "Worst Joint F1 (%)",
    "worst_joint_relative_drop_pct": (
        "Worst Relative Drop (%)"
    ),
}

for source_column, target_column in (
    percentage_columns.items()
):
    if source_column == (
        "worst_joint_relative_drop_pct"
    ):
        paper_main_results_df[
            target_column
        ] = paper_main_results_df[
            source_column
        ].round(3)

    else:
        paper_main_results_df[
            target_column
        ] = (
            paper_main_results_df[
                source_column
            ]
            * 100
        ).round(3)


paper_main_results_df[
    "Statistically Superior"
] = "No"

paper_main_results_df[
    "Main Interpretation"
] = [
    (
        "Highest clean and worst-case "
        "joint F1"
    ),
    (
        "Highest mean joint F1 and "
        "smallest relative drop"
    ),
    (
        "Simple fixed-weight baseline "
        "with competitive performance"
    ),
]


paper_main_results_df = (
    paper_main_results_df[
        [
            "Model",
            "Clean F1 (%)",
            "J1 F1 (%)",
            "J2 F1 (%)",
            "J3 F1 (%)",
            "Mean Joint F1 (%)",
            "Worst Joint F1 (%)",
            "Worst Relative Drop (%)",
            "Statistically Superior",
            "Main Interpretation",
        ]
    ]
)


paper_main_results_path = (
    summary_dir
    / "paper_ready_main_results.csv"
)

paper_main_results_df.to_csv(
    paper_main_results_path,
    index=False
)


print(
    "=== PAPER-READY MAIN "
    "ROBUSTNESS RESULTS ==="
)

display(paper_main_results_df)

print(
    "\nSaved to:",
    paper_main_results_path
)

# Publication notebook cell pub-088
research_question_summary = [
    {
        "research_question": "RQ1",
        "focus": (
            "Dampak text obfuscation dan image corruption "
            "terhadap performa model"
        ),
        "main_finding": (
            "Semua model mengalami penurunan performa setelah "
            "perturbation. Text-only paling terdampak oleh text "
            "obfuscation dengan relative F1 drop 3.243%, sedangkan "
            "image-only paling terdampak oleh compound image "
            "corruption dengan relative F1 drop 2.262%."
        ),
        "supporting_result": (
            "Text-only: 95.471% clean menjadi 92.374% pada "
            "text obfuscation. Image-only: 93.869% clean menjadi "
            "91.744% pada blur r5 kemudian pixelation 12.5%."
        ),
    },
    {
        "research_question": "RQ2",
        "focus": (
            "Perturbation dan severity yang paling merusak"
        ),
        "main_finding": (
            "Blur radius 5 lebih merusak daripada radius 2, "
            "sedangkan pixelation 12.5% lebih merusak daripada "
            "50% dan 25%. Compound blur r5 kemudian pixelation "
            "12.5% menjadi gangguan visual paling berat."
        ),
        "supporting_result": (
            "Compound image corruption menjadi kondisi single "
            "terburuk untuk Raw Concatenation, Projection, "
            "Contrastive Similarity, dan Image-only."
        ),
    },
    {
        "research_question": "RQ3",
        "focus": (
            "Strategi fusion paling robust pada joint perturbation"
        ),
        "main_finding": (
            "Learnable Weighted Fusion memperoleh clean F1 dan "
            "worst-case joint F1 tertinggi. Raw Concatenation "
            "memperoleh mean joint F1 tertinggi serta relative "
            "drop terkecil."
        ),
        "supporting_result": (
            "Weighted: clean 96.289%, worst joint 94.616%. "
            "Raw Concatenation: mean joint 94.851%, relative "
            "worst-case drop 1.678%. Paired bootstrap menunjukkan "
            "top-three fusion tidak berbeda signifikan."
        ),
    },
    {
        "research_question": "RQ4",
        "focus": (
            "Perbedaan robustness berdasarkan platform"
        ),
        "main_finding": (
            "Facebook dan X mempertahankan performa tinggi, "
            "sedangkan TikTok menjadi platform paling sulit. "
            "Kegagalan TikTok terutama berasal dari false negative "
            "dan rendahnya recall pada sampel positif."
        ),
        "supporting_result": (
            "Raw Concatenation menjadi model terbaik pada TikTok "
            "untuk J1, J2, dan J3. Pada J3, TikTok F1 hanya 60.226%, "
            "dibandingkan Facebook 98.836% dan X sekitar 97–98%."
        ),
    },
]


research_question_summary_df = pd.DataFrame(
    research_question_summary
)


rq_summary_path = (
    summary_dir
    / "paper_ready_research_question_summary.csv"
)

rq_summary_json_path = (
    config_dir
    / "final_research_question_findings.json"
)


research_question_summary_df.to_csv(
    rq_summary_path,
    index=False
)

with rq_summary_json_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        research_question_summary,
        file,
        indent=2,
        ensure_ascii=False
    )


print(
    "=== PAPER-READY RESEARCH "
    "QUESTION SUMMARY ==="
)

display(research_question_summary_df)

print("\nSaved to:")
print(rq_summary_path)
print(rq_summary_json_path)

# Publication notebook cell pub-090
critical_output_paths = {
    "training_protocol": (
        config_dir / "final_training_protocol.json"
    ),
    "clean_test_results": (
        summary_dir / "clean_test_model_aggregate.csv"
    ),
    "robustness_results": (
        summary_dir
        / "robustness_model_condition_aggregate.csv"
    ),
    "main_robustness_results": (
        summary_dir
        / "final_main_robustness_results.csv"
    ),
    "paper_main_results": (
        summary_dir
        / "paper_ready_main_results.csv"
    ),
    "research_question_summary": (
        summary_dir
        / "paper_ready_research_question_summary.csv"
    ),
    "paired_bootstrap": (
        bootstrap_dir
        / "paired_bootstrap_top3_primary_conditions.csv"
    ),
    "platform_results": (
        per_platform_dir
        / "primary_conditions_platform_summary.csv"
    ),
    "all_predictions": (
        prediction_dir
        / "robustness_all_predictions.csv"
    ),
}


integrity_rows = []

for output_name, output_path in (
    critical_output_paths.items()
):
    exists = output_path.exists()

    integrity_rows.append({
        "output_name": output_name,
        "exists": exists,
        "file_size_mb": (
            round(
                output_path.stat().st_size
                / (1024 ** 2),
                3
            )
            if exists
            else None
        ),
        "sha256": (
            calculate_file_sha256(
                output_path
            )
            if exists
            else None
        ),
        "file_path": str(output_path),
    })


final_output_integrity_df = pd.DataFrame(
    integrity_rows
)

if not final_output_integrity_df[
    "exists"
].all():
    display(final_output_integrity_df)

    raise FileNotFoundError(
        "Ada output final yang belum tersimpan."
    )


experiment_counts = {
    "models": int(
        robustness_aggregate_df[
            "model_name"
        ].nunique()
    ),
    "random_seeds": int(
        robustness_per_seed_df[
            "seed"
        ].nunique()
    ),
    "test_conditions": int(
        robustness_per_seed_df[
            "condition"
        ].nunique()
    ),
    "metric_evaluations": int(
        len(robustness_per_seed_df)
    ),
    "prediction_rows": int(
        len(robustness_predictions_df)
    ),
    "expected_metric_evaluations": 385,
    "expected_prediction_rows": 289135,
}


if experiment_counts["models"] != 7:
    raise ValueError("Jumlah model final bukan 7.")

if experiment_counts["random_seeds"] != 5:
    raise ValueError("Jumlah seed final bukan 5.")

if experiment_counts["test_conditions"] != 11:
    raise ValueError(
        "Jumlah test condition final bukan 11."
    )

if (
    experiment_counts["metric_evaluations"]
    != experiment_counts[
        "expected_metric_evaluations"
    ]
):
    raise ValueError(
        "Jumlah metric evaluation tidak sesuai."
    )

if (
    experiment_counts["prediction_rows"]
    != experiment_counts[
        "expected_prediction_rows"
    ]
):
    raise ValueError(
        "Jumlah prediction rows tidak sesuai."
    )


final_manifest_path = (
    summary_dir
    / "final_experiment_output_manifest.csv"
)

final_integrity_report_path = (
    config_dir
    / "final_experiment_integrity_report.json"
)


final_output_integrity_df.to_csv(
    final_manifest_path,
    index=False
)

with final_integrity_report_path.open(
    "w",
    encoding="utf-8"
) as file:
    json.dump(
        {
            "experiment_version": (
                experiment_version
            ),
            "status": "locked_and_complete",
            "counts": experiment_counts,
            "critical_outputs": (
                final_output_integrity_df
                .set_index("output_name")
                .to_dict(orient="index")
            ),
            "primary_test_labels_changed": False,
            "test_time_retraining": False,
            "perturbation_threshold_tuning": False,
        },
        file,
        indent=2,
        ensure_ascii=False
    )


print(
    "=== FINAL V2 EXPERIMENT "
    "INTEGRITY CHECK ==="
)

display(
    final_output_integrity_df[
        [
            "output_name",
            "exists",
            "file_size_mb",
        ]
    ]
)

print("\nModels:", experiment_counts["models"])
print(
    "Random seeds:",
    experiment_counts["random_seeds"]
)
print(
    "Test conditions:",
    experiment_counts["test_conditions"]
)
print(
    "Metric evaluations:",
    experiment_counts[
        "metric_evaluations"
    ]
)
print(
    "Prediction rows:",
    experiment_counts[
        "prediction_rows"
    ]
)
print("Final status: locked_and_complete")

print("\nSaved to:")
print(final_manifest_path)
print(final_integrity_report_path)
