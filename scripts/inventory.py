"""Create or verify the publication file inventory (new packaging utility).
Run after any approved payload edit. Generated runtime files under work/ and
Python caches are excluded; prohibited artifacts elsewhere cause a failure.
"""
import argparse, collections, csv, hashlib, io
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
INVENTORIES={'MANIFEST.md','MANIFEST.csv','SHA256SUMS'}


def payload_files():
    result=[]
    for p in ROOT.rglob('*'):
        r=p.relative_to(ROOT)
        if any(x in r.parts for x in ['work','.venv','__pycache__','.git','.ipynb_checkpoints']):continue
        if not p.is_file() or str(r) in INVENTORIES:continue
        assert not p.is_symlink(), 'Publication payload must not contain symlinks'
        assert p.suffix.lower() not in {'.npy','.npz','.pt','.pth','.ckpt','.safetensors','.bin','.jpg','.jpeg','.webp','.pdf'}, 'Excluded artifact: '+str(r)
        assert r.parts[0]!='image' and 'images' not in r.parts, 'Raw/corrupted image directory found'
        assert str(r)!='dataset.csv', 'Full dataset must remain external'
        result.append(p)
    return sorted(result,key=lambda p:str(p.relative_to(ROOT)))


def category(path):
    if path.startswith('results/final_v2/predictions/'):return 'Original prediction CSV, lossless gzip'
    if path.startswith('results/'):return 'FINAL V2 numerical output/configuration/aggregate plot'
    if path.startswith('splits/'):return 'Caption-free frozen split metadata'
    if path.startswith('perturbations/'):return 'FINAL V2 corruption metadata or caption-free text projection'
    if path.startswith('models/'):return 'Fixed fusion descriptor; no checkpoint binary'
    if path.startswith('notebooks/'):return 'Sanitized research source'
    if path.startswith('src/'):return 'Extracted research definitions'
    if path.startswith('scripts/'):return 'Research-source export or labeled publication tooling'
    if path.startswith('provenance/'):return 'Source/provenance metadata'
    if path.startswith('docs/'):return 'Documentation and audit evidence'
    return 'Repository documentation/configuration'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    files=payload_files()
    rows=[{'path':str(p.relative_to(ROOT)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'purpose':category(str(p.relative_to(ROOT)))} for p in files]
    if args.check:
        with (ROOT/'MANIFEST.csv').open() as f:expected=list(csv.DictReader(f))
        assert [r['path'] for r in rows]==[r['path'] for r in expected], 'Inventory paths changed'
        for a,b in zip(rows,expected):assert a['bytes']==int(b['bytes']) and a['sha256']==b['sha256'],a['path']
        checksums=''.join(f"{r['sha256']}  {r['path']}\n" for r in rows)
        assert (ROOT/'SHA256SUMS').read_text()==checksums,'SHA256SUMS changed'
        print(f"Inventory verified: {len(rows)} payload files plus 3 inventory files.")
        return
    with (ROOT/'MANIFEST.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['path','bytes','sha256','purpose'],lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (ROOT/'SHA256SUMS').write_text(''.join(f"{r['sha256']}  {r['path']}\n" for r in rows))
    groups=collections.defaultdict(list)
    for r in rows:groups[r['path'].split('/')[0] if '/' in r['path'] else '(root)'].append(r)
    total=sum(r['bytes'] for r in rows)
    text=f'''# Publication file manifest\n\n{len(rows)} payload files, {total:,} bytes ({total/1048576:.2f} MiB), plus the three inventory files `MANIFEST.md`, `MANIFEST.csv`, and `SHA256SUMS`. All listed files are intended for publication. No runtime `work/`, original dataset, source image or model/embedding binary belongs in the release.\n\n`MANIFEST.csv` records every payload size, SHA-256 digest and purpose. `SHA256SUMS` lists the same payload. The three inventories exclude themselves to avoid circular checksums. Source-to-publication transformations are in `provenance/source_artifacts.json`; gzip files decompress to original prediction bytes.\n\nVerify with `python scripts/inventory.py --check` or `shasum -a 256 -c SHA256SUMS`. Regenerate with `python scripts/inventory.py` only after an approved payload change.\n\n| Area | Files | Bytes |\n| --- | ---: | ---: |\n'''
    for g,rs in sorted(groups.items()):text+=f"| `{g}` | {len(rs)} | {sum(r['bytes'] for r in rs):,} |\n"
    text+='\n## Complete inventory\n\n| File | Bytes | Purpose |\n| --- | ---: | --- |\n'
    for r in rows:text+=f"| [{r['path']}]({r['path']}) | {r['bytes']:,} | {r['purpose']} |\n"
    (ROOT/'MANIFEST.md').write_text(text)
    print(f"Wrote {len(rows)} payload entries plus 3 inventories; payload {total:,} bytes.")

if __name__=='__main__':main()
