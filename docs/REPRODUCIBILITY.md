# Reproducibility boundaries

## Direct, offline replay

All test per-example outputs are present: 289,135 robustness rows across the complete 7-model/5-seed/11-condition grid, plus the separate 26,285-row clean prediction export. Gzip decompression yields byte-identical original CSVs. Predictions contain ID, platform, label, model, seed, condition, logit, probability, threshold, class prediction and correctness; the robustness file also has condition category. There are no raw captions or image contents in these files.

The evaluation verifier recalculates metrics from these predictions. Aggregation is by seed, never by pooling duplicated seed rows as independent samples. The standalone bootstrap utility retains the exact FINAL V2 cell 73, with only a file-loading/output adapter. Numeric table transcriptions in `provenance/manuscript_tables.json` are independent comparison targets from the author-selected PDF, not additional measurements.

Per-example validation outputs are available only for text-only MLP, image-only MLP and average-logit fusion (15 files, 753 rows each). They allow independent threshold-grid replay. The other four fusion configurations have validation summaries and training histories but no saved per-example validation outputs. Their threshold choices agree with the test exports, but their full validation threshold search cannot be replayed from the bundled prediction files alone.

## Regeneration and retraining

The original dataset is external and pinned to the audited local dataset commit. Public split files omit `text_caption` and `normalized_caption`, preserving all other original columns in original row order. The preparation adapter joins metadata by ID and applies the exact original caption-normalization function. Reconstructed full splits must match the original four SHA-256 digests, including the manifest `767d106d78a12fa02860336ccaba3b2621fbf4640b02b01035cf3252c3422888`. All four reconstructions passed locally using pinned matching dataset metadata. Encoder arrays, corruption images and trained checkpoints must be regenerated in `work/`.

The original workspace has 13 relevant arrays: six clean (three splits, two modalities), one obfuscated-text test array, six corrupted-image test arrays. Their original digests are in `provenance/dataset.json`. They were inspected read-only for shape, finite values and near-unit norms. Clean SigLIP vectors show small half-precision normalization error (maximum about 0.00055), consistent with the historical code normalizing before float32 storage. Perturbed image extraction normalizes in float32. Do not silently normalize the stored clean vectors again and claim byte-identical historical inputs.

The clean-embedding script is extracted from `01_dataset_audit.ipynb` cells 25–28, 40 and 43–45, adapting paths and backend choice. It has not been executed through encoder inference during this audit. FINAL V2 assumes the shared clean arrays already exist; it is not an independent clean-feature extraction source. Shared setup code is included without importing any pilot prediction or metric result.

No encoder revision hashes were recorded in the original source/configs, and no experiment-time transitive environment lock was found. Model identifiers alone do not pin historical weights/tokenizer/processor versions. Requirements pin the **observed packaging-time workspace versions**; only Python/PyTorch versions are also corroborated by saved runtime metadata. A clean install of those pins has not been tested. MPS/CUDA/CPU precision, training nondeterminism and library versions can change newly generated outputs. A bit-exact end-to-end retraining guarantee is unsupported.

## Source adaptations

- Cleared all notebook outputs, execution counts, attachments and transient metadata. Preserved the scientific code through the locked experiment and removed free-form interpretation cells and ad hoc plotting/debugging cells after that point. Source cell indices/digests are recorded.
- Redirected generated artifacts to `work/`, retaining published reference artifacts outside it.
- Preserved the initial pixelation preview and the subsequent locked 12.5% override; the final dispatched conditions use 0.125 explicitly.
- Exported the sanitized final notebook to `scripts/run_final_v2.py` and verbatim definitions to `src/`. The standalone definition module defaults to CPU and exposes the original global training settings; the full runner retains the original backend selection.
- Extracted plotting cells 113 and 115 with portable input paths and separate `work/` output paths.
- Added preparation, verification, bootstrap input adapter and checksum/inventory utilities as **new publication tooling**, never represented as previously existing research scripts.
- Replaced absolute paths in published metadata with relative paths, normalized nonstandard JSON `NaN` to `null`, and kept historical source hashes separately from transformed publication hashes.
- Removed captions and qualitative reviewer prose from published diagnostic tables. Categorical judgments/counts and stable sample IDs remain. Repeating the human review requires the external data; it cannot be inferred solely from predictions.

The complete downstream training pipeline, encoder extraction, all image regenerations and a fresh dependency installation were not run as part of the lightweight audit. No missing scientific results, models, validation predictions or historical environment details were invented.
