# Multimodal Gambling Robustness

Research artifacts for **Robustness Evaluation of Unimodal and Multimodal Models for Indonesian Online Gambling Promotion Detection on Social Media**, JOSCEX Paper ID **338**. This repository contains the **FINAL V2** experiment, exact split identifiers, saved predictions, and evaluation outputs. It supports research on content moderation.

**Verification reference:** the author-selected `JOSCEX_338_REVISED_2026_PREVIEW.pdf`, SHA-256 `6261461d10039f479f0dce99de850ca199ffe449c7f59d6079a25cab7ac7bd3f`. The manuscript is not redistributed. Tables 3–5 and the checked experimental/statistical/platform claims agree with the saved FINAL V2 predictions. See [verification](docs/VERIFICATION.md) for scope and limitations. This does not certify the journal's submitted-file record or completion of the preview's author declarations.

## What can be reproduced

| Task | Bundled inputs | Further computation |
| --- | --- | --- |
| Recompute 385 test metric evaluations and 77 model/condition aggregates | Complete saved test predictions; test IDs and labels | CPU evaluation only; no dataset or encoders |
| Check manuscript Tables 3–5 and platform/error statistics | Predictions, original tables, transcribed manuscript comparison values | Offline verification |
| Recompute all 12 paired bootstrap comparisons | Predictions, test order, seed 42 | 5,000 CPU resamples per pair; no training |
| Regenerate analysis plots and manuscript Figure 3 | Saved aggregate/platform CSVs and extracted plotting code | Plotting only |
| Reconstruct the original full split CSVs | Caption-free IDs, image hashes, groups and folds | Download pinned Hugging Face metadata; checksum-check reconstruction |
| Regenerate corruptions and embeddings; train/evaluate all seven models | Original-source pipeline and frozen experiment configuration | Download dataset and encoders, generate six clean and seven perturbed embedding arrays, retrain downstream models |
| Re-run the qualitative review of six TikTok examples | Sample IDs and original categorical annotations | Download original content and conduct a new human review; original annotations are evidence, not an automated result |

Saved predictions reproduce the reported evaluation directly. Fresh model training has **not** been rerun for this release. Exact training outputs across devices or encoder revisions are not guaranteed. Four fusion models have validation summaries but no saved per-example validation prediction files in the source workspace; none were fabricated. See [limitations](docs/REPRODUCIBILITY.md).

## Experiment

The source dataset has 5,028 paired captions/images: 2,195 positive gambling-promotion and 2,833 negative samples, from Facebook (1,419), TikTok (1,442), and X (2,167). Duplicate-linked samples were grouped using normalized-caption equality and exact image SHA-256 equality with transitive union. Empty normalized captions were excluded from caption linking. There are 4,457 leakage groups; 228 multi-row groups contain 799 samples, with maximum size 23. Original labels, including conflicting annotations, remain unchanged.

The fixed split is **3,524 train / 753 validation / 751 test**. IDs and their row order are preserved in `splits/`. The original algorithm uses `StratifiedGroupKFold(n_splits=20, shuffle=True, random_state=42)` with platform-label strata; folds 0–13 are train, 14–16 validation, and 17–19 test. Use the published IDs instead of creating a new split. Neither IDs nor leakage groups cross splits. Caption normalization is for deduplication only; original captions are encoded.

Frozen encoders are `intfloat/multilingual-e5-large` (1,024 dimensions, `query: ` prefix, normalized embeddings) and `google/siglip2-so400m-patch14-384` (1,152 dimensions, RGB images). Clean text extraction uses batch size 8 and clean image extraction batch size 4; perturbed text/image extraction uses 32/16. The historical MPS run used half-precision SigLIP inference. Clean image vectors were normalized in half precision before float32 storage; FINAL V2 perturbed image vectors were normalized after conversion to float32. The original code preserves this distinction. Encoder model commit revisions were not recorded.

Seven configurations are evaluated: Text-only MLP, Image-only MLP, Average Logit Fusion, Learnable Weighted Logit Fusion, Raw Concatenation Fusion, Projection Fusion, and Contrastive Similarity Fusion. The MLPs use 256 hidden units, ReLU and dropout 0.30. Projection-based fusion uses 512-dimensional projections. Average fusion combines logits with fixed 0.5/0.5 weights before sigmoid. Learnable fusion trains two global softmax weight parameters while the component MLPs remain fixed. Contrastive Similarity Fusion uses normalized projected cosine similarity and a trainable scale under binary cross-entropy; it is not a separate contrastive pretraining stage.

Models train only on clean embeddings, using seeds **42, 52, 62, 72, 82**, batch size 64, AdamW (learning rate `1e-3`, weight decay `1e-4`), and `BCEWithLogitsLoss`. Maximum training is 50 epochs, with early stopping after five epochs without validation-loss improvement and restoration of the lowest-loss checkpoint. Thresholds are selected from 0.05–0.95 in steps of 0.01 using clean-validation positive-class F1; ties resolve by higher precision, higher recall, then lower threshold. Checkpoints and thresholds stay fixed during test evaluation. The historical environment reports Python 3.14.4, PyTorch 2.13.0 and Apple MPS; the manuscript describes an Apple M5 MacBook Pro. Hardware identity was not independently audited.

### Test conditions

| Code | Saved condition | Transformation |
| --- | --- | --- |
| C0 | `clean` | Original captions and images |
| P1 | `text_obfuscated` | Insert periods between characters of matched lexicon terms |
| P2 | `gaussian_blur_r2` | Gaussian blur radius 2 |
| P3 | `gaussian_blur_r5` | Gaussian blur radius 5 |
| P4 | `pixelation_p50` | Reduce both dimensions to 50%, then restore |
| P5 | `pixelation_p25` | Reduce both dimensions to 25%, then restore |
| P6 | `pixelation_p12_5` | Reduce both dimensions to 12.5%, then restore |
| P7 | `blur_r5_pixelation_p12_5` | Blur radius 5, then pixelation at 12.5% |
| J1 | `joint_j1_text_obfuscated_blur_r5` | P1 + P3 |
| J2 | `joint_j2_text_obfuscated_pixelation_p12_5` | P1 + P6 |
| J3 | `joint_j3_text_obfuscated_blur_r5_pixelation_p12_5` | P1 + P7 |

Text matching uses 17 locked terms and a development-selected map of approved compound tokens. It avoids unrelated substrings, resolves overlapping matches by longest span, and preserves character order and case. It changed 314 of 751 test captions (4,540 target occurrences). Lexicon/compound selection uses train and validation content. Visual severity selection uses validation inspection. Pixelation uses `max(1, round(dimension * scale))`, BOX downsampling and NEAREST upsampling. Corrupted images are PNG with compression level 1. The final severe pixelation scale is **0.125**; the notebook's initial 0.25 preview is explicitly superseded before test generation. Call the low-level compound helper with `pixelation_scale=0.125`; its historical default is retained in the extracted source.

Evaluation comprises **7 × 5 × 11 = 385** metric evaluations and **289,135** test-prediction rows. F1 is positive-class F1, averaged across seeds; SD is sample SD (`ddof=1`). PR-AUC uses scikit-learn average precision. Other saved metrics include precision, recall, accuracy, ROC-AUC, Brier score and confusion counts. Main joint summaries take the mean/minimum of the three condition-wise seed means. Main-table relative drop is `(clean mean F1 - worst joint mean F1) / clean mean F1 × 100`. The per-condition aggregate CSV additionally retains the historical mean of per-seed relative drops, which can differ slightly from a ratio of means.

| Model | Clean F1 (%) | Mean joint F1 (%) | Worst joint F1 (%) |
| --- | ---: | ---: | ---: |
| Learnable Weighted Logit Fusion | 96.29 | 94.70 | 94.62 |
| Raw Concatenation Fusion | 96.19 | 94.85 | 94.57 |
| Average Logit Fusion | 96.26 | 94.75 | 94.53 |

The three leading fusion models have small numerical differences. All 12 pairwise bootstrap confidence intervals include zero; none establishes statistical superiority. Bootstrap samples the same test-post indices for both models and all five seeds, averages per-seed F1, uses NumPy RNG seed 42, 5,000 resamples per pair and percentile 95% intervals. It is post-level resampling, not leakage-group or seed resampling; do not describe it as a test of all sources of training uncertainty. TikTok has 216 test posts and only 30 positives. Twenty distinct positive TikTok IDs meet the persistent-false-negative criterion (missed by at least four seeds in a leading-model/primary-condition pair). No labels were changed after qualitative inspection.

## Quick start: saved-result verification

Run commands from the repository root. No Git operation is required to inspect or verify this directory.

```sh
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_publication.py
```

The verifier reads bundled files and writes `work/verification.json`. It checks compressed predictions against original uncompressed checksums, the complete evaluation grid, split order/leakage, fixed thresholds, all 385 global and 1,155 platform metric evaluations, aggregates, manuscript tables, and notebook/source hygiene. The requirements reflect packages observed in the workspace at packaging time, not a recovered historical lockfile. Fresh dependency installation was not tested.

To recompute the original bootstrap comparisons or regenerate plots, install the experiment dependencies:

```sh
python -m pip install -r requirements-experiment.txt
python scripts/recompute_bootstrap.py
MPLBACKEND=Agg python scripts/render_figures.py
```

Both commands use saved reference predictions/tables and write under `work/`. The bootstrap utility asserts equality with the published original table. The plot named `figure_2_robustness_top3` is an analysis graph; it is **not** the manuscript's Figure 2, which contains dataset image examples and is excluded. `figure_3_platform_clean_vs_j3` corresponds to the platform analysis.

## Full reproduction from the original dataset

Download the [original dataset on Hugging Face](https://huggingface.co/datasets/0xRafie/indonesian-gambling-multimodal-dataset) at revision `93f00571002bf1bf0dfea67fef68db2a03f0f58a`. The public pinned `dataset.csv` checksum was verified against the workspace. Raw captions and images are not redistributed here.

```sh
python -m pip install -r requirements-experiment.txt
python scripts/prepare_data.py --download
python scripts/extract_clean_embeddings.py
python scripts/run_final_v2.py
```

Alternatively, use `python scripts/prepare_data.py --dataset-root /path/to/downloaded/dataset` with a dataset directory containing `dataset.csv` and `image/`. This reads that source and copies runtime data into `work/`; it does not alter the source. Preparation verifies metadata/image checksums and reconstructs all four full split CSVs in the original order, requiring byte-identical original checksums. `--metadata-only` checks the split reconstruction without copying images.

Full reproduction is computationally expensive and downloads large pretrained models. The clean-feature script is extracted from the shared data-audit notebook; the FINAL V2 runner is exported from the authoritative FINAL V2 notebook. The same runner is available as `notebooks/02_final_v2_full_pipeline.ipynb`. Its outputs are cleared. It uses `work/` as its project root, so regenerated results do not overwrite the bundled reference tables. The optional `01_dataset_audit_and_split.ipynb` documents the original grouping/split algorithm and writes exploratory regenerated partitions separately.

The full pipeline reuses existing generated arrays/checkpoints if present. For an independent fresh run, begin with a fresh `work/` and prepare the data again; keep the published IDs. Before comparing fresh model predictions to the paper, compare regenerated embedding checksums to `provenance/dataset.json` and inspect numeric/device differences. No checkpoints or embedding arrays are included. For reproducing the historical trained predictions, the saved prediction files are the verified evidence; this release does not claim independently recreated weights.

## Repository contents and provenance

- `notebooks/`, `scripts/`, `src/`: sanitized original research source and explicitly labeled publication/validation adapters.
- `splits/`: caption-free exact IDs, labels, platforms, image hashes, leakage groups, folds, split names and original order.
- `perturbations/final_v2/manifests/`: image transformation/hash metadata and a text-manifest projection without captions.
- `results/final_v2/`: original prediction bytes compressed losslessly, numerical summaries, statistical/platform outputs, selected training diagnostics, sanitized configuration, and two aggregate plots.
- `models/final_v2/*.json`: lightweight fixed-fusion descriptors only; referenced component checkpoints are deliberately excluded.
- `provenance/`: source-to-publication mappings, source checksums, manuscript numeric comparison targets, dataset revision, and observed dependency versions.
- `docs/`: verification evidence, audit findings, reproduction limits and proposed publication procedure.

See [MANIFEST.md](MANIFEST.md) for the complete file inventory and [MANIFEST.csv](MANIFEST.csv) for per-file sizes and SHA-256 digests. `SHA256SUMS` excludes the inventory files to avoid circular hashes. Original artifact hashes inside historical integrity reports refer to pre-sanitization source files; publication hashes and transformations are recorded separately. The final evaluation metrics and prediction content were not edited.

## Dataset attribution and release status

The dataset is maintained by D. R. Ekaputra on Hugging Face and described by Syukron and Ekaputra in *Multimodal Deep Learning for Online Gambling Promotion Detection on Indonesian Social Media* (2026), DOI [10.32665/statkom.v5i1.6412](https://doi.org/10.32665/statkom.v5i1.6412). Consult the source dataset card for its CC BY 4.0 declaration and terms. This repository does not relicense the dataset. No separate software license has yet been selected by the authors.

The preparation and verification reports in `docs/` describe the pre-publication audit. The authors have reserved the software licensing decision.
