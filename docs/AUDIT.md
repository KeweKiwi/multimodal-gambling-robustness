# Publication audit

## Original Git repository (read-only)

The original branch is `main`, tracking `upstream/main`, at `93f00571002bf1bf0dfea67fef68db2a03f0f58a`. `origin` already targets `https://github.com/KeweKiwi/multimodal-gambling-robustness.git`; `upstream` targets `https://huggingface.co/datasets/0xRafie/indonesian-gambling-multimodal-dataset`. The GitHub connector reported the target public repository size as zero. No remote mutation or GitHub write was performed.

The complete seven-commit history is dataset provenance: `3e7fe64` initial commit; `d4bde14` initial dataset upload; `8cd1e02` removal of source URLs from public metadata; `ebb065e` dataset-card metadata; `768cbf4` dataset-card wording; `6a9691f` image_100 repair; `93f0057` dataset schema inference. It includes the dataset/images and older metadata. It is intentionally not copied into this publication directory. The original has 5,031 tracked files (5,028 images plus dataset metadata, dataset card and attributes).

Untracked directories include embeddings, models, notebooks, perturbations and splits; `.gitignore` itself is untracked. Most result files are ignored, so an ordinary `git status` does not describe the available research artifacts. The audit enumerated ignored files directly rather than relying on tracked files. There were no standalone original research Python scripts outside the environment; the authoritative implementations are notebook cells.

The original `.gitignore` contains literal pasted shell commands (`cd`, `cp`, a heredoc command and its terminator). It ignores the complete final result tree before attempting selected exceptions, omits essential test predictions, and its unanchored `image/` rule can also hide image-manifest directories. It is not reused. The publication ignore rules explicitly retain predictions and manifests while ignoring local runtime data, embeddings, raw images, checkpoint binaries, caches and common credential files. Read-only Git ignore checks verified no deliverable payload was hidden and excluded-data probes were ignored.

## Authoritative research evidence

`notebooks/02_final_v2_full_pipeline.ipynb` is authoritative because its locked `final_v2` protocol, integrity report, 7×5×11 evaluation grid, prediction hashes and numerical manuscript tables agree. Its six primary/severity image conditions include 12.5% pixelation and blur-then-12.5% pixelation. The earlier 25% compound preview and pilot text-separator perturbation are not FINAL V2 test conditions.

The shared clean data audit, group construction, split algorithm and clean embedding extraction are in `01_dataset_audit.ipynb`; an overlapping `01_dataset_audit_and_pilot.ipynb` is excluded. Exact source notebook hashes and original-to-publication cell maps are recorded under `provenance/`.

The split manifest and original train/validation/test files are frozen input evidence. The FINAL V2 test prediction export, per-seed summaries, model/condition aggregates, bootstrap table, platform tables, training protocol and locked integrity report are the result evidence. All nine critical source-file hashes from the locked report were independently checked. Pilot predictions, checkpoints and metrics were excluded.

## Exclusions

No original Git history, full dataset CSV, raw image, corrupted image, embedding array, model checkpoint, environment directory, cache, credentials, source URL list, visual post-review collage or transient notebook output is included. Two aggregate plot PNGs are included; they contain plotted metrics, not source images. Text manifests and split tables omit raw/normalized captions. Diagnostic tables retain only the fields needed for numerical review or categorical sample-audit evidence. Machine usernames, absolute local paths and contact information are not required in the publication payload.

## Outstanding author decisions and limitations

1. Select a software license; the source dataset license does not establish the authors' intended license for this research code. No license was invented.
2. The selected PDF is a revised preview with outstanding author-confirmation placeholders, including data/code availability, authorship/funding/conflict and AI declarations. Its journal submission identity was not verified against a submission system. Use the proposed availability wording only after this repository is actually published.
3. Historical encoder revisions and a full experiment-time dependency lock were not found. Fresh embeddings/retraining are an executable source workflow, not an independently revalidated bit-exact reproduction claim.
4. Missing per-example validation predictions for four fusion models remain missing. No checkpoint binaries are released, so recovering their historical validation predictions requires further author-provided evidence or retraining.
5. The audit verifies experiment evidence and checked manuscript numerical claims. It does not verify external literature claims, hardware identity, qualitative label judgments, or the authors' declarations.

The original workspace was not edited, checked out, stashed, committed or pushed. The publication directory has no `.git` directory. See the separate local handoff report for exact proposed Git commands; execute them only after author approval.
