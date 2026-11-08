# Publication file manifest

200 payload files, 12,737,809 bytes (12.15 MiB), plus the three inventory files `MANIFEST.md`, `MANIFEST.csv`, and `SHA256SUMS`. All listed files are intended for publication. No runtime `work/`, original dataset, source image or model/embedding binary belongs in the release.

`MANIFEST.csv` records every payload size, SHA-256 digest and purpose. `SHA256SUMS` lists the same payload. The three inventories exclude themselves to avoid circular checksums. Source-to-publication transformations are in `provenance/source_artifacts.json`; gzip files decompress to original prediction bytes.

Verify with `python scripts/inventory.py --check` or `shasum -a 256 -c SHA256SUMS`. Regenerate with `python scripts/inventory.py` only after an approved payload change.

| Area | Files | Bytes |
| --- | ---: | ---: |
| `(root)` | 4 | 15,461 |
| `docs` | 12 | 463,138 |
| `models` | 5 | 3,306 |
| `notebooks` | 2 | 502,413 |
| `perturbations` | 8 | 2,162,378 |
| `provenance` | 10 | 110,242 |
| `results` | 144 | 7,946,926 |
| `scripts` | 7 | 363,023 |
| `splits` | 4 | 1,145,044 |
| `src` | 4 | 25,878 |

## Complete inventory

| File | Bytes | Purpose |
| --- | ---: | --- |
| [.gitignore](.gitignore) | 395 | Repository documentation/configuration |
| [README.md](README.md) | 14,626 | Repository documentation/configuration |
| [docs/AUDIT.md](docs/AUDIT.md) | 5,305 | Documentation and audit evidence |
| [docs/PROPOSED_AVAILABILITY.md](docs/PROPOSED_AVAILABILITY.md) | 1,314 | Documentation and audit evidence |
| [docs/PUBLISHING.md](docs/PUBLISHING.md) | 1,364 | Documentation and audit evidence |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | 5,511 | Documentation and audit evidence |
| [docs/VERIFICATION.md](docs/VERIFICATION.md) | 3,839 | Documentation and audit evidence |
| [docs/bootstrap_validation.json](docs/bootstrap_validation.json) | 287 | Documentation and audit evidence |
| [docs/model_smoke_validation.json](docs/model_smoke_validation.json) | 381 | Documentation and audit evidence |
| [docs/payload_validation.json](docs/payload_validation.json) | 260 | Documentation and audit evidence |
| [docs/source_validation.json](docs/source_validation.json) | 3,017 | Documentation and audit evidence |
| [docs/split_validation.json](docs/split_validation.json) | 299 | Documentation and audit evidence |
| [docs/verification.json](docs/verification.json) | 441,115 | Documentation and audit evidence |
| [docs/workspace_preservation.json](docs/workspace_preservation.json) | 446 | Documentation and audit evidence |
| [models/final_v2/average_logit_fusion_seed_42.json](models/final_v2/average_logit_fusion_seed_42.json) | 675 | Fixed fusion descriptor; no checkpoint binary |
| [models/final_v2/average_logit_fusion_seed_52.json](models/final_v2/average_logit_fusion_seed_52.json) | 675 | Fixed fusion descriptor; no checkpoint binary |
| [models/final_v2/average_logit_fusion_seed_62.json](models/final_v2/average_logit_fusion_seed_62.json) | 652 | Fixed fusion descriptor; no checkpoint binary |
| [models/final_v2/average_logit_fusion_seed_72.json](models/final_v2/average_logit_fusion_seed_72.json) | 652 | Fixed fusion descriptor; no checkpoint binary |
| [models/final_v2/average_logit_fusion_seed_82.json](models/final_v2/average_logit_fusion_seed_82.json) | 652 | Fixed fusion descriptor; no checkpoint binary |
| [notebooks/01_dataset_audit_and_split.ipynb](notebooks/01_dataset_audit_and_split.ipynb) | 23,622 | Sanitized research source |
| [notebooks/02_final_v2_full_pipeline.ipynb](notebooks/02_final_v2_full_pipeline.ipynb) | 478,791 | Sanitized research source |
| [perturbations/final_v2/manifests/image/test_all_image_corruptions.csv](perturbations/final_v2/manifests/image/test_all_image_corruptions.csv) | 1,050,375 | FINAL V2 corruption metadata or caption-free text projection |
| [perturbations/final_v2/manifests/image/test_blur_r5_pixelation_p12_5.csv](perturbations/final_v2/manifests/image/test_blur_r5_pixelation_p12_5.csv) | 200,813 | FINAL V2 corruption metadata or caption-free text projection |
| [perturbations/final_v2/manifests/image/test_gaussian_blur_r2.csv](perturbations/final_v2/manifests/image/test_gaussian_blur_r2.csv) | 170,895 | FINAL V2 corruption metadata or caption-free text projection |
| [perturbations/final_v2/manifests/image/test_gaussian_blur_r5.csv](perturbations/final_v2/manifests/image/test_gaussian_blur_r5.csv) | 170,839 | FINAL V2 corruption metadata or caption-free text projection |
| [perturbations/final_v2/manifests/image/test_pixelation_p12_5.csv](perturbations/final_v2/manifests/image/test_pixelation_p12_5.csv) | 173,031 | FINAL V2 corruption metadata or caption-free text projection |
| [perturbations/final_v2/manifests/image/test_pixelation_p25.csv](perturbations/final_v2/manifests/image/test_pixelation_p25.csv) | 168,169 | FINAL V2 corruption metadata or caption-free text projection |
| [perturbations/final_v2/manifests/image/test_pixelation_p50.csv](perturbations/final_v2/manifests/image/test_pixelation_p50.csv) | 167,613 | FINAL V2 corruption metadata or caption-free text projection |
| [perturbations/final_v2/manifests/text/test_text_obfuscated_fullchar_period.csv](perturbations/final_v2/manifests/text/test_text_obfuscated_fullchar_period.csv) | 60,643 | FINAL V2 corruption metadata or caption-free text projection |
| [provenance/audit_source_cells.json](provenance/audit_source_cells.json) | 2,549 | Source/provenance metadata |
| [provenance/dataset.json](provenance/dataset.json) | 2,482 | Source/provenance metadata |
| [provenance/manuscript.json](provenance/manuscript.json) | 476 | Source/provenance metadata |
| [provenance/manuscript_tables.json](provenance/manuscript_tables.json) | 3,812 | Source/provenance metadata |
| [provenance/observed_environment.json](provenance/observed_environment.json) | 552 | Source/provenance metadata |
| [provenance/original_notebooks.json](provenance/original_notebooks.json) | 695 | Source/provenance metadata |
| [provenance/source_artifacts.json](provenance/source_artifacts.json) | 67,446 | Source/provenance metadata |
| [provenance/source_cells.json](provenance/source_cells.json) | 25,643 | Source/provenance metadata |
| [provenance/source_definitions.json](provenance/source_definitions.json) | 5,097 | Source/provenance metadata |
| [provenance/stage_sources.json](provenance/stage_sources.json) | 1,490 | Source/provenance metadata |
| [requirements-experiment.txt](requirements-experiment.txt) | 315 | Repository documentation/configuration |
| [requirements.txt](requirements.txt) | 125 | Repository documentation/configuration |
| [results/final_v2/bootstrap/paired_bootstrap_top3_primary_conditions.csv](results/final_v2/bootstrap/paired_bootstrap_top3_primary_conditions.csv) | 2,991 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/average_logit_fusion_config.json](results/final_v2/configs/average_logit_fusion_config.json) | 321 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/clean_embedding_validation.json](results/final_v2/configs/clean_embedding_validation.json) | 772 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/contrastive_similarity_fusion_config.json](results/final_v2/configs/contrastive_similarity_fusion_config.json) | 356 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/final_experiment_integrity_report.json](results/final_v2/configs/final_experiment_integrity_report.json) | 2,734 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/final_experimental_findings.json](results/final_v2/configs/final_experimental_findings.json) | 849 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/final_research_question_findings.json](results/final_v2/configs/final_research_question_findings.json) | 2,066 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/final_training_protocol.json](results/final_v2/configs/final_training_protocol.json) | 1,558 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/image_corruption_config.json](results/final_v2/configs/image_corruption_config.json) | 1,047 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/image_corruption_generation_report.json](results/final_v2/configs/image_corruption_generation_report.json) | 3,260 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/image_only_mlp_architecture.json](results/final_v2/configs/image_only_mlp_architecture.json) | 280 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/perturbed_image_embedding_report.json](results/final_v2/configs/perturbed_image_embedding_report.json) | 4,389 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/robustness_test_conditions.json](results/final_v2/configs/robustness_test_conditions.json) | 1,909 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/runtime_environment.json](results/final_v2/configs/runtime_environment.json) | 327 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/siglip2_clean_reproduction_report.json](results/final_v2/configs/siglip2_clean_reproduction_report.json) | 343 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/split_validation.json](results/final_v2/configs/split_validation.json) | 298 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/text_obfuscated_embedding_report.json](results/final_v2/configs/text_obfuscated_embedding_report.json) | 480 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/text_obfuscation_lexicon.json](results/final_v2/configs/text_obfuscation_lexicon.json) | 998 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/text_obfuscation_matcher_config.json](results/final_v2/configs/text_obfuscation_matcher_config.json) | 2,720 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/text_obfuscation_matcher_map.json](results/final_v2/configs/text_obfuscation_matcher_map.json) | 3,099 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/text_obfuscation_test_report.json](results/final_v2/configs/text_obfuscation_test_report.json) | 428 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/configs/text_only_mlp_architecture.json](results/final_v2/configs/text_only_mlp_architecture.json) | 275 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/text_obfuscation_matcher_unit_tests.csv](results/final_v2/diagnostics/text_obfuscation_matcher_unit_tests.csv) | 413 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/threshold_search_average_logit_fusion_seed_42.csv](results/final_v2/diagnostics/threshold_search_average_logit_fusion_seed_42.csv) | 8,903 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/threshold_search_average_logit_fusion_seed_52.csv](results/final_v2/diagnostics/threshold_search_average_logit_fusion_seed_52.csv) | 8,785 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/threshold_search_average_logit_fusion_seed_62.csv](results/final_v2/diagnostics/threshold_search_average_logit_fusion_seed_62.csv) | 8,889 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/threshold_search_average_logit_fusion_seed_72.csv](results/final_v2/diagnostics/threshold_search_average_logit_fusion_seed_72.csv) | 8,955 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/threshold_search_average_logit_fusion_seed_82.csv](results/final_v2/diagnostics/threshold_search_average_logit_fusion_seed_82.csv) | 8,789 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/threshold_search_image_only_mlp_seed_42.csv](results/final_v2/diagnostics/threshold_search_image_only_mlp_seed_42.csv) | 8,973 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/threshold_search_image_only_mlp_seed_52.csv](results/final_v2/diagnostics/threshold_search_image_only_mlp_seed_52.csv) | 8,987 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/threshold_search_image_only_mlp_seed_62.csv](results/final_v2/diagnostics/threshold_search_image_only_mlp_seed_62.csv) | 8,961 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/threshold_search_image_only_mlp_seed_72.csv](results/final_v2/diagnostics/threshold_search_image_only_mlp_seed_72.csv) | 8,939 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/threshold_search_image_only_mlp_seed_82.csv](results/final_v2/diagnostics/threshold_search_image_only_mlp_seed_82.csv) | 8,937 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/threshold_search_text_only_mlp_seed_42.csv](results/final_v2/diagnostics/threshold_search_text_only_mlp_seed_42.csv) | 8,805 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/threshold_search_text_only_mlp_seed_52.csv](results/final_v2/diagnostics/threshold_search_text_only_mlp_seed_52.csv) | 8,801 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/threshold_search_text_only_mlp_seed_62.csv](results/final_v2/diagnostics/threshold_search_text_only_mlp_seed_62.csv) | 8,879 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/threshold_search_text_only_mlp_seed_72.csv](results/final_v2/diagnostics/threshold_search_text_only_mlp_seed_72.csv) | 8,853 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/threshold_search_text_only_mlp_seed_82.csv](results/final_v2/diagnostics/threshold_search_text_only_mlp_seed_82.csv) | 8,821 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/tiktok_label_audit_category_summary.csv](results/final_v2/diagnostics/tiktok_label_audit_category_summary.csv) | 114 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/tiktok_label_audit_issue_summary.csv](results/final_v2/diagnostics/tiktok_label_audit_issue_summary.csv) | 52 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/tiktok_label_audit_overview.csv](results/final_v2/diagnostics/tiktok_label_audit_overview.csv) | 156 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/tiktok_persistent_false_negative_summary.csv](results/final_v2/diagnostics/tiktok_persistent_false_negative_summary.csv) | 752 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/tiktok_persistent_false_negatives.csv](results/final_v2/diagnostics/tiktok_persistent_false_negatives.csv) | 23,970 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/tiktok_persistent_fn_label_audit.csv](results/final_v2/diagnostics/tiktok_persistent_fn_label_audit.csv) | 314 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/tiktok_top3_confusion_summary.csv](results/final_v2/diagnostics/tiktok_top3_confusion_summary.csv) | 1,805 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_contrastive_similarity_fusion_seed_42.csv](results/final_v2/diagnostics/training_history_contrastive_similarity_fusion_seed_42.csv) | 713 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_contrastive_similarity_fusion_seed_52.csv](results/final_v2/diagnostics/training_history_contrastive_similarity_fusion_seed_52.csv) | 707 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_contrastive_similarity_fusion_seed_62.csv](results/final_v2/diagnostics/training_history_contrastive_similarity_fusion_seed_62.csv) | 789 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_contrastive_similarity_fusion_seed_72.csv](results/final_v2/diagnostics/training_history_contrastive_similarity_fusion_seed_72.csv) | 791 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_contrastive_similarity_fusion_seed_82.csv](results/final_v2/diagnostics/training_history_contrastive_similarity_fusion_seed_82.csv) | 628 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_image_only_mlp_seed_42.csv](results/final_v2/diagnostics/training_history_image_only_mlp_seed_42.csv) | 913 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_image_only_mlp_seed_52.csv](results/final_v2/diagnostics/training_history_image_only_mlp_seed_52.csv) | 1,037 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_image_only_mlp_seed_62.csv](results/final_v2/diagnostics/training_history_image_only_mlp_seed_62.csv) | 1,240 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_image_only_mlp_seed_72.csv](results/final_v2/diagnostics/training_history_image_only_mlp_seed_72.csv) | 906 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_image_only_mlp_seed_82.csv](results/final_v2/diagnostics/training_history_image_only_mlp_seed_82.csv) | 1,108 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_learnable_weighted_logit_fusion_seed_42.csv](results/final_v2/diagnostics/training_history_learnable_weighted_logit_fusion_seed_42.csv) | 407 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_learnable_weighted_logit_fusion_seed_52.csv](results/final_v2/diagnostics/training_history_learnable_weighted_logit_fusion_seed_52.csv) | 1,237 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_learnable_weighted_logit_fusion_seed_62.csv](results/final_v2/diagnostics/training_history_learnable_weighted_logit_fusion_seed_62.csv) | 617 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_learnable_weighted_logit_fusion_seed_72.csv](results/final_v2/diagnostics/training_history_learnable_weighted_logit_fusion_seed_72.csv) | 612 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_learnable_weighted_logit_fusion_seed_82.csv](results/final_v2/diagnostics/training_history_learnable_weighted_logit_fusion_seed_82.csv) | 1,074 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_projection_fusion_seed_42.csv](results/final_v2/diagnostics/training_history_projection_fusion_seed_42.csv) | 680 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_projection_fusion_seed_52.csv](results/final_v2/diagnostics/training_history_projection_fusion_seed_52.csv) | 675 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_projection_fusion_seed_62.csv](results/final_v2/diagnostics/training_history_projection_fusion_seed_62.csv) | 675 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_projection_fusion_seed_72.csv](results/final_v2/diagnostics/training_history_projection_fusion_seed_72.csv) | 683 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_projection_fusion_seed_82.csv](results/final_v2/diagnostics/training_history_projection_fusion_seed_82.csv) | 541 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_raw_concatenation_fusion_seed_42.csv](results/final_v2/diagnostics/training_history_raw_concatenation_fusion_seed_42.csv) | 1,277 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_raw_concatenation_fusion_seed_52.csv](results/final_v2/diagnostics/training_history_raw_concatenation_fusion_seed_52.csv) | 1,351 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_raw_concatenation_fusion_seed_62.csv](results/final_v2/diagnostics/training_history_raw_concatenation_fusion_seed_62.csv) | 1,588 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_raw_concatenation_fusion_seed_72.csv](results/final_v2/diagnostics/training_history_raw_concatenation_fusion_seed_72.csv) | 1,356 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_raw_concatenation_fusion_seed_82.csv](results/final_v2/diagnostics/training_history_raw_concatenation_fusion_seed_82.csv) | 1,126 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_text_only_mlp_seed_42.csv](results/final_v2/diagnostics/training_history_text_only_mlp_seed_42.csv) | 1,804 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_text_only_mlp_seed_52.csv](results/final_v2/diagnostics/training_history_text_only_mlp_seed_52.csv) | 1,679 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_text_only_mlp_seed_62.csv](results/final_v2/diagnostics/training_history_text_only_mlp_seed_62.csv) | 1,928 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_text_only_mlp_seed_72.csv](results/final_v2/diagnostics/training_history_text_only_mlp_seed_72.csv) | 1,487 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/diagnostics/training_history_text_only_mlp_seed_82.csv](results/final_v2/diagnostics/training_history_text_only_mlp_seed_82.csv) | 1,934 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/figures/figure_2_robustness_top3.png](results/final_v2/figures/figure_2_robustness_top3.png) | 471,326 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/figures/figure_3_platform_clean_vs_j3.png](results/final_v2/figures/figure_3_platform_clean_vs_j3.png) | 392,770 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/per_platform/best_model_per_joint_platform.csv](results/final_v2/per_platform/best_model_per_joint_platform.csv) | 1,394 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/per_platform/joint_platform_robustness_drop.csv](results/final_v2/per_platform/joint_platform_robustness_drop.csv) | 8,706 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/per_platform/primary_conditions_platform_summary.csv](results/final_v2/per_platform/primary_conditions_platform_summary.csv) | 15,280 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/per_platform/robustness_platform_aggregate.csv](results/final_v2/per_platform/robustness_platform_aggregate.csv) | 39,076 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/per_platform/robustness_platform_per_seed.csv](results/final_v2/per_platform/robustness_platform_per_seed.csv) | 170,947 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/predictions/clean_test_all_model_predictions.csv.gz](results/final_v2/predictions/clean_test_all_model_predictions.csv.gz) | 502,037 | Original prediction CSV, lossless gzip |
| [results/final_v2/predictions/robustness_all_predictions.csv.gz](results/final_v2/predictions/robustness_all_predictions.csv.gz) | 5,644,265 | Original prediction CSV, lossless gzip |
| [results/final_v2/predictions/validation_average_logit_fusion_seed_42.csv.gz](results/final_v2/predictions/validation_average_logit_fusion_seed_42.csv.gz) | 22,735 | Original prediction CSV, lossless gzip |
| [results/final_v2/predictions/validation_average_logit_fusion_seed_52.csv.gz](results/final_v2/predictions/validation_average_logit_fusion_seed_52.csv.gz) | 22,712 | Original prediction CSV, lossless gzip |
| [results/final_v2/predictions/validation_average_logit_fusion_seed_62.csv.gz](results/final_v2/predictions/validation_average_logit_fusion_seed_62.csv.gz) | 22,704 | Original prediction CSV, lossless gzip |
| [results/final_v2/predictions/validation_average_logit_fusion_seed_72.csv.gz](results/final_v2/predictions/validation_average_logit_fusion_seed_72.csv.gz) | 22,725 | Original prediction CSV, lossless gzip |
| [results/final_v2/predictions/validation_average_logit_fusion_seed_82.csv.gz](results/final_v2/predictions/validation_average_logit_fusion_seed_82.csv.gz) | 22,763 | Original prediction CSV, lossless gzip |
| [results/final_v2/predictions/validation_image_only_mlp_seed_42.csv.gz](results/final_v2/predictions/validation_image_only_mlp_seed_42.csv.gz) | 14,327 | Original prediction CSV, lossless gzip |
| [results/final_v2/predictions/validation_image_only_mlp_seed_52.csv.gz](results/final_v2/predictions/validation_image_only_mlp_seed_52.csv.gz) | 14,333 | Original prediction CSV, lossless gzip |
| [results/final_v2/predictions/validation_image_only_mlp_seed_62.csv.gz](results/final_v2/predictions/validation_image_only_mlp_seed_62.csv.gz) | 14,390 | Original prediction CSV, lossless gzip |
| [results/final_v2/predictions/validation_image_only_mlp_seed_72.csv.gz](results/final_v2/predictions/validation_image_only_mlp_seed_72.csv.gz) | 14,305 | Original prediction CSV, lossless gzip |
| [results/final_v2/predictions/validation_image_only_mlp_seed_82.csv.gz](results/final_v2/predictions/validation_image_only_mlp_seed_82.csv.gz) | 14,334 | Original prediction CSV, lossless gzip |
| [results/final_v2/predictions/validation_text_only_mlp_seed_42.csv.gz](results/final_v2/predictions/validation_text_only_mlp_seed_42.csv.gz) | 14,356 | Original prediction CSV, lossless gzip |
| [results/final_v2/predictions/validation_text_only_mlp_seed_52.csv.gz](results/final_v2/predictions/validation_text_only_mlp_seed_52.csv.gz) | 14,285 | Original prediction CSV, lossless gzip |
| [results/final_v2/predictions/validation_text_only_mlp_seed_62.csv.gz](results/final_v2/predictions/validation_text_only_mlp_seed_62.csv.gz) | 14,369 | Original prediction CSV, lossless gzip |
| [results/final_v2/predictions/validation_text_only_mlp_seed_72.csv.gz](results/final_v2/predictions/validation_text_only_mlp_seed_72.csv.gz) | 14,377 | Original prediction CSV, lossless gzip |
| [results/final_v2/predictions/validation_text_only_mlp_seed_82.csv.gz](results/final_v2/predictions/validation_text_only_mlp_seed_82.csv.gz) | 14,396 | Original prediction CSV, lossless gzip |
| [results/final_v2/summaries/average_logit_fusion_validation_aggregate.csv](results/final_v2/summaries/average_logit_fusion_validation_aggregate.csv) | 421 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/average_logit_fusion_validation_summary.csv](results/final_v2/summaries/average_logit_fusion_validation_summary.csv) | 1,399 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/clean_embedding_audit.csv](results/final_v2/summaries/clean_embedding_audit.csv) | 1,454 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/clean_test_model_aggregate.csv](results/final_v2/summaries/clean_test_model_aggregate.csv) | 1,300 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/clean_test_per_seed_summary.csv](results/final_v2/summaries/clean_test_per_seed_summary.csv) | 6,244 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/contrastive_similarity_fusion_validation_aggregate.csv](results/final_v2/summaries/contrastive_similarity_fusion_validation_aggregate.csv) | 468 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/contrastive_similarity_fusion_validation_summary.csv](results/final_v2/summaries/contrastive_similarity_fusion_validation_summary.csv) | 1,289 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/final_clean_validation_model_comparison.csv](results/final_v2/summaries/final_clean_validation_model_comparison.csv) | 1,026 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/final_experiment_output_manifest.csv](results/final_v2/summaries/final_experiment_output_manifest.csv) | 1,475 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/final_main_robustness_results.csv](results/final_v2/summaries/final_main_robustness_results.csv) | 1,393 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/final_model_selection_summary.csv](results/final_v2/summaries/final_model_selection_summary.csv) | 895 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/final_training_input_audit.csv](results/final_v2/summaries/final_training_input_audit.csv) | 258 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/image_corruption_final_qc_summary.csv](results/final_v2/summaries/image_corruption_final_qc_summary.csv) | 222 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/image_corruption_generation_summary.csv](results/final_v2/summaries/image_corruption_generation_summary.csv) | 1,361 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/image_corruption_severity_results.csv](results/final_v2/summaries/image_corruption_severity_results.csv) | 7,751 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/image_only_mlp_training_summary.csv](results/final_v2/summaries/image_only_mlp_training_summary.csv) | 1,612 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/image_only_mlp_validation_aggregate.csv](results/final_v2/summaries/image_only_mlp_validation_aggregate.csv) | 389 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/joint_robustness_aggregate.csv](results/final_v2/summaries/joint_robustness_aggregate.csv) | 6,171 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/learnable_weighted_logit_fusion_validation_aggregate.csv](results/final_v2/summaries/learnable_weighted_logit_fusion_validation_aggregate.csv) | 407 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/learnable_weighted_logit_fusion_validation_summary.csv](results/final_v2/summaries/learnable_weighted_logit_fusion_validation_summary.csv) | 1,182 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/most_damaging_single_corruption.csv](results/final_v2/summaries/most_damaging_single_corruption.csv) | 1,125 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/paper_ready_main_results.csv](results/final_v2/summaries/paper_ready_main_results.csv) | 521 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/paper_ready_research_question_summary.csv](results/final_v2/summaries/paper_ready_research_question_summary.csv) | 1,718 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/perturbed_image_embedding_audit.csv](results/final_v2/summaries/perturbed_image_embedding_audit.csv) | 1,838 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/projection_fusion_validation_aggregate.csv](results/final_v2/summaries/projection_fusion_validation_aggregate.csv) | 348 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/projection_fusion_validation_summary.csv](results/final_v2/summaries/projection_fusion_validation_summary.csv) | 1,009 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/raw_concatenation_fusion_validation_aggregate.csv](results/final_v2/summaries/raw_concatenation_fusion_validation_aggregate.csv) | 369 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/raw_concatenation_fusion_validation_summary.csv](results/final_v2/summaries/raw_concatenation_fusion_validation_summary.csv) | 1,042 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/robustness_condition_input_audit.csv](results/final_v2/summaries/robustness_condition_input_audit.csv) | 972 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/robustness_model_condition_aggregate.csv](results/final_v2/summaries/robustness_model_condition_aggregate.csv) | 20,015 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/robustness_per_seed_summary.csv](results/final_v2/summaries/robustness_per_seed_summary.csv) | 133,858 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/robustness_worst_case_summary.csv](results/final_v2/summaries/robustness_worst_case_summary.csv) | 1,591 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/single_corruption_severity_summary.csv](results/final_v2/summaries/single_corruption_severity_summary.csv) | 9,741 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/split_label_distribution.csv](results/final_v2/summaries/split_label_distribution.csv) | 112 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/split_platform_label_distribution.csv](results/final_v2/summaries/split_platform_label_distribution.csv) | 407 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/split_summary.csv](results/final_v2/summaries/split_summary.csv) | 134 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/text_obfuscated_embedding_audit.csv](results/final_v2/summaries/text_obfuscated_embedding_audit.csv) | 645 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/text_obfuscation_approved_compound_forms.csv](results/final_v2/summaries/text_obfuscation_approved_compound_forms.csv) | 3,137 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/text_obfuscation_approved_compound_summary.csv](results/final_v2/summaries/text_obfuscation_approved_compound_summary.csv) | 301 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/text_obfuscation_development_match_summary.csv](results/final_v2/summaries/text_obfuscation_development_match_summary.csv) | 539 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/text_obfuscation_final_lexicon_audit.csv](results/final_v2/summaries/text_obfuscation_final_lexicon_audit.csv) | 1,355 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/text_obfuscation_test_coverage.csv](results/final_v2/summaries/text_obfuscation_test_coverage.csv) | 118 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/text_obfuscation_test_term_frequency.csv](results/final_v2/summaries/text_obfuscation_test_term_frequency.csv) | 482 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/text_only_mlp_training_summary.csv](results/final_v2/summaries/text_only_mlp_training_summary.csv) | 1,606 | FINAL V2 numerical output/configuration/aggregate plot |
| [results/final_v2/summaries/text_only_mlp_validation_aggregate.csv](results/final_v2/summaries/text_only_mlp_validation_aggregate.csv) | 398 | FINAL V2 numerical output/configuration/aggregate plot |
| [scripts/extract_clean_embeddings.py](scripts/extract_clean_embeddings.py) | 14,295 | Research-source export or labeled publication tooling |
| [scripts/inventory.py](scripts/inventory.py) | 4,844 | Research-source export or labeled publication tooling |
| [scripts/prepare_data.py](scripts/prepare_data.py) | 3,945 | Research-source export or labeled publication tooling |
| [scripts/recompute_bootstrap.py](scripts/recompute_bootstrap.py) | 8,614 | Research-source export or labeled publication tooling |
| [scripts/render_figures.py](scripts/render_figures.py) | 10,017 | Research-source export or labeled publication tooling |
| [scripts/run_final_v2.py](scripts/run_final_v2.py) | 304,805 | Research-source export or labeled publication tooling |
| [scripts/verify_publication.py](scripts/verify_publication.py) | 16,503 | Research-source export or labeled publication tooling |
| [splits/split_manifest.csv](splits/split_manifest.csv) | 572,452 | Caption-free frozen split metadata |
| [splits/test.csv](splits/test.csv) | 84,775 | Caption-free frozen split metadata |
| [splits/train.csv](splits/train.csv) | 398,344 | Caption-free frozen split metadata |
| [splits/validation.csv](splits/validation.csv) | 89,473 | Caption-free frozen split metadata |
| [src/__init__.py](src/__init__.py) | 0 | Extracted research definitions |
| [src/final_v2_core.py](src/final_v2_core.py) | 18,779 | Extracted research definitions |
| [src/perturbations.py](src/perturbations.py) | 6,365 | Extracted research definitions |
| [src/split_normalization.py](src/split_normalization.py) | 734 | Extracted research definitions |
