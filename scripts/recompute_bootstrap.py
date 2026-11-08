"""Exact FINAL V2 bootstrap cell 73 with a read-only saved-prediction adapter.
All 5000 resamples per pair are recomputed; no models or embeddings are loaded.
Output goes to work/bootstrap/ and is compared to the published original table.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from IPython.display import display
ROOT=Path(__file__).resolve().parents[1]
test_df=pd.read_csv(ROOT/"splits/test.csv")
y_test=test_df.label.to_numpy(dtype=np.int64)
TRAINING_SEEDS=[42,52,62,72,82]
robustness_predictions_df=pd.read_csv(ROOT/"results/final_v2/predictions/robustness_all_predictions.csv.gz")
bootstrap_dir=ROOT/"work/bootstrap"
bootstrap_dir.mkdir(parents=True,exist_ok=True)
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
expected=pd.read_csv(ROOT/"results/final_v2/bootstrap/paired_bootstrap_top3_primary_conditions.csv")
actual=paired_bootstrap_top3_df
assert actual.shape==expected.shape
for column in expected:
    if pd.api.types.is_numeric_dtype(expected[column]):
        assert np.allclose(actual[column],expected[column],rtol=0,atol=1e-12),column
    else:
        assert actual[column].equals(expected[column]),column
print("All 12 bootstrap comparisons match the original 5000-resample FINAL V2 table.")
