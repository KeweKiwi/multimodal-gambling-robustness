"""Figure code extracted from FINAL V2 cells 113 and 115; portable paths only.
Reads reference aggregate CSVs. Writes plots under work/results/final_v2/figures.
The source filename figure_2_robustness_top3 is an analysis graph, not manuscript Figure 2.
"""
from pathlib import Path
from IPython.display import display
_REPO=Path(__file__).resolve().parents[1]

# Original FINAL V2 cell 113
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATH
# ============================================================

ROOT = _REPO

RESULT_PATH = (
    ROOT
    / "results"
    / "final_v2"
    / "summaries"
    / "robustness_model_condition_aggregate.csv"
)

OUTPUT_DIR = ROOT / "work" / "results" / "final_v2" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD FINAL V2 RESULTS
# ============================================================

df = pd.read_csv(RESULT_PATH)


# Hanya tiga fusion model utama
model_names = {
    "learnable_weighted_logit_fusion":
        "Learnable Weighted Logit Fusion",

    "raw_concatenation_fusion":
        "Raw Concatenation Fusion",

    "average_logit_fusion":
        "Average Logit Fusion",
}


# Urutan kondisi mengikuti desain eksperimen
condition_order = [
    "clean",
    "text_obfuscated",
    "gaussian_blur_r2",
    "gaussian_blur_r5",
    "pixelation_p50",
    "pixelation_p25",
    "pixelation_p12_5",
    "blur_r5_pixelation_p12_5",
    "joint_j1_text_obfuscated_blur_r5",
    "joint_j2_text_obfuscated_pixelation_p12_5",
    "joint_j3_text_obfuscated_blur_r5_pixelation_p12_5",
]


condition_labels = {
    "clean": "Clean",
    "text_obfuscated": "Text\nobfuscation",
    "gaussian_blur_r2": "Blur\nr2",
    "gaussian_blur_r5": "Blur\nr5",
    "pixelation_p50": "Pixelation\n50%",
    "pixelation_p25": "Pixelation\n25%",
    "pixelation_p12_5": "Pixelation\n12.5%",
    "blur_r5_pixelation_p12_5":
        "Blur r5 +\nPixelation 12.5%",
    "joint_j1_text_obfuscated_blur_r5": "J1",
    "joint_j2_text_obfuscated_pixelation_p12_5": "J2",
    "joint_j3_text_obfuscated_blur_r5_pixelation_p12_5": "J3",
}


# ============================================================
# FILTER + PREPARE
# ============================================================

plot_df = df[
    df["model_name"].isin(model_names.keys())
    & df["condition"].isin(condition_order)
].copy()

plot_df["condition"] = pd.Categorical(
    plot_df["condition"],
    categories=condition_order,
    ordered=True,
)

plot_df = plot_df.sort_values("condition")

# Ubah F1 dari 0–1 menjadi persen
plot_df["f1_percent"] = plot_df["mean_test_f1"] * 100


# ============================================================
# PLOT
# ============================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
})

fig, ax = plt.subplots(figsize=(7.4, 4.2))

for model_key, model_label in model_names.items():
    model_df = plot_df[
        plot_df["model_name"] == model_key
    ].sort_values("condition")

    ax.plot(
        range(len(model_df)),
        model_df["f1_percent"],
        marker="o",
        linewidth=1.8,
        markersize=4.5,
        label=model_label,
    )


# ============================================================
# AXIS
# ============================================================

ax.set_xticks(range(len(condition_order)))

ax.set_xticklabels(
    [condition_labels[c] for c in condition_order],
    rotation=35,
    ha="right",
)

ax.set_ylabel("F1-score (%)")
ax.set_xlabel("Test condition")

# Sedikit zoom agar perbedaan terlihat,
# tetapi range tetap ditampilkan jelas.
ax.set_ylim(93.5, 96.7)

ax.grid(
    axis="y",
    linestyle="--",
    linewidth=0.6,
    alpha=0.5,
)

ax.legend(
    loc="lower left",
    frameon=True,
)

# Tidak perlu title karena paper sudah punya caption
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

fig.tight_layout()


# ============================================================
# SAVE
# ============================================================

png_path = OUTPUT_DIR / "figure_2_robustness_top3.png"
pdf_path = OUTPUT_DIR / "figure_2_robustness_top3.pdf"

fig.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    pdf_path,
    bbox_inches="tight",
)

plt.show()

print("Saved:")
print(png_path)
print(pdf_path)

# Original FINAL V2 cell 115
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Figure 3 — Platform-level F1: Clean vs J3
# Three-panel journal-ready version
# ============================================================

PROJECT_ROOT = _REPO

platform_path = (
    PROJECT_ROOT
    / "results"
    / "final_v2"
    / "per_platform"
    / "joint_platform_robustness_drop.csv"
)

# ------------------------------------------------------------
# 1. Load FINAL V2 results
# ------------------------------------------------------------

platform_df = pd.read_csv(platform_path)

TOP_MODELS = [
    "learnable_weighted_logit_fusion",
    "raw_concatenation_fusion",
    "average_logit_fusion",
]

MODEL_LABELS = {
    "learnable_weighted_logit_fusion": "Learnable\nWeighted",
    "raw_concatenation_fusion": "Raw\nConcatenation",
    "average_logit_fusion": "Average\nLogit",
}

PLATFORMS = [
    "Facebook",
    "TikTok",
    "X",
]

# ------------------------------------------------------------
# 2. Select J3 results
# clean_f1 = corresponding clean performance
# mean_f1  = J3 performance
# ------------------------------------------------------------

plot_df = (
    platform_df[
        (platform_df["joint_condition"] == "J3")
        & (platform_df["model_name"].isin(TOP_MODELS))
        & (platform_df["platform"].isin(PLATFORMS))
    ]
    .copy()
)

plot_df["Model"] = plot_df["model_name"].map(MODEL_LABELS)

plot_df["Clean"] = plot_df["clean_f1"] * 100
plot_df["J3"] = plot_df["mean_f1"] * 100

# Check
assert len(plot_df) == 9, (
    f"Expected 9 rows, found {len(plot_df)}"
)

display(
    plot_df[
        [
            "platform",
            "Model",
            "samples",
            "positive_samples",
            "Clean",
            "J3",
        ]
    ]
    .sort_values(["platform", "Model"])
)


# ------------------------------------------------------------
# 3. Plot
# ------------------------------------------------------------

fig, axes = plt.subplots(
    1,
    3,
    figsize=(11, 4.5),
    sharey=True,
)

x = np.arange(len(TOP_MODELS))
width = 0.34

for ax, platform in zip(axes, PLATFORMS):

    subset = (
        plot_df[
            plot_df["platform"] == platform
        ]
        .set_index("model_name")
        .loc[TOP_MODELS]
        .reset_index()
    )

    clean_values = subset["Clean"].values
    j3_values = subset["J3"].values

    # Clean
    clean_bars = ax.bar(
        x - width / 2,
        clean_values,
        width,
        label="Clean",
        edgecolor="black",
        linewidth=0.7,
    )

    # J3
    j3_bars = ax.bar(
        x + width / 2,
        j3_values,
        width,
        label="J3",
        hatch="//",
        edgecolor="black",
        linewidth=0.7,
    )

    # --------------------------------------------------------
    # Value labels
    # --------------------------------------------------------

    for bars in [clean_bars, j3_bars]:
        for bar in bars:

            value = bar.get_height()

            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + 1.0,
                f"{value:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    # --------------------------------------------------------
    # Platform title + positive count
    # --------------------------------------------------------

    positive_count = int(
        subset["positive_samples"].iloc[0]
    )

    total_count = int(
        subset["samples"].iloc[0]
    )

    ax.set_title(
        f"{platform}\n"
        f"{positive_count} positives / {total_count} samples",
        fontsize=10,
        pad=8,
    )

    # --------------------------------------------------------
    # Axis formatting
    # --------------------------------------------------------

    ax.set_xticks(x)

    ax.set_xticklabels(
        [
            "Learnable\nWeighted",
            "Raw\nConcatenation",
            "Average\nLogit",
        ],
        fontsize=8.5,
    )

    ax.set_ylim(0, 105)

    ax.grid(
        axis="y",
        linestyle="--",
        linewidth=0.6,
        alpha=0.3,
    )

    ax.set_axisbelow(True)


# ------------------------------------------------------------
# 4. Shared labels
# ------------------------------------------------------------

axes[0].set_ylabel(
    "F1-score (%)",
    fontsize=10,
)

# One shared legend only
handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="upper center",
    ncol=2,
    frameon=False,
    bbox_to_anchor=(0.5, 1.03),
    fontsize=9,
)

plt.tight_layout(
    rect=[0, 0, 1, 0.94]
)


# ------------------------------------------------------------
# 5. Save publication-quality outputs
# ------------------------------------------------------------

FIGURE_DIR = (
    PROJECT_ROOT
    / "work"
    / "results"
    / "final_v2"
    / "figures"
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

png_path = (
    FIGURE_DIR
    / "figure_3_platform_clean_vs_j3.png"
)

pdf_path = (
    FIGURE_DIR
    / "figure_3_platform_clean_vs_j3.pdf"
)

plt.savefig(
    png_path,
    dpi=600,
    bbox_inches="tight",
)

plt.savefig(
    pdf_path,
    bbox_inches="tight",
)

plt.show()

print("Saved:")
print(png_path)
print(pdf_path)
