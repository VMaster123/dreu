import os
import json
import glob
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PATHS
# ============================================================

# ============================================================
# PATHS
# ============================================================

# This file lives in:
# codebase/results/analyze_results.py

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

ANALYSIS_DIR = os.path.join(
    RESULTS_DIR,
    "analysis",
)

os.makedirs(
    ANALYSIS_DIR,
    exist_ok=True,
)


# ============================================================
# LOAD RESULTS
# ============================================================


def load_results():
    """
    Load results.json from every benchmark directory.
    """

    pattern = os.path.join(
        RESULTS_DIR,
        "*",
        "results.json",
    )

    files = sorted(glob.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No results.json files found under {RESULTS_DIR}")

    records = []

    for path in files:

        benchmark_dir = os.path.basename(os.path.dirname(path))

        with open(path, "r") as f:
            result = json.load(f)

        result["benchmark"] = result.get(
            "benchmark",
            benchmark_dir,
        )

        records.append(result)

    return records


# ============================================================
# SUMMARY TABLE
# ============================================================


def create_summary(records):

    rows = []

    for result in records:

        train_history = result.get("train", [])
        val_history = result.get("val", [])

        best_epoch = None

        if val_history:
            best_epoch = int(np.argmin(val_history)) + 1

        rows.append(
            {
                "benchmark": result["benchmark"],
                "best_val_mse": result.get("best_val", np.nan),
                "test_mse": result.get("test_mse", np.nan),
                "test_rmse": result.get("test_rmse", np.nan),
                "test_mse_original": result.get(
                    "test_mse_original",
                    np.nan,
                ),
                "test_rmse_original": result.get(
                    "test_rmse_original",
                    np.nan,
                ),
                "best_epoch": best_epoch,
                "epochs_recorded": len(train_history),
            }
        )

    df = pd.DataFrame(rows)

    if len(df) > 0:

        # ----------------------------------------------------
        # Baseline-relative metrics
        # ----------------------------------------------------

        # A normalized target with MSE ~= 1 is roughly the
        # variance-level prediction error. This gives us a
        # useful reference point.
        df["test_mse_reduction_vs_variance"] = 1.0 - df["test_mse"]

        df["test_rmse_reduction_vs_variance"] = 1.0 - df["test_rmse"]

        # ----------------------------------------------------
        # Ranking
        # ----------------------------------------------------

        df["test_mse_rank"] = (
            df["test_mse"]
            .rank(
                method="min",
                ascending=True,
            )
            .astype(int)
        )

    return df


# ============================================================
# PRINT SUMMARY
# ============================================================


def print_summary(df):

    print("\n")
    print("=" * 80)
    print("RESULT SUMMARY")
    print("=" * 80)

    columns = [
        "benchmark",
        "best_val_mse",
        "test_mse",
        "test_rmse",
        "test_mse_original",
        "test_rmse_original",
        "best_epoch",
    ]

    display_df = df[columns].copy()

    for column in columns[1:]:

        if column != "best_epoch":
            display_df[column] = display_df[column].map(
                lambda x: f"{x:.6f}" if pd.notna(x) else "N/A"
            )

    print(display_df.to_string(index=False))


# ============================================================
# PLOT 1
# TEST MSE BY BENCHMARK
# ============================================================


def plot_test_mse(df):

    plt.figure(figsize=(10, 6))

    x = np.arange(len(df))

    plt.bar(
        x,
        df["test_mse"],
    )

    plt.xticks(
        x,
        df["benchmark"],
        rotation=20,
        ha="right",
    )

    plt.ylabel("Test MSE")
    plt.xlabel("Benchmark")
    plt.title("Test MSE Across Quantum Transfer Benchmarks")

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    path = os.path.join(
        ANALYSIS_DIR,
        "test_mse_by_benchmark.png",
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return path


# ============================================================
# PLOT 2
# TEST RMSE BY BENCHMARK
# ============================================================


def plot_test_rmse(df):

    plt.figure(figsize=(10, 6))

    x = np.arange(len(df))

    plt.bar(
        x,
        df["test_rmse"],
    )

    plt.xticks(
        x,
        df["benchmark"],
        rotation=20,
        ha="right",
    )

    plt.ylabel("Test RMSE (normalized energy)")
    plt.xlabel("Benchmark")
    plt.title("Test RMSE Across Quantum Transfer Benchmarks")

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    path = os.path.join(
        ANALYSIS_DIR,
        "test_rmse_by_benchmark.png",
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return path


# ============================================================
# PLOT 3
# VALIDATION VS TEST
# ============================================================


def plot_validation_vs_test(df):

    plt.figure(figsize=(8, 7))

    plt.scatter(
        df["best_val_mse"],
        df["test_mse"],
        s=100,
    )

    for _, row in df.iterrows():

        plt.annotate(
            row["benchmark"],
            (
                row["best_val_mse"],
                row["test_mse"],
            ),
            xytext=(6, 6),
            textcoords="offset points",
        )

    minimum = min(
        df["best_val_mse"].min(),
        df["test_mse"].min(),
    )

    maximum = max(
        df["best_val_mse"].max(),
        df["test_mse"].max(),
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        alpha=0.5,
    )

    plt.xlabel("Best Validation MSE")
    plt.ylabel("Test MSE")
    plt.title("Validation Error vs. Test Error")

    plt.grid(alpha=0.3)

    plt.tight_layout()

    path = os.path.join(
        ANALYSIS_DIR,
        "validation_vs_test.png",
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return path


# ============================================================
# PLOT 4
# TRAINING CURVES
# ============================================================


def plot_training_curves(records):

    for result in records:

        benchmark = result["benchmark"]

        train = result.get("train", [])
        val = result.get("val", [])

        if not train:
            continue

        epochs = np.arange(
            1,
            len(train) + 1,
        )

        plt.figure(figsize=(10, 6))

        plt.plot(
            epochs,
            train,
            label="Training MSE",
        )

        if val:

            plt.plot(
                np.arange(
                    1,
                    len(val) + 1,
                ),
                val,
                label="Validation MSE",
            )

        plt.xlabel("Epoch")
        plt.ylabel("MSE")

        plt.title(f"Training and Validation Curves: {benchmark}")

        plt.legend()

        plt.grid(alpha=0.3)

        plt.tight_layout()

        path = os.path.join(
            ANALYSIS_DIR,
            f"{benchmark}_training_curve.png",
        )

        plt.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close()


# ============================================================
# PLOT 5
# TRAIN/VALIDATION GAP
# ============================================================


def calculate_generalization_gap(records):

    rows = []

    for result in records:

        train = result.get("train", [])
        val = result.get("val", [])

        if not train or not val:
            continue

        best_epoch = int(np.argmin(val))

        best_train = train[best_epoch]
        best_val = val[best_epoch]

        gap = best_val - best_train

        ratio = best_val / max(best_train, 1e-12)

        rows.append(
            {
                "benchmark": result["benchmark"],
                "best_train_mse": best_train,
                "best_val_mse": best_val,
                "generalization_gap": gap,
                "validation_to_train_ratio": ratio,
                "best_epoch": best_epoch + 1,
            }
        )

    return pd.DataFrame(rows)


def plot_generalization_gap(gap_df):

    if gap_df.empty:
        return None

    plt.figure(figsize=(10, 6))

    x = np.arange(len(gap_df))

    plt.bar(
        x,
        gap_df["generalization_gap"],
    )

    plt.axhline(
        0,
        linestyle="--",
        alpha=0.5,
    )

    plt.xticks(
        x,
        gap_df["benchmark"],
        rotation=20,
        ha="right",
    )

    plt.ylabel("Validation MSE - Training MSE")

    plt.xlabel("Benchmark")

    plt.title("Generalization Gap at Best Validation Epoch")

    plt.grid(
        axis="y",
        alpha=0.3,
    )

    plt.tight_layout()

    path = os.path.join(
        ANALYSIS_DIR,
        "generalization_gap.png",
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close()

    return path


# ============================================================
# ANALYSIS
# ============================================================


def analyze_results(df, gap_df):

    lines = []

    lines.append("# Quantum Transfer Benchmark Analysis")

    lines.append("")

    lines.append("## 1. Overview")

    lines.append("")

    lines.append(
        "The experiment evaluates a shared classical encoder and "
        "surrogate model across four quantum benchmarks: "
        "architecture transfer, Hamiltonian variation, noise "
        "transfer, and random instances."
    )

    lines.append("")

    lines.append(
        "The reported errors are based on the normalized energy "
        "target used during training. Therefore, a normalized "
        "MSE near 1 indicates performance near the scale of the "
        "target variance, while substantially smaller values "
        "indicate that the model explains more of the energy "
        "variation."
    )

    lines.append("")

    # --------------------------------------------------------
    # Best benchmark
    # --------------------------------------------------------

    if not df.empty:

        best = df.loc[df["test_mse"].idxmin()]

        worst = df.loc[df["test_mse"].idxmax()]

        lines.append("## 2. Benchmark Performance")

        lines.append("")

        lines.append(
            f"The lowest test MSE was obtained on "
            f"**{best['benchmark']}**, with a normalized test "
            f"MSE of **{best['test_mse']:.6f}** and a normalized "
            f"RMSE of **{best['test_rmse']:.6f}**."
        )

        lines.append("")

        lines.append(
            f"The highest test MSE was obtained on "
            f"**{worst['benchmark']}**, with a normalized test "
            f"MSE of **{worst['test_mse']:.6f}**."
        )

        lines.append("")

        # ----------------------------------------------------
        # Ranking
        # ----------------------------------------------------

        ranked = df.sort_values("test_mse")

        lines.append("Performance ranking by test MSE:")

        lines.append("")

        for i, (_, row) in enumerate(
            ranked.iterrows(),
            start=1,
        ):

            lines.append(
                f"{i}. **{row['benchmark']}** — " f"MSE = {row['test_mse']:.6f}"
            )

        lines.append("")

    # --------------------------------------------------------
    # Generalization
    # --------------------------------------------------------

    if not gap_df.empty:

        lines.append("## 3. Generalization Behavior")

        lines.append("")

        for _, row in gap_df.iterrows():

            lines.append(
                f"For **{row['benchmark']}**, the best validation "
                f"epoch was {int(row['best_epoch'])}. At that point, "
                f"training MSE was {row['best_train_mse']:.6f}, "
                f"while validation MSE was "
                f"{row['best_val_mse']:.6f}, producing a "
                f"generalization gap of "
                f"{row['generalization_gap']:.6f}."
            )

            lines.append("")

        lines.append(
            "A large positive validation-training gap indicates "
            "that the model is fitting the training distribution "
            "more strongly than it generalizes to validation "
            "samples. This distinction is particularly important "
            "for the transfer benchmarks, where low training error "
            "alone is not evidence of successful transfer."
        )

        lines.append("")

    # --------------------------------------------------------
    # Overfitting interpretation
    # --------------------------------------------------------

    lines.append("## 4. Interpretation of Training Behavior")

    lines.append("")

    lines.append(
        "The training curves should be interpreted together with "
        "the validation curves. A decreasing training error "
        "combined with a flat or increasing validation error is "
        "evidence of overfitting rather than improved "
        "generalization."
    )

    lines.append("")

    lines.append(
        "In particular, if the model reaches very low training "
        "MSE while validation MSE remains close to its initial "
        "level, the encoder-surrogate system has sufficient "
        "capacity to represent the training samples but has not "
        "learned a transferable representation of the underlying "
        "quantum problem."
    )

    lines.append("")

    # --------------------------------------------------------
    # Important caveat
    # --------------------------------------------------------

    lines.append("## 5. Important Experimental Caveat")

    lines.append("")

    lines.append(
        "The benchmark results should not be described as evidence "
        "of successful quantum transfer solely from the training "
        "loss. The critical quantity is performance on held-out "
        "instances whose quantum properties differ from those "
        "represented during training."
    )

    lines.append("")

    lines.append(
        "The current experiment also uses a classical surrogate "
        "trained directly against simulated energy values. The "
        "encoder is therefore learning a representation of the "
        "provided classical description of the quantum instance; "
        "it should not be described as a physics-informed encoder "
        "unless explicit physical constraints or quantum "
        "equations are incorporated into its objective or "
        "architecture."
    )

    lines.append("")

    # --------------------------------------------------------
    # Suggested paper language
    # --------------------------------------------------------

    lines.append("## 6. Suggested Results-Section Language")

    lines.append("")

    if not df.empty:

        best_name = best["benchmark"]

        lines.append(
            f"The proposed encoder-surrogate architecture was "
            f"evaluated across four quantum benchmark settings. "
            f"Among the evaluated tasks, {best_name} produced the "
            f"lowest held-out prediction error. However, the "
            f"training and validation curves reveal a substantial "
            f"distinction between fitting the observed training "
            f"samples and generalizing to unseen quantum "
            f"instances. In several settings, the training error "
            f"decreased substantially while validation error "
            f"remained comparatively high, indicating that model "
            f"capacity alone was sufficient to fit the training "
            f"distribution but did not guarantee transfer."
        )

        lines.append("")

        lines.append(
            "These results motivate evaluating transfer according "
            "to held-out quantum structure rather than training "
            "error alone. In particular, architecture, Hamiltonian, "
            "and noise-transfer experiments should be interpreted "
            "as tests of whether the learned latent representation "
            "captures features that remain predictive when the "
            "underlying quantum instance changes."
        )

    lines.append("")

    return "\n".join(lines)


# ============================================================
# SAVE EVERYTHING
# ============================================================


def save_outputs(df, gap_df, records):

    # --------------------------------------------------------
    # Summary CSV
    # --------------------------------------------------------

    summary_path = os.path.join(
        ANALYSIS_DIR,
        "benchmark_summary.csv",
    )

    df.to_csv(
        summary_path,
        index=False,
    )

    # --------------------------------------------------------
    # Generalization CSV
    # --------------------------------------------------------

    gap_path = os.path.join(
        ANALYSIS_DIR,
        "generalization_analysis.csv",
    )

    gap_df.to_csv(
        gap_path,
        index=False,
    )

    # --------------------------------------------------------
    # Full report
    # --------------------------------------------------------

    report = analyze_results(
        df,
        gap_df,
    )

    report_path = os.path.join(
        ANALYSIS_DIR,
        "analysis_report.md",
    )

    with open(
        report_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(report)

    return (
        summary_path,
        gap_path,
        report_path,
    )


# ============================================================
# MAIN
# ============================================================


def main():

    print("\n" + "=" * 80)
    print("QUANTUM TRANSFER RESULT ANALYSIS")
    print("=" * 80)

    print(
        "\nResults directory:",
        RESULTS_DIR,
    )

    records = load_results()

    print("\nLoaded benchmarks:")

    for result in records:

        print(
            " -",
            result["benchmark"],
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    df = create_summary(records)

    print_summary(df)

    # --------------------------------------------------------
    # Generalization
    # --------------------------------------------------------

    gap_df = calculate_generalization_gap(records)

    # --------------------------------------------------------
    # Graphs
    # --------------------------------------------------------

    print("\nGenerating graphs...")

    plot_test_mse(df)

    plot_test_rmse(df)

    plot_validation_vs_test(df)

    plot_training_curves(records)

    plot_generalization_gap(gap_df)

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    (
        summary_path,
        gap_path,
        report_path,
    ) = save_outputs(
        df,
        gap_df,
        records,
    )

    print("\nGenerated:")

    print(
        " ",
        summary_path,
    )

    print(
        " ",
        gap_path,
    )

    print(
        " ",
        report_path,
    )

    print("\nGraphs saved to:")

    print(
        " ",
        ANALYSIS_DIR,
    )

    print("\nDone.")


if __name__ == "__main__":
    main()
