"""
run_benchmarks.py
=================
Runs the full duplicate-detection pipeline on synthetic data and generates
professional benchmark charts for the README.

Steps:
  1. Generate synthetic PDFs (if not already present)
  2. Run detect_duplicates on every client folder
  3. Collect timing, accuracy, and storage metrics
  4. Generate gradient-styled charts saved as PNGs in benchmarks/

Usage:
    python run_benchmarks.py
"""

import os
import sys
import time
import shutil
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyBboxPatch
import matplotlib.ticker as ticker

# ---------------------------------------------------------------------------
# Style constants
# ---------------------------------------------------------------------------
BG_COLOR = "#0d1117"
CARD_COLOR = "#161b22"
TEXT_COLOR = "#e6edf3"
GRID_COLOR = "#21262d"
ACCENT_TEAL = "#58a6ff"
ACCENT_GREEN = "#3fb950"
ACCENT_ORANGE = "#d29922"
ACCENT_RED = "#f85149"
ACCENT_PURPLE = "#bc8cff"

GRAD_TEAL = mcolors.LinearSegmentedColormap.from_list("teal", ["#1a535c", "#4ecdc4"])
GRAD_GREEN = mcolors.LinearSegmentedColormap.from_list("green", ["#1b4332", "#52b788"])
GRAD_RED = mcolors.LinearSegmentedColormap.from_list("red", ["#6a040f", "#e5383b"])
GRAD_BLUE = mcolors.LinearSegmentedColormap.from_list("blue", ["#023e8a", "#48cae4"])
GRAD_ORANGE = mcolors.LinearSegmentedColormap.from_list("orange", ["#7f4f24", "#dda15e"])

plt.rcParams.update({
    "figure.facecolor": BG_COLOR,
    "axes.facecolor": CARD_COLOR,
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "text.color": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    "grid.color": GRID_COLOR,
    "font.family": "sans-serif",
    "font.size": 11,
})


def _apply_gradient_to_bars(ax, bars, cmap, orientation="vertical"):
    """Fill matplotlib bars with a gradient."""
    for bar in bars:
        if orientation == "vertical":
            x0, y0 = bar.get_x(), bar.get_y()
            w, h = bar.get_width(), bar.get_height()
            gradient = np.linspace(0, 1, 256).reshape(256, 1)
            ax.imshow(gradient, extent=[x0, x0 + w, y0, y0 + h],
                      aspect="auto", cmap=cmap, zorder=bar.get_zorder() + 1,
                      clip_path=bar, clip_on=True)
        else:  # horizontal
            x0, y0 = bar.get_x(), bar.get_y()
            w, h = bar.get_width(), bar.get_height()
            gradient = np.linspace(0, 1, 256).reshape(1, 256)
            ax.imshow(gradient, extent=[x0, x0 + w, y0, y0 + h],
                      aspect="auto", cmap=cmap, zorder=bar.get_zorder() + 1,
                      clip_path=bar, clip_on=True)
    # Make original bars transparent so gradient shows
    for bar in bars:
        bar.set_facecolor("none")
        bar.set_edgecolor(TEXT_COLOR)
        bar.set_linewidth(0.5)


# ---------------------------------------------------------------------------
# Ground truth for accuracy calculation
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    # (folder, pdf1, pdf2) -> expected duplicate? True/False
    ("CLIENT_1001", "DOC_0001.pdf", "DOC_0002.pdf"): True,
    ("CLIENT_1001", "DOC_0001.pdf", "DOC_0003.pdf"): False,
    ("CLIENT_1001", "DOC_0002.pdf", "DOC_0003.pdf"): False,
    ("CLIENT_1002", "DOC_0004.pdf", "DOC_0005.pdf"): True,
    ("CLIENT_1002", "DOC_0004.pdf", "DOC_0006.pdf"): False,
    ("CLIENT_1002", "DOC_0005.pdf", "DOC_0006.pdf"): False,
    ("CLIENT_1003", "DOC_0007.pdf", "DOC_0008.pdf"): False,  # partial only
    ("CLIENT_1003", "DOC_0007.pdf", "DOC_0009.pdf"): False,
    ("CLIENT_1003", "DOC_0008.pdf", "DOC_0009.pdf"): False,
    ("CLIENT_1004", "DOC_0010.pdf", "DOC_0011.pdf"): False,
    ("CLIENT_1004", "DOC_0010.pdf", "DOC_0012.pdf"): False,
    ("CLIENT_1004", "DOC_0011.pdf", "DOC_0012.pdf"): False,
    ("CLIENT_1005", "DOC_0013.pdf", "DOC_0014.pdf"): True,
    ("CLIENT_1005", "DOC_0013.pdf", "DOC_0015.pdf"): False,  # near-dup but below 98%
    ("CLIENT_1005", "DOC_0013.pdf", "DOC_0016.pdf"): False,
    ("CLIENT_1005", "DOC_0014.pdf", "DOC_0015.pdf"): False,
    ("CLIENT_1005", "DOC_0014.pdf", "DOC_0016.pdf"): False,
    ("CLIENT_1005", "DOC_0015.pdf", "DOC_0016.pdf"): False,
    ("CLIENT_2001", "DL_COPY.pdf", "DL_ORIG.pdf"): True,
    ("CLIENT_2001", "DL_COPY.pdf", "DL_OTHER.pdf"): False,
    ("CLIENT_2001", "DL_ORIG.pdf", "DL_OTHER.pdf"): False,
    ("CLIENT_2002", "BC_ORIGINAL.pdf", "BC_ROTATED_180.pdf"): True,  # ideally detected
    ("CLIENT_2002", "BC_ORIGINAL.pdf", "BC_ROTATED_90.pdf"): True,   # ideally detected
    ("CLIENT_2002", "BC_ROTATED_180.pdf", "BC_ROTATED_90.pdf"): True, # ideally detected
    ("CLIENT_2003", "TAX_ORIGINAL.pdf", "TAX_SCANNED.pdf"): True,
    ("CLIENT_2003", "TAX_ORIGINAL.pdf", "TAX_SHIFTED.pdf"): True,    # ideally detected
    ("CLIENT_2003", "TAX_SCANNED.pdf", "TAX_SHIFTED.pdf"): False,
    ("CLIENT_2004", "NOTICE_FULL.pdf", "NOTICE_SCALED_60.pdf"): True,
    ("CLIENT_2004", "NOTICE_FULL.pdf", "NOTICE_SCALED_80.pdf"): True,
    ("CLIENT_2004", "NOTICE_SCALED_60.pdf", "NOTICE_SCALED_80.pdf"): True,
    ("CLIENT_2005", "GOV_BC.pdf", "GOV_DL.pdf"): False,
    ("CLIENT_2005", "GOV_BC.pdf", "GOV_NOTICE.pdf"): False,
    ("CLIENT_2005", "GOV_BC.pdf", "GOV_TAX.pdf"): False,
    ("CLIENT_2005", "GOV_DL.pdf", "GOV_NOTICE.pdf"): False,
    ("CLIENT_2005", "GOV_DL.pdf", "GOV_TAX.pdf"): False,
    ("CLIENT_2005", "GOV_NOTICE.pdf", "GOV_TAX.pdf"): False,
}
# CLIENT_2006: all 10 pairs are true duplicates
for i in range(1, 6):
    for j in range(i + 1, 6):
        GROUND_TRUTH[("CLIENT_2006", f"BATCH_{i:02d}.pdf", f"BATCH_{j:02d}.pdf")] = True


# Simulated storage cost per PDF (KB) for impact analysis
STORAGE_PER_DOC_KB = {
    "DOC_0001.pdf": 420, "DOC_0002.pdf": 420, "DOC_0003.pdf": 380,
    "DOC_0004.pdf": 520, "DOC_0005.pdf": 530, "DOC_0006.pdf": 340,
    "DOC_0007.pdf": 780, "DOC_0008.pdf": 790, "DOC_0009.pdf": 360,
    "DOC_0010.pdf": 450, "DOC_0011.pdf": 410, "DOC_0012.pdf": 280,
    "DOC_0013.pdf": 560, "DOC_0014.pdf": 560, "DOC_0015.pdf": 570, "DOC_0016.pdf": 390,
    "DL_ORIG.pdf": 85, "DL_COPY.pdf": 85, "DL_OTHER.pdf": 88,
    "BC_ORIGINAL.pdf": 310, "BC_ROTATED_90.pdf": 315, "BC_ROTATED_180.pdf": 312,
    "TAX_ORIGINAL.pdf": 290, "TAX_SCANNED.pdf": 295, "TAX_SHIFTED.pdf": 292,
    "NOTICE_FULL.pdf": 340, "NOTICE_SCALED_80.pdf": 320, "NOTICE_SCALED_60.pdf": 280,
    "GOV_DL.pdf": 85, "GOV_BC.pdf": 310, "GOV_TAX.pdf": 290, "GOV_NOTICE.pdf": 340,
    "BATCH_01.pdf": 85, "BATCH_02.pdf": 85, "BATCH_03.pdf": 85,
    "BATCH_04.pdf": 85, "BATCH_05.pdf": 85,
}


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

def run_pipeline(script_dir):
    """Generate data + run detection, return list of result dicts."""
    print("=" * 60)
    print("BENCHMARK PIPELINE")
    print("=" * 60)

    # Step 1 — generate synthetic data
    test_dir = os.path.join(script_dir, "test")
    results_dir = os.path.join(script_dir, "results")
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)
    if os.path.exists(results_dir):
        shutil.rmtree(results_dir)

    print("\n[1/3] Generating synthetic PDFs...")
    from generate_synthetic_data import main as gen_main
    gen_main()

    # Step 2 — run detector and collect per-pair timings
    print("\n[2/3] Running duplicate detection with timing...")
    from detect_duplicates import (
        convert_pdf_to_images, compare_page_to_all, save_comparison_images,
        rename_output_folder, quantize_image, process_folders
    )
    import concurrent.futures, csv

    os.makedirs(results_dir, exist_ok=True)
    all_metrics = []

    folders = sorted([
        d for d in os.listdir(test_dir)
        if os.path.isdir(os.path.join(test_dir, d))
    ])

    for folder_name in folders:
        folder_path = os.path.join(test_dir, folder_name)
        pdfs = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".pdf"))
        pairs = [(pdfs[i], pdfs[j]) for i in range(len(pdfs)) for j in range(i+1, len(pdfs))]

        for pdf1, pdf2 in pairs:
            t0 = time.time()

            p1 = os.path.join(folder_path, pdf1)
            p2 = os.path.join(folder_path, pdf2)
            imgs1, pg1 = convert_pdf_to_images(p1)
            imgs2, pg2 = convert_pdf_to_images(p2)

            if len(imgs1) > len(imgs2):
                imgs1, imgs2 = imgs2, imgs1
                pg1, pg2 = pg2, pg1

            sims = [compare_page_to_all(i, img, imgs2) for i, img in enumerate(imgs1)]
            avg_sim = sum(s for s, _ in sims) / len(sims) * 100

            elapsed = time.time() - t0

            gt_key = (folder_name, pdf1, pdf2)
            expected_dup = GROUND_TRUTH.get(gt_key, None)
            predicted_dup = avg_sim >= 98.0

            all_metrics.append({
                "client": folder_name,
                "pdf1": pdf1,
                "pdf2": pdf2,
                "similarity": round(avg_sim, 2),
                "pages": pg1,
                "time_sec": round(elapsed, 3),
                "expected_dup": expected_dup,
                "predicted_dup": predicted_dup,
                "correct": expected_dup == predicted_dup if expected_dup is not None else None,
                "pdf1_kb": STORAGE_PER_DOC_KB.get(pdf1, 200),
                "pdf2_kb": STORAGE_PER_DOC_KB.get(pdf2, 200),
            })

    # Also run the full pipeline for reports
    process_folders(test_dir, results_dir, 98.0)

    print(f"\n  Collected metrics for {len(all_metrics)} pairs.")
    return all_metrics


# ---------------------------------------------------------------------------
# Chart generators
# ---------------------------------------------------------------------------

def chart_similarity_distribution(metrics, out_dir):
    """Horizontal bar chart of all pair similarities, gradient-filled."""
    fig, ax = plt.subplots(figsize=(14, max(8, len(metrics) * 0.35)))

    sorted_m = sorted(metrics, key=lambda x: x["similarity"])
    labels = [f"{m['pdf1']} vs {m['pdf2']}\n({m['client']})" for m in sorted_m]
    values = [m["similarity"] for m in sorted_m]
    colors = [ACCENT_GREEN if v >= 98 else ACCENT_ORANGE if v >= 70 else ACCENT_RED for v in values]

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, height=0.7, color=colors, edgecolor="none")

    # Gradient fill per bar
    for bar, val in zip(bars, values):
        cmap = GRAD_GREEN if val >= 98 else GRAD_BLUE if val >= 70 else GRAD_RED
        x0, y0 = bar.get_x(), bar.get_y()
        w, h = bar.get_width(), bar.get_height()
        if w > 0:
            gradient = np.linspace(0.2, 1.0, 256).reshape(1, 256)
            ax.imshow(gradient, extent=[x0, x0 + w, y0, y0 + h],
                      aspect="auto", cmap=cmap, zorder=2)

    for bar in bars:
        bar.set_facecolor("none")
        bar.set_edgecolor("none")

    # Value labels
    for i, v in enumerate(values):
        ax.text(v + 1, i, f"{v:.1f}%", va="center", fontsize=9,
                color=TEXT_COLOR, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("SSIM Similarity (%)", fontsize=12, fontweight="bold")
    ax.set_title("Similarity Score Distribution — All Pair Comparisons",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0, 110)
    ax.axvline(98, color=ACCENT_TEAL, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(98.5, len(labels) - 0.5, "98% threshold", color=ACCENT_TEAL, fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "chart_similarity_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [OK] {path}")


def chart_accuracy_by_scenario(metrics, out_dir):
    """Grouped accuracy bars per client folder."""
    from collections import defaultdict
    client_stats = defaultdict(lambda: {"tp": 0, "tn": 0, "fp": 0, "fn": 0})

    for m in metrics:
        c = m["client"]
        if m["expected_dup"] is None:
            continue
        if m["expected_dup"] and m["predicted_dup"]:
            client_stats[c]["tp"] += 1
        elif not m["expected_dup"] and not m["predicted_dup"]:
            client_stats[c]["tn"] += 1
        elif not m["expected_dup"] and m["predicted_dup"]:
            client_stats[c]["fp"] += 1
        elif m["expected_dup"] and not m["predicted_dup"]:
            client_stats[c]["fn"] += 1

    clients = sorted(client_stats.keys())
    accuracies = []
    for c in clients:
        s = client_stats[c]
        total = s["tp"] + s["tn"] + s["fp"] + s["fn"]
        acc = (s["tp"] + s["tn"]) / total * 100 if total else 0
        accuracies.append(acc)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(clients))
    bars = ax.bar(x, accuracies, width=0.6, edgecolor="none")

    for bar, acc in zip(bars, accuracies):
        cmap = GRAD_GREEN if acc >= 90 else GRAD_ORANGE if acc >= 60 else GRAD_RED
        x0, y0 = bar.get_x(), bar.get_y()
        w, h = bar.get_width(), bar.get_height()
        if h > 0:
            gradient = np.linspace(0.3, 1.0, 256).reshape(256, 1)
            ax.imshow(gradient, extent=[x0, x0 + w, y0, y0 + h],
                      aspect="auto", cmap=cmap, zorder=2)
        bar.set_facecolor("none")
        bar.set_edgecolor(TEXT_COLOR)
        bar.set_linewidth(0.5)

    for i, v in enumerate(accuracies):
        ax.text(i, v + 1.5, f"{v:.0f}%", ha="center", fontsize=10,
                fontweight="bold", color=TEXT_COLOR)

    ax.set_xticks(x)
    ax.set_xticklabels(clients, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Detection Accuracy (%)", fontsize=12, fontweight="bold")
    ax.set_title("Detection Accuracy by Client Scenario",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 115)
    ax.axhline(100, color=ACCENT_GREEN, linestyle=":", linewidth=1, alpha=0.5)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "chart_accuracy_by_scenario.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [OK] {path}")


def chart_storage_impact(metrics, out_dir):
    """Stacked bar: unique vs wasted storage per client."""
    from collections import defaultdict
    client_storage = defaultdict(lambda: {"unique_kb": 0, "wasted_kb": 0, "total_docs": 0, "dup_docs": 0})

    # Track which PDFs are "wasted" (duplicates)
    seen_content = {}  # client -> set of canonical doc names
    for m in metrics:
        c = m["client"]
        if m["predicted_dup"] and m["similarity"] >= 98:
            # The second doc in the pair is the wasted one
            client_storage[c]["wasted_kb"] += m["pdf2_kb"]
            client_storage[c]["dup_docs"] += 1

    # Calculate unique storage per client
    from collections import defaultdict
    client_all_docs = defaultdict(set)
    for m in metrics:
        client_all_docs[m["client"]].add(m["pdf1"])
        client_all_docs[m["client"]].add(m["pdf2"])

    for c, docs in client_all_docs.items():
        total = sum(STORAGE_PER_DOC_KB.get(d, 200) for d in docs)
        client_storage[c]["unique_kb"] = total - client_storage[c]["wasted_kb"]
        client_storage[c]["total_docs"] = len(docs)

    clients = sorted(client_storage.keys())
    unique = [client_storage[c]["unique_kb"] / 1024 for c in clients]  # MB
    wasted = [client_storage[c]["wasted_kb"] / 1024 for c in clients]  # MB

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(clients))

    bars_unique = ax.bar(x, unique, 0.6, label="Unique Storage", color="#2d6a4f", edgecolor=TEXT_COLOR, linewidth=0.5)
    bars_wasted = ax.bar(x, wasted, 0.6, bottom=unique, label="Wasted (Duplicates)", color="#d00000", edgecolor=TEXT_COLOR, linewidth=0.5)

    # Gradient on wasted bars
    for bar in bars_wasted:
        x0, y0 = bar.get_x(), bar.get_y()
        w, h = bar.get_width(), bar.get_height()
        if h > 0:
            gradient = np.linspace(0.4, 1.0, 256).reshape(256, 1)
            ax.imshow(gradient, extent=[x0, x0 + w, y0, y0 + h],
                      aspect="auto", cmap=GRAD_RED, zorder=2)
            bar.set_facecolor("none")

    for bar in bars_unique:
        x0, y0 = bar.get_x(), bar.get_y()
        w, h = bar.get_width(), bar.get_height()
        if h > 0:
            gradient = np.linspace(0.3, 1.0, 256).reshape(256, 1)
            ax.imshow(gradient, extent=[x0, x0 + w, y0, y0 + h],
                      aspect="auto", cmap=GRAD_GREEN, zorder=2)
            bar.set_facecolor("none")

    # Value labels
    for i in range(len(clients)):
        total_val = unique[i] + wasted[i]
        if wasted[i] > 0.001:
            ax.text(i, total_val + 0.02, f"+{wasted[i]:.2f} MB\nwasted",
                    ha="center", fontsize=8, color=ACCENT_RED, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(clients, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Storage (MB)", fontsize=12, fontweight="bold")
    ax.set_title("Storage Impact Analysis — Unique vs Wasted (Duplicate) Storage",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right", facecolor=CARD_COLOR, edgecolor=GRID_COLOR)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "chart_storage_impact.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [OK] {path}")


def chart_kpi_inflation(metrics, out_dir):
    """Before/after comparison: docs processed with vs without dedup."""
    from collections import defaultdict
    client_docs = defaultdict(lambda: {"total": 0, "unique": 0})

    client_all = defaultdict(set)
    client_dups = defaultdict(set)
    for m in metrics:
        c = m["client"]
        client_all[c].add(m["pdf1"])
        client_all[c].add(m["pdf2"])
        if m["predicted_dup"] and m["similarity"] >= 98:
            client_dups[c].add(m["pdf2"])

    for c in client_all:
        client_docs[c]["total"] = len(client_all[c])
        client_docs[c]["unique"] = len(client_all[c]) - len(client_dups[c])

    clients = sorted(client_docs.keys())
    total = [client_docs[c]["total"] for c in clients]
    unique = [client_docs[c]["unique"] for c in clients]

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(clients))
    width = 0.35

    bars_before = ax.bar(x - width/2, total, width, label="Before Dedup (Inflated)", edgecolor=TEXT_COLOR, linewidth=0.5)
    bars_after = ax.bar(x + width/2, unique, width, label="After Dedup (Actual)", edgecolor=TEXT_COLOR, linewidth=0.5)

    # Gradients
    for bar in bars_before:
        x0, y0 = bar.get_x(), bar.get_y()
        w, h = bar.get_width(), bar.get_height()
        if h > 0:
            gradient = np.linspace(0.3, 1.0, 256).reshape(256, 1)
            ax.imshow(gradient, extent=[x0, x0 + w, y0, y0 + h],
                      aspect="auto", cmap=GRAD_ORANGE, zorder=2)
            bar.set_facecolor("none")

    for bar in bars_after:
        x0, y0 = bar.get_x(), bar.get_y()
        w, h = bar.get_width(), bar.get_height()
        if h > 0:
            gradient = np.linspace(0.3, 1.0, 256).reshape(256, 1)
            ax.imshow(gradient, extent=[x0, x0 + w, y0, y0 + h],
                      aspect="auto", cmap=GRAD_GREEN, zorder=2)
            bar.set_facecolor("none")

    # Inflation labels
    for i in range(len(clients)):
        if total[i] > unique[i]:
            inflation = ((total[i] - unique[i]) / unique[i]) * 100 if unique[i] > 0 else 0
            ax.text(i, max(total[i], unique[i]) + 0.3,
                    f"v{total[i]-unique[i]} docs\n({inflation:.0f}% inflated)",
                    ha="center", fontsize=8, color=ACCENT_ORANGE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(clients, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Documents Processed", fontsize=12, fontweight="bold")
    ax.set_title("KPI Inflation Risk — Document Count Before vs After Deduplication",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(loc="upper right", facecolor=CARD_COLOR, edgecolor=GRID_COLOR)
    ax.grid(axis="y", alpha=0.3)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    path = os.path.join(out_dir, "chart_kpi_inflation.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [OK] {path}")


def chart_processing_time(metrics, out_dir):
    """Scatter plot: processing time vs page count, coloured by similarity."""
    fig, ax = plt.subplots(figsize=(12, 6))

    pages = [m["pages"] for m in metrics]
    times = [m["time_sec"] for m in metrics]
    sims = [m["similarity"] for m in metrics]

    scatter = ax.scatter(pages, times, c=sims, cmap=GRAD_BLUE, s=80,
                          edgecolors=TEXT_COLOR, linewidth=0.5, vmin=0, vmax=100, zorder=3)

    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Similarity (%)", fontsize=11, fontweight="bold")
    cbar.ax.yaxis.set_tick_params(color=TEXT_COLOR)
    cbar.outline.set_edgecolor(GRID_COLOR)

    ax.set_xlabel("Pages per Document", fontsize=12, fontweight="bold")
    ax.set_ylabel("Processing Time (seconds)", fontsize=12, fontweight="bold")
    ax.set_title("Processing Time vs Page Count (coloured by similarity)",
                 fontsize=14, fontweight="bold", pad=15)
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.tight_layout()
    path = os.path.join(out_dir, "chart_processing_time.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [OK] {path}")


def chart_transform_robustness(metrics, out_dir):
    """Grouped bar chart showing detection scores for rotation, scale, shift tests."""
    transform_pairs = {
        "Original vs\nRotated 90°": ("CLIENT_2002", "BC_ORIGINAL.pdf", "BC_ROTATED_90.pdf"),
        "Original vs\nRotated 180°": ("CLIENT_2002", "BC_ORIGINAL.pdf", "BC_ROTATED_180.pdf"),
        "Original vs\nScan Noise": ("CLIENT_2003", "TAX_ORIGINAL.pdf", "TAX_SCANNED.pdf"),
        "Original vs\nShifted Margins": ("CLIENT_2003", "TAX_ORIGINAL.pdf", "TAX_SHIFTED.pdf"),
        "Full Size vs\nScaled 80%": ("CLIENT_2004", "NOTICE_FULL.pdf", "NOTICE_SCALED_80.pdf"),
        "Full Size vs\nScaled 60%": ("CLIENT_2004", "NOTICE_FULL.pdf", "NOTICE_SCALED_60.pdf"),
    }

    labels = list(transform_pairs.keys())
    values = []
    for label, (client, p1, p2) in transform_pairs.items():
        found = [m for m in metrics
                 if m["client"] == client and
                 ((m["pdf1"] == p1 and m["pdf2"] == p2) or (m["pdf1"] == p2 and m["pdf2"] == p1))]
        values.append(found[0]["similarity"] if found else 0)

    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, values, 0.6, edgecolor=TEXT_COLOR, linewidth=0.5)

    for bar, val in zip(bars, values):
        cmap = GRAD_GREEN if val >= 98 else GRAD_BLUE if val >= 70 else GRAD_RED
        x0, y0 = bar.get_x(), bar.get_y()
        w, h = bar.get_width(), bar.get_height()
        if h > 0:
            gradient = np.linspace(0.3, 1.0, 256).reshape(256, 1)
            ax.imshow(gradient, extent=[x0, x0 + w, y0, y0 + h],
                      aspect="auto", cmap=cmap, zorder=2)
            bar.set_facecolor("none")

    for i, v in enumerate(values):
        status = "DETECTED" if v >= 98 else "PARTIAL" if v >= 70 else "MISSED"
        color = ACCENT_GREEN if v >= 98 else ACCENT_ORANGE if v >= 70 else ACCENT_RED
        ax.text(i, v + 2, f"{v:.1f}%\n{status}", ha="center", fontsize=9,
                fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("SSIM Similarity (%)", fontsize=12, fontweight="bold")
    ax.set_title("Transform Robustness — Rotation, Scale, Shift, and Noise Detection",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_ylim(0, 115)
    ax.axhline(98, color=ACCENT_TEAL, linestyle="--", linewidth=1.5, alpha=0.7)
    ax.text(len(labels) - 0.5, 99, "98% threshold", color=ACCENT_TEAL, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(out_dir, "chart_transform_robustness.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=BG_COLOR)
    plt.close(fig)
    print(f"  [OK] {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    bench_dir = os.path.join(script_dir, "benchmarks")
    os.makedirs(bench_dir, exist_ok=True)

    metrics = run_pipeline(script_dir)

    print("\n[3/3] Generating benchmark charts...")
    chart_similarity_distribution(metrics, bench_dir)
    chart_accuracy_by_scenario(metrics, bench_dir)
    chart_storage_impact(metrics, bench_dir)
    chart_kpi_inflation(metrics, bench_dir)
    chart_processing_time(metrics, bench_dir)
    chart_transform_robustness(metrics, bench_dir)

    # Save raw metrics as JSON (convert numpy types for serialization)
    json_path = os.path.join(bench_dir, "benchmark_metrics.json")

    def _serializable(obj):
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj

    clean_metrics = [
        {k: _serializable(v) for k, v in m.items()} for m in metrics
    ]
    with open(json_path, "w") as f:
        json.dump(clean_metrics, f, indent=2)
    print(f"  [OK] {json_path}")

    # Print summary table
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    total_pairs = len(metrics)
    correct = sum(1 for m in metrics if m["correct"] is not None and bool(m["correct"]))
    incorrect = sum(1 for m in metrics if m["correct"] is not None and not bool(m["correct"]))
    unknown = sum(1 for m in metrics if m["correct"] is None)
    accuracy = correct / (correct + incorrect) * 100 if (correct + incorrect) > 0 else 0

    flagged = sum(1 for m in metrics if m["predicted_dup"])
    avg_time = np.mean([m["time_sec"] for m in metrics])
    total_wasted = sum(m["pdf2_kb"] for m in metrics if m["predicted_dup"]) / 1024

    print(f"  Total pairs analysed  : {total_pairs}")
    print(f"  Flagged as duplicate  : {flagged}")
    print(f"  Overall accuracy      : {accuracy:.1f}%  ({correct}/{correct+incorrect})")
    print(f"  Avg processing time   : {avg_time:.2f}s per pair")
    print(f"  Total wasted storage  : {total_wasted:.2f} MB")
    print(f"  Charts saved to       : {bench_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
