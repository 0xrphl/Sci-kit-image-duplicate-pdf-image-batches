"""
detect_duplicates.py
====================
Scans a directory of PDF files organized by client folders and detects
duplicate or near-duplicate pages using pixel-map comparison (SSIM).

Algorithm overview:
  1. For each client folder, list all PDF files.
  2. Convert every PDF page to a quantized grayscale image (black / gray / white).
  3. Compare every pair of PDFs page-by-page using Structural Similarity (SSIM).
  4. Flag pairs that exceed configurable similarity thresholds.
  5. Save comparison images and generate CSV + text reports.

Usage:
    python detect_duplicates.py                    # defaults: input=test/ output=results/
    python detect_duplicates.py --input my_pdfs    # custom input folder
    python detect_duplicates.py --threshold 0.95   # custom similarity threshold
"""

import os
import sys
import shutil
import argparse
import csv
import time
import multiprocessing as mp
import concurrent.futures

import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Image processing helpers
# ---------------------------------------------------------------------------

def quantize_image(image):
    """
    Quantize an image into three intensity levels: black (0), gray (128),
    white (255).  This reduces noise sensitivity and focuses comparison on
    the structural layout of the page.

    Parameters
    ----------
    image : numpy.ndarray or PIL.Image.Image

    Returns
    -------
    numpy.ndarray  –  quantized grayscale image
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    if len(image.shape) == 3:
        image = np.mean(image, axis=2).astype(np.uint8)

    lower_threshold = 85
    upper_threshold = 170
    quantized = np.zeros_like(image)
    quantized[image < lower_threshold] = 0
    quantized[(image >= lower_threshold) & (image < upper_threshold)] = 128
    quantized[image >= upper_threshold] = 255
    return quantized


def process_page(page, size):
    """Render a single fitz.Page to a quantized grayscale numpy array."""
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    img = img.resize(size).convert("L")
    return quantize_image(img)


def convert_pdf_to_images(pdf_path, size=(1000, 1440)):
    """
    Convert all pages of a PDF to quantized grayscale images.

    Returns
    -------
    tuple(list[numpy.ndarray], int)  –  (list of page images, page count)
    """
    doc = fitz.open(pdf_path)
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_page, page, size) for page in doc]
        images = [f.result() for f in concurrent.futures.as_completed(futures)]
    page_count = len(doc)
    doc.close()
    return images, page_count


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

def compare_images(img1, img2):
    """Return the SSIM score (0-1) between two grayscale numpy arrays."""
    return ssim(img1, img2)


def compare_page_to_all(i, img1, images2):
    """Find the best-matching page in images2 for img1."""
    similarities = [compare_images(img1, img2) for img2 in images2]
    best = max(similarities)
    best_idx = similarities.index(best)
    return best, best_idx


def save_comparison_images(i, img1, img2, output_folder):
    """Save side-by-side comparison PNGs for a page pair."""
    Image.fromarray(img1).save(os.path.join(output_folder, f"page_{i+1}_pdf1.png"))
    Image.fromarray(img2).save(os.path.join(output_folder, f"page_{i+1}_pdf2.png"))


def rename_output_folder(output_folder, formatted_similarity):
    """Prefix the comparison folder name with the similarity score."""
    time.sleep(0.10)
    base = os.path.basename(output_folder)
    parent = os.path.dirname(output_folder)
    new_name = os.path.join(parent, f"{formatted_similarity}_{base}")
    counter = 1
    candidate = new_name
    while os.path.exists(candidate):
        candidate = f"{new_name}_{counter}"
        counter += 1
    os.rename(output_folder, candidate)
    return candidate


# ---------------------------------------------------------------------------
# PDF pair processing
# ---------------------------------------------------------------------------

def compare_pdfs(pdf1_path, pdf2_path, output_folder):
    """
    Compare two PDFs page-by-page.  Returns similarity metrics and saves
    comparison images.

    Returns
    -------
    tuple  –  (similarities, pages1, pages2, identical_count, output_folder)
    """
    images1, pages1 = convert_pdf_to_images(pdf1_path)
    images2, pages2 = convert_pdf_to_images(pdf2_path)

    # Always iterate over the shorter document
    if len(images1) > len(images2):
        images1, images2 = images2, images1
        pages1, pages2 = pages2, pages1

    similarities = [
        compare_page_to_all(i, img1, images2)
        for i, img1 in enumerate(images1)
    ]

    identical_count = sum(1 for sim, _ in similarities if sim > 0.99)

    # Save comparison images in parallel
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(
                save_comparison_images, i, images1[i],
                images2[best_idx], output_folder
            )
            for i, (_, best_idx) in enumerate(similarities)
        ]
        concurrent.futures.wait(futures)

    avg_sim = sum(s for s, _ in similarities) / len(similarities) * 100
    formatted = f"{int(avg_sim):03d}"
    new_folder = rename_output_folder(output_folder, formatted)

    return [s for s, _ in similarities], pages1, pages2, identical_count, new_folder


def save_exact_match_to_csv(pdf1_name, pdf2_name, folder_path):
    """Append an exact-match pair to exact_matches.csv inside the folder."""
    csv_path = os.path.join(os.path.dirname(folder_path), "exact_matches.csv")
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["PDF1 Name", "PDF2 Name"])
        writer.writerow([pdf1_name, pdf2_name])


def process_pdf_pair(pair, folder_path, duplicate_threshold, results_dir):
    """
    Process a single pair of PDFs: compare them and write results.

    Parameters
    ----------
    pair             : tuple(str, str)  –  filenames of the two PDFs
    folder_path      : str              –  directory containing the PDFs
    duplicate_threshold : float         –  similarity % above which a pair is flagged
    results_dir      : str              –  root results directory

    Returns
    -------
    dict or None  –  comparison metrics (None if skipped)
    """
    pdf1, pdf2 = pair
    pdf1_path = os.path.join(folder_path, pdf1)
    pdf2_path = os.path.join(folder_path, pdf2)
    pdf1_name = os.path.splitext(pdf1)[0]
    pdf2_name = os.path.splitext(pdf2)[0]

    # Skip very large PDFs to keep demo fast
    p1_count = fitz.open(pdf1_path).page_count
    p2_count = fitz.open(pdf2_path).page_count
    if p1_count > 100 or p2_count > 100:
        return None

    output_folder = os.path.join(folder_path, f"compared_{pdf1_name}_{pdf2_name}")
    os.makedirs(output_folder, exist_ok=True)

    similarities, pages1, pages2, identical_count, new_folder = compare_pdfs(
        pdf1_path, pdf2_path, output_folder
    )

    avg_sim = sum(similarities) / len(similarities) * 100
    similar_98 = sum(1 for s in similarities if s > 0.98)

    # Record exact matches
    if any(s == 1.0 for s in similarities):
        save_exact_match_to_csv(pdf1_name, pdf2_name, folder_path)

    # Copy high-similarity comparison folders into results/duplicates/
    if avg_sim >= duplicate_threshold:
        _move_duplicate_folders(
            folder_path, pdf1_name, pdf2_name,
            avg_sim, pages1, results_dir,
        )

    return {
        "pair": (pdf1, pdf2),
        "folder": folder_path,
        "average_similarity": avg_sim,
        "identical_pages": identical_count,
        "similar_98": similar_98,
        "extra_pages": abs(pages2 - pages1),
        "output_folder": new_folder,
    }


def _move_duplicate_folders(folder_path, pdf1_name, pdf2_name,
                            avg_sim, pages, results_dir):
    """Copy high-similarity comparison folders to results/duplicates/."""
    dup_dir = os.path.join(results_dir, "duplicates")
    os.makedirs(dup_dir, exist_ok=True)

    label = f"{os.path.basename(folder_path)}_{pdf1_name}_{pdf2_name}_{int(avg_sim)}"
    dest = os.path.join(dup_dir, label)
    os.makedirs(dest, exist_ok=True)

    for item in os.listdir(folder_path):
        if item.startswith(("100_compared", "099_compared", "098_compared")):
            src = os.path.join(folder_path, item)
            shutil.copytree(src, os.path.join(dest, item), dirs_exist_ok=True)

    # Append to duplicated_doc_id_pairs.csv
    csv_path = os.path.join(results_dir, "duplicated_doc_id_pairs.csv")
    exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow([
                "PDF1 Name", "PDF2 Name", "Average Similarity",
                "Pages", "Similarity per Page",
            ])
        writer.writerow([
            pdf1_name, pdf2_name, f"{avg_sim:.2f}",
            pages, f"{avg_sim / pages:.2f}" if pages else 0,
        ])


# ---------------------------------------------------------------------------
# Report generators
# ---------------------------------------------------------------------------

def generate_folder_report(folder_path, results):
    """Write a per-folder summary report."""
    report = os.path.join(folder_path, "folder_summary_report.txt")
    with open(report, "w") as f:
        f.write("Folder Summary Report\n")
        f.write("=" * 50 + "\n\n")
        for key in ("average_similarity", "identical_pages", "similar_98"):
            f.write(f"\nRanking by {key}:\n")
            for i, r in enumerate(
                sorted(results, key=lambda x: x[key], reverse=True), 1
            ):
                f.write(
                    f"  {i}. {r['pair'][0]} vs {r['pair'][1]}: "
                    f"{r[key]:.2f}  (extra pages: {r['extra_pages']})\n"
                )
                f.write(f"     Folder: {r['output_folder']}\n")


def generate_comprehensive_report(root_dir, all_results):
    """Write an overall summary report covering every client folder."""
    report = os.path.join(root_dir, "comprehensive_summary_report.txt")
    with open(report, "w") as f:
        f.write("Comprehensive Summary Report\n")
        f.write("=" * 50 + "\n\n")
        for key in ("average_similarity", "identical_pages", "similar_98"):
            f.write(f"\nRanking all files by {key}:\n")
            for i, r in enumerate(
                sorted(all_results, key=lambda x: x[key], reverse=True), 1
            ):
                folder_name = os.path.basename(r["folder"])
                f.write(
                    f"  {i}. {r['pair'][0]} vs {r['pair'][1]} "
                    f"(Client: {folder_name}): "
                    f"{r[key]:.2f}  (extra pages: {r['extra_pages']})\n"
                )
                f.write(f"     Folder: {r['output_folder']}\n")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def process_folders(root_dir, results_dir, duplicate_threshold=98.0):
    """
    Walk every client folder inside *root_dir*, compare all PDF pairs,
    and write reports to *results_dir*.
    """
    all_results = []
    os.makedirs(results_dir, exist_ok=True)

    folders = [
        os.path.join(root_dir, d)
        for d in sorted(os.listdir(root_dir))
        if os.path.isdir(os.path.join(root_dir, d))
    ]

    for folder in tqdm(folders, desc="Processing client folders"):
        pdfs = sorted(f for f in os.listdir(folder) if f.lower().endswith(".pdf"))
        pairs = [
            (pdfs[i], pdfs[j])
            for i in range(len(pdfs))
            for j in range(i + 1, len(pdfs))
        ]

        folder_results = []
        for pair in tqdm(pairs, desc=f"  {os.path.basename(folder)}", leave=False):
            result = process_pdf_pair(pair, folder, duplicate_threshold, results_dir)
            if result:
                folder_results.append(result)

        if folder_results:
            generate_folder_report(folder, folder_results)
        all_results.extend(folder_results)

    generate_comprehensive_report(root_dir, all_results)
    return all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Detect duplicate PDF pages using pixel-map SSIM comparison.",
    )
    parser.add_argument(
        "--input", "-i",
        default="test",
        help="Directory containing client folders with PDFs (default: test/)",
    )
    parser.add_argument(
        "--output", "-o",
        default="results",
        help="Directory for reports and duplicate copies (default: results/)",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=98.0,
        help="Average similarity %% above which a pair is flagged as duplicate (default: 98)",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Delete input and output folders before running (fresh start).",
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, args.input) if not os.path.isabs(args.input) else args.input
    output_dir = os.path.join(script_dir, args.output) if not os.path.isabs(args.output) else args.output

    if args.clean:
        for d in (input_dir, output_dir):
            if os.path.exists(d):
                shutil.rmtree(d)
                print(f"Deleted: {d}")

    if not os.path.isdir(input_dir):
        print(f"Error: Input directory not found: {input_dir}")
        print("Run  python generate_synthetic_data.py  first to create test data.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Input folder  : {input_dir}")
    print(f"Output folder : {output_dir}")
    print(f"Threshold     : {args.threshold}%\n")

    mp.freeze_support()  # Required on Windows
    results = process_folders(input_dir, output_dir, args.threshold)

    # Print summary
    flagged = sum(1 for r in results if r["average_similarity"] >= args.threshold)
    print(f"\n{'=' * 50}")
    print(f"Scan complete!")
    print(f"  Total pairs compared : {len(results)}")
    print(f"  Flagged as duplicate : {flagged}")
    print(f"  Reports written to   : {output_dir}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
