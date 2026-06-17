"""
generate_synthetic_data.py
==========================
Generates synthetic PDF test data with pixel-map images for duplicate detection testing.

Each PDF page contains a programmatically generated image built from:
  - Color block grids (pixel-art style maps)
  - Geometric patterns (lines, rectangles, circles)
  - Gradient regions
  - Random noise patches
  - Gov-style document layouts (driver license, birth certificate, tax form, etc.)

NO personal data, NO real documents — purely synthetic pixel content.

Test scenarios (11 client folders):
  CLIENT_1001 — Exact duplicates (identical PDFs)
  CLIENT_1002 — Near duplicates (same content + minor noise)
  CLIENT_1003 — Partial overlap (multi-page, some pages shared)
  CLIENT_1004 — Completely different documents
  CLIENT_1005 — Mixed: exact dup + near dup + unique in one client pool
  CLIENT_2001 — Driver License: exact duplicate + different license
  CLIENT_2002 — Birth Certificate: original + rotated 90° + rotated 180°
  CLIENT_2003 — Tax Form: original + scan noise + shifted margins
  CLIENT_2004 — Immigration Notice: same doc at different page sizes
  CLIENT_2005 — Mixed gov docs (all different — negative test)
  CLIENT_2006 — Batch inflation: same DL submitted 5 times
"""

import os
import sys
import io
import numpy as np
from PIL import Image, ImageDraw
import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Pixel-map image generators  (basic patterns)
# ---------------------------------------------------------------------------

def _seed(name: str) -> int:
    """Derive a deterministic seed from a string so results are reproducible."""
    return int.from_bytes(name.encode(), "little") % (2**31)


def generate_color_block_grid(width=800, height=1100, block_size=50, seed=0):
    """Grid of solid-colour blocks (pixel-art map)."""
    rng = np.random.RandomState(seed)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    rows = height // block_size
    cols = width // block_size
    palette = rng.randint(0, 256, size=(rows * cols, 3), dtype=np.uint8)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * block_size, (r + 1) * block_size
            x0, x1 = c * block_size, (c + 1) * block_size
            img[y0:y1, x0:x1] = palette[idx]
            idx += 1
    return Image.fromarray(img)


def generate_geometric_pattern(width=800, height=1100, seed=0):
    """Random rectangles, ellipses, and lines on white canvas."""
    rng = np.random.RandomState(seed)
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    for _ in range(30):
        shape = rng.choice(["rect", "ellipse", "line"])
        color = tuple(rng.randint(0, 256, 3).tolist())
        x0, x1 = sorted(rng.randint(0, width, 2).tolist())
        y0, y1 = sorted(rng.randint(0, height, 2).tolist())
        if shape == "rect":
            draw.rectangle([x0, y0, x1, y1], fill=color)
        elif shape == "ellipse":
            draw.ellipse([x0, y0, x1, y1], fill=color)
        else:
            lw = int(rng.randint(1, 6))
            draw.line([x0, y0, x1, y1], fill=color, width=lw)
    return img


def generate_gradient(width=800, height=1100, seed=0):
    """Smooth vertical gradient between two random colours."""
    rng = np.random.RandomState(seed)
    c1 = rng.randint(0, 256, 3).astype(np.float64)
    c2 = rng.randint(0, 256, 3).astype(np.float64)
    arr = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        t = y / max(height - 1, 1)
        arr[y, :] = ((1 - t) * c1 + t * c2).astype(np.uint8)
    return Image.fromarray(arr)


def generate_noise_patch(width=800, height=1100, seed=0):
    """Full-frame random noise."""
    rng = np.random.RandomState(seed)
    arr = rng.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def add_noise_to_image(pil_img, intensity=10, seed=None):
    """Add Gaussian noise (simulates scan/compression artefacts)."""
    rng = np.random.RandomState(seed)
    arr = np.array(pil_img, dtype=np.int16)
    noise = rng.normal(0, intensity, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


# ---------------------------------------------------------------------------
# Gov-style document template generators (pixel-map layouts)
# ---------------------------------------------------------------------------

def generate_driver_license(width=800, height=500, seed=0):
    """
    Synthetic driver license card layout:
      - Header bar with state colour
      - Photo placeholder (coloured block)
      - Horizontal data-line bars
      - Barcode strip at bottom
    All purely geometric — no text, no personal data.
    """
    rng = np.random.RandomState(seed)
    img = Image.new("RGB", (width, height), (245, 245, 240))
    draw = ImageDraw.Draw(img)

    # Header bar
    hdr_color = tuple(rng.randint(30, 150, 3).tolist())
    draw.rectangle([0, 0, width, 60], fill=hdr_color)

    # State seal circle
    draw.ellipse([30, 8, 52, 52], fill=(255, 215, 0), outline=(180, 150, 0))

    # Photo placeholder
    photo_color = tuple(rng.randint(100, 200, 3).tolist())
    draw.rectangle([30, 80, 230, 320], fill=photo_color, outline=(80, 80, 80))

    # Data lines (simulate name, DOB, address fields)
    for i, y_off in enumerate([90, 130, 170, 210, 250, 290]):
        line_w = rng.randint(150, 400)
        gray = rng.randint(40, 120)
        draw.rectangle([260, y_off, 260 + line_w, y_off + 18],
                        fill=(gray, gray, gray))

    # Signature line
    draw.line([260, 340, 550, 340], fill=(100, 100, 100), width=2)

    # Barcode strip
    x = 30
    while x < width - 30:
        bar_w = rng.randint(2, 6)
        if rng.random() > 0.4:
            draw.rectangle([x, height - 80, x + bar_w, height - 20],
                            fill=(20, 20, 20))
        x += bar_w + rng.randint(1, 4)

    # Coloured stripe
    stripe_color = tuple(rng.randint(50, 200, 3).tolist())
    draw.rectangle([0, height - 10, width, height], fill=stripe_color)

    return img


def generate_birth_certificate(width=800, height=1100, seed=0):
    """
    Synthetic birth certificate layout:
      - Ornate double border
      - Header block with decorative patterns
      - Grid of field rows
      - Seal/stamp circle
      - Signature line at bottom
    """
    rng = np.random.RandomState(seed)
    bg_color = (255, 253, 245)
    img = Image.new("RGB", (width, height), bg_color)
    draw = ImageDraw.Draw(img)

    # Ornate double border
    border_color = tuple(rng.randint(80, 160, 3).tolist())
    draw.rectangle([15, 15, width - 15, height - 15], outline=border_color, width=3)
    draw.rectangle([25, 25, width - 25, height - 25], outline=border_color, width=1)

    # Decorative corner flourishes
    for (cx, cy) in [(35, 35), (width - 35, 35), (35, height - 35), (width - 35, height - 35)]:
        draw.ellipse([cx - 10, cy - 10, cx + 10, cy + 10],
                      fill=border_color, outline=border_color)

    # Header band
    hdr_color = tuple(rng.randint(40, 120, 3).tolist())
    draw.rectangle([40, 50, width - 40, 120], fill=hdr_color)

    # Decorative header accent
    accent = tuple(rng.randint(150, 230, 3).tolist())
    draw.rectangle([40, 120, width - 40, 130], fill=accent)

    # Field rows (name, DOB, place, parents, etc.)
    field_y = 160
    for i in range(12):
        # Label block
        label_w = rng.randint(100, 200)
        draw.rectangle([60, field_y, 60 + label_w, field_y + 20],
                        fill=(180, 180, 180))
        # Value bar
        val_w = rng.randint(200, 500)
        gray = rng.randint(50, 100)
        draw.rectangle([60 + label_w + 20, field_y, 60 + label_w + 20 + val_w, field_y + 20],
                        fill=(gray, gray, gray))
        # Underline
        draw.line([60, field_y + 28, width - 60, field_y + 28],
                   fill=(200, 200, 200), width=1)
        field_y += 55

    # Official seal circle
    seal_color = tuple(rng.randint(140, 220, 3).tolist())
    seal_x, seal_y = width - 160, height - 220
    draw.ellipse([seal_x - 50, seal_y - 50, seal_x + 50, seal_y + 50],
                  outline=seal_color, width=3)
    draw.ellipse([seal_x - 35, seal_y - 35, seal_x + 35, seal_y + 35],
                  outline=seal_color, width=1)
    # Star in centre
    for angle_offset in range(0, 360, 72):
        rad = np.radians(angle_offset)
        dx, dy = int(20 * np.cos(rad)), int(20 * np.sin(rad))
        draw.line([seal_x, seal_y, seal_x + dx, seal_y + dy],
                   fill=seal_color, width=2)

    # Signature line
    draw.line([60, height - 100, 350, height - 100], fill=(80, 80, 80), width=2)
    draw.rectangle([60, height - 90, 200, height - 75], fill=(160, 160, 160))

    return img


def generate_tax_form(width=800, height=1100, seed=0):
    """
    Synthetic tax form layout:
      - Dense grid with numbered section boxes
      - Shaded header bands per section
      - Multiple columns of data fields
      - Footer with checkboxes
    """
    rng = np.random.RandomState(seed)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Top header band
    hdr_color = tuple(rng.randint(20, 80, 3).tolist())
    draw.rectangle([0, 0, width, 50], fill=hdr_color)

    # Form number box
    draw.rectangle([width - 200, 5, width - 10, 45], fill=(255, 255, 255),
                    outline=(200, 200, 200))
    draw.rectangle([width - 195, 10, width - 120, 40],
                    fill=(100, 100, 100))

    # Section blocks
    section_y = 70
    for section in range(5):
        # Section header
        sec_color = tuple((np.array(hdr_color) + rng.randint(-20, 40, 3)).clip(0, 255).tolist())
        draw.rectangle([10, section_y, width - 10, section_y + 30],
                        fill=sec_color)

        # Section number circle
        draw.ellipse([15, section_y + 3, 39, section_y + 27],
                      fill=(255, 255, 255))

        row_y = section_y + 40
        num_rows = rng.randint(3, 7)
        for row in range(num_rows):
            # Line number
            draw.rectangle([15, row_y, 35, row_y + 16], fill=(220, 220, 220))

            # Field boxes (2-3 columns)
            num_cols = rng.randint(2, 4)
            col_w = (width - 80) // num_cols
            for col in range(num_cols):
                x0 = 50 + col * col_w
                fw = rng.randint(col_w // 3, col_w - 10)
                gray = rng.randint(60, 140)
                draw.rectangle([x0, row_y, x0 + fw, row_y + 16],
                                fill=(gray, gray, gray))
                draw.rectangle([x0, row_y, x0 + fw, row_y + 16],
                                outline=(180, 180, 180))
            row_y += 28

        section_y = row_y + 15

    # Footer checkboxes
    for i in range(4):
        bx = 30 + i * 180
        draw.rectangle([bx, height - 60, bx + 16, height - 44],
                        outline=(80, 80, 80), width=2)
        draw.rectangle([bx + 24, height - 58, bx + 24 + rng.randint(60, 140), height - 44],
                        fill=(120, 120, 120))

    # Bottom line
    draw.line([10, height - 20, width - 10, height - 20], fill=(0, 0, 0), width=1)

    return img


def generate_immigration_notice(width=800, height=1100, seed=0):
    """
    Synthetic immigration notice / government letter layout:
      - Letterhead bar with agency logo placeholder
      - Body paragraph lines
      - Reference number block
      - Footer with stamps/logo blocks
    """
    rng = np.random.RandomState(seed)
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Letterhead bar
    bar_color = tuple(rng.randint(20, 100, 3).tolist())
    draw.rectangle([0, 0, width, 80], fill=bar_color)

    # Logo placeholder (eagle-like shield shape)
    draw.rectangle([30, 10, 90, 70], fill=(255, 255, 255), outline=(200, 200, 200))
    draw.ellipse([40, 15, 80, 55], fill=tuple(rng.randint(100, 200, 3).tolist()))

    # Agency name bar
    draw.rectangle([110, 20, 500, 40], fill=(220, 220, 220))
    draw.rectangle([110, 48, 400, 62], fill=(180, 180, 180))

    # Reference number block
    draw.rectangle([width - 250, 100, width - 30, 150], outline=(150, 150, 150))
    draw.rectangle([width - 245, 108, width - 80, 122], fill=(100, 100, 100))
    draw.rectangle([width - 245, 128, width - 120, 142], fill=(130, 130, 130))

    # Date line
    draw.rectangle([60, 120, 250, 138], fill=(140, 140, 140))

    # Body paragraph lines
    line_y = 190
    for para in range(4):
        num_lines = rng.randint(3, 8)
        for ln in range(num_lines):
            line_w = rng.randint(500, width - 80)
            gray = rng.randint(60, 110)
            draw.rectangle([60, line_y, 60 + line_w, line_y + 12],
                            fill=(gray, gray, gray))
            line_y += 22
        line_y += 20  # paragraph gap

    # Signature block
    draw.line([60, line_y + 30, 300, line_y + 30], fill=(80, 80, 80), width=2)
    draw.rectangle([60, line_y + 40, 220, line_y + 55], fill=(150, 150, 150))
    draw.rectangle([60, line_y + 60, 180, line_y + 72], fill=(170, 170, 170))

    # Footer stamps
    for i in range(3):
        sx = 60 + i * 250
        stamp_color = tuple(rng.randint(100, 200, 3).tolist())
        draw.rectangle([sx, height - 80, sx + 60, height - 30],
                        fill=stamp_color, outline=(80, 80, 80))

    # Bottom rule
    draw.rectangle([0, height - 10, width, height], fill=bar_color)

    return img


# ---------------------------------------------------------------------------
# Transform helpers
# ---------------------------------------------------------------------------

def rotate_image(pil_img, degrees):
    """Rotate image by given degrees, expanding canvas to fit."""
    return pil_img.rotate(degrees, expand=True, fillcolor=(255, 255, 255))


def shift_image(pil_img, dx=20, dy=30):
    """Shift image content by (dx, dy) pixels, adding white margin."""
    w, h = pil_img.size
    new_img = Image.new("RGB", (w, h), (255, 255, 255))
    new_img.paste(pil_img, (dx, dy))
    return new_img


def scale_image(pil_img, scale_factor=0.8):
    """Scale image content while keeping original canvas size."""
    w, h = pil_img.size
    new_w, new_h = int(w * scale_factor), int(h * scale_factor)
    scaled = pil_img.resize((new_w, new_h), Image.LANCZOS)
    new_img = Image.new("RGB", (w, h), (255, 255, 255))
    paste_x = (w - new_w) // 2
    paste_y = (h - new_h) // 2
    new_img.paste(scaled, (paste_x, paste_y))
    return new_img


# ---------------------------------------------------------------------------
# PDF creation helpers
# ---------------------------------------------------------------------------

def images_to_pdf(images: list, pdf_path: str):
    """Convert a list of PIL Images into a multi-page PDF."""
    doc = fitz.open()
    for img in images:
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)
        png_bytes = buf.read()
        img_doc = fitz.open(stream=png_bytes, filetype="png")
        rect = img_doc[0].rect
        page = doc.new_page(width=rect.width, height=rect.height)
        page.insert_image(page.rect, stream=png_bytes)
        img_doc.close()
    doc.save(pdf_path)
    doc.close()


# ---------------------------------------------------------------------------
# Scenario builders — Original (CLIENT_1001–1005)
# ---------------------------------------------------------------------------

def build_client_1001(out_dir):
    """Exact duplicates. DOC_0001==DOC_0002, DOC_0003 different."""
    folder = os.path.join(out_dir, "CLIENT_1001")
    os.makedirs(folder, exist_ok=True)
    shared = [generate_color_block_grid(seed=1001), generate_geometric_pattern(seed=1002)]
    images_to_pdf(shared, os.path.join(folder, "DOC_0001.pdf"))
    images_to_pdf(shared, os.path.join(folder, "DOC_0002.pdf"))
    images_to_pdf([generate_gradient(seed=2001), generate_noise_patch(seed=2002)],
                  os.path.join(folder, "DOC_0003.pdf"))
    print(f"  CLIENT_1001: 3 PDFs (exact dup + different)")


def build_client_1002(out_dir):
    """Near duplicates. DOC_0004 original, DOC_0005 noisy copy, DOC_0006 different."""
    folder = os.path.join(out_dir, "CLIENT_1002")
    os.makedirs(folder, exist_ok=True)
    pages = [generate_color_block_grid(seed=3001), generate_geometric_pattern(seed=3002),
             generate_gradient(seed=3003)]
    images_to_pdf(pages, os.path.join(folder, "DOC_0004.pdf"))
    noisy = [add_noise_to_image(p, intensity=8, seed=42 + i) for i, p in enumerate(pages)]
    images_to_pdf(noisy, os.path.join(folder, "DOC_0005.pdf"))
    images_to_pdf([generate_noise_patch(seed=4001), generate_color_block_grid(seed=4002, block_size=25)],
                  os.path.join(folder, "DOC_0006.pdf"))
    print(f"  CLIENT_1002: 3 PDFs (near dup + different)")


def build_client_1003(out_dir):
    """Partial page overlap. DOC_0007 & DOC_0008 share 2 pages."""
    folder = os.path.join(out_dir, "CLIENT_1003")
    os.makedirs(folder, exist_ok=True)
    s1 = generate_color_block_grid(seed=5001)
    s2 = generate_geometric_pattern(seed=5002)
    images_to_pdf([s1, generate_gradient(seed=5003), s2, generate_noise_patch(seed=5004)],
                  os.path.join(folder, "DOC_0007.pdf"))
    images_to_pdf([generate_color_block_grid(seed=5005, block_size=100), s1,
                   generate_geometric_pattern(seed=5006), s2],
                  os.path.join(folder, "DOC_0008.pdf"))
    images_to_pdf([generate_gradient(seed=5007), generate_noise_patch(seed=5008)],
                  os.path.join(folder, "DOC_0009.pdf"))
    print(f"  CLIENT_1003: 3 PDFs (partial overlap)")


def build_client_1004(out_dir):
    """All different — no duplicates."""
    folder = os.path.join(out_dir, "CLIENT_1004")
    os.makedirs(folder, exist_ok=True)
    images_to_pdf([generate_color_block_grid(seed=6001), generate_gradient(seed=6002)],
                  os.path.join(folder, "DOC_0010.pdf"))
    images_to_pdf([generate_geometric_pattern(seed=6003), generate_noise_patch(seed=6004)],
                  os.path.join(folder, "DOC_0011.pdf"))
    images_to_pdf([generate_color_block_grid(seed=6005, block_size=20)],
                  os.path.join(folder, "DOC_0012.pdf"))
    print(f"  CLIENT_1004: 3 PDFs (all different)")


def build_client_1005(out_dir):
    """Mixed: exact dup + near dup + unique."""
    folder = os.path.join(out_dir, "CLIENT_1005")
    os.makedirs(folder, exist_ok=True)
    pages = [generate_color_block_grid(seed=7001, block_size=40),
             generate_geometric_pattern(seed=7002), generate_gradient(seed=7003)]
    images_to_pdf(pages, os.path.join(folder, "DOC_0013.pdf"))
    images_to_pdf(pages, os.path.join(folder, "DOC_0014.pdf"))
    noisy = [add_noise_to_image(p, intensity=15, seed=80 + i) for i, p in enumerate(pages)]
    images_to_pdf(noisy, os.path.join(folder, "DOC_0015.pdf"))
    images_to_pdf([generate_noise_patch(seed=8001), generate_color_block_grid(seed=8002, block_size=80)],
                  os.path.join(folder, "DOC_0016.pdf"))
    print(f"  CLIENT_1005: 4 PDFs (mixed scenario)")


# ---------------------------------------------------------------------------
# Scenario builders — Gov-style documents (CLIENT_2001–2006)
# ---------------------------------------------------------------------------

def build_client_2001(out_dir):
    """
    Driver License: DL_ORIG and DL_COPY are exact duplicates,
    DL_OTHER is a different person's license.
    """
    folder = os.path.join(out_dir, "CLIENT_2001")
    os.makedirs(folder, exist_ok=True)
    dl1 = generate_driver_license(seed=9001)
    images_to_pdf([dl1], os.path.join(folder, "DL_ORIG.pdf"))
    images_to_pdf([dl1], os.path.join(folder, "DL_COPY.pdf"))
    dl2 = generate_driver_license(seed=9002)
    images_to_pdf([dl2], os.path.join(folder, "DL_OTHER.pdf"))
    print(f"  CLIENT_2001: 3 PDFs (driver license — exact dup + different)")


def build_client_2002(out_dir):
    """
    Birth Certificate: original + rotated 90° + rotated 180°.
    Tests rotation robustness.
    """
    folder = os.path.join(out_dir, "CLIENT_2002")
    os.makedirs(folder, exist_ok=True)
    bc = generate_birth_certificate(seed=9101)
    images_to_pdf([bc], os.path.join(folder, "BC_ORIGINAL.pdf"))
    images_to_pdf([rotate_image(bc, 90)], os.path.join(folder, "BC_ROTATED_90.pdf"))
    images_to_pdf([rotate_image(bc, 180)], os.path.join(folder, "BC_ROTATED_180.pdf"))
    print(f"  CLIENT_2002: 3 PDFs (birth cert — rotation test)")


def build_client_2003(out_dir):
    """
    Tax Form: original + scan noise + shifted margins.
    Tests noise and margin shift robustness.
    """
    folder = os.path.join(out_dir, "CLIENT_2003")
    os.makedirs(folder, exist_ok=True)
    tf = generate_tax_form(seed=9201)
    images_to_pdf([tf], os.path.join(folder, "TAX_ORIGINAL.pdf"))
    noisy_tf = add_noise_to_image(tf, intensity=12, seed=9202)
    images_to_pdf([noisy_tf], os.path.join(folder, "TAX_SCANNED.pdf"))
    shifted_tf = shift_image(tf, dx=25, dy=35)
    images_to_pdf([shifted_tf], os.path.join(folder, "TAX_SHIFTED.pdf"))
    print(f"  CLIENT_2003: 3 PDFs (tax form — noise + shift test)")


def build_client_2004(out_dir):
    """
    Immigration Notice: same doc at original size + scaled down 80% + scaled down 60%.
    Tests scale invariance.
    """
    folder = os.path.join(out_dir, "CLIENT_2004")
    os.makedirs(folder, exist_ok=True)
    notice = generate_immigration_notice(seed=9301)
    images_to_pdf([notice], os.path.join(folder, "NOTICE_FULL.pdf"))
    images_to_pdf([scale_image(notice, 0.80)], os.path.join(folder, "NOTICE_SCALED_80.pdf"))
    images_to_pdf([scale_image(notice, 0.60)], os.path.join(folder, "NOTICE_SCALED_60.pdf"))
    print(f"  CLIENT_2004: 3 PDFs (immigration notice — scale test)")


def build_client_2005(out_dir):
    """
    Mixed gov docs — all different types, no duplicates.
    Negative test: algorithm should report low similarity.
    """
    folder = os.path.join(out_dir, "CLIENT_2005")
    os.makedirs(folder, exist_ok=True)
    images_to_pdf([generate_driver_license(seed=9401)],
                  os.path.join(folder, "GOV_DL.pdf"))
    images_to_pdf([generate_birth_certificate(seed=9402)],
                  os.path.join(folder, "GOV_BC.pdf"))
    images_to_pdf([generate_tax_form(seed=9403)],
                  os.path.join(folder, "GOV_TAX.pdf"))
    images_to_pdf([generate_immigration_notice(seed=9404)],
                  os.path.join(folder, "GOV_NOTICE.pdf"))
    print(f"  CLIENT_2005: 4 PDFs (mixed gov — all different, negative test)")


def build_client_2006(out_dir):
    """
    Batch inflation: same driver license submitted 5 times.
    Simulates an employee inflating processing metrics.
    """
    folder = os.path.join(out_dir, "CLIENT_2006")
    os.makedirs(folder, exist_ok=True)
    dl = generate_driver_license(seed=9501)
    for i in range(1, 6):
        images_to_pdf([dl], os.path.join(folder, f"BATCH_{i:02d}.pdf"))
    print(f"  CLIENT_2006: 5 PDFs (same DL x5 -- batch inflation test)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "test")

    if os.path.exists(out_dir):
        import shutil
        shutil.rmtree(out_dir)

    os.makedirs(out_dir, exist_ok=True)
    print(f"Generating synthetic PDF test data in: {out_dir}\n")

    # Original scenarios
    build_client_1001(out_dir)
    build_client_1002(out_dir)
    build_client_1003(out_dir)
    build_client_1004(out_dir)
    build_client_1005(out_dir)

    # Gov-style document scenarios
    build_client_2001(out_dir)
    build_client_2002(out_dir)
    build_client_2003(out_dir)
    build_client_2004(out_dir)
    build_client_2005(out_dir)
    build_client_2006(out_dir)

    # Summary
    total = sum(
        len([f for f in os.listdir(os.path.join(out_dir, d)) if f.endswith(".pdf")])
        for d in os.listdir(out_dir)
        if os.path.isdir(os.path.join(out_dir, d))
    )
    folders = len([d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))])
    print(f"\n[OK] Done! Generated {total} synthetic PDFs across {folders} client folders.")
    print(f"  Output directory: {out_dir}")
    print("\nNext step: run  python detect_duplicates.py  to scan for duplicates.")
    print("     or:   run  python run_benchmarks.py    for full benchmarks + charts.")


if __name__ == "__main__":
    main()
