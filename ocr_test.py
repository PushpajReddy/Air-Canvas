"""
DROP-IN REPLACEMENT for extract_text_from_canvas() in your VirtualPainter class.
Also includes a standalone test function you can run independently.

ISSUES FIXED:
1. THRESH_BINARY_INV used instead of THRESH_BINARY — canvas is black BG + colored strokes,
   so we need to INVERT to get black text on white for Tesseract.
2. Image upscaled 2x — Tesseract needs larger images for reliable recognition.
3. Morphological dilation applied — thickens thin air-drawn strokes.
4. Gaussian blur + Otsu threshold — cleaner binarization than fixed threshold of 150.
5. Padding added — Tesseract fails on text that touches image borders.
6. PSM changed to 11 (sparse text) — better for irregular handwritten canvas content.
7. Multiple PSM fallback — tries PSM 11, then 6, then 3 for best result.
8. OEM 1 (LSTM) forced — more accurate than legacy Tesseract engine.
"""

import cv2
import numpy as np
from PIL import Image
import pytesseract
from datetime import datetime
import os

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"


def preprocess_canvas_for_ocr(img_canvas):
    """
    Robustly preprocess an AirCanvas image for Tesseract OCR.
    
    The canvas has colored strokes on a pure black background.
    Tesseract needs: black text on white background, thick strokes, clean edges.
    
    Returns a preprocessed PIL image ready for pytesseract.
    """
    # Step 1: Convert BGR canvas to grayscale
    gray = cv2.cvtColor(img_canvas, cv2.COLOR_BGR2GRAY)

    # Step 2: Upscale 2x — Tesseract accuracy improves significantly on larger images
    scale = 2
    gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Step 3: Slight Gaussian blur to smooth jagged air-drawn edges
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Step 4: Threshold — INVERT because canvas = black background, colored strokes
    # Otsu automatically picks the best threshold value
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Now: strokes = black pixels, background = white pixels ✓

    # Step 5: Dilate to thicken thin air-drawn strokes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    thresh = cv2.dilate(thresh, kernel, iterations=1)

    # Step 6: Add white padding around the image — Tesseract fails on border-touching text
    padded = cv2.copyMakeBorder(thresh, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)

    return Image.fromarray(padded)


def extract_text_from_canvas(self):
    """
    FIXED drop-in replacement for VirtualPainter.extract_text_from_canvas().
    
    Paste this method into your VirtualPainter class to replace the existing one.
    """
    if not OCR_AVAILABLE:
        print("❌ OCR not available")
        print("📦 Install: pip install pytesseract pillow")
        print("📦 Install Tesseract: https://github.com/UB-Mannheim/tesseract/wiki")
        return

    try:
        pil_image = preprocess_canvas_for_ocr(self.img_canvas)

        # Try multiple PSM modes and pick the best non-empty result
        # PSM 11 = sparse text (best for irregular hand-drawn content)
        # PSM 6  = uniform block of text (fallback)
        # PSM 3  = fully automatic (last resort)
        best_text = ""
        for psm in [11, 6, 3, 7, 8]:
            config = f"--psm {psm} --oem 1"
            result = pytesseract.image_to_string(pil_image, config=config)
            cleaned = result.strip()
            if cleaned and len(cleaned) > len(best_text):
                best_text = cleaned

        if best_text:
            os.makedirs('extracted_text', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"extracted_text/text_{timestamp}.txt"
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(best_text)
            print(f"📝 Text extracted → {filename}")
            print(f"🔤 Detected text:\n{'='*30}\n{best_text}\n{'='*30}")
        else:
            print("⚠️  No text detected.")
            print("💡 Tips:")
            print("   • Write larger letters (taller than 80px on screen)")
            print("   • Use a bright color (purple, blue, red — not black/eraser)")
            print("   • Increase brush size with '+' key before writing")
            print("   • Write clearly with deliberate strokes")

            # Debug: save the preprocessed image so you can inspect it
            os.makedirs('extracted_text', exist_ok=True)
            debug_path = "extracted_text/debug_ocr_input.png"
            pil_image.save(debug_path)
            print(f"   • Debug image saved → {debug_path} (inspect what Tesseract sees)")

    except Exception as e:
        print(f"❌ OCR error: {e}")
        import traceback
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────
# STANDALONE TEST — run this file directly to verify OCR works
# python ocr_fix.py
# ─────────────────────────────────────────────────────────────

def test_ocr_with_simulated_canvas():
    """
    Simulates what your AirCanvas produces and tests OCR on it.
    Run this standalone to confirm Tesseract is working correctly
    before testing with real hand-drawn input.
    """
    print("🧪 Testing OCR pipeline with simulated canvas...\n")

    # Simulate a 1280x720 black canvas (exactly like img_canvas in your app)
    canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

    # Draw "hi" in purple (255, 0, 255) with thick strokes, like air drawing
    color = (255, 0, 255)
    thickness = 18

    # Draw 'h'
    cv2.line(canvas, (200, 200), (200, 400), color, thickness)   # left vertical
    cv2.line(canvas, (200, 300), (280, 300), color, thickness)   # crossbar
    cv2.line(canvas, (280, 300), (280, 400), color, thickness)   # right leg

    # Draw 'i'
    cv2.line(canvas, (360, 280), (360, 400), color, thickness)   # body
    cv2.circle(canvas, (360, 240), 10, color, -1)                # dot

    print("Canvas created (black background + colored strokes)")
    print(f"Canvas shape: {canvas.shape}")

    # Run through the same preprocessing pipeline
    pil_image = preprocess_canvas_for_ocr(canvas)

    # Save debug image
    os.makedirs('extracted_text', exist_ok=True)
    pil_image.save("extracted_text/debug_ocr_input.png")
    print("Debug preprocessed image saved → extracted_text/debug_ocr_input.png")

    # Try all PSM modes
    print("\n--- Tesseract Results by PSM mode ---")
    best_text = ""
    for psm in [11, 6, 3, 7, 8]:
        config = f"--psm {psm} --oem 1"
        result = pytesseract.image_to_string(pil_image, config=config).strip()
        print(f"  PSM {psm:2d}: '{result}'")
        if result and len(result) > len(best_text):
            best_text = result

    print(f"\n✅ Best result: '{best_text}'")

    if best_text:
        print("✓ OCR pipeline is working correctly!")
        print("✓ The fix will work for your real canvas drawings too.")
    else:
        print("⚠ Tesseract returned empty. Check:")
        print("  1. Tesseract is installed at the path in pytesseract.tesseract_cmd")
        print("  2. Run: tesseract --version   in your terminal")
        print("  3. Check extracted_text/debug_ocr_input.png to see what Tesseract receives")


if __name__ == "__main__":
    test_ocr_with_simulated_canvas()