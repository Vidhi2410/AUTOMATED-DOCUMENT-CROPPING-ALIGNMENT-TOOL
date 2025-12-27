from PIL import Image
import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog

# DPI values
PIXEL_MARGIN = 55          # ≈ 4mm at 350 DPI
FALLBACK_MARGIN = 70       # ≈ 5mm at 350 DPI
NO_BORDER_MARGIN = 90      # ≈ 7mm at 350 DPI

def save_tiff_with_dpi(image_cv, save_path, dpi=(350, 350)):
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    pil_image.save(save_path, format='TIFF', dpi=dpi, compression='tiff_deflate')

def detect_border_intensity(gray, x, y, w, h):
    borders = {}
    if y > 10:
        top_strip = gray[max(0, y - 10):y, x:x + w]
        borders['top'] = np.mean(top_strip)
    if y + h + 10 < gray.shape[0]:
        bottom_strip = gray[y + h:y + h + 10, x:x + w]
        borders['bottom'] = np.mean(bottom_strip)
    if x > 10:
        left_strip = gray[y:y + h, max(0, x - 10):x]
        borders['left'] = np.mean(left_strip)
    if x + w + 10 < gray.shape[1]:
        right_strip = gray[y:y + h, x + w:x + w + 10]
        borders['right'] = np.mean(right_strip)
    return borders

def crop_and_align(image_path, output_folder):
    print(f"🔍 Processing: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Unable to read image")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("❌ No contour found")
        return

    doc_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(doc_contour)

    borders = detect_border_intensity(gray, x, y, w, h)

    margin_top = PIXEL_MARGIN if borders.get('top', 255) < 50 else NO_BORDER_MARGIN
    margin_bottom = PIXEL_MARGIN if borders.get('bottom', 255) < 50 else NO_BORDER_MARGIN
    margin_left = PIXEL_MARGIN if borders.get('left', 255) < 50 else NO_BORDER_MARGIN
    margin_right = PIXEL_MARGIN if borders.get('right', 255) < 50 else NO_BORDER_MARGIN

    x_new = max(0, x - margin_left)
    y_new = max(0, y - margin_top)
    w_new = min(image.shape[1] - x_new, w + margin_left + margin_right)
    h_new = min(image.shape[0] - y_new, h + margin_top + margin_bottom)

    cropped = image[y_new:y_new + h_new, x_new:x_new + w_new]

    # Alignment
    gray_crop = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
    blur_crop = cv2.GaussianBlur(gray_crop, (5, 5), 0)
    _, thresh_crop = cv2.threshold(blur_crop, 50, 255, cv2.THRESH_BINARY)
    contours_crop, _ = cv2.findContours(thresh_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours_crop:
        print("❌ No contour in cropped image")
        return

    rect = cv2.minAreaRect(max(contours_crop, key=cv2.contourArea))
    angle = rect[-1]
    if angle < -45:
        angle += 90
    elif angle > 45:
        angle -= 90

    if abs(angle) > 1.0:
        (h_c, w_c) = cropped.shape[:2]
        center = (w_c // 2, h_c // 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        aligned = cv2.warpAffine(cropped, rot_mat, (w_c, h_c), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    else:
        aligned = cropped.copy()

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f"{base_name}.tiff")
    save_tiff_with_dpi(aligned, output_path)

    print(f"✅ Saved: {output_path} with 350 DPI")

def process_folder():
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="📂 Select Folder with Images")

    if not folder_path:
        print("🚫 No folder selected.")
        return

    output_folder = os.path.join(folder_path, "Cropped Output")
    os.makedirs(output_folder, exist_ok=True)

    supported_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(supported_exts)]

    if not image_files:
        print("❌ No supported image files found.")
        return

    for filename in image_files:
        full_path = os.path.join(folder_path, filename)
        crop_and_align(full_path, output_folder)

    print(f"\n🎉 All images saved to: {output_folder}")

if __name__ == "__main__":
    process_folder()
