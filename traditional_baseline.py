import os
import cv2
import numpy as np

# ======================================================
# Path 설정
# ======================================================
INPUT_DIR = "./testing/images"

OUT_OTSU = "./testing/Otsu_Threshold"
OUT_ADAPTIVE = "./testing/Adaptive_Threshold"
OUT_CANNY = "./testing/Canny_Morphology"

for d in [OUT_OTSU, OUT_ADAPTIVE, OUT_CANNY]:
    os.makedirs(d, exist_ok=True)

# ======================================================
# Utils
# ======================================================
def read_gray(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


# ======================================================
# Baseline methods
# ======================================================
def otsu_threshold(gray):
    _, mask = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return mask


def adaptive_threshold(gray):
    mask = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=5
    )
    return mask


def canny_morphology(gray):
    # 1. Canny edge
    edges = cv2.Canny(gray, 50, 150)

    # 2. Morphology close to connect edges
    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

    # 3. Fill contours
    contours, _ = cv2.findContours(
        closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mask = np.zeros_like(gray, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    return mask


# ======================================================
# Main
# ======================================================
def run_baselines():
    files = sorted(os.listdir(INPUT_DIR))
    count = 0

    for fname in files:
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue

        img_path = os.path.join(INPUT_DIR, fname)

        try:
            gray = read_gray(img_path)
        except Exception as e:
            print(e)
            continue

        # --- Baselines ---
        mask_otsu = otsu_threshold(gray)
        mask_adapt = adaptive_threshold(gray)
        mask_canny = canny_morphology(gray)

        # --- Save ---
        cv2.imwrite(os.path.join(OUT_OTSU, fname), mask_otsu)
        cv2.imwrite(os.path.join(OUT_ADAPTIVE, fname), mask_adapt)
        cv2.imwrite(os.path.join(OUT_CANNY, fname), mask_canny)

        count += 1

    print(f"\n完成：共處理 {count} 張影像")
    print("輸出資料夾：")
    print(f"  {OUT_OTSU}")
    print(f"  {OUT_ADAPTIVE}")
    print(f"  {OUT_CANNY}")


if __name__ == "__main__":
    run_baselines()