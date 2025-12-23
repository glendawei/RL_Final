import os
import cv2
import numpy as np
import csv

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ======================================================
# Evaluation config（所有方法一次列清楚）
# ======================================================
DATA_CONFIG = [
          {
        "name": "dqn",
        "gt_dir": os.path.join(SCRIPT_DIR, "testing", "masks"),
        "pred_dir": os.path.join(SCRIPT_DIR,"results_DQN", "251221_DQN", "prediction_mask"),
                "suffix": "",
                "requie_mask_suffix": True
    },
      {
        "name": "a2c",
        "gt_dir": os.path.join(SCRIPT_DIR, "testing", "masks"),
        "pred_dir": os.path.join(SCRIPT_DIR,"results_A2C", "251221_A2C", "prediction_mask"),
                "suffix": "",
                "requie_mask_suffix": True
    },
          {
        "name": "ppo",
        "gt_dir": os.path.join(SCRIPT_DIR, "testing", "masks"),
        "pred_dir": os.path.join(SCRIPT_DIR,"results_ppo", "251213_useLessPoint", "prediction_mask"),
                "suffix": "",
                "requie_mask_suffix": True
    },
]

# ======================================================
# Utils
# ======================================================
def binarize_mask(img):
    if img is None:
        raise ValueError("img is None")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    maxv = float(img.max()) if img.size else 0.0
    thr = 0.5 if maxv <= 1.0 else 127
    return (img > thr).astype(np.uint8)


def boundary_iou(pred_mask, gt_mask, dilation_ratio=0.02):
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)

    h, w = pred.shape
    diag = (h*h + w*w) ** 0.5
    dp = max(1, int(diag * dilation_ratio))
    k_size = 2 * dp + 1

    k3 = np.ones((3,3), np.uint8)
    kk = np.ones((k_size, k_size), np.uint8)

    pred_b = cv2.subtract(pred, cv2.erode(pred, k3))
    gt_b   = cv2.subtract(gt,   cv2.erode(gt,   k3))

    pred_d = cv2.dilate(pred_b, kk).astype(bool)
    gt_d   = cv2.dilate(gt_b,   kk).astype(bool)

    inter = np.logical_and(pred_d, gt_d).sum()
    union = np.logical_or(pred_d, gt_d).sum()
    return float(inter / union) if union > 0 else 0.0


def compute_metrics(pred_mask, gt_mask):
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    TP = np.logical_and(pred, gt).sum()
    FP = np.logical_and(pred, ~gt).sum()
    FN = np.logical_and(~pred, gt).sum()
    TN = np.logical_and(~pred, ~gt).sum()

    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    accuracy = (TP + TN) / (TP + FP + FN + TN) if (TP + FP + FN + TN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    b_iou = boundary_iou(pred_mask, gt_mask)

    return {
        "IoU": iou,
        "BoundaryIoU": b_iou,
        "Precision": precision,
        "Recall": recall,
        "Accuracy": accuracy,
        "F1": f1,
    }


def get_pred_path(pred_dir, img_name, suffix):
    names_to_try = [img_name]
    # try adding _mask (predictions may be saved with _mask suffix)
    if not img_name.endswith("_mask"):
        names_to_try.append(img_name + "_mask")
    # also try stripping _mask if given
    if img_name.endswith("_mask"):
        names_to_try.append(img_name[:-5])

    for base in names_to_try:
        for ext in [".png", ".jpg"]:
            p = os.path.join(pred_dir, base + suffix + ext)
            if os.path.isfile(p):
                return p
    return None


# ======================================================
# Main evaluation
# ======================================================
def evaluate(cfg):
    totals = {k: 0.0 for k in ["IoU","BoundaryIoU","Precision","Recall","Accuracy","F1"]}
    count = 0

    for fname in os.listdir(cfg["gt_dir"]):
        if not fname.lower().endswith((".png", ".jpg")):
            continue

        img_name = os.path.splitext(fname)[0]
        gt_path = os.path.join(cfg["gt_dir"], fname)
        pred_path = get_pred_path(cfg["pred_dir"], img_name, cfg["suffix"])

        if pred_path is None:
            continue

        gt_img = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED)
        pred_img = cv2.imread(pred_path, cv2.IMREAD_UNCHANGED)

        try:
            gt_mask = binarize_mask(gt_img)
            pred_mask = binarize_mask(pred_img)
        except Exception:
            continue

        if gt_mask.shape != pred_mask.shape:
            continue

        metrics = compute_metrics(pred_mask, gt_mask)
        for k in totals:
            totals[k] += metrics[k]
        count += 1

    return {k: totals[k] / count if count > 0 else 0.0 for k in totals}


# ======================================================
# Run + CSV
# ======================================================
if __name__ == "__main__":
    metrics = {k: [] for k in ["IoU","BoundaryIoU","Precision","Recall","Accuracy","F1"]}
    names = []

    for cfg in DATA_CONFIG:
        print(f"Evaluating {cfg['name']} ...")
        result = evaluate(cfg)
        names.append(cfg["name"])
        for k in metrics:
            metrics[k].append(result[k])

    csv_path = os.path.join(SCRIPT_DIR, "evaluation_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Metric"] + names)
        for k in metrics:
            writer.writerow([k] + [f"{v:.6f}" for v in metrics[k]])

    print(f"\nCSV 已輸出：{csv_path}")