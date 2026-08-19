"""
Per-sample test-set analysis for the proposed U-Net++ model.

Mirrors m11_unet_test_sample_analysis.ipynb (which covers the plain-U-Net baseline,
output to results/analysis_results/unet/) but loads the U-Net++ checkpoint trained in
notebooks/model_comparison/model_comparison.ipynb, output to results/analysis_results/unetpp/.
Produces the same artifacts: full_performance.csv, per-sample PDF+TIFF panels, and a
combined all_test_samples.pdf, so the two model folders are directly comparable.
"""
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm.auto import tqdm
import segmentation_models_pytorch as smp

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class Config:
    MODEL_PATH = REPO_ROOT / "models" / "comparison" / "UNetPlusPlus_Pretrained.pth"
    DATA_SPLIT_PATH = REPO_ROOT / "models" / "data_split.json"
    OUTPUT_DIR = REPO_ROOT / "results" / "analysis_results" / "unetpp"
    ENCODER_NAME = "resnet34"
    INPUT_SIZE = (512, 512)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CLASS_NAMES = ["Background", "Tissue", "OS", "Vaginal"]
    CLASS_COLORS = {0: [0, 0, 0], 1: [0, 0, 255], 2: [0, 255, 0], 3: [255, 0, 0]}


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def extract_m11_and_mask(npz_path: Path) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            m11 = None
            if "nM11s" in data:
                m11 = np.array(data["nM11s"])
            elif "M11s" in data:
                raw = np.array(data["M11s"])
                m11 = (raw - raw.min()) / (raw.max() - raw.min()) if raw.max() > raw.min() else np.zeros_like(raw)
            if m11 is None or m11.ndim != 2:
                return None
            m11 = np.nan_to_num(m11, nan=0.0, posinf=0.0, neginf=0.0)

            tissue_mask = None
            for key in ["tissue_mask", "annotation_mask"]:
                if key in data and data[key].size > 0:
                    tissue_mask = np.array(data[key]) > 0
                    break
            if tissue_mask is None:
                return None

            os_mask = np.array(data["os_mask"]) > 0 if ("os_mask" in data and data["os_mask"].size > 0) else None
            vag_mask = None
            for key in ["vaginal_mask", "vaginal_wall", "vaginal_wall_mask"]:
                if key in data and data[key].size > 0:
                    vag_mask = np.array(data[key]) > 0
                    break

            combined = np.zeros_like(tissue_mask, dtype=np.int64)
            combined[tissue_mask] = 1
            if os_mask is not None:
                combined[os_mask] = 2
            if vag_mask is not None:
                combined[vag_mask] = 3
            return m11.astype(np.float32), combined.astype(np.int64)
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return None


def preprocess_m11(m11: np.ndarray, target_size) -> torch.Tensor:
    if m11.shape != target_size:
        t = torch.from_numpy(m11).float().unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=target_size, mode="bilinear", align_corners=True)
        m11 = t.squeeze().numpy()
    rgb = np.stack([m11, m11, m11], axis=0)
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb).float().unsqueeze(0)


def calculate_dice_coefficient(pred, gt, class_id):
    p, g = (pred == class_id), (gt == class_id)
    inter, union = np.logical_and(p, g).sum(), p.sum() + g.sum()
    return 1.0 if union == 0 else (2.0 * inter) / union


def calculate_iou(pred, gt, class_id):
    p, g = (pred == class_id), (gt == class_id)
    inter, union = np.logical_and(p, g).sum(), np.logical_or(p, g).sum()
    return 1.0 if union == 0 else inter / union


def calculate_f1_score(pred, gt, class_id):
    p, g = (pred == class_id), (gt == class_id)
    tp = np.logical_and(p, g).sum()
    fp = np.logical_and(p, ~g).sum()
    fn = np.logical_and(~p, g).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0


def calculate_all_metrics(pred, gt, num_classes) -> Dict:
    metrics = {"overall": {}, "per_class": {}}
    for c in range(num_classes):
        metrics["per_class"][c] = {
            "dice": calculate_dice_coefficient(pred, gt, c),
            "iou": calculate_iou(pred, gt, c),
            "f1_score": calculate_f1_score(pred, gt, c),
        }
    metrics["overall"]["pixel_accuracy"] = float(np.mean(pred == gt))
    metrics["overall"]["mean_tissue_dice"] = float(
        np.mean([metrics["per_class"][c]["dice"] for c in range(1, num_classes)])
    )
    return metrics


@torch.no_grad()
def run_inference(npz_path: Path, model, device, input_size) -> Optional[Dict]:
    result = extract_m11_and_mask(npz_path)
    if result is None:
        return None
    m11, gt_mask = result
    original_shape = m11.shape
    x = preprocess_m11(m11, input_size).to(device)
    logits = model(x)
    pred = torch.argmax(logits, dim=1)
    pred = F.interpolate(pred.unsqueeze(1).float(), size=original_shape, mode="nearest").squeeze().long()
    pred_mask = pred.cpu().numpy()
    num_classes = len(Config.CLASS_NAMES)
    metrics = calculate_all_metrics(pred_mask, gt_mask, num_classes)
    return {"m11": m11, "ground_truth": gt_mask, "prediction": pred_mask, "metrics": metrics}


def main():
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(Config.MODEL_PATH, map_location=Config.DEVICE, weights_only=False)
    model = smp.UnetPlusPlus(encoder_name=Config.ENCODER_NAME, encoder_weights=None,
                              in_channels=3, classes=len(Config.CLASS_NAMES))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(Config.DEVICE).eval()
    print(f"Loaded U-Net++ checkpoint: epoch {checkpoint.get('epoch')}, val_loss={checkpoint.get('val_loss'):.4f}")

    with open(Config.DATA_SPLIT_PATH) as f:
        data_split = json.load(f)

    test_results = []
    for info in tqdm(data_split["test"], desc="Test samples"):
        r = run_inference(Path(info["path"]), model, Config.DEVICE, Config.INPUT_SIZE)
        if r is not None:
            r["sample_name"] = info["name"]
            test_results.append(r)
        else:
            print(f"Failed to process: {info['name']}")

    # --- full_performance.csv (same layout/convention as the plain-U-Net notebook) ---
    all_pixel_acc = [r["metrics"]["overall"]["pixel_accuracy"] for r in test_results]
    all_tissue_dice = [r["metrics"]["overall"]["mean_tissue_dice"] for r in test_results]
    rows = [
        {"Class": "Overall Pixel Accuracy", "Dice": "-",
         "IoU": f"{np.mean(all_pixel_acc):.4f} ± {np.std(all_pixel_acc):.4f}", "F1": "-"},
        {"Class": "Overall Tissue DSC",
         "Dice": f"{np.mean(all_tissue_dice):.4f} ± {np.std(all_tissue_dice):.4f}", "IoU": "-", "F1": "-"},
    ]
    for class_id, class_name in enumerate(Config.CLASS_NAMES):
        dice_v = [r["metrics"]["per_class"][class_id]["dice"] for r in test_results]
        iou_v = [r["metrics"]["per_class"][class_id]["iou"] for r in test_results]
        f1_v = [r["metrics"]["per_class"][class_id]["f1_score"] for r in test_results]
        rows.append({
            "Class": class_name,
            "Dice": f"{np.mean(dice_v):.4f} ± {np.std(dice_v):.4f}",
            "IoU": f"{np.mean(iou_v):.4f} ± {np.std(iou_v):.4f}",
            "F1": f"{np.mean(f1_v):.4f} ± {np.std(f1_v):.4f}",
        })
    table_df = pd.DataFrame(rows)
    table_df.to_csv(Config.OUTPUT_DIR / "full_performance.csv", index=False)
    print(table_df.to_string(index=False))

    # --- per-sample PDF + TIFF panels, and combined all_test_samples.pdf ---
    pdf_path = Config.OUTPUT_DIR / "all_test_samples.pdf"
    with PdfPages(pdf_path) as pdf:
        for idx, result in enumerate(tqdm(test_results, desc="Creating visualizations")):
            m11, gt_mask, pred_mask = result["m11"], result["ground_truth"], result["prediction"]
            sample_name = result["sample_name"]

            fig, axes = plt.subplots(3, 1, figsize=(8, 12))
            axes[0].imshow(m11, cmap="gray", vmin=0, vmax=1)
            axes[0].set_title("(A) M11 Image", fontweight="bold", fontsize=12)
            axes[0].axis("off")

            gt_rgb = np.zeros((*gt_mask.shape, 3), dtype=np.uint8)
            for class_id, color in Config.CLASS_COLORS.items():
                gt_rgb[gt_mask == class_id] = color
            axes[1].imshow(gt_rgb)
            axes[1].set_title("(B) Ground Truth", fontweight="bold", fontsize=12)
            axes[1].axis("off")

            pred_rgb = np.zeros((*pred_mask.shape, 3), dtype=np.uint8)
            for class_id, color in Config.CLASS_COLORS.items():
                pred_rgb[pred_mask == class_id] = color
            axes[2].imshow(pred_rgb)
            axes[2].set_title("(C) Prediction (U-Net++)", fontweight="bold", fontsize=12)
            axes[2].axis("off")

            dsc = result["metrics"]["overall"]["mean_tissue_dice"]
            tissue_dsc = result["metrics"]["per_class"][1]["dice"]
            os_dsc = result["metrics"]["per_class"][2]["dice"]
            vaginal_dsc = result["metrics"]["per_class"][3]["dice"]
            fig.suptitle(
                f"Sample {idx + 1}/{len(test_results)}: {sample_name}\n"
                f"Overall DSC: {dsc:.4f} | Tissue: {tissue_dsc:.4f} | OS: {os_dsc:.4f} | Vaginal: {vaginal_dsc:.4f}",
                fontsize=13, fontweight="bold", y=0.995,
            )
            plt.tight_layout(rect=[0, 0, 1, 0.98])

            plt.savefig(Config.OUTPUT_DIR / f"sample_{idx + 1:02d}_{sample_name}.pdf",
                        dpi=300, bbox_inches="tight", format="pdf")
            plt.savefig(Config.OUTPUT_DIR / f"sample_{idx + 1:02d}_{sample_name}.tiff",
                        dpi=300, bbox_inches="tight", format="tiff", pil_kwargs={"compression": "tiff_lzw"})
            pdf.savefig(fig, dpi=300, bbox_inches="tight")
            plt.close(fig)

    print(f"\nSaved {len(test_results)} per-sample panels + combined PDF to: {Config.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
