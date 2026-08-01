"""
Benchmark GPU/CPU inference latency and parameter count for the five segmentation
methods compared in model_comparison.ipynb. Loads the already-trained checkpoints
from models/best_model.pth and models/comparison/*, times a single 512x512 forward
pass (30 runs after 5 warmup iterations), and writes results to
results/model_comparison/inference_speed.{json,csv}.

Run this after model_comparison.ipynb has produced all five checkpoints.
"""
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import segmentation_models_pytorch as smp

REPO = Path(__file__).resolve().parents[2]
COMPARISON_DIR = REPO / "models" / "comparison"
RESULTS_DIR = REPO / "results" / "model_comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_CLASSES = 4
ENCODER_NAME = 'resnet34'
INPUT_SIZE = (512, 512)
N_WARMUP = 5
N_RUNS = 30


class PublishedUNet(nn.Module):
    """Matches run_trained/model_utils.py / notebooks/m11_unet training architecture exactly."""

    def __init__(self, num_classes):
        super().__init__()
        self.encoder = models.resnet34(weights=None)
        enc_layers = list(self.encoder.children())
        self.enc1 = nn.Sequential(*enc_layers[:3])
        self.enc2 = nn.Sequential(*enc_layers[3:5])
        self.enc3 = enc_layers[5]
        self.enc4 = enc_layers[6]
        self.enc5 = enc_layers[7]
        self.dec4 = self._block(512 + 256, 256)
        self.dec3 = self._block(256 + 128, 128)
        self.dec2 = self._block(128 + 64, 64)
        self.dec1 = self._block(64 + 64, 32)
        self.final = nn.Conv2d(32, num_classes, 1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1), nn.BatchNorm2d(out_c), nn.ReLU())

    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(e1); e3 = self.enc3(e2); e4 = self.enc4(e3); e5 = self.enc5(e4)
        d4 = F.interpolate(e5, size=e4.shape[2:], mode='bilinear')
        d4 = self.dec4(torch.cat([d4, e4], 1))
        d3 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear')
        d3 = self.dec3(torch.cat([d3, e3], 1))
        d2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear')
        d2 = self.dec2(torch.cat([d2, e2], 1))
        d1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear')
        d1 = self.dec1(torch.cat([d1, e1], 1))
        out = F.interpolate(d1, scale_factor=2, mode='bilinear')
        return self.final(out)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def load_checkpoint_state(path):
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    return ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt


def build_models():
    m = {}

    ref = PublishedUNet(num_classes=NUM_CLASSES)
    ref.load_state_dict(load_checkpoint_state(REPO / "models" / "best_model.pth"))
    m['UNet_Pretrained_Published'] = ref

    unet_scratch = smp.Unet(encoder_name=ENCODER_NAME, encoder_weights=None, in_channels=3, classes=NUM_CLASSES)
    unet_scratch.load_state_dict(load_checkpoint_state(COMPARISON_DIR / 'UNet_NoPretrained.pth'))
    m['UNet_NoPretrained'] = unet_scratch

    unet_pp = smp.UnetPlusPlus(encoder_name=ENCODER_NAME, encoder_weights=None, in_channels=3, classes=NUM_CLASSES)
    unet_pp.load_state_dict(load_checkpoint_state(COMPARISON_DIR / 'UNetPlusPlus_Pretrained.pth'))
    m['UNetPlusPlus_Pretrained'] = unet_pp

    deeplab = smp.DeepLabV3Plus(encoder_name=ENCODER_NAME, encoder_weights=None, in_channels=3, classes=NUM_CLASSES)
    deeplab.load_state_dict(load_checkpoint_state(COMPARISON_DIR / 'DeepLabV3Plus_Pretrained.pth'))
    m['DeepLabV3Plus_Pretrained'] = deeplab

    for model in m.values():
        model.eval()
    return m


def benchmark_torch_model(model, device):
    model = model.to(device)
    x = torch.randn(1, 3, *INPUT_SIZE, device=device)

    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()

        times = []
        for _ in range(N_RUNS):
            t0 = time.perf_counter()
            _ = model(x)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    times = np.array(times)
    return times.mean(), times.std()


def benchmark_rf():
    rf_ckpt = joblib.load(COMPARISON_DIR / 'RandomForest_Classical.joblib')
    rf_model = rf_ckpt['model']
    n_feat = len(rf_ckpt['feature_names'])
    h, w = INPUT_SIZE
    x = np.random.rand(h * w, n_feat).astype(np.float32)

    for _ in range(3):
        _ = rf_model.predict(x)

    times = []
    for _ in range(10):
        t0 = time.perf_counter()
        _ = rf_model.predict(x)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    times = np.array(times)

    n_nodes = sum(est.tree_.node_count for est in rf_model.estimators_)
    return times.mean(), times.std(), n_nodes


def main():
    print("Loading models...")
    models_dict = build_models()

    print(f"\n{'Model':<28} {'Params (M)':>12} {'GPU ms/img':>14} {'CPU ms/img':>14}")
    print("-" * 72)

    cuda_available = torch.cuda.is_available()
    device_cpu = torch.device('cpu')

    results = []
    for name, model in models_dict.items():
        n_params = count_params(model) / 1e6
        cpu_mean, cpu_std = benchmark_torch_model(model, device_cpu)
        if cuda_available:
            gpu_mean, gpu_std = benchmark_torch_model(model, torch.device('cuda'))
        else:
            gpu_mean, gpu_std = float('nan'), float('nan')
        print(f"{name:<28} {n_params:>12.2f} {gpu_mean:>10.2f}+/-{gpu_std:<4.2f} {cpu_mean:>10.2f}+/-{cpu_std:<4.2f}")
        results.append({'model': name, 'params_M': n_params, 'gpu_ms': gpu_mean, 'gpu_std': gpu_std,
                         'cpu_ms': cpu_mean, 'cpu_std': cpu_std})

    rf_mean, rf_std, rf_nodes = benchmark_rf()
    print(f"{'RandomForest_Classical':<28} {'(nodes=' + str(rf_nodes) + ')':>12} {'N/A':>14} {rf_mean:>10.2f}+/-{rf_std:<4.2f}")
    results.append({'model': 'RandomForest_Classical', 'params_M': None, 'tree_nodes': rf_nodes,
                     'gpu_ms': None, 'gpu_std': None, 'cpu_ms': rf_mean, 'cpu_std': rf_std})

    with open(RESULTS_DIR / "inference_speed.json", "w") as f:
        json.dump(results, f, indent=2)

    fields = ['model', 'params_M', 'tree_nodes', 'gpu_ms', 'gpu_std', 'cpu_ms', 'cpu_std']
    with open(RESULTS_DIR / "inference_speed.csv", "w", newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in results:
            w.writerow({k: row.get(k, '') for k in fields})

    print(f"\nSaved to {RESULTS_DIR / 'inference_speed.json'} and inference_speed.csv")
    print(f"GPU: {torch.cuda.get_device_name(0) if cuda_available else 'N/A'}")


if __name__ == "__main__":
    main()
