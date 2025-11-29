"""
Binary inference (tissue vs background), matching training preprocessing:
1) per-image min–max -> [0,1]
2) resize to 512x512 (bilinear)
3) replicate to 3 channels
4) ImageNet mean/std normalization
5) softmax; confidence = P(class==1)
6) upsample back to original HxW
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def preprocess_m11(m11: np.ndarray, target_size=(512, 512)) -> torch.Tensor:
    m11 = m11.astype(np.float32)
    mn, mx = float(m11.min()), float(m11.max())
    m11 = (m11 - mn) / (mx - mn + 1e-6)
    x = torch.from_numpy(m11).unsqueeze(0).unsqueeze(0)          # [1,1,H,W]
    x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=True)
    x = x.squeeze(0).repeat(3, 1, 1).numpy()                     # [3,H,W]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3,1,1)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3,1,1)
    x = (x - mean) / std
    return torch.from_numpy(x).unsqueeze(0).float()              # [1,3,H,W]

class UNetWithPretrainedEncoder(nn.Module):
    def __init__(self, encoder_name='resnet34', num_classes=2):
        super().__init__()
        import torchvision.models as models
        if encoder_name == 'resnet18':
            enc = models.resnet18(weights=None); ch = [64, 64, 128, 256, 512]
        elif encoder_name == 'resnet34':
            enc = models.resnet34(weights=None); ch = [64, 64, 128, 256, 512]
        elif encoder_name == 'resnet50':
            enc = models.resnet50(weights=None); ch = [64, 256, 512, 1024, 2048]
        else:
            enc = models.resnet34(weights=None); ch = [64, 64, 128, 256, 512]

        self.encoder0 = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.encoder1 = nn.Sequential(enc.maxpool, enc.layer1)
        self.encoder2 = enc.layer2
        self.encoder3 = enc.layer3
        self.encoder4 = enc.layer4

        def dec(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

        self.decoder4 = dec(ch[4] + ch[3], ch[3])
        self.decoder3 = dec(ch[3] + ch[2], ch[2])
        self.decoder2 = dec(ch[2] + ch[1], ch[1])
        self.decoder1 = dec(ch[1] + ch[0], ch[0])
        self.decoder0 = dec(ch[0], 64)
        self.final_conv = nn.Conv2d(64, 2, 1)  # binary: 2 logits

    def forward(self, x):
        e0 = self.encoder0(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        d4 = F.interpolate(e4, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d4 = self.decoder4(torch.cat([d4, e3], dim=1))
        d3 = F.interpolate(d4, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.decoder3(torch.cat([d3, e2], dim=1))
        d2 = F.interpolate(d3, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d2 = self.decoder2(torch.cat([d2, e1], dim=1))
        d1 = F.interpolate(d2, size=e0.shape[2:], mode='bilinear', align_corners=True)
        d1 = self.decoder1(torch.cat([d1, e0], dim=1))
        d0 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=True)
        d0 = self.decoder0(d0)
        return self.final_conv(d0)

def run_inference(m11_mat_path: str, out_mat_path: str, model_path: str = '../../models/best_model.pth'):
    from scipy.io import loadmat, savemat

    data = loadmat(m11_mat_path)
    if 'm11' not in data:
        raise KeyError("MAT file must contain variable 'm11'")
    m11 = data['m11']
    if m11.ndim != 2:
        raise ValueError(f"Expected 2D m11, got shape {m11.shape}")
    H0, W0 = m11.shape

    x = preprocess_m11(m11, target_size=(512, 512))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        ckpt = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(model_path, map_location=device)

    enc_name = ckpt.get('config', {}).get('encoder_name', 'resnet34')
    model = UNetWithPretrainedEncoder(encoder_name=enc_name, num_classes=2).to(device)
    model.load_state_dict(ckpt['model_state_dict'], strict=True)
    model.eval()

    with torch.no_grad():
        logits = model(x.to(device))                # [1,2,512,512]
        probs  = torch.softmax(logits, dim=1)       # [1,2,512,512]
        pred   = probs[:, 1, ...] >= probs[:, 0, ...]     # class-1 is tissue
        conf   = probs[:, 1, ...]                          # P(tissue)

    # back to original size
    pred_full = F.interpolate(pred.float(), size=(H0, W0), mode='nearest').squeeze(0)
    conf_full = F.interpolate(conf,        size=(H0, W0), mode='bilinear', align_corners=True).squeeze(0)

    pred_np = (pred_full > 0.5).cpu().numpy().astype(np.uint8)   # [H,W], {0,1}
    conf_np = conf_full.cpu().numpy().astype(np.float32)         # [H,W], [0,1]

    tissue_pct = float(pred_np.mean() * 100.0)

    savemat(out_mat_path, {
        'predicted_mask': pred_np,        # uint8 [H,W] 0/1
        'confidence_map': conf_np,        # float32 [H,W] in [0,1]
        'tissue_percentage': tissue_pct
    }, do_compression=True)

    return pred_np, conf_np

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python inference.py <m11_mat_file> <output_mat_file> [model_path]')
        sys.exit(1)
    in_mat = sys.argv[1]
    out_mat = sys.argv[2]
    model_p = sys.argv[3] if len(sys.argv) > 3 else '../../models/best_model.pth'
    run_inference(in_mat, out_mat, model_p)
