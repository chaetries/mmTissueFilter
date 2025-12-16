import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class UNet(nn.Module):
    """Simple UNet - matches training architecture exactly"""

    def __init__(self, num_classes):
        super().__init__()
        # Using simple ResNet34 encoder
        self.encoder = models.resnet34(weights=None) # No internet usually, so weights=None or 'IMAGENET1K_V1' if cached
        self.enc_layers = list(self.encoder.children())

        self.enc1 = nn.Sequential(*self.enc_layers[:3])  # 64
        self.enc2 = nn.Sequential(*self.enc_layers[3:5])  # 64
        self.enc3 = self.enc_layers[5]  # 128
        self.enc4 = self.enc_layers[6]  # 256
        self.enc5 = self.enc_layers[7]  # 512

        self.dec4 = self._block(512 + 256, 256)
        self.dec3 = self._block(256 + 128, 128)
        self.dec2 = self._block(128 + 64, 64)
        self.dec1 = self._block(64 + 64, 32)

        self.final = nn.Conv2d(32, num_classes, 1)

    def _block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c), nn.ReLU()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)

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
