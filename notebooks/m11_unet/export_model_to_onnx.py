"""
Export Trained PyTorch Model to ONNX for MATLAB
================================================
This script exports your trained U-Net model to ONNX format,
which can be imported and used in MATLAB.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UNetWithPretrainedEncoder(nn.Module):
    """U-Net with pretrained encoder - same as training code"""

    def __init__(self, encoder_name='resnet34', num_classes=2, pretrained=False):
        super().__init__()

        self.encoder_name = encoder_name
        self.num_classes = num_classes

        if encoder_name == 'resnet18':
            encoder = models.resnet18(weights=None)
            encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name == 'resnet34':
            encoder = models.resnet34(weights=None)
            encoder_channels = [64, 64, 128, 256, 512]
        elif encoder_name == 'resnet50':
            encoder = models.resnet50(weights=None)
            encoder_channels = [64, 256, 512, 1024, 2048]
        elif encoder_name == 'mobilenet_v2':
            encoder = models.mobilenet_v2(weights=None)
            encoder_channels = [16, 24, 32, 96, 1280]
        else:
            raise ValueError(f"Unsupported encoder: {encoder_name}")

        if 'resnet' in encoder_name:
            self.encoder0 = nn.Sequential(encoder.conv1, encoder.bn1, encoder.relu)
            self.encoder1 = nn.Sequential(encoder.maxpool, encoder.layer1)
            self.encoder2 = encoder.layer2
            self.encoder3 = encoder.layer3
            self.encoder4 = encoder.layer4
        elif encoder_name == 'mobilenet_v2':
            features = encoder.features
            self.encoder0 = features[0:2]
            self.encoder1 = features[2:4]
            self.encoder2 = features[4:7]
            self.encoder3 = features[7:14]
            self.encoder4 = features[14:]

        self.enc_channels = encoder_channels

        self.decoder4 = self._decoder_block(encoder_channels[4] + encoder_channels[3], encoder_channels[3])
        self.decoder3 = self._decoder_block(encoder_channels[3] + encoder_channels[2], encoder_channels[2])
        self.decoder2 = self._decoder_block(encoder_channels[2] + encoder_channels[1], encoder_channels[1])
        self.decoder1 = self._decoder_block(encoder_channels[1] + encoder_channels[0], encoder_channels[0])
        self.decoder0 = self._decoder_block(encoder_channels[0], 64)

        self.final_conv = nn.Conv2d(64, num_classes, 1)

    def _decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        enc0 = self.encoder0(x)
        enc1 = self.encoder1(enc0)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        dec4 = F.interpolate(enc4, size=enc3.shape[2:], mode='bilinear', align_corners=True)
        dec4 = torch.cat([dec4, enc3], dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = F.interpolate(dec4, size=enc2.shape[2:], mode='bilinear', align_corners=True)
        dec3 = torch.cat([dec3, enc2], dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = F.interpolate(dec3, size=enc1.shape[2:], mode='bilinear', align_corners=True)
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = F.interpolate(dec2, size=enc0.shape[2:], mode='bilinear', align_corners=True)
        dec1 = torch.cat([dec1, enc0], dim=1)
        dec1 = self.decoder1(dec1)

        dec0 = F.interpolate(dec1, size=x.shape[2:], mode='bilinear', align_corners=True)
        dec0 = self.decoder0(dec0)

        out = self.final_conv(dec0)

        return out


def export_to_onnx(
        checkpoint_path: str,
        output_path: str,
        encoder_name: str = 'resnet34',
        num_classes: int = 2,
        input_size: tuple = (512, 512),
        opset_version: int = 11
):
    """
    Export trained PyTorch model to ONNX format

    Args:
        checkpoint_path: Path to the .pth checkpoint file
        output_path: Path where to save the .onnx file
        encoder_name: Name of encoder used during training
        num_classes: Number of output classes
        input_size: Input image size (H, W)
        opset_version: ONNX opset version (11 is widely compatible)
    """

    logger.info(f"Loading checkpoint from: {checkpoint_path}")

    # Load model
    model = UNetWithPretrainedEncoder(
        encoder_name=encoder_name,
        num_classes=num_classes,
        pretrained=False
    )

    # Load trained weights
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model loaded successfully (Val Acc: {checkpoint.get('val_acc', 'N/A')})")

    # Create dummy input (batch_size=1, channels=3, height, width)
    dummy_input = torch.randn(1, 3, input_size[0], input_size[1])

    # Export to ONNX
    logger.info(f"Exporting to ONNX (opset {opset_version})...")
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size', 2: 'height', 3: 'width'},
            'output': {0: 'batch_size', 2: 'height', 3: 'width'}
        }
    )

    logger.info(f"✓ Model exported successfully to: {output_path}")
    logger.info(f"  Input shape: [batch, 3, {input_size[0]}, {input_size[1]}]")
    logger.info(f"  Output shape: [batch, {num_classes}, {input_size[0]}, {input_size[1]}]")

    # Verify the exported model
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("✓ ONNX model verified successfully")


if __name__ == "__main__":
    # Configuration - adjust these to match your trained model
    CHECKPOINT_PATH = "models/best_model.pth"
    OUTPUT_PATH = "models/model.onnx"
    ENCODER_NAME = 'resnet34'
    NUM_CLASSES = 2
    INPUT_SIZE = (512, 512)

    export_to_onnx(
        checkpoint_path=CHECKPOINT_PATH,
        output_path=OUTPUT_PATH,
        encoder_name=ENCODER_NAME,
        num_classes=NUM_CLASSES,
        input_size=INPUT_SIZE,
        opset_version=11
    )