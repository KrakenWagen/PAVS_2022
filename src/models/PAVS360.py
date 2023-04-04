# Import torch library
import torch
import torch.nn as nn
from utils.resnet3D import resnet18
import torch.nn.functional as F
from spherenet.sphere_cnn_3d import SphereConv3D, SphereMaxPool3D

# Spherical Block = Spherical Conv + Norm + ReLU
class SphereBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1, stride=1, groups=1, is_bn=True,
                 is_relu=True):
        super(SphereBlock3D, self).__init__()
        # Spherical Convolution
        self.conv = SphereConv3D(in_channels, out_channels, stride=stride, bias=False)
        # Batch normalization
        self.bn = nn.BatchNorm3d(out_channels, eps=1e-4)
        # ReLU activation
        self.relu = nn.ReLU(inplace=True)
        # If no BN or ReLU indicated, then the step is avoided
        if is_bn is False: self.bn = None
        if is_relu is False: self.relu = None

    def forward(self, x):
        # Convolve input
        x = self.conv(x)
        # Batch normalize if the layer exists
        if self.bn is not None: x = self.bn(x)
        # ReLU if the layer exists
        if self.relu is not None: x = self.relu(x)
        return x

# Spherical Encoder
class SphereEncoder3D(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3):
        super(SphereEncoder3D, self).__init__()
        # Calculate padding
        padding = (kernel_size - 1) // 2
        # Block with two Spherical Blocks
        self.encode = nn.Sequential(
            SphereBlock3D(x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
                        groups=1),
            SphereBlock3D(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1, stride=1,
                        groups=1),
        )
        # Spherical pooling
        self.pool = SphereMaxPool3D(stride=2)

    def forward(self, x):
        # Forward input
        y = self.encode(x)
        # Pooling
        y_pooled = self.pool(y)
        # y_pooled = F.max_pool2d(y, kernel_size=2, stride=2)
        return y, y_pooled

# Mesh Decoder
class SphereDecoder3D(nn.Module):
    def __init__(self, x_channels, y_channels, kernel_size=3):
        super(SphereDecoder3D, self).__init__()
        # Calculate padding
        padding = (kernel_size - 1) // 2
        # Block with three Spherical Blocks
        self.decode = nn.Sequential(
            SphereBlock3D(2 * x_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1,
                        stride=1, groups=1),
            SphereBlock3D(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1,
                        stride=1, groups=1),
            SphereBlock3D(y_channels, y_channels, kernel_size=kernel_size, padding=padding, dilation=1,
                        stride=1, groups=1),
        )

    def forward(self, down, x):
        N, C, T, H, W = down.size()
        # Upsampling
        y = F.interpolate(x, size=(T, H, W), mode='trilinear', align_corners=True)
        y = torch.cat([y, down], 1)
        # Forward input
        y = self.decode(y)
        return y

# PAVS MODEL
class PAVS_V7(nn.Module):
    def __init__(self):
        super(PAVS_V7, self).__init__()
        self.down1 = SphereEncoder3D(4, 48, kernel_size=3)
        self.down2 = SphereEncoder3D(48, 128, kernel_size=3)
        self.down3 = SphereEncoder3D(128, 256, kernel_size=3)  # [B, 256, 8, 16]
        self.down4 = SphereEncoder3D(256, 512, kernel_size=3)  # [B, 256, 8, 16]
        self.center = nn.Sequential(
            SphereBlock3D(512, 512, kernel_size=3, padding=1, stride=1),
        )  # [B, 512, 4, 8]

        self.combiner = nn.Conv2d(1024, 512, kernel_size=1, padding=0)

        self.up4 = SphereDecoder3D(512, 256, kernel_size=3)
        self.up3 = SphereDecoder3D(256, 128, kernel_size=3)
        self.up2 = SphereDecoder3D(128, 48, kernel_size=3)
        self.up1 = SphereDecoder3D(48, 48, kernel_size=3)
        self.end = SphereConv3D(48, 1, stride=1, bias=True)
        self.end3d = nn.Conv2d(6,1, kernel_size=1, padding=0)

        self.audioBranch = resnet18(shortcut_type='A', sample_size=64, sample_duration=16, num_classes=12,
                                        last_fc=False, last_pool=True)

    def forward(self, frame, v_cube, a, aem, eq_b):
        # v_cube & eq_b are inputs not used in this model,
        # they are present as different models were tested
        # along this one and I decided to keep the inputs
        # equal for all :)

        # Audio branch
        xA = self.audioBranch(a)  # xA = [B, 256, 1, 1, 1]
        xA = xA.squeeze(2).squeeze(2).squeeze(2)

        # Frame/AEM down
        out = torch.cat((frame, aem), dim=1)  # frame[B, 4, TEMPORAL_DEPTH, H, W]
        down1, out = self.down1(out)
        down2, out = self.down2(out)
        down3, out = self.down3(out)
        down4, out = self.down4(out)

        # Center
        out = self.center(out)

        # Combinator
        xA = xA.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # xA = [B, 256, 1, 1, 1]
        xA = xA.expand_as(out)  # xA = [B, 512, 1, 1, 1] -> [B, 512, 1, 8, 16]
        audioVisual = torch.cat((out, xA), dim=1)  # audioVisual = [B, 512, 1, 8, 16]

        audioVisual = audioVisual.squeeze(2)
        out = self.combiner(audioVisual)  # out = [B, 512, 4, 8]
        out = out.unsqueeze(2)

        # UP branch
        out = self.up4(down4, out)
        out = self.up3(down3, out)
        out = self.up2(down2, out)
        out = self.up1(down1, out)

        # End
        out = self.end(out)  # frame[B, 1, TEMPORAL_DEPTH, H, W]
        out = out.squeeze(1)
        out = self.end3d(out)

        return out