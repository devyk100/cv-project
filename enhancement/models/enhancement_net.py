import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EnhancementNet(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()

        self.kernel_size = kernel_size
        k = kernel_size * kernel_size

        # -------------------------
        # Encoder (U-Net Downsampling)
        # -------------------------

        self.conv1 = ConvBlock(1, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = ConvBlock(128, 256)

        # -------------------------
        # Dynamic Filter Head
        # -------------------------

        # self.filter_fc = nn.Linear(256 * 28 * 28, k)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.filter_fc = nn.Linear(256, k)

        # -------------------------
        # Exposure Map Decoder
        # -------------------------

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = ConvBlock(256, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.exposure_conv = nn.Conv2d(32, 1, 1)

    def forward(self, y):

        # -------------------------
        # Encoder
        # -------------------------

        c1 = self.conv1(y)
        p1 = self.pool1(c1)

        c2 = self.conv2(p1)
        p2 = self.pool2(c2)

        c3 = self.conv3(p2)
        p3 = self.pool3(c3)

        c4 = self.conv4(p3)

        # -------------------------
        # Dynamic Filter Generation
        # -------------------------

        # b = c4.shape[0]

        # k = c4.view(b, -1)
        # k = self.filter_fc(k)

        # k = k.view(b, 1, self.kernel_size, self.kernel_size)

        b = c4.shape[0]

        k = self.gap(c4)      # shape: B × 256 × 1 × 1
        k = k.view(b, 256)

        k = self.filter_fc(k)
        k = k.view(b, 1, self.kernel_size, self.kernel_size)

        # -------------------------
        # Exposure Map Decoder
        # -------------------------

        u3 = self.up3(c4)
        u3 = torch.cat([u3, c3], dim=1)
        d3 = self.dec3(u3)

        u2 = self.up2(d3)
        u2 = torch.cat([u2, c2], dim=1)
        d2 = self.dec2(u2)

        u1 = self.up1(d2)
        u1 = torch.cat([u1, c1], dim=1)
        d1 = self.dec1(u1)

        e = torch.sigmoid(self.exposure_conv(d1))

        # resize exposure map to input size
        e = F.interpolate(e, size=y.shape[-2:], mode="bilinear", align_corners=False)

        # -------------------------
        # Apply Dynamic Filter
        # -------------------------

        outputs = []

        for i in range(b):
            yi = y[i:i+1]
            ki = k[i]

            yi = F.conv2d(yi, ki.unsqueeze(0), padding=self.kernel_size//2)
            outputs.append(yi)

        y_filtered = torch.cat(outputs, dim=0)

        # -------------------------
        # Hadamard Product
        # -------------------------

        y_enhanced = y_filtered * e

        return y_enhanced