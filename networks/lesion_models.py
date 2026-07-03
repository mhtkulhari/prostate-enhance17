import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_ch, out_ch),
            ConvBNReLU(out_ch, out_ch),
        )

    def forward(self, x):
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv2(self.conv1(x)) + self.skip(x))


class AttentionGate(nn.Module):
    def __init__(self, gate_ch, skip_ch, inter_ch):
        super().__init__()
        self.gate_proj = nn.Sequential(
            nn.Conv2d(gate_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.skip_proj = nn.Sequential(
            nn.Conv2d(skip_ch, inter_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_ch),
        )
        self.psi = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(inter_ch, 1, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, gate, skip):
        if gate.shape[-2:] != skip.shape[-2:]:
            gate = F.interpolate(gate, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        attn = self.psi(self.gate_proj(gate) + self.skip_proj(skip))
        return skip * attn


class UpBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch, residual=False, attention=False):
        super().__init__()
        self.attention = AttentionGate(in_ch, skip_ch, max(out_ch // 2, 16)) if attention else None
        self.conv = ResidualBlock(in_ch + skip_ch, out_ch) if residual else DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        if self.attention is not None:
            skip = self.attention(x, skip)
        return self.conv(torch.cat([x, skip], dim=1))


class ResUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, base_ch=64):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]
        self.e1 = ResidualBlock(n_channels, ch[0])
        self.e2 = ResidualBlock(ch[0], ch[1])
        self.e3 = ResidualBlock(ch[1], ch[2])
        self.e4 = ResidualBlock(ch[2], ch[3])
        self.b = ResidualBlock(ch[3], ch[4])
        self.pool = nn.MaxPool2d(2)
        self.u4 = UpBlock(ch[4], ch[3], ch[3], residual=True)
        self.u3 = UpBlock(ch[3], ch[2], ch[2], residual=True)
        self.u2 = UpBlock(ch[2], ch[1], ch[1], residual=True)
        self.u1 = UpBlock(ch[1], ch[0], ch[0], residual=True)
        self.out = nn.Conv2d(ch[0], n_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.b(self.pool(e4))
        x = self.u4(b, e4)
        x = self.u3(x, e3)
        x = self.u2(x, e2)
        x = self.u1(x, e1)
        return self.out(x)


class AttUNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, base_ch=64):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]
        self.e1 = DoubleConv(n_channels, ch[0])
        self.e2 = DoubleConv(ch[0], ch[1])
        self.e3 = DoubleConv(ch[1], ch[2])
        self.e4 = DoubleConv(ch[2], ch[3])
        self.b = DoubleConv(ch[3], ch[4])
        self.pool = nn.MaxPool2d(2)
        self.u4 = UpBlock(ch[4], ch[3], ch[3], attention=True)
        self.u3 = UpBlock(ch[3], ch[2], ch[2], attention=True)
        self.u2 = UpBlock(ch[2], ch[1], ch[1], attention=True)
        self.u1 = UpBlock(ch[1], ch[0], ch[0], attention=True)
        self.out = nn.Conv2d(ch[0], n_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.b(self.pool(e4))
        x = self.u4(b, e4)
        x = self.u3(x, e3)
        x = self.u2(x, e2)
        x = self.u1(x, e1)
        return self.out(x)


class UNetPP(nn.Module):
    def __init__(self, n_channels=1, n_classes=2, base_ch=64):
        super().__init__()
        nb = base_ch
        self.pool = nn.MaxPool2d(2)
        self.x00 = DoubleConv(n_channels, nb)
        self.x10 = DoubleConv(nb, nb * 2)
        self.x20 = DoubleConv(nb * 2, nb * 4)
        self.x30 = DoubleConv(nb * 4, nb * 8)
        self.x40 = DoubleConv(nb * 8, nb * 16)

        self.x01 = DoubleConv(nb + nb * 2, nb)
        self.x11 = DoubleConv(nb * 2 + nb * 4, nb * 2)
        self.x21 = DoubleConv(nb * 4 + nb * 8, nb * 4)
        self.x31 = DoubleConv(nb * 8 + nb * 16, nb * 8)

        self.x02 = DoubleConv(nb * 2 + nb * 2, nb)
        self.x12 = DoubleConv(nb * 4 + nb * 4, nb * 2)
        self.x22 = DoubleConv(nb * 8 + nb * 8, nb * 4)

        self.x03 = DoubleConv(nb * 3 + nb * 2, nb)
        self.x13 = DoubleConv(nb * 6 + nb * 4, nb * 2)

        self.x04 = DoubleConv(nb * 4 + nb * 2, nb)
        self.out = nn.Conv2d(nb, n_classes, kernel_size=1)

    @staticmethod
    def _up(x, ref):
        return F.interpolate(x, size=ref.shape[-2:], mode='bilinear', align_corners=False)

    def forward(self, x):
        x00 = self.x00(x)
        x10 = self.x10(self.pool(x00))
        x20 = self.x20(self.pool(x10))
        x30 = self.x30(self.pool(x20))
        x40 = self.x40(self.pool(x30))

        x01 = self.x01(torch.cat([x00, self._up(x10, x00)], dim=1))
        x11 = self.x11(torch.cat([x10, self._up(x20, x10)], dim=1))
        x21 = self.x21(torch.cat([x20, self._up(x30, x20)], dim=1))
        x31 = self.x31(torch.cat([x30, self._up(x40, x30)], dim=1))

        x02 = self.x02(torch.cat([x00, x01, self._up(x11, x00)], dim=1))
        x12 = self.x12(torch.cat([x10, x11, self._up(x21, x10)], dim=1))
        x22 = self.x22(torch.cat([x20, x21, self._up(x31, x20)], dim=1))

        x03 = self.x03(torch.cat([x00, x01, x02, self._up(x12, x00)], dim=1))
        x13 = self.x13(torch.cat([x10, x11, x12, self._up(x22, x10)], dim=1))

        x04 = self.x04(torch.cat([x00, x01, x02, x03, self._up(x13, x00)], dim=1))
        return self.out(x04)


class SCSEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )
        self.sse = nn.Sequential(
            nn.Conv2d(channels, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.cse(x) + x * self.sse(x)


class SCSEConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.scse = SCSEBlock(out_ch)

    def forward(self, x):
        return self.scse(self.conv(x))


class SCSEUNet(nn.Module):
    """Lightweight attention U-Net using spatial+channel SE blocks.

    This is used as the replacement for TransUNet here because it is usually
    safer for small, low-contrast prostate lesions when training data is limited.
    """
    def __init__(self, n_channels=1, n_classes=2, base_ch=64):
        super().__init__()
        ch = [base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16]
        self.pool = nn.MaxPool2d(2)
        self.e1 = SCSEConv(n_channels, ch[0])
        self.e2 = SCSEConv(ch[0], ch[1])
        self.e3 = SCSEConv(ch[1], ch[2])
        self.e4 = SCSEConv(ch[2], ch[3])
        self.b = SCSEConv(ch[3], ch[4])
        self.u4 = UpBlock(ch[4], ch[3], ch[3])
        self.a4 = SCSEBlock(ch[3])
        self.u3 = UpBlock(ch[3], ch[2], ch[2])
        self.a3 = SCSEBlock(ch[2])
        self.u2 = UpBlock(ch[2], ch[1], ch[1])
        self.a2 = SCSEBlock(ch[1])
        self.u1 = UpBlock(ch[1], ch[0], ch[0])
        self.a1 = SCSEBlock(ch[0])
        self.out = nn.Conv2d(ch[0], n_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        b = self.b(self.pool(e4))
        x = self.a4(self.u4(b, e4))
        x = self.a3(self.u3(x, e3))
        x = self.a2(self.u2(x, e2))
        x = self.a1(self.u1(x, e1))
        return self.out(x)


def _weights_arg(value):
    if value is None:
        return None
    if str(value).strip().lower() in {'none', 'null', 'no', ''}:
        return None
    return value


def build_segmentation_model(model_name, n_channels, n_classes, patch_size=224,
                             encoder_name=None, encoder_weights=None):
    name = str(model_name).lower()

    if name == 'unet':
        from networks.unet_model import UNet
        return UNet(n_channels=n_channels, n_classes=n_classes)

    if name == 'unetpp':
        return UNetPP(n_channels=n_channels, n_classes=n_classes, base_ch=64)

    if name == 'attunet':
        return AttUNet(n_channels=n_channels, n_classes=n_classes, base_ch=64)

    if name == 'resunet':
        return ResUNet(n_channels=n_channels, n_classes=n_classes, base_ch=64)

    if name == 'scseunet':
        return SCSEUNet(n_channels=n_channels, n_classes=n_classes, base_ch=64)

    if name in {'fpn', 'deeplabv3plus', 'segformer'}:
        try:
            import segmentation_models_pytorch as smp
        except ImportError as exc:
            raise ImportError(
                "Model '{}' needs segmentation_models_pytorch and timm. Install with: "
                "pip install segmentation-models-pytorch timm".format(name)
            ) from exc

        weights = _weights_arg(encoder_weights)
        if name == 'segformer':
            enc = encoder_name or 'mit_b0'
            return smp.Segformer(
                encoder_name=enc,
                encoder_weights=weights,
                in_channels=n_channels,
                classes=n_classes,
                activation=None,
            )
        enc = encoder_name or 'resnet34'
        if name == 'fpn':
            return smp.FPN(
                encoder_name=enc,
                encoder_weights=weights,
                in_channels=n_channels,
                classes=n_classes,
                activation=None,
            )
        return smp.DeepLabV3Plus(
            encoder_name=enc,
            encoder_weights=weights,
            in_channels=n_channels,
            classes=n_classes,
            activation=None,
        )

    raise ValueError(
        "Unknown model '{}'. Use one of: unet, fpn, deeplabv3plus, segformer, "
        "unetpp, attunet, resunet, scseunet.".format(model_name)
    )
