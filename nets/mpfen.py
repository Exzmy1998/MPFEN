"""Paper-first MPFEN. See README-LWTYPE.md for every underspecified choice.

Standalone: this file does not import or modify the previous MPFEN implementation.
Public entry: MPFENLWTYPE; returns two-class logits, not a thresholded mask.
"""
import torch
from torch import nn
from torch.nn import functional as F


def resize(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ConvBNReLU(nn.Sequential):
    def __init__(self, cin, cout, kernel=1, stride=1, groups=1, dilation=1, relu6=False):
        super().__init__(
            nn.Conv2d(cin, cout, kernel, stride, padding=dilation*(kernel//2),
                      dilation=dilation, groups=groups, bias=False),
            nn.BatchNorm2d(cout),
            nn.ReLU6(inplace=False) if relu6 else nn.ReLU(inplace=False),
        )


class InvertedResidual(nn.Module):
    """MobileNetV2 expansion -> depthwise -> linear projection, optional shortcut."""
    def __init__(self, cin, cout, stride=1, expansion=6):
        super().__init__()
        hidden = cin * expansion
        layers = []
        if expansion != 1:
            layers.append(ConvBNReLU(cin, hidden, relu6=True))
        layers.extend([
            ConvBNReLU(hidden, hidden, 3, stride, groups=hidden, relu6=True),
            nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout),
        ])
        self.branch = nn.Sequential(*layers)
        self.residual = stride == 1 and cin == cout

    def forward(self, x):
        y = self.branch(x)
        return x + y if self.residual else y


class PaperBackbone(nn.Module):
    """MobileNetV2 through the 96-channel stage; explicit 32->48 lateral adapter.

    Output order follows the manuscript: F0 (coarse), F1, F2 (fine).
    At 512: [N,96,32,32], [N,48,64,64], [N,24,128,128].
    """
    def __init__(self):
        super().__init__()
        self.stem = ConvBNReLU(3, 32, 3, 2, relu6=True)
        stages, cin = [], 32
        for expansion, cout, repeats, stride in [(1,16,1,1), (6,24,2,2),
                                                (6,32,3,2), (6,64,4,2), (6,96,3,1)]:
            blocks = []
            for i in range(repeats):
                blocks.append(InvertedResidual(cin, cout, stride if i == 0 else 1, expansion))
                cin = cout
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.ModuleList(stages)
        self.middle_adapter = ConvBNReLU(32, 48)

    def forward(self, x):
        x = self.stem(x)
        x = self.stages[0](x)
        f2 = self.stages[1](x)
        raw_middle = self.stages[2](f2)
        f1 = self.middle_adapter(raw_middle)
        f0 = self.stages[4](self.stages[3](raw_middle))
        return f0, f1, f2


class SobelMagnitude(nn.Module):
    """Fixed channelwise Sobel, registered as buffers so initialization cannot overwrite it."""
    def __init__(self):
        super().__init__()
        self.register_buffer("kx", torch.tensor([[-1.,0.,1.],[-2.,0.,2.],[-1.,0.,1.]])[None,None])
        self.register_buffer("ky", torch.tensor([[-1.,-2.,-1.],[0.,0.,0.],[1.,2.,1.]])[None,None])

    def forward(self, x):
        c = x.shape[1]
        padded = F.pad(x, (1,1,1,1), mode="replicate")
        gx = F.conv2d(padded, self.kx.expand(c,1,3,3), groups=c)
        gy = F.conv2d(padded, self.ky.expand(c,1,3,3), groups=c)
        # vector_norm implements sqrt(gx^2+gy^2) with a finite zero subgradient.
        return torch.linalg.vector_norm(torch.stack((gx, gy), dim=0), dim=0)


class BPU(nn.Module):
    """Eq.3 and following prose: cat(input, edge) -> 1x1 -> BN -> ReLU."""
    def __init__(self, channels):
        super().__init__()
        self.sobel = SobelMagnitude()
        self.fusion = ConvBNReLU(2*channels, channels)

    def forward(self, x):
        return self.fusion(torch.cat((x, self.sobel(x)), dim=1))


class DiagonalConv(nn.Module):
    """Learned kernel with a permanent main-/anti-diagonal support mask."""
    def __init__(self, cin, cout, kernel=7, anti=False):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(cout, cin, kernel, kernel))
        mask = torch.eye(kernel)
        self.register_buffer("mask", (mask.flip(1) if anti else mask)[None,None])
        self.padding = kernel // 2
        nn.init.kaiming_normal_(self.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        return F.conv2d(x, self.weight*self.mask, padding=self.padding)


class DGG(nn.Module):
    """Four directional branches; concatenate -> 1x1 fusion -> BN -> ReLU."""
    def __init__(self, channels=48, kernel=7):
        super().__init__()
        if channels % 4:
            raise ValueError("DGG channels must be divisible by four")
        quarter = channels // 4
        self.branches = nn.ModuleList([
            nn.Conv2d(channels, quarter, (1,kernel), padding=(0,kernel//2), bias=False),
            nn.Conv2d(channels, quarter, (kernel,1), padding=(kernel//2,0), bias=False),
            DiagonalConv(channels, quarter, kernel),
            DiagonalConv(channels, quarter, kernel, anti=True),
        ])
        self.fusion = ConvBNReLU(channels, channels)

    def forward(self, x):
        return self.fusion(torch.cat([branch(x) for branch in self.branches], dim=1))


class TDE(nn.Module):
    """Text Eq.5: I is each channel of the incoming high-level feature.

    Wt=sigmoid(alpha*abs(I-local_mean(I))); output=I*Wt, no residual blend.
    """
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def weights(self, x):
        mean = F.avg_pool2d(F.pad(x, (1,1,1,1), mode="replicate"), 3, stride=1)
        return torch.sigmoid(self.alpha * (x-mean).abs())

    def forward(self, x):
        return x * self.weights(x)


class MFEM(nn.Module):
    """BPU -> middle-scale fusion + IRB -> DGG -> fine-scale fusion + IRB -> TDE.

    Explicit adapters resolve missing cross-scale operators in Fig.2.
    Flow is the resized/projected BPU branch in Fig.1; Fhigh is 96 channels at /16.
    """
    def __init__(self):
        super().__init__()
        self.bpu = BPU(96)
        self.low_adapter = ConvBNReLU(96, 24)
        self.middle_fusion = nn.Sequential(ConvBNReLU(96+48, 48), InvertedResidual(48,48))
        self.dgg = DGG(48)
        self.fine_fusion = nn.Sequential(ConvBNReLU(48+24, 24), InvertedResidual(24,24))
        self.high_adapter = ConvBNReLU(24, 96)
        self.tde = TDE()

    def forward(self, f0, f1, f2):
        boundary = self.bpu(f0)
        low = self.low_adapter(resize(boundary, f2.shape[-2:]))
        middle = self.middle_fusion(torch.cat((resize(boundary, f1.shape[-2:]), f1), dim=1))
        directional = self.dgg(middle)
        fine = self.fine_fusion(torch.cat((resize(directional, f2.shape[-2:]), f2), dim=1))
        high_input = self.high_adapter(F.adaptive_avg_pool2d(fine, f0.shape[-2:]))
        return low, self.tde(high_input)


class MSFE(nn.Module):
    """Eq.6: each ordinary dilated 3x3 sees all previous outputs plus H0."""
    def __init__(self, cin=96, cout=256, growth=32, rates=(3,6,12,18,24)):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvBNReLU(cin+i*growth, growth, 3, dilation=rate)
            for i, rate in enumerate(rates)
        ])
        self.fusion = ConvBNReLU(cin+len(rates)*growth, cout)

    def forward(self, x):
        features = [x]
        for block in self.blocks:
            features.append(block(torch.cat(features, dim=1)))
        return self.fusion(torch.cat(features, dim=1))


class HSA(nn.Module):
    """Eqs.7-8: shared MLP(GAP)+MLP(GMP), then spatial mean/max and 7x7."""
    def __init__(self, channels=256, reduction=16):
        super().__init__()
        hidden = max(1, channels//reduction)
        self.shared_mlp = nn.Sequential(nn.Conv2d(channels,hidden,1), nn.ReLU(),
                                        nn.Conv2d(hidden,channels,1))
        self.spatial = nn.Conv2d(2,1,7,padding=3,bias=False)

    def forward(self, x):
        wc = torch.sigmoid(self.shared_mlp(F.adaptive_avg_pool2d(x,1)) +
                           self.shared_mlp(F.adaptive_max_pool2d(x,1)))
        channel_feature = x * wc
        spatial_input = torch.cat((channel_feature.mean(1,keepdim=True),
                                    channel_feature.amax(1,keepdim=True)), dim=1)
        ws = torch.sigmoid(self.spatial(spatial_input))
        return channel_feature * ws


class LCFFEM(nn.Sequential):
    def __init__(self):
        super().__init__(MSFE(), HSA())


class EGFFM(nn.Module):
    """Eqs.9-11: low-feature Sobel -> Norm -> gate upsampled deep -> cat low + residual."""
    def __init__(self, low_channels=24, deep_channels=256, out_channels=256):
        super().__init__()
        self.sobel = SobelMagnitude()
        self.fusion = nn.Sequential(ConvBNReLU(low_channels+deep_channels,out_channels,3),
                                     ConvBNReLU(out_channels,out_channels,3))
        self.residual = nn.Conv2d(low_channels,out_channels,1,bias=False)

    def edge_weights(self, low):
        magnitude = self.sobel(low).mean(1,keepdim=True)
        minimum = magnitude.amin(dim=(-2,-1),keepdim=True)
        maximum = magnitude.amax(dim=(-2,-1),keepdim=True)
        return (magnitude-minimum)/(maximum-minimum).clamp_min(1e-6)

    def forward(self, low, deep):
        deep_attention = resize(deep, low.shape[-2:]) * self.edge_weights(low)
        return self.fusion(torch.cat((low, deep_attention),dim=1)) + self.residual(low)


class PlainFusion(nn.Module):
    """Ablation replacement: same output shape without Sobel gating or low residual."""
    def __init__(self):
        super().__init__()
        self.fusion = nn.Sequential(ConvBNReLU(24+256,256,3),ConvBNReLU(256,256,3))

    def forward(self, low, deep):
        return self.fusion(torch.cat((low,resize(deep,low.shape[-2:])),dim=1))


class MPFENLWTYPE(nn.Module):
    architecture = "MPFEN-LWTYPE-text-v1"

    def __init__(self, num_classes=2, use_mfem=True, use_lcffem=True, use_egffm=True):
        super().__init__()
        if num_classes < 2:
            raise ValueError("Use >=2 logits for CE; binary segmentation uses two classes")
        self.config = dict(num_classes=num_classes, use_mfem=use_mfem,
                           use_lcffem=use_lcffem, use_egffm=use_egffm)
        self.backbone = PaperBackbone()
        self.mfem = MFEM() if use_mfem else None
        self.lcffem = LCFFEM() if use_lcffem else ConvBNReLU(96,256)
        self.decoder = EGFFM() if use_egffm else PlainFusion()
        self.classifier = nn.Conv2d(256,num_classes,1)

    def forward(self, x, return_features=False):
        if x.ndim != 4 or x.shape[1] != 3:
            raise ValueError("Expected NCHW RGB input")
        f0, f1, f2 = self.backbone(x)
        low, high = self.mfem(f0,f1,f2) if self.mfem is not None else (f2,f0)
        deep = self.lcffem(high)
        fused = self.decoder(low,deep)
        logits = resize(self.classifier(fused),x.shape[-2:])
        if return_features:
            return logits, dict(F0=f0,F1=f1,F2=f2,F_low=low,F_high=high,F_deep=deep,F_fused=fused)
        return logits


class MPFEN(MPFENLWTYPE):
    """Paper architecture behind the unchanged repository training/prediction API.

    downsample_factor is retained for call compatibility; paper scales are fixed
    at 1/16, 1/8 and 1/4. Both legacy values construct identical checkpoints.
    """
    def __init__(self, num_classes=2, backbone="mobilenet", pretrained=False,
                 downsample_factor=16, use_crack_aware=True,
                 use_directional=True, use_texture=True):
        if backbone != "mobilenet":
            raise ValueError("The confirmed paper architecture uses mobilenet only")
        if pretrained:
            raise ValueError("No compatible pretrained paper-backbone weights; use pretrained=False")
        if downsample_factor not in (8,16):
            raise ValueError("Legacy downsample_factor must be 8 or 16")
        super().__init__(num_classes=num_classes,use_mfem=use_crack_aware,
                         use_lcffem=use_crack_aware,use_egffm=use_crack_aware)
        self.use_crack_aware=use_crack_aware
        self.use_directional=use_directional and use_crack_aware
        self.use_texture=use_texture and use_crack_aware
        self.requested_downsample_factor=downsample_factor
        self.effective_downsample_factor=16
        if self.mfem is not None:
            if not use_directional:self.mfem.dgg=nn.Identity()
            if not use_texture:self.mfem.tde=nn.Identity()


def create_crack_aware_mpfen(num_classes=2, backbone="mobilenet", pretrained=False,
                            downsample_factor=16,use_directional=True,use_texture=True):
    return MPFEN(num_classes,backbone,pretrained,downsample_factor,True,use_directional,use_texture)


def create_standard_mpfen(num_classes=2, backbone="mobilenet", pretrained=False,downsample_factor=16):
    return MPFEN(num_classes,backbone,pretrained,downsample_factor,False)
