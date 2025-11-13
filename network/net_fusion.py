import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入现有的编码器和解码器
from .net_autoencoder import Encoder, Decoder, ConvLayer




# -----------------------------
# 注意力融合（稳定初始化 + 温度）
# -----------------------------
class AttnFusion(nn.Module):
    """
    att = sigmoid(Conv([ir, vis]) / T)
    fused = att * ir + (1 - att) * vis
    """
    def __init__(self, channels, temperature=2.0):
        super().__init__()
        self.temperature = temperature
        self.conv = nn.Conv2d(channels * 2, channels, kernel_size=3, padding=1, bias=True)
        nn.init.zeros_(self.conv.bias)  # 初始 att≈0.5 防止早期塌缩

    def forward(self, fir, fvis):
        x = torch.cat([fir, fvis], dim=1)
        logit = self.conv(x)
        att = torch.sigmoid(logit / self.temperature)
        fused = att * fir + (1.0 - att) * fvis
        return fused, att

# -----------------------------
# 你的可学习加权（保留）
# -----------------------------
class LearnableFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight_net = nn.Sequential(
            ConvLayer(channels * 2, channels, kernel_size=3),
            nn.Sigmoid()
        )
        self.conv = ConvLayer(channels, channels, kernel_size=3)

    def forward(self, fir, fvis):
        w = self.weight_net(torch.cat([fir, fvis], dim=1))
        fused = w * fir + (1 - w) * fvis
        fused = F.relu(self.conv(fused))
        return fused

# -----------------------------
# 你的简单加权（保留）
# -----------------------------
class SimpleAdditionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.weight_ir = nn.Parameter(torch.tensor(0.5))
        self.weight_vi = nn.Parameter(torch.tensor(0.5))

    def forward(self, fir, fvis):
        if fir.shape != fvis.shape:
            raise ValueError("红外线和可见光特征图的尺寸必须相同")
        fused_features = self.weight_ir * fir + self.weight_vi * fvis
        return fused_features

# 红外引导融合--可见光为主体，红外线做残差注入
class IRGuidedFusion(nn.Module):
    """
    以可见光特征为主体，IR 只通过注意力在局部注入
    fused = fvis + M * (fir - fvis)
    """
    def __init__(self, channels):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, fir, fvis):
        # 注意力完全由 IR 特征决定（只看 IR 哪些区域重要）
        M = self.att(fir)             # [B,C,H,W] ∈ (0,1)

        # 残差式融合：VIS 为主体，IR 在 M 高的地方“拉过去”
        fused = fvis + M * (fir - fvis)
        return fused, M

# -----------------------------
# 主模型：双编码器 + 层级融合 + 解码器
# -----------------------------
class Fusion(nn.Module):
    def __init__(self,
                 in_channels=1,
                 en_out_conv=32,
                 dense_out=128,
                 out_channels=1,
                 dense_Layer_out=64,
                 dense_layers=3,
                 kernel_size=3,
                 debug=False,
                 fusion_type="simple",
                 attn_temperature=2.0):
        super().__init__()
        self.debug = debug
        self.fusion_type = fusion_type

        # 两个编码器
        self.ir_encoder = Encoder(
            in_channels=in_channels,
            out_channels=dense_out,
            en_out_conv=en_out_conv,
            dense_Layer_out=dense_Layer_out,
            dense_layers=dense_layers,
            dense_out=dense_out,
            kernel_size=kernel_size,
            debug=debug
        )
        self.vi_encoder = Encoder(
            in_channels=in_channels,
            out_channels=dense_out,
            en_out_conv=en_out_conv,
            dense_Layer_out=dense_Layer_out,
            dense_layers=dense_layers,
            dense_out=dense_out,
            kernel_size=kernel_size,
            debug=debug
        )

        # 层级融合（支持 simple / attn / learnable）
        if fusion_type == "simple":
            self.shallow_fusion = SimpleAdditionFusion(en_out_conv)
            self.deep_fusion = SimpleAdditionFusion(dense_out)
        elif fusion_type == "attn":
            self.shallow_fusion = AttnFusion(en_out_conv, temperature=attn_temperature)
            self.deep_fusion = AttnFusion(dense_out, temperature=attn_temperature)
        elif fusion_type == "learnable":
            self.shallow_fusion = LearnableFusion(en_out_conv)
            self.deep_fusion = LearnableFusion(dense_out)
        elif fusion_type == "ir_guided":
            self.shallow_fusion = IRGuidedFusion(en_out_conv)  # 32 通道
            self.deep_fusion = IRGuidedFusion(dense_out)  # 128 通道
        else:
            raise ValueError(f"未知的融合类型: {fusion_type}")

        # 解码器
        self.decoder = Decoder(
            in_channels=dense_out,
            kernel_size=kernel_size,
            stride=1,
            debug=debug
        )

    def forward(self, ir_image, vi_image):
        ir_conv_out, ir_dense_out = self.ir_encoder(ir_image)
        vi_conv_out, vi_dense_out = self.vi_encoder(vi_image)

        if self.fusion_type == "simple":
            fused_conv_out = self.shallow_fusion(ir_conv_out, vi_conv_out)
            fused_dense_out = self.deep_fusion(ir_dense_out, vi_dense_out)
            shallow = deep = None
        elif self.fusion_type == "attn" or self.fusion_type == "ir_guided":
            fused_conv_out, shallow = self.shallow_fusion(ir_conv_out, vi_conv_out)
            fused_dense_out, deep = self.deep_fusion(ir_dense_out, vi_dense_out)
        else:  # learnable
            fused_conv_out = self.shallow_fusion(ir_conv_out, vi_conv_out)
            fused_dense_out = self.deep_fusion(ir_dense_out, vi_dense_out)
            shallow = deep = None

        reconstructed = self.decoder(fused_conv_out, fused_dense_out)
        if self.debug:
            return reconstructed, {"shallow": shallow, "deep": deep}
        return reconstructed
