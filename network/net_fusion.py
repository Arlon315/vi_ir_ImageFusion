import torch
import torch.nn as nn
import torch.nn.functional as F
# 导入现有的编码器和解码器
from .net_autoencoder import Encoder, Decoder,ConvLayer

# yolo引导的融合模块
class YoloGuidedFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.yolo_feature_map = None  # 存放YOLO特征图（外部输入）
        self.conv_attention = nn.Sequential(
            ConvLayer(channels, channels, kernel_size=3),
            nn.Sigmoid()
        )

    def forward(self, fir, fvis, yolo_feat=None):
        if yolo_feat is not None:
            # 把YOLO特征映射到融合通道
            attention = self.conv_attention(yolo_feat)
        else:
            # 若YOLO特征未提供，退化为均匀融合
            attention = torch.ones_like(fir)

        # 计算加权融合
        fused = attention * fvis + (1 - attention) * fir
        fused = F.relu(fused)
        return fused


# 通道自注意力模块
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


# 简单的特征值相加融合方式
class SimpleAdditionFusion(nn.Module):
    def __init__(self, channels):
        super().__init__()
        """
        简单的特征值相加融合方式

        Args:
            channels (int): 输入特征图的通道数
        """
        # 可学习的权重参数，用于调整两个输入的贡献
        self.weight_ir = nn.Parameter(torch.tensor(0.5))
        self.weight_vi = nn.Parameter(torch.tensor(0.5))

    def forward(self, fir, fvis):
        """
        通过简单的加权求和融合红外线和可见光特征图

        Args:
            fir (torch.Tensor): 红外线编码器输出的特征图
            fvis (torch.Tensor): 可见光编码器输出的特征图

        Returns:
            torch.Tensor: 融合后的特征图
        """
        # 检查输入特征图的尺寸是否匹配
        if fir.shape != fvis.shape:
            raise ValueError("红外线和可见光特征图的尺寸必须相同")

        # 简单的加权求和
        fused_features = self.weight_ir * fir + self.weight_vi * fvis

        # 应用ReLU激活函数
        # fused_features = F.relu(fused_features)

        return fused_features

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
                 fusion_type="simple"):
        self.debug = debug
        self.fusion_type = fusion_type
        super().__init__()
        # self.learnable_fusion = LearnableFusion(channels)
        # 红外编码器
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

        # 可见光编码器
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
        if fusion_type == "simple":
            # 浅层特征融合 (32通道)
            self.shallow_fusion = SimpleAdditionFusion(en_out_conv)

            # 深层特征融合 (128通道)
            self.deep_fusion = SimpleAdditionFusion(dense_out)
        # 解码器
        self.decoder = Decoder(
            in_channels=dense_out,
            kernel_size=kernel_size,
            stride=1,
            debug=debug
        )

    def forward(self, ir_image, vi_image):
        # learnable_fused = self.learnable_fusion(fir, fvis)
        # 编码器提取特征
        # 红外图像特征提取
        ir_conv_out, ir_dense_out = self.ir_encoder(ir_image)
        if self.debug:
            print(f"[红外编码器] 浅层特征: {ir_conv_out.shape}, 深层特征: {ir_dense_out.shape}")

        # 可见光图像特征提取
        vi_conv_out, vi_dense_out = self.vi_encoder(vi_image)
        if self.debug:
            print(f"[可见光编码器] 浅层特征: {vi_conv_out.shape}, 深层特征: {vi_dense_out.shape}")

        # 特征融合
        if self.fusion_type == "simple":
            # 浅层特征融合
            fused_conv_out = self.shallow_fusion(ir_conv_out, vi_conv_out)
            if self.debug:
                print(f"[浅层融合] 融合特征: {fused_conv_out.shape}")

            # 深层特征融合
            fused_dense_out = self.deep_fusion(ir_dense_out, vi_dense_out)
            if self.debug:
                print(f"[深层融合] 融合特征: {fused_dense_out.shape}")
        # 解码器重建图像
        reconstructed = self.decoder(fused_conv_out, fused_dense_out)
        if self.debug:
            print(f"[解码器] 重建图像: {reconstructed.shape}")

        return reconstructed
