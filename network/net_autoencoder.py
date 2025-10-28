custom_config_auto = {
    "in_channels": 1,           # 输入通道数（灰度图像）
    "out_channels": 1,         # 输出通道数
    "en_out_channels1": 32,    # 编码器第一层输出通道
    "en_out_channels": 64,     # 编码器密集块输出通道
    "num_layers": 3,           # 每个密集块的层数
    "dense_out": 128,         # 密集块最终输出通道
    "part_out": 128,          # 编码器最终输出通道
    "train_flag": True,
}

import torch
import torch.nn as nn
import numpy as np


# === 基础卷积层（带反射填充、BN归一化、ReLU） ===
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_bn=False, use_relu=False):
        super().__init__()
        padding = int(np.floor(kernel_size / 2))
        layers = [nn.ReflectionPad2d(padding),
                  nn.Conv2d(in_channels, out_channels, kernel_size, stride)]

        if use_bn:
            layers.append(nn.InstanceNorm2d(out_channels))
        if use_relu:
            layers.append(nn.LeakyReLU(negative_slope= 0.1, inplace=True))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv(x)


# === 密集块中的单层卷积层 ===
class DenseConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_bn=False, use_relu=False):
        super().__init__()
        padding = int(np.floor(kernel_size / 2))
        layers = [
            nn.ReflectionPad2d(padding),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        ]
        if use_bn:
            layers.append(nn.InstanceNorm2d(out_channels))
        if use_relu:
            layers.append(nn.LeakyReLU(negative_slope= 0.1, inplace=True))
        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer(x)
        out = torch.cat((x, out), dim=1)  # 通道维拼接
        return out


# === 密集块（由多个 DenseConvLayer 组成） ===
class DenseBlock(nn.Module):
    def __init__(self, in_channels, dense_Layer_out=32, num_layers=3, out_channels=None, kernel_size=3, stride=1):
        super().__init__()
        self.num_layers = num_layers
        current_channels = in_channels

        for i in range(num_layers):
            layer = DenseConvLayer(
                in_channels=current_channels,
                out_channels=dense_Layer_out,
                kernel_size=kernel_size,
                stride=stride
            )
            self.add_module(f"dense_conv{i+1}", layer)
            current_channels += dense_Layer_out  # 每层通道数增加

        # 输出层调整通道数（用于控制最终输出维度）
        self.adjust_conv = ConvLayer(current_channels, out_channels or current_channels, kernel_size=kernel_size)

    def forward(self, x):
        out = x
        for i in range(self.num_layers):
            dense_conv = getattr(self, f"dense_conv{i+1}")
            out = dense_conv(out)
        out = self.adjust_conv(out)
        return out


# custom_config_auto = {
#     "in_channels": 1,           # 输入通道数（灰度图像）
#     "out_channels": 1,         # 输出通道数
#     "en_out_channels1": 32,    # 编码器第一层输出通道
#     "en_out_channels": 64,     # 编码器密集块输出通道
#     "num_layers": 3,           # 每个密集块的层数
#     "dense_out": 128,         # 密集块最终输出通道
#     "part_out": 128,          # 编码器最终输出通道
#     "train_flag": True,
# }

# === 编码器结构 ===
class Encoder(nn.Module):
    def __init__(self,
                 in_channels=1,# 输入通道数（灰度图像）
                 out_channels=128,# 输出通道数
                 en_out_conv=32, # 编码器第一层输出通道
                 dense_Layer_out=64,# 编码器密集块输出通道
                 dense_layers=3,# 每个密集块的层数
                 dense_out=128,# 编码器最终输出通道
                 kernel_size=3,
                 debug=True):
        super().__init__()
        self.debug = debug

        # 初始卷积层
        self.conv1 = ConvLayer(in_channels, out_channels=en_out_conv, kernel_size=kernel_size,stride=1)

        # 池化层（下采样一次尺寸减半）
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 密集块层
        self.dense_block =  nn.Sequential(
            DenseBlock(
                in_channels=en_out_conv,
                dense_Layer_out=dense_Layer_out,
                num_layers=dense_layers,
                out_channels=dense_out,
                kernel_size=kernel_size
            ),
            # (尺寸减半)
            nn.MaxPool2d(2, 2),
            DenseBlock(
                in_channels=dense_out,
                dense_Layer_out=dense_Layer_out,
                num_layers=dense_layers,
                out_channels=dense_out,
                kernel_size=kernel_size
            ),
            # （尺寸减半）
            nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        conv_out = self.conv1(x)
        if self.debug:
            print(f"[Encoder] conv1_out: {conv_out.shape}")

        out_pooled = self.pool(conv_out)

        dense_out = self.dense_block(out_pooled)
        if self.debug:
            print(f"[Encoder] dense_out: {dense_out.shape}")

        return conv_out, dense_out  # 返回浅层与深层特征

# 解码器
class Decoder(nn.Module):
    def __init__(self,
                 in_channels=128, # 输入通道数
                 kernel_size=3,
                 stride=1,
                 debug=False):
        super().__init__()
        self.debug = debug
        self.conv1 = ConvLayer(int(in_channels / 4), int(in_channels / 4), kernel_size=kernel_size, stride=stride)

        self.up1 = nn.Sequential(
            ConvLayer(in_channels, int(in_channels / 2), kernel_size=kernel_size, stride=stride),
            # (放大一倍)
            nn.Upsample(scale_factor=2),
        )

        self.up2 = nn.Sequential(
            ConvLayer(int(in_channels / 2), int(in_channels / 2), kernel_size=kernel_size, stride=stride),
            # (放大一倍)
            nn.Upsample(scale_factor=2),
        )

        self.up3 = nn.Sequential(
            ConvLayer(int(in_channels / 2), int(in_channels / 4), kernel_size=kernel_size, stride=stride),
        # (放大一倍)
            nn.Upsample(scale_factor=2),
        )

        # self.conv_block = nn.Sequential(
        #     ConvLayer(int(in_channels / 2), int(in_channels / 2), kernel_size, stride),
        #     nn.Upsample(scale_factor=2),
        #     ConvLayer(int(in_channels / 2), int(in_channels / 4), kernel_size, stride),
        #     nn.Upsample(scale_factor=2),
        #     # ConvLayer(int(in_channels / 4), int(in_channels / 8), self.kernel_size, self.stride),
        #     # nn.Upsample(scale_factor=2),
        #     # ConvLayer(int(in_channels / 4), out_channels, self.kernel_size, self.stride)
        # )
        self.conv_last = ConvLayer(int(in_channels / 4), out_channels=1, kernel_size=kernel_size, stride=stride)


    def forward(self,conv_out, dense_out):
        w = [1.0, 1.0]
        # x1 = x1 + w[0] * c_ad
        out = self.up1(dense_out)
        # out = self.conv1(x1)
        # out = out + w[0] * c_ad
        out = self.up2(out)
        out = self.up3(out)
        c1_matched = self.conv1(conv_out)
        out = out + w[1] * c1_matched
        out = self.conv_last(out)

        return out


# ==========================
# 自编码器模型
# ==========================
class AutoEncoder(nn.Module):
    def __init__(self, 
                 in_channels=1,#输入通道
                 out_channels=1,#输出通道
                 en_out_conv=32,#编码器第一层卷积输出通道
                 dense_Layer_out=64,#编码器密集块输出通道
                 dense_layers=3,#每个密集块的层数
                 dense_out=128,#编码器最终输出通道
                 kernel_size=3,#卷积核大小
                 debug=False):
        super().__init__()
        self.debug = debug
        
        # 编码器
        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=dense_out,
            en_out_conv=en_out_conv,
            dense_Layer_out=dense_Layer_out,
            dense_layers=dense_layers,
            dense_out=dense_out,
            kernel_size=kernel_size,
            debug=False
        )
        
        # 解码器
        self.decoder = Decoder(
            in_channels=dense_out,
            kernel_size=kernel_size,
            stride=1,
            debug=False
        )
        
    def forward(self, x):
        # 编码器提取特征
        conv_out, dense_out = self.encoder(x)
        
        # 解码器重建图像
        reconstructed = self.decoder(conv_out, dense_out)
        
        return reconstructed