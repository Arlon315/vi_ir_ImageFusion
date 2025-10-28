import torch
import torch.nn as nn
import numpy as np


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


# class ConvLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, stride, kernel_size):
#         super().__init__()
#         self.reflection = nn.ReflectionPad2d(1)
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride),
#             # nn.BatchNorm2d(out_channels)
#         )
#
#     def forward(self, x):
#         out = self.reflection(x)
#         out = self.conv(out)
#         return out
#
#
# # 密集快的卷积层
# class Dense_ConvLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         super().__init__()
#         reflection_padding = int(np.floor(kernel_size / 2))
#         self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
#         self.conv2d = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size, stride),
#             # nn.BatchNorm2d(out_channels)
#         )
#
#     def forward(self, x):
#         out = self.reflection_pad(x)
#         out = self.conv2d(out)
#         out = torch.cat((x, out), 1)
#         return out
#
#
# class DenseLayer(nn.Module):
#     def __init__(self, in_channels, out_channels, stride, kernel_size, num_layers, dense_out):
#         super().__init__()
#         self.num_layers = num_layers
#         for i in range(num_layers):
#             self.add_module('dense_conv' + str(i),
#                             Dense_ConvLayer(
#                                 in_channels=in_channels + i * out_channels,
#                                 out_channels=out_channels,
#                                 kernel_size=kernel_size,
#                                 stride=stride
#                             )
#                             )
#         self.adjust_conv = ConvLayer(in_channels=in_channels + num_layers * out_channels,
#                                      out_channels=dense_out,
#                                      kernel_size=kernel_size, stride=stride
#                                      )
#
#     def forward(self, x):
#         # 密集块的前向传播
#         out = x
#         print('密集快')
#         for i in range(self.num_layers):
#             print('num_block - ' + str(i))
#             dense_conv = getattr(self, 'dense_conv' + str(i))
#             out = dense_conv(out)
#         out = self.adjust_conv(out)
#         return out
#
#
# class Encoder(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()
#
#         self.kernel_size = 3
#         self.stride = 1
#         self.num_layers = 3  # 密集块的层数
#         self.dense_out = 128  # 密集块的输出通道数
#
#         # 卷积层
#         self.conv1 = ConvLayer(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             stride=self.stride,
#             kernel_size=self.kernel_size
#         )
#         self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
#         # 密集块层
#         self.dense_layer = DenseLayer(
#             in_channels=out_channels,
#             out_channels=self.dense_out,
#             stride=self.stride,
#             kernel_size=self.kernel_size,
#             num_layers=self.num_layers,
#             dense_out=self.dense_out
#         )
#
#     def forward(self, x):
#         conv1_out = self.conv1(x)
#         print(f'conv1_out: {conv1_out.shape}')
#         # 池化层
#         out = self.pool1(conv1_out)
#         print(f'pool1_out: {out.shape}')
#         # 密集块层
#         dense_out = self.dense_layer(out)
#         print(f'dense_out: {dense_out.shape}')
#
#         return conv1_out, dense_out

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




# # ==========================
# # 2️⃣ 模型定义（Hybrid Upsample + ConvTranspose）
# # ==========================
# class AutoEncoder(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # --- Encoder ---
#         self.enc1 = nn.Sequential(
#             nn.Conv2d(1, 16, 3, 2, 1),
#             nn.ReLU(inplace=True)
#         )
#         self.enc2 = nn.Sequential(
#             nn.Conv2d(16, 32, 3, 2, 1),
#             nn.ReLU(inplace=True)
#         )
#         self.enc3 = nn.Sequential(
#             nn.Conv2d(32, 64, 3, 2, 1),
#             nn.ReLU(inplace=True)
#         )
#
#         # --- Decoder ---
#         self.dec3 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(64, 32, 3, 1, 1),
#             nn.ReLU()
#         )
#         self.dec2 = nn.Sequential(
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(32 + 32, 16, 3, 1, 1),
#             nn.ReLU()
#         )
#         self.dec1 = nn.Sequential(
#             # ✅ 再次上采样到原图尺寸
#             nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
#             nn.Conv2d(16 + 16, 1, 3, 1, 1),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         e1 = self.enc1(x)  # [B,16,64,64]
#         e2 = self.enc2(e1)  # [B,32,32,32]
#         e3 = self.enc3(e2)  # [B,64,16,16]
#
#         d3 = self.dec3(e3)  # [B,32,32,32]
#         d2 = self.dec2(torch.cat([d3, e2], dim=1))  # [B,16,64,64]
#         out = self.dec1(torch.cat([d2, e1], dim=1))  # [B,1,128,128]
#         return out
