#%%
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from network.net_autoencoder import Encoder
#%%
# 图像预处理转换
transform = transforms.Compose([
    # transforms.Resize((128, 128)),  # 调整到模型输入尺寸
    transforms.Grayscale(),         # 转换为灰度图
    transforms.ToTensor(),          # 转换为张量
])

#%%
def load_and_preprocess_image(image_path):
    """加载并预处理单张图片"""
    image = Image.open(image_path)
    print(f"原始图片尺寸: {image.size}")
    print(f"原始图片模式: {image.mode}")
    image_tensor = transform(image).unsqueeze(0)  # 添加batch维度
    print(f"预处理后的图片尺寸: {image_tensor.shape}")
    return image_tensor

#%%
def visualize_feature_maps(tensor, title, num_channels=10):
    """可视化特征图"""
    tensor = tensor[0, :num_channels, :, :].detach().cpu().numpy()
    plt.figure(figsize=(12, 4))
    for i in range(num_channels):
        plt.subplot(2, 5, i + 1)
        plt.imshow(tensor[i], cmap='gray')
        plt.title(f"{title} - Ch {i+1}")
        plt.axis('off')
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()
#%%
def test_encoder_output():
    """测试encoder输出"""
    # 1. 加载模型
    model = Encoder(in_channels=1, base_channels=32)
    model.eval()  # 设置为评估模式


    # 2. 加载图片（请替换为您的图片路径）
    # 示例：使用项目中的测试图片或指定路径
    image_path = r"../image/testNet/portrait.jpg"  # 请修改为实际路径
    try:
        input_tensor = load_and_preprocess_image(image_path)
        print(f"输入图片尺寸: {input_tensor.shape}")
    except FileNotFoundError:
        print(f"图片路径不存在: {image_path}")
        # 创建一个随机测试图片
        input_tensor = torch.randn(1, 1, 128, 128)
        print("使用随机测试数据")

    # GPU测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    model.to(device)
    input_tensor = input_tensor.to(device)



    # 3. 前向传播
    with torch.no_grad():
        conv1_out, dense_out = model(input_tensor)
        # print(f"conv1_out 尺寸: {conv1_out.shape}")
        # print(f"dense_out 尺寸: {dense_out.shape}")
        # 取前10个通道
        conv1_out = conv1_out[:, :10, :, :]
        dense_out = dense_out[:, :10, :, :]
        # 可视化 conv1_out 的每个通道
        # visualize_feature_maps(conv1_out, "Conv1 输出特征", num_channels=10)
        # visualize_feature_maps(dense_out, "Dense 输出特征", num_channels=10)

        plt.hist(conv1_out.cpu().flatten(), bins=50)
        plt.title("Conv1 Feature Distribution")
        plt.show()
        import torch.nn.functional as F

        feat_map = dense_out.mean(dim=1, keepdim=True)  # 所有通道取平均
        feat_map = F.interpolate(feat_map, size=(128, 128), mode='bilinear')
        plt.imshow(feat_map[0,0].cpu(), cmap='jet')
        plt.title("Feature Activation Heatmap")
        plt.colorbar()
        plt.show()
        plt.imshow(conv1_out.mean(1)[0].cpu(), cmap='jet')
        plt.title("Conv1 Mean Feature")
        plt.colorbar()
        plt.show()

        print("mean:", conv1_out.mean().item())
        print("std:", conv1_out.std().item())
        print("sparsity:", torch.mean((conv1_out==0).float()).item())

        plt.hist(conv1_out.flatten().cpu().numpy(), bins=100)
        plt.title("Conv1 Activation Distribution (After LeakyReLU + InstanceNorm)")
        plt.show()




if __name__ == "__main__":
    test_encoder_output()