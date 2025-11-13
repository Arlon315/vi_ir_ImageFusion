import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from network.net_fusion import Fusion


def load_and_preprocess_image(image_path, extract_y_channel=True):
    """加载并预处理单张图像"""
    # 打开图像
    image = Image.open(image_path)

    # 保存原始尺寸
    original_size = image.size  # (width, height)

    # 如果需要提取Y通道，则转换为YCrCb并提取Y分量
    if extract_y_channel and image.mode == 'RGB':
        # 转换为numpy数组
        image_np = np.array(image)
        # 转换为YCrCb色彩空间
        image_ycrcb = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
        # 提取Y通道（亮度通道）
        y_channel = image_ycrcb[:, :, 0]
        # 转换回PIL图像
        image = Image.fromarray(y_channel)
    else:
        # 转换为灰度图
        image = image.convert('L')

    # 定义预处理步骤
    transform = transforms.Compose([
        # transforms.Resize((256, 256)),  # 调整图像尺寸为256x256
        transforms.ToTensor(),  # 转换为tensor并归一化到[0,1]
    ])

    # 应用预处理
    tensor_image = transform(image)

    # 添加batch维度 (1, 1, H, W)
    tensor_image = tensor_image.unsqueeze(0)

    return tensor_image, image, original_size


def restore_color_from_visible(fused_y, visible_original, brightness_factor=1.2):
    """使用可见光图像的色彩信息恢复融合图像的颜色"""
    # 确保输入是numpy数组
    if torch.is_tensor(fused_y):
        fused_y = fused_y.cpu().numpy()

    # 调整融合图像的亮度
    fused_y = np.clip(fused_y * brightness_factor, 0, 1)

    # 获取可见光图像的尺寸
    orig_h, orig_w = visible_original.shape[:2]

    # 调整融合图像尺寸以匹配原始可见光图像
    fused_y_resized = cv2.resize(fused_y, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

    # 将融合结果转换为0-255范围的uint8类型
    fused_y_uint8 = (fused_y_resized * 255).astype(np.uint8)

    # 将可见光图像转换为LAB色彩空间
    visible_lab = cv2.cvtColor(visible_original, cv2.COLOR_RGB2LAB)

    # 用融合的Y通道替换LAB图像的L通道
    visible_lab[:, :, 0] = fused_y_uint8

    # 转换回RGB色彩空间
    color_fused = cv2.cvtColor(visible_lab, cv2.COLOR_LAB2RGB)

    # 转换为PIL图像
    color_fused_pil = Image.fromarray(color_fused)

    return color_fused_pil


def predict_fusion(ir_image_path, vi_image_path, model_path, output_path="fused_result.png", brightness_factor=1.2):
    """使用训练好的融合模型进行预测"""
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建融合模型
    model = Fusion(
        in_channels=1,
        out_channels=1,
        en_out_conv=32,
        dense_Layer_out=64,
        dense_layers=3,
        dense_out=128,
        kernel_size=3,
        debug=False,
        fusion_type="attn",
        attn_temperature=2.0
    )

    # 加载训练好的融合模型权重
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print("已加载融合模型权重")
    else:
        print(f"错误：未找到模型权重文件 {model_path}")
        return

    model.to(device)
    model.eval()

    # 加载并预处理红外和可见光图像
    # 对于红外图像，使用灰度图
    ir_tensor_image, ir_original_image, ir_original_size = load_and_preprocess_image(ir_image_path,
                                                                                     extract_y_channel=False)

    # 对于可见光图像，提取Y通道用于融合，并保留原始彩色图像
    vi_tensor_image, vi_processed_image, vi_original_size = load_and_preprocess_image(vi_image_path,
                                                                                      extract_y_channel=True)

    # 保存原始可见光图像用于颜色恢复
    vi_original = np.array(Image.open(vi_image_path).convert('RGB'))

    ir_tensor_image = ir_tensor_image.to(device)
    vi_tensor_image = vi_tensor_image.to(device)

    print(f"红外图像尺寸: {ir_tensor_image.shape}")
    print(f"可见光图像尺寸: {vi_tensor_image.shape}")

    # 使用模型进行预测
    with torch.no_grad():
        fused_image = model(ir_tensor_image, vi_tensor_image)

    print(f"融合图像尺寸: {fused_image.shape}")

    # 将融合结果转换为numpy数组用于颜色恢复
    fused_np = fused_image.squeeze(0).squeeze(0).cpu().numpy()

    # 调整融合结果到原始尺寸
    fused_np_resized = cv2.resize(fused_np, (vi_original_size[0], vi_original_size[1]), interpolation=cv2.INTER_LINEAR)

    # 使用可见光图像的色彩信息恢复融合图像的颜色
    color_fused = restore_color_from_visible(fused_np, vi_original, brightness_factor)

    # 保存彩色融合图像
    color_fused.save(output_path)
    print(f"彩色融合图像已保存到: {output_path}")

    # 保存对比图像（包含红外、可见光和融合图像）
    # 创建对比图像
    comparison_output_path = output_path.replace(".png", "_comparison.png")

    # 转换红外图像到原始尺寸
    ir_original_np = np.array(ir_original_image)
    ir_resized = cv2.resize(ir_original_np, (vi_original_size[0], vi_original_size[1]), interpolation=cv2.INTER_LINEAR)

    # 转换可见光处理后的图像到原始尺寸
    vi_processed_np = np.array(vi_processed_image)
    vi_resized = cv2.resize(vi_processed_np, (vi_original_size[0], vi_original_size[1]), interpolation=cv2.INTER_LINEAR)

    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # 显示红外图像
    axes[0].imshow(ir_resized, cmap='gray')
    axes[0].set_title('Infrared Image')
    axes[0].axis('off')

    # 显示可见光图像
    axes[1].imshow(vi_resized, cmap='gray')
    axes[1].set_title('Visible Image')
    axes[1].axis('off')

    # 显示融合图像
    axes[2].imshow(fused_np_resized, cmap='gray')
    axes[2].set_title('Fused Image')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(comparison_output_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"对比图像已保存到: {comparison_output_path}")

    return fused_image


if __name__ == "__main__":
    # 模型路径
    model_path = r"weights/fusion/fusion_model_epoch02_16.pth"  # 使用weights/fusion目录中的模型

    # 测试图像路径（请根据实际情况修改）
    ir_image_path = r"image/testNet/260518_ir.jpg"  # 红外图像路径
    vi_image_path = r"image/testNet/260518_vi.jpg"  # 可见光图像路径

    # 输出图像路径
    output_path = r"output/predict/fusion_result.png"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 亮度增强因子
    brightness_factor = 1.2  # 增加亮度20%

    # 进行融合预测
    try:
        print("开始图像融合预测...")
        fused_result = predict_fusion(
            ir_image_path, vi_image_path,
            model_path, output_path,
            brightness_factor
        )
        print("图像融合预测完成!")
    except Exception as e:
        print(f"预测过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
