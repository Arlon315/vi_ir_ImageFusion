import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np

# 导入融合网络
from network.net_fusion import Fusion


def enhance_brightness(image, factor=1.2):
    """
    增强图像亮度
    Args:
        image: 输入图像 (numpy数组)
        factor: 亮度增强因子，大于1表示增加亮度
    Returns:
        enhanced_image: 增强后的图像
    """
    # 确保factor在合理范围内
    factor = max(0.1, min(factor, 3.0))

    # 增强亮度
    enhanced_image = image.astype(np.float32) * factor

    # 确保像素值在[0, 255]范围内
    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)

    return enhanced_image


def restore_color_from_visible(fused_gray, visible_rgb, brightness_factor=1.2):
    """
    使用可见光的颜色信息恢复灰度融合图像的色彩 (使用LAB色彩空间)
    Args:
        fused_gray: 融合后的灰度图 (numpy数组，范围0~1或0~255)
        visible_rgb: 可见光RGB图像 (numpy数组，BGR或RGB皆可)
        brightness_factor: 亮度增强因子，默认1.2使图像更亮一些
    Returns:
        fused_rgb_img: 彩色融合图像 (PIL格式)
    """
    # === 1️⃣ 确保灰度范围在 [0,255] ===
    if fused_gray.max() <= 1.0:
        # 对融合结果进行归一化处理，增强对比度
        fused_gray = (fused_gray - fused_gray.min()) / (fused_gray.max() - fused_gray.min() + 1e-8)
        fused_gray = (fused_gray * 255).astype(np.uint8)
    else:
        fused_gray = fused_gray.astype(np.uint8)

    # === 2️⃣ 增强亮度 ===
    fused_gray = enhance_brightness(fused_gray, brightness_factor)

    # === 3️⃣ 转换可见光到 LAB 空间 ===
    vis_bgr = cv2.cvtColor(visible_rgb, cv2.COLOR_RGB2BGR)
    vis_lab = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2LAB)
    l_vis, a_vis, b_vis = cv2.split(vis_lab)

    # === 4️⃣ 调整尺寸匹配 ===
    fused_L = cv2.resize(fused_gray, (l_vis.shape[1], l_vis.shape[0]))

    # === 5️⃣ 合并 LAB 通道（使用融合后的亮度 + 可见光的颜色） ===
    fused_lab = cv2.merge([fused_L, a_vis, b_vis])

    # === 6️⃣ 转回 RGB 空间 ===
    fused_bgr = cv2.cvtColor(fused_lab, cv2.COLOR_LAB2BGR)
    fused_rgb = cv2.cvtColor(fused_bgr, cv2.COLOR_BGR2RGB)

    # === 7️⃣ 转为 PIL 图像输出 ===
    return Image.fromarray(fused_rgb)


def restore_color_from_visible_ycrcb(fused_Y, visible_rgb, brightness_factor=1.2):
    """
    将融合结果作为亮度通道，结合可见光图像的色度信息生成彩色图像
    Args:
        fused_Y: 融合后的亮度图像 (numpy数组，范围0~1或0~255)
        visible_rgb: 可见光RGB图像 (numpy数组，RGB格式)
        brightness_factor: 亮度增强因子，默认1.2使图像更亮一些
    Returns:
        fused_rgb_img: 彩色融合图像 (PIL格式)
    """
    # === 1️⃣ 确保融合亮度范围在 [0,255] ===
    if fused_Y.max() <= 1.0:
        # 对融合结果进行归一化处理，增强对比度
        fused_Y = (fused_Y - fused_Y.min()) / (fused_Y.max() - fused_Y.min() + 1e-8)
        fused_Y = (fused_Y * 255).astype(np.uint8)
    else:
        fused_Y = fused_Y.astype(np.uint8)

    # === 2️⃣ 增强亮度 ===
    fused_Y = enhance_brightness(fused_Y, brightness_factor)

    # === 3️⃣ 转换可见光到 YCrCb 空间 ===
    vis_bgr = cv2.cvtColor(visible_rgb, cv2.COLOR_RGB2BGR)
    vis_ycrcb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2YCrCb)
    y_vis, cr_vis, cb_vis = cv2.split(vis_ycrcb)

    # === 4️⃣ 调整融合亮度尺寸匹配 ===
    fused_y_resized = cv2.resize(fused_Y, (y_vis.shape[1], y_vis.shape[0]))

    # === 5️⃣ 合并 YCrCb 通道（使用融合后的Y + 可见光的Cr/Cb） ===
    # 这里直接使用融合结果作为亮度通道，保留可见光的色度信息
    fused_ycrcb = cv2.merge([fused_y_resized, cr_vis, cb_vis])

    # === 6️⃣ 转回 RGB 空间 ===
    fused_bgr = cv2.cvtColor(fused_ycrcb, cv2.COLOR_YCrCb2BGR)
    fused_rgb = cv2.cvtColor(fused_bgr, cv2.COLOR_BGR2RGB)

    # === 7️⃣ 转为 PIL 图像输出 ===
    return Image.fromarray(fused_rgb)


def load_and_preprocess_image(image_path, extract_y_channel=True, return_original=False):
    """加载并预处理单张图像，支持彩色图像处理"""
    # 打开图像
    image = Image.open(image_path)
    original_image = None

    if return_original:
        original_image = np.array(image)

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
        transforms.ToTensor(),  # 转换为tensor并归一化到[0,1]
    ])

    # 应用预处理
    tensor_image = transform(image)

    # 添加batch维度 (1, 1, H, W)
    tensor_image = tensor_image.unsqueeze(0)

    if return_original:
        return tensor_image, image, original_image
    else:
        return tensor_image, image


def save_comparison_fusion(ir_image, vi_image, fused_image, save_path):
    """保存红外、可见光和融合图像的对比"""
    # 转换为numpy数组并移除batch维度
    ir_np = ir_image.squeeze(0).squeeze(0).cpu().numpy()
    vi_np = vi_image.squeeze(0).squeeze(0).cpu().numpy()
    fused_np = fused_image.squeeze(0).squeeze(0).cpu().numpy()

    # 对融合图像进行归一化处理，确保显示效果
    fused_np = (fused_np - fused_np.min()) / (fused_np.max() - fused_np.min() + 1e-8)

    # 创建对比图
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 显示红外图像
    axes[0].imshow(ir_np, cmap='gray')
    axes[0].set_title('Infrared Image')
    axes[0].axis('off')

    # 显示可见光图像
    axes[1].imshow(vi_np, cmap='gray')
    axes[1].set_title('Visible Image')
    axes[1].axis('off')

    # 显示融合图像
    axes[2].imshow(fused_np, cmap='gray')
    axes[2].set_title('Fused Image')
    axes[2].axis('off')

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"融合对比图像已保存到: {save_path}")


def calculate_entropy(image):
    """计算图像的信息熵"""
    # 确保图像值在[0, 1]范围内
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    # 计算直方图
    hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 1), density=True)
    # 移除零值以避免log(0)
    hist = hist[hist > 0]
    # 计算熵
    entropy = -np.sum(hist * np.log2(hist))
    return entropy


def calculate_spatial_frequency(image):
    """计算图像的空间频率"""
    # 计算水平和垂直方向的梯度
    sf_horizontal = np.mean(np.abs(np.diff(image, axis=1)))
    sf_vertical = np.mean(np.abs(np.diff(image, axis=0)))
    # 计算空间频率
    spatial_frequency = np.sqrt(sf_horizontal ** 2 + sf_vertical ** 2)
    return spatial_frequency


def evaluate_fusion_performance(ir_image, vi_image, fused_image):
    """评估融合图像的性能"""
    # 转换为numpy数组
    ir_np = ir_image.squeeze(0).squeeze(0).cpu().numpy()
    vi_np = vi_image.squeeze(0).squeeze(0).cpu().numpy()
    fused_np = fused_image.squeeze(0).squeeze(0).cpu().numpy()

    # 计算评估指标
    entropy = calculate_entropy(fused_np)
    spatial_freq = calculate_spatial_frequency(fused_np)

    # 计算与源图像的相关性
    corr_ir = np.corrcoef(fused_np.flatten(), ir_np.flatten())[0, 1]
    corr_vi = np.corrcoef(fused_np.flatten(), vi_np.flatten())[0, 1]

    print(f"融合图像评估结果:")
    print(f"  - 信息熵: {entropy:.4f}")
    print(f"  - 空间频率: {spatial_freq:.4f}")
    print(f"  - 与红外图像相关性: {corr_ir:.4f}")
    print(f"  - 与可见光图像相关性: {corr_vi:.4f}")

    return {
        "entropy": entropy,
        "spatial_frequency": spatial_freq,
        "correlation_ir": corr_ir,
        "correlation_vi": corr_vi
    }


def simple_fusion_evaluation(ir_encoder_path, vi_encoder_path,
                             ir_image_path, vi_image_path, output_path, brightness_factor=1.2):
    """简单的融合网络评估方法"""
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
        fusion_type="simple"
    )

    # 加载训练好的各部分权重
    # 加载红外编码器权重
    if os.path.exists(ir_encoder_path):
        model.ir_encoder.load_state_dict(torch.load(ir_encoder_path, map_location=device))
        print("已加载红外编码器权重")
    else:
        print("警告：未找到红外编码器权重文件")

    # 加载可见光编码器权重
    if os.path.exists(vi_encoder_path):
        model.vi_encoder.load_state_dict(torch.load(vi_encoder_path, map_location=device))
        print("已加载可见光编码器权重")
    else:
        print("警告：未找到可见光编码器权重文件")

    model.to(device)
    model.eval()

    # 加载并预处理红外和可见光图像
    # 对于红外图像，仍然使用灰度图
    ir_tensor_image, _, ir_original = load_and_preprocess_image(ir_image_path, extract_y_channel=False,
                                                                return_original=True)
    # 对于可见光图像，提取Y通道用于融合，并保留原始彩色图像
    vi_tensor_image, _, vi_original = load_and_preprocess_image(vi_image_path, extract_y_channel=True,
                                                                return_original=True)

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

    # 使用原有的LAB方法进行颜色恢复（增加亮度）
    color_fused = restore_color_from_visible(fused_np, vi_original, brightness_factor)
    color_fused.save("fused_color_result.png")
    # color_fused.show()  # 注释掉自动显示，避免阻塞

    # 使用新的YCrCb方法进行颜色恢复（将融合结果作为亮度通道，增加亮度）
    color_fused_ycrcb = restore_color_from_visible_ycrcb(fused_np, vi_original, brightness_factor)
    color_fused_ycrcb.save("fused_color_result_ycrcb.png")
    # color_fused_ycrcb.show()  # 注释掉自动显示，避免阻塞

    # 保存对比结果
    save_comparison_fusion(ir_tensor_image, vi_tensor_image, fused_image, output_path)

    # 评估融合性能
    metrics = evaluate_fusion_performance(ir_tensor_image, vi_tensor_image, fused_image)

    return ir_tensor_image, vi_tensor_image, fused_image, metrics


if __name__ == "__main__":
    # 模型路径（请根据实际情况修改）
    ir_encoder_path = r"weights/encoder_final.pth"  # 红外编码器权重路径
    vi_encoder_path = r"weights/encoder_final.pth"  # 可见光编码器权重路径（这里假设使用相同的编码器）

    # 测试图像路径（请根据实际情况修改）
    ir_image_path = r"image/testNet/250256_ir.jpg"  # 红外图像路径
    vi_image_path = r"image/testNet/250256_vi.jpg"  # 可见光图像路径

    # 输出图像路径
    output_path = r"output/fusion/simple_fusion_evaluation.png"

    # 亮度增强因子，可以调整这个值来改变亮度
    brightness_factor = 1.3  # 增加亮度30%

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 进行简单的融合评估
    try:
        print("开始简单的融合网络评估...")
        ir_tensor, vi_tensor, fused_tensor, metrics = simple_fusion_evaluation(
            ir_encoder_path, vi_encoder_path,
            ir_image_path, vi_image_path, output_path, brightness_factor
        )
        print("简单的融合网络评估完成!")
        print(f"评估结果已保存到: {output_path}")
    except Exception as e:
        print(f"评估过程中出现错误: {e}")
        import traceback

        traceback.print_exc()