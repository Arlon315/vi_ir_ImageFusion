import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import numpy as np
from PIL.ImageCms import Flags

from network.net_autoencoder import AutoEncoder, Encoder, Decoder


def load_and_preprocess_image(image_path):
    """加载并预处理单张图像"""
    # 打开图像并转换为灰度图
    image = Image.open(image_path).convert('L')

    # 定义预处理步骤
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为tensor并归一化到[0,1]
    ])

    # 应用预处理
    tensor_image = transform(image)

    # 添加batch维度 (1, 1, H, W)
    tensor_image = tensor_image.unsqueeze(0)

    return tensor_image, image


def load_and_preprocess_color_image(image_path):
    """加载并预处理彩色图像"""
    # 打开图像并转换为RGB
    image = Image.open(image_path).convert('RGB')

    # 转换为LAB颜色空间
    lab_image = image.convert('LAB')

    # 分离L通道(亮度)和AB通道(色彩)
    l_channel, a_channel, b_channel = lab_image.split()

    # 对L通道进行预处理
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为tensor并归一化到[0,1]
    ])

    # 应用预处理到L通道
    l_tensor = transform(l_channel)

    # 添加batch维度 (1, 1, H, W)
    l_tensor = l_tensor.unsqueeze(0)

    return l_tensor, image, lab_image


def post_process_white_areas(original, reconstructed, threshold=0.9, boost_factor=1.2):
    """
    后处理函数，增强重建图像中白色区域的亮度

    Args:
        original: 原始图像张量
        reconstructed: 重建图像张量
        threshold: 白色区域阈值 (0-1)
        boost_factor: 白色区域增强因子
    """
    # 将张量转换为numpy数组
    original_np = original.squeeze(0).squeeze(0).cpu().numpy()
    reconstructed_np = reconstructed.squeeze(0).squeeze(0).cpu().numpy()

    # 创建白色区域掩码
    white_mask = original_np > threshold

    # 对白色区域进行增强
    reconstructed_np[white_mask] = np.minimum(reconstructed_np[white_mask] * boost_factor, 1.0)

    # 转换回张量
    reconstructed_processed = torch.from_numpy(reconstructed_np).unsqueeze(0).unsqueeze(0)

    return reconstructed_processed


def convert_gray_to_color(gray_tensor, original_lab):
    """将处理后的灰度图像与原始色彩信息结合生成彩色图像"""
    # 将处理后的灰度图像转换为PIL图像
    gray_np = gray_tensor.squeeze(0).squeeze(0).cpu().numpy()
    gray_pil = Image.fromarray((gray_np * 255).astype(np.uint8), mode='L')

    # 获取原始图像的色彩通道
    orig_l, orig_a, orig_b = original_lab.split()

    # 将处理后的亮度与原始色彩结合
    reconstructed_lab = Image.merge('LAB', (gray_pil, orig_a, orig_b))

    # 转换回RGB
    reconstructed_rgb = reconstructed_lab.convert('RGB')

    return reconstructed_rgb


def save_comparison(original, reconstructed, save_path):
    """保存原始图像和重建图像的对比"""
    # 转换为numpy数组并移除batch维度
    original_np = original.squeeze(0).squeeze(0).cpu().numpy()
    reconstructed_np = reconstructed.squeeze(0).squeeze(0).cpu().numpy()

    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 显示原始图像
    axes[0].imshow(original_np, cmap='gray')
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 显示重建图像
    axes[1].imshow(reconstructed_np, cmap='gray')
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"对比图像已保存到: {save_path}")


def save_color_comparison(original_rgb, reconstructed_rgb, save_path):
    """保存原始彩色图像和重建彩色图像的对比"""
    # 创建对比图
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 显示原始图像
    axes[0].imshow(original_rgb)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 显示重建图像
    axes[1].imshow(reconstructed_rgb)
    axes[1].set_title('Reconstructed Image')
    axes[1].axis('off')

    # 保存图像
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"彩色对比图像已保存到: {save_path}")


def predict_single_image(model_path, image_path, output_path, enhance_white=False):
    """使用训练好的自编码器模型对单张图像进行预测"""
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型
    model = AutoEncoder(
        in_channels=1,
        out_channels=1,
        en_out_conv=32,
        dense_Layer_out=64,
        dense_layers=3,
        dense_out=128,
        kernel_size=3,
        debug=False
    )

    # 加载训练好的模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 加载并预处理图像
    tensor_image, original_pil = load_and_preprocess_image(image_path)
    tensor_image = tensor_image.to(device)

    print(f"输入图像尺寸: {tensor_image.shape}")

    # 使用模型进行预测
    with torch.no_grad():
        reconstructed = model(tensor_image)

    # 如果需要增强白色区域
    if enhance_white:
        reconstructed = post_process_white_areas(tensor_image, reconstructed)

    print(f"重建图像尺寸: {reconstructed.shape}")

    # 保存对比结果
    save_comparison(tensor_image, reconstructed, output_path)

    return tensor_image, reconstructed


def predict_with_separate_models(encoder_path, decoder_path, image_path, output_path, enhance_white=False):
    """使用训练好的编码器和解码器模型对单张图像进行预测"""
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建编码器和解码器
    encoder = Encoder(
        in_channels=1,
        out_channels=128,
        en_out_conv=32,
        dense_Layer_out=64,
        dense_layers=3,
        dense_out=128,
        kernel_size=3,
        debug=False
    )

    decoder = Decoder(
        in_channels=128,
        kernel_size=3,
        stride=1,
        debug=False
    )

    # 加载训练好的编码器和解码器权重
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    encoder.to(device)
    decoder.to(device)

    encoder.eval()
    decoder.eval()

    # 加载并预处理图像
    tensor_image, original_pil = load_and_preprocess_image(image_path)
    tensor_image = tensor_image.to(device)

    print(f"输入图像尺寸: {tensor_image.shape}")

    # 使用编码器和解码器进行预测
    with torch.no_grad():
        conv_out, dense_out = encoder(tensor_image)
        reconstructed = decoder(conv_out, dense_out)

    # 如果需要增强白色区域
    if enhance_white:
        reconstructed = post_process_white_areas(tensor_image, reconstructed)

    print(f"重建图像尺寸: {reconstructed.shape}")

    # 保存对比结果
    save_comparison(tensor_image, reconstructed, output_path)

    return tensor_image, reconstructed


def predict_single_color_image(model_path, image_path, output_path, enhance_white=False):
    """使用训练好的自编码器模型对彩色图像进行预测"""
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建模型
    model = AutoEncoder(
        in_channels=1,
        out_channels=1,
        en_out_conv=32,
        dense_Layer_out=64,
        dense_layers=3,
        dense_out=128,
        kernel_size=3,
        debug=False
    )

    # 加载训练好的模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # 加载并预处理图像
    l_tensor, original_rgb, original_lab = load_and_preprocess_color_image(image_path)
    l_tensor = l_tensor.to(device)

    print(f"输入图像尺寸: {l_tensor.shape}")

    # 使用模型对L通道进行预测
    with torch.no_grad():
        reconstructed_l = model(l_tensor)

    # 如果需要增强白色区域
    if enhance_white:
        reconstructed_l = post_process_white_areas(l_tensor, reconstructed_l)

    print(f"重建L通道尺寸: {reconstructed_l.shape}")

    # 将处理后的L通道与原始色彩信息结合
    reconstructed_rgb = convert_gray_to_color(reconstructed_l, original_lab)

    # 保存对比结果
    save_color_comparison(original_rgb, reconstructed_rgb, output_path)

    # 同时保存重建的彩色图像
    output_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    reconstructed_image_path = os.path.join(output_dir, f"240135_ir_{base_name}_reconstructed_enhance.png")
    reconstructed_rgb.save(reconstructed_image_path)
    print(f"重建彩色图像已保存到: {reconstructed_image_path}")

    return original_rgb, reconstructed_rgb


def predict_color_with_separate_models(encoder_path, decoder_path, image_path, output_path, enhance_white=False):
    """使用训练好的编码器和解码器模型对彩色图像进行预测"""
    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建编码器和解码器
    encoder = Encoder(
        in_channels=1,
        out_channels=128,
        en_out_conv=32,
        dense_Layer_out=64,
        dense_layers=3,
        dense_out=128,
        kernel_size=3,
        debug=False
    )

    decoder = Decoder(
        in_channels=128,
        kernel_size=3,
        stride=1,
        debug=False
    )

    # 加载训练好的编码器和解码器权重
    encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    decoder.load_state_dict(torch.load(decoder_path, map_location=device))

    encoder.to(device)
    decoder.to(device)

    encoder.eval()
    decoder.eval()

    # 加载并预处理图像
    l_tensor, original_rgb, original_lab = load_and_preprocess_color_image(image_path)
    l_tensor = l_tensor.to(device)

    print(f"输入图像尺寸: {l_tensor.shape}")

    # 使用编码器和解码器对L通道进行预测
    with torch.no_grad():
        conv_out, dense_out = encoder(l_tensor)
        reconstructed_l = decoder(conv_out, dense_out)

    # 如果需要增强白色区域
    if enhance_white:
        reconstructed_l = post_process_white_areas(l_tensor, reconstructed_l)

    print(f"重建L通道尺寸: {reconstructed_l.shape}")

    # 将处理后的L通道与原始色彩信息结合
    reconstructed_rgb = convert_gray_to_color(reconstructed_l, original_lab)

    # 保存对比结果
    save_color_comparison(original_rgb, reconstructed_rgb, output_path)

    # 同时保存重建的彩色图像
    output_dir = os.path.dirname(output_path)
    base_name = os.path.splitext(os.path.basename(output_path))[0]
    reconstructed_image_path = os.path.join(output_dir, f"240135_ir_{base_name}_reconstructed_enhance.png")
    reconstructed_rgb.save(reconstructed_image_path)
    print(f"重建彩色图像已保存到: {reconstructed_image_path}")

    return original_rgb, reconstructed_rgb


if __name__ == "__main__":
    # 模型路径
    model_path = "weights/autoencoder_final.pth"
    encoder_path = "weights/encoder_final.pth"
    decoder_path = "weights/decoder_final.pth"

    # 测试图像路径（您可以根据需要修改）
    image_path = "image/testNet/260528.jpg"

    # 输出图像路径
    output_path = "output/predict/260528_vi_single_image_comparison_enhance.png"
    separate_output_path = "output/predict/260528_vi_separate_models_comparison_enhance.png"

    # 确保输出目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 检查输入图像是否为彩色图像
    img = Image.open(image_path)
    is_color = img.mode == 'RGB'

    # 进行预测
    try:
        if is_color:
            print("检测到彩色图像，使用彩色处理模式...")
            # print("使用完整自编码器模型:")
            # original, reconstructed = predict_single_color_image(model_path, image_path, output_path,
            #                                                      enhance_white=True)

            print("\n使用分离的编码器和解码器模型:")
            original_sep, reconstructed_sep = predict_color_with_separate_models(
                encoder_path, decoder_path, image_path, separate_output_path, enhance_white=True)
        else:
            print("检测到灰度图像，使用灰度处理模式...")
            # print("使用完整自编码器模型:")
            # original, reconstructed = predict_single_image(model_path, image_path, output_path, enhance_white=True)

            print("\n使用分离的编码器和解码器模型:")
            original_sep, reconstructed_sep = predict_with_separate_models(
                encoder_path, decoder_path, image_path, separate_output_path, enhance_white=True)
        print("单张图像测试完成!")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")