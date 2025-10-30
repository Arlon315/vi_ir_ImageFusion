import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from PIL import Image
import torch.nn.functional as F
import torchvision.models as models

# 导入网络结构
from network.net_autoencoder import AutoEncoder, Encoder, Decoder
from datasets.LLVIPDataset import LLVIPDataset
# 导入自定义损失函数
from loss import SSIM_L1_Loss, MultiLoss


# 服务器训练配置
class ServerTrainConfig:
    def __init__(self):
        # 数据路径
        self.data_path = os.path.expanduser(r"~/autodl-tmp/dataSets/LLVIP")  # 服务器数据根目录
        self.ir_data_path = os.path.join(self.data_path, "infrared", "train")  # 红外图像路径
        self.vi_data_path = os.path.join(self.data_path, "visible", "train")  # 可见光图像路径

        # 训练参数
        self.batch_size = 32  # 增加batch size提高训练效率
        self.num_epochs = 30  # 增加训练轮数
        self.learning_rate = 0.001
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 优化参数
        self.log_interval = 100  # 每100个batch输出一次日志
        self.save_interval = 10  # 每10个epoch保存一次检查点
        self.image_save_interval = 500  # 每500个batch保存一次对比图像

        # 模型保存路径
        self.checkpoint_dir = "weights/server/checkpoints"
        self.final_model_dir = "weights/server"
        self.output_dir = "output/server"

        # 创建目录
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.final_model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)


# 数据预处理
def get_transforms():
    return transforms.Compose([
        transforms.Resize((128, 128)),  # 调整图像尺寸
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(mean=[0.5], std=[0.5])  # 归一化到[-1, 1]
    ])


# 反归一化函数
def denormalize(tensor):
    """将归一化的张量转换回[0, 1]范围"""
    return (tensor + 1) / 2


# 保存图像对比 (优化版本，仅保存少量图像)
def save_comparison_optimized(original, reconstructed, epoch, batch_idx, output_dir, max_images=4):
    """保存原始图像和重建图像的对比，仅保存少量图像以提高效率"""
    # 仅保存前max_images张图像
    if batch_idx % 500 != 0:  # 每500个batch才保存一次
        return

    # 反归一化
    original = denormalize(original)
    reconstructed = denormalize(reconstructed)

    # 限制值范围在[0, 1]
    original = torch.clamp(original, 0, 1)
    reconstructed = torch.clamp(reconstructed, 0, 1)

    # 仅保存前max_images张图像
    original = original[:max_images]
    reconstructed = reconstructed[:max_images]

    # 创建对比图像
    comparison = torch.cat([original, reconstructed], dim=3)  # 水平拼接

    # 保存图像
    save_path = os.path.join(output_dir, f"epoch_{epoch:03d}_batch_{batch_idx:05d}.png")
    save_image(comparison, save_path, nrow=2, padding=2, normalize=False)


# 服务器优化训练函数
def train_autoencoder_server():
    config = ServerTrainConfig()
    print(f"[{datetime.now()}] 使用设备: {config.device}")

    # 数据预处理
    transform = get_transforms()

    # 创建数据集 - 使用红外图像作为输入和目标
    try:
        train_dataset = LLVIPDataset(
            root_ir=config.ir_data_path,
            root_vi=config.ir_data_path,  # 使用红外图像作为目标
            transform=transform
        )
        print(f"[{datetime.now()}] 训练数据集大小: {len(train_dataset)}")
    except Exception as e:
        print(f"[{datetime.now()}] 数据加载错误: {e}")
        print("请检查数据路径是否正确")
        return

    # 创建数据加载器 (增加num_workers提高效率)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,  # 增加多进程加载数据
        pin_memory=True  # 使用锁页内存加速GPU传输
    )

    # 创建模型 (关闭debug模式)
    model = AutoEncoder(
        in_channels=1,
        out_channels=1,
        en_out_conv=32,
        dense_Layer_out=64,
        dense_layers=3,
        dense_out=128,
        kernel_size=3,
        debug=False  # 关闭调试信息
    )
    model.to(config.device)

    # 使用loss.py中的混合损失函数
    criterion = MultiLoss(w_pix=1.0, w_gra=1.0, w_mean=0.5, w_fea=0.1)

    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)  # 调整学习率衰减

    # 训练记录
    train_losses = []

    print(f"[{datetime.now()}] 开始训练自编码器...")
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        batch_count = 0

        for batch_idx, (ir_images, target_images) in enumerate(train_loader):
            # 使用红外图像作为输入和目标（自编码器模式）
            inputs = ir_images.to(config.device, non_blocking=True)  # 使用non_blocking加速传输
            targets = target_images.to(config.device, non_blocking=True)

            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)

            # 计算损失
            loss_total, loss_pix, loss_gra, loss_mean, loss_fea = criterion(outputs, targets)
            loss = loss_total

            # 反向传播
            loss.backward()

            # 梯度裁剪防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1

            # 定期输出日志 (减少输出频率)
            if batch_idx % config.log_interval == 0:
                print(f'[{datetime.now()}] Epoch: {epoch + 1}/{config.num_epochs}, '
                      f'Batch: {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.6f}')

        # 计算平均损失
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0
        train_losses.append(avg_loss)

        # 更新学习率
        scheduler.step()

        print(f'[{datetime.now()}] Epoch [{epoch + 1}/{config.num_epochs}], '
              f'Average Loss: {avg_loss:.6f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')

        # 定期保存检查点
        if (epoch + 1) % config.save_interval == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir,
                f'autoencoder_epoch_{epoch + 1:03d}.pth'
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'lr': scheduler.get_last_lr()[0]
            }, checkpoint_path)
            print(f'[{datetime.now()}] 检查点已保存: {checkpoint_path}')

        # 仅在特定epoch保存少量对比图像
        if (epoch + 1) % 5 == 0:  # 每5个epoch保存一次对比图像
            with torch.no_grad():
                model.eval()
                # 获取一个batch的数据进行可视化
                sample_data = next(iter(train_loader))
                sample_inputs = sample_data[0][:4].to(config.device)  # 仅取前4张图像
                sample_outputs = model(sample_inputs)
                save_comparison_optimized(
                    sample_inputs.cpu(),
                    sample_outputs.cpu(),
                    epoch + 1,
                    0,
                    config.output_dir
                )
                model.train()

    # 保存最终模型 - 分别保存编码器和解码器
    print(f"[{datetime.now()}] 正在保存最终模型...")

    # 保存完整的自编码器模型
    final_autoencoder_path = os.path.join(config.final_model_dir, 'autoencoder_final.pth')
    torch.save(model.state_dict(), final_autoencoder_path)
    print(f'[{datetime.now()}] 完整自编码器模型已保存: {final_autoencoder_path}')

    # 单独保存编码器模型
    final_encoder_path = os.path.join(config.final_model_dir, 'encoder_final.pth')
    torch.save(model.encoder.state_dict(), final_encoder_path)
    print(f'[{datetime.now()}] 编码器模型已保存: {final_encoder_path}')

    # 单独保存解码器模型
    final_decoder_path = os.path.join(config.final_model_dir, 'decoder_final.pth')
    torch.save(model.decoder.state_dict(), final_decoder_path)
    print(f'[{datetime.now()}] 解码器模型已保存: {final_decoder_path}')

    # 绘制损失曲线 (不显示，直接保存)
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, config.num_epochs + 1), train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    loss_plot_path = os.path.join(config.output_dir, 'training_loss.png')
    plt.savefig(loss_plot_path)
    plt.close()  # 关闭图形避免内存泄漏

    print(f"[{datetime.now()}] 训练完成!")


# 测试函数 - 用于评估自编码器性能
def test_autoencoder_server(model_path="weights/server/autoencoder_final.pth"):
    config = ServerTrainConfig()

    # 加载模型
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
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    model.to(config.device)
    model.eval()

    # 数据预处理
    transform = get_transforms()

    # 创建测试数据集 - 使用红外图像作为输入和目标
    try:
        test_dataset = LLVIPDataset(
            root_ir=config.ir_data_path,
            root_vi=config.ir_data_path,  # 使用红外图像作为目标
            transform=transform
        )
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)
    except Exception as e:
        print(f"[{datetime.now()}] 测试数据加载错误: {e}")
        return

    # 使用loss.py中的混合损失函数进行测试
    criterion = MultiLoss(w_pix=1.0, w_gra=1.0, w_mean=0.5, w_fea=0.1)

    print(f"[{datetime.now()}] 开始测试...")
    with torch.no_grad():
        total_loss = 0.0
        batch_count = 0

        for batch_idx, (ir_images, target_images) in enumerate(test_loader):
            inputs = ir_images.to(config.device)
            targets = target_images.to(config.device)  # 目标是输入图像本身

            outputs = model(inputs)
            loss_total, loss_pix, loss_gra, loss_mean, loss_fea = criterion(outputs, targets)
            total_loss += loss_total.item()
            batch_count += 1

            # 仅保存少量测试结果
            if batch_idx < 2:  # 仅保存前2个batch的结果
                save_comparison_optimized(inputs.cpu(), outputs.cpu(), 0, batch_idx, config.output_dir)

        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        print(f'[{datetime.now()}] 测试平均总损失: {avg_loss:.6f}')


if __name__ == "__main__":
    # 服务器训练自编码器
    train_autoencoder_server()

    # 测试自编码器（可选）
    # test_autoencoder_server()