import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import time
from tqdm import tqdm
import argparse

# 导入网络结构
from network.net_fusion import Fusion
# 导入损失函数
from loss import MultiLoss
# 导入数据集
from datasets.LLVIPDataset import LLVIPDataset


def train_fusion_network(args):
    """
    训练融合网络和解码器，同时冻结预训练的编码器

    Args:
        args: 训练参数配置
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])

    # 创建数据集和数据加载器
    train_dataset = LLVIPDataset(
        root_ir=args.ir_train_path,
        root_vi=args.vi_train_path,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )

    # 创建融合网络模型
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
    ).to(device)

    # 加载预训练的编码器权重
    if os.path.exists(args.ir_encoder_path):
        model.ir_encoder.load_state_dict(torch.load(args.ir_encoder_path, map_location=device))
        print("已加载红外编码器权重")
    else:
        print("警告：未找到红外编码器权重文件")

    if os.path.exists(args.vi_encoder_path):
        model.vi_encoder.load_state_dict(torch.load(args.vi_encoder_path, map_location=device))
        print("已加载可见光编码器权重")
    else:
        print("警告：未找到可见光编码器权重文件")

    # 冻结编码器参数
    for param in model.ir_encoder.parameters():
        param.requires_grad = False

    for param in model.vi_encoder.parameters():
        param.requires_grad = False

    # 设置需要训练的参数（融合网络和解码器）
    trainable_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_params.append(param)
            print(f"训练参数: {name}")

    # 定义优化器
    optimizer = optim.Adam(trainable_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    # 定义学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

    # 定义损失函数
    criterion = MultiLoss(
        w_pix=args.w_pix,
        w_gra=args.w_gra,
        w_mean=args.w_mean,
        w_fea=args.w_fea
    ).to(device)

    # 创建保存目录
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # 训练循环
    model.train()
    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_pix_loss = 0.0
        epoch_gra_loss = 0.0
        epoch_mean_loss = 0.0
        epoch_fea_loss = 0.0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader),
                            desc=f"Epoch {epoch + 1}/{args.num_epochs}")

        for batch_idx, (ir_images, vi_images) in progress_bar:
            ir_images = ir_images.to(device)
            vi_images = vi_images.to(device)

            # 前向传播
            optimizer.zero_grad()
            fused_images = model(ir_images, vi_images)

            # 计算损失（使用可见光图像作为目标）
            total_loss, pix_loss, gra_loss, mean_loss, fea_loss = criterion(fused_images, vi_images)

            # 反向传播
            total_loss.backward()
            optimizer.step()

            # 累计损失
            epoch_loss += total_loss.item()
            epoch_pix_loss += pix_loss.item()
            epoch_gra_loss += gra_loss.item()
            epoch_mean_loss += mean_loss.item()
            epoch_fea_loss += fea_loss.item()

            # 更新进度条
            progress_bar.set_postfix({
                "Loss": f"{total_loss.item():.4f}",
                "Pix": f"{pix_loss.item():.4f}",
                "Gra": f"{gra_loss.item():.4f}",
                "Mean": f"{mean_loss.item():.4f}",
                "Fea": f"{fea_loss.item():.4f}"
            })

        # 更新学习率
        scheduler.step()

        # 计算平均损失
        avg_loss = epoch_loss / len(train_loader)
        avg_pix_loss = epoch_pix_loss / len(train_loader)
        avg_gra_loss = epoch_gra_loss / len(train_loader)
        avg_mean_loss = epoch_mean_loss / len(train_loader)
        avg_fea_loss = epoch_fea_loss / len(train_loader)

        print(f"Epoch [{epoch + 1}/{args.num_epochs}] 完成:")
        print(f"  平均总损失: {avg_loss:.4f}")
        print(f"  像素损失: {avg_pix_loss:.4f}")
        print(f"  梯度损失: {avg_gra_loss:.4f}")
        print(f"  均值损失: {avg_mean_loss:.4f}")
        print(f"  特征损失: {avg_fea_loss:.4f}")
        print(f"  当前学习率: {scheduler.get_last_lr()[0]:.6f}")

        # 保存检查点
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"fusion_model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"模型已保存到: {checkpoint_path}")

    # 保存最终模型
    final_model_path = os.path.join(args.checkpoint_dir, "fusion_model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存到: {final_model_path}")


def main():
    parser = argparse.ArgumentParser(description="训练图像融合网络")

    # 数据集参数
    parser.add_argument("--ir_train_path", type=str, default="E:\\workspace\\python_work\\dataSet\\LLVIP\\infrared\\train", help="红外图像训练数据路径")
    parser.add_argument("--vi_train_path", type=str, default="E:\\workspace\\python_work\\dataSet\\LLVIP\\visible\\train", help="可见光图像训练数据路径")
    parser.add_argument("--image_size", type=int, default=256, help="图像尺寸")

    # 模型参数
    parser.add_argument("--ir_encoder_path", type=str, default="weights/server/encoder_final.pth",
                        help="红外编码器权重路径")
    parser.add_argument("--vi_encoder_path", type=str, default="weights/server/encoder_final.pth",
                        help="可见光编码器权重路径")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16, help="批处理大小")
    parser.add_argument("--num_epochs", type=int, default=2, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="权重衰减")
    parser.add_argument("--lr_decay_step", type=int, default=30, help="学习率衰减步长")
    parser.add_argument("--lr_decay_gamma", type=float, default=0.5, help="学习率衰减因子")
    parser.add_argument("--num_workers", type=int, default=4, help="数据加载器的工作线程数")

    # 损失函数权重
    parser.add_argument("--w_pix", type=float, default=1.0, help="像素损失权重")
    parser.add_argument("--w_gra", type=float, default=5.0, help="梯度损失权重")
    parser.add_argument("--w_mean", type=float, default=0.5, help="均值损失权重")
    parser.add_argument("--w_fea", type=float, default=0.1, help="特征损失权重")

    # 保存参数
    parser.add_argument("--checkpoint_dir", type=str, default="weights/fusion", help="模型保存目录")
    parser.add_argument("--save_interval", type=int, default=10, help="模型保存间隔（epoch）")

    args = parser.parse_args()

    # 开始训练
    train_fusion_network(args)


if __name__ == "__main__":
    main()