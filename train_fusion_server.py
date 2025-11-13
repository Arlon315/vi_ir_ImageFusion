import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from tqdm import tqdm
import argparse

# 导入网络结构
from network.net_fusion import Fusion
# 导入损失函数
from loss import MultiTargetLoss, VisMainIRHighlightLoss
# 导入数据集
from datasets.LLVIPDataset import LLVIPDataset


def group_params(model):
    fuse_params, deco_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ("shallow_fusion" in n) or ("deep_fusion" in n) or ("AttnFusion" in n) or ("fuse" in n):
            fuse_params.append(p)
        elif "decoder" in n:
            deco_params.append(p)
    return fuse_params, deco_params


def train_fusion_network(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor()
    ])

    # 数据
    train_dataset = LLVIPDataset(
        root_ir=args.ir_train_path,
        root_vi=args.vi_train_path,
        transform=transform
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # 模型（支持 simple / attn / learnable）
    model = Fusion(
        in_channels=1,
        out_channels=1,
        en_out_conv=32,
        dense_Layer_out=64,
        dense_layers=3,
        dense_out=128,
        kernel_size=3,
        debug=False,
        fusion_type=args.fusion_type,        # 新增参数
        # attn_temperature=args.attn_temp      # 新增参数
    ).to(device)

    # 加载编码器权重（若有）
    if os.path.exists(args.ir_encoder_path):
        model.ir_encoder.load_state_dict(torch.load(args.ir_encoder_path, map_location=device))
        print("已加载红外编码器权重")
    if os.path.exists(args.vi_encoder_path):
        model.vi_encoder.load_state_dict(torch.load(args.vi_encoder_path, map_location=device))
        print("已加载可见光编码器权重")

    # 编码器：冻结 + eval（停BN/IN统计）
    model.ir_encoder.eval()
    model.vi_encoder.eval()
    for p in model.ir_encoder.parameters(): p.requires_grad = False
    for p in model.vi_encoder.parameters(): p.requires_grad = False

    # （可选）加载解码器预训练权重
    if os.path.exists(args.decoder_path):
        model.decoder.load_state_dict(torch.load(args.decoder_path, map_location=device))
        print("已加载解码器权重")

    # 阶段A：先冻结解码器（只训融合层）
    for p in model.decoder.parameters(): p.requires_grad = False

    # 损失函数（无 VGG 版），可选注意力正则
    # criterion = MultiTargetLoss(
    #     w_pix=args.w_pix, w_gra=args.w_gra, w_mean=args.w_mean,
    #     w_ir=args.w_ir, w_vi=args.w_vi, attn_reg_lambda=args.attn_reg
    # ).to(device)

    # 损失函数，非对称版（红外线更重视高亮区域）
    criterion = VisMainIRHighlightLoss(
        w_pix_vis=1.0,
        w_pix_ir=0.3,  # 想更强调 IR 高亮区域，可以慢慢加到 0.5
        w_gra_vis=1.0,
        w_gra_ir=0.7,
        w_mean_vis=0.2,
        w_mean_ir=0.1,
        mask_gamma=1.5  # >1 会更突出“特别亮”的 IR 区域
    ).to(device)

    # 优化器分组
    def make_optimizer():
        fuse_params, deco_params = group_params(model)
        groups = []
        if len(fuse_params) > 0:
            groups.append({"params": fuse_params, "lr": args.lr_fuse, "weight_decay": 0.0})
        if len(deco_params) > 0:
            groups.append({"params": deco_params, "lr": args.lr_deco, "weight_decay": args.weight_decay})
        return optim.Adam(groups, betas=(0.9, 0.999))

    optimizer = make_optimizer()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma)

    # 训练
    for epoch in range(args.num_epochs):
        model.train()  # 注意：encoder 已经 eval，仅融合/解码器处于 train

        # 到达阶段B：解冻解码器，联合训练
        if epoch == args.stageA_epochs:
            for p in model.decoder.parameters(): p.requires_grad = True
            optimizer = make_optimizer()
            print(">>> 进入 Stage B：开始联合训练解码器")

        epoch_loss = epoch_pix_loss = epoch_gra_loss = epoch_mean_loss = 0.0

        pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{args.num_epochs}")
        for batch_idx, (ir_images, vi_images) in pbar:
            ir_images = ir_images.to(device, non_blocking=True)
            vi_images = vi_images.to(device, non_blocking=True)

            optimizer.zero_grad()

            # 前向
            if args.fusion_type == "attn" and model.debug:
                fused, aux = model(ir_images, vi_images)  # 若你想把注意力传进 loss，可把 model.debug=True 并取 aux
                total_loss, pix_loss, gra_loss, mean_loss = criterion(fused, ir_images, vi_images, aux["deep"])
            else:
                fused = model(ir_images, vi_images)
                total_loss, pix_loss, gra_loss, mean_loss = criterion(fused, ir_images, vi_images)

            # 反向
            total_loss.backward()

            # 放宽梯度裁剪（更不易平台）
            trainable = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=5.0)

            optimizer.step()

            # 统计
            epoch_loss += total_loss.item()
            epoch_pix_loss += pix_loss.item()
            epoch_gra_loss += gra_loss.item()
            epoch_mean_loss += mean_loss.item()

            pbar.set_postfix({
                "Loss": f"{total_loss.item():.4f}",
                "Pix": f"{pix_loss.item():.4f}",
                "Gra": f"{gra_loss.item():.4f}",
                "Mean": f"{mean_loss.item():.4f}"
            })

        scheduler.step()

        n = len(train_loader)
        print(f"Epoch [{epoch + 1}/{args.num_epochs}] 完成:")
        print(f"  平均总损失: {epoch_loss / n:.4f}")
        print(f"  像素损失: {epoch_pix_loss / n:.4f}")
        print(f"  梯度损失: {epoch_gra_loss / n:.4f}")
        print(f"  均值损失: {epoch_mean_loss / n:.4f}")
        print(f"  当前学习率: {scheduler.get_last_lr()}")

        # 保存
        if (epoch + 1) % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f"fusion_model_epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"模型已保存到: {checkpoint_path}")

    # 保存最终模型
    final_model_path = os.path.join(args.checkpoint_dir, "fusion_model_final.pth")
    torch.save(model.state_dict(), final_model_path)
    print(f"最终模型已保存到: {final_model_path}")


def main():
    parser = argparse.ArgumentParser(description="训练图像融合网络（稳定版）")

    # 数据路径（保持你原来的默认）
    data_path = os.path.expanduser(r"~/autodl-tmp/dataSets/LLVIP")
    # data_path = os.path.expanduser(r"E:\workspace\python_work\dataSet\LLVIP")
    ir_train_path = os.path.join(data_path, "infrared", "train")
    vi_train_path = os.path.join(data_path, "visible", "train")

    # 数据集参数
    parser.add_argument("--ir_train_path", type=str, default=ir_train_path)
    parser.add_argument("--vi_train_path", type=str, default=vi_train_path)
    parser.add_argument("--image_size", type=int, default=256)

    # 预训练权重
    parser.add_argument("--ir_encoder_path", type=str, default="weights/server/encoder_final.pth")
    parser.add_argument("--vi_encoder_path", type=str, default="weights/server/encoder_final.pth")
    parser.add_argument("--decoder_path", type=str, default="weights/server/decoder_final.pth")

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--stageA_epochs", type=int, default=2, help="Stage A：仅训练融合层的 epoch 数")

    # 学习率与正则（分组）
    parser.add_argument("--lr_fuse", type=float, default=5e-4, help="融合层学习率")
    parser.add_argument("--lr_deco", type=float, default=1e-4, help="解码器学习率")
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--lr_decay_step", type=int, default=10, help="学习率衰减周期")
    parser.add_argument("--lr_decay_gamma", type=float, default=0.5, help="学习率衰减因子")
    parser.add_argument("--num_workers", type=int, default=4)

    # 损失权重（建议起步）
    # parser.add_argument("--w_pix", type=float, default=1.0)
    # parser.add_argument("--w_gra", type=float, default=2.0)
    # parser.add_argument("--w_mean", type=float, default=0.3)
    # parser.add_argument("--w_ir", type=float, default=0.4)
    # parser.add_argument("--w_vi", type=float, default=0.6)
    # parser.add_argument("--attn_reg", type=float, default=0.0, help="注意力正则（0~1e-3 可尝试）")

    # 模型开关
    parser.add_argument("--fusion_type", type=str, default="ir_guided", choices=["simple", "attn", "learnable", "ir_guided"])
    # parser.add_argument("--attn_temp", type=float, default=2.0)

    # 保存
    parser.add_argument("--checkpoint_dir", type=str, default="weights/fusion")
    parser.add_argument("--save_interval", type=int, default=2)

    args = parser.parse_args()
    train_fusion_network(args)


if __name__ == "__main__":
    main()
