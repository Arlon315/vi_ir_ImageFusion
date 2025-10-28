import os
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import  DataLoader
from tqdm import tqdm
from loss import SSIM_L1_Loss
from datasets.LLVIPDataset import LLVIPDataset
from network.net_autoencoder import AutoEncoder

# ==========================
# 4️⃣ 训练设置
# ==========================
def main():
    # 数据路径
    dataset_base_path = r"E:\workspace\python_work\dataSet\LLVIP"  # ⚠️修改为你自己的路径
    train_ir_path = os.path.join(dataset_base_path, "infrared", "train")
    train_vi_path = os.path.join(dataset_base_path, "visible", "train")

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    trainset = LLVIPDataset(train_ir_path, train_vi_path, transform)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=2)

    # 设备选择
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 模型、优化器、损失
    AE_ir = AutoEncoder().to(device)
    AE_vi = AutoEncoder().to(device)
    optimizer_ir = optim.Adam(AE_ir.parameters(), lr=1e-3)
    optimizer_vi = optim.Adam(AE_vi.parameters(), lr=1e-3)
    criterion = SSIM_L1_Loss(alpha=0.5).to(device)

    # ✅ 学习率调度器（10个epoch后下降10倍）
    scheduler_ir = optim.lr_scheduler.StepLR(optimizer_ir, step_size=10, gamma=0.1)
    scheduler_vi = optim.lr_scheduler.StepLR(optimizer_vi, step_size=10, gamma=0.1)

    os.makedirs("weights", exist_ok=True)

    # ==========================
    # 5️⃣ 训练循环
    # ==========================
    epochs = 20
    for epoch in range(epochs):
        AE_ir.train()
        AE_vi.train()
        total_loss_ir, total_loss_vi = 0, 0

        for ir, vi in tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}"):
            ir, vi = ir.to(device), vi.to(device)

            # IR 自编码器
            recon_ir = AE_ir(ir)
            loss_ir = criterion(recon_ir, ir)
            optimizer_ir.zero_grad()
            loss_ir.backward()
            optimizer_ir.step()
            total_loss_ir += loss_ir.item()

            # VI 自编码器
            recon_vi = AE_vi(vi)
            loss_vi = criterion(recon_vi, vi)
            optimizer_vi.zero_grad()
            loss_vi.backward()
            optimizer_vi.step()
            total_loss_vi += loss_vi.item()

        # 每个epoch打印平均loss
        avg_ir = total_loss_ir / len(trainloader)
        avg_vi = total_loss_vi / len(trainloader)
        print(f"Epoch {epoch+1}/{epochs} | IR Loss: {avg_ir:.4f} | VI Loss: {avg_vi:.4f}")

        # ✅ 更新学习率
        scheduler_ir.step()
        scheduler_vi.step()
        # 查看当前学习率（可选）
        lr_now = scheduler_ir.get_last_lr()[0]
        print(f"当前学习率: {lr_now:.6f}")

        # 保存中间权重
        torch.save({
            'epoch': epoch + 1,
            'AE_ir_state_dict': AE_ir.state_dict(),
            'AE_vi_state_dict': AE_vi.state_dict(),
        }, f"weights/checkpoints/epoch_{epoch+1:02d}.pth")

    # ==========================
    # 6️⃣ 保存最终模型
    # ==========================
    torch.save(AE_ir.state_dict(), "weights/AE_IR_final_hybrid.pth")
    torch.save(AE_vi.state_dict(), "weights/AE_VI_final_hybrid.pth")
    print("✅ 最终模型已保存")

if __name__ == "__main__":
    main()
