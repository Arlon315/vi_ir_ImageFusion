# ==========================
# 修复OpenMP库冲突问题
# ==========================
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from PIL import Image




from network.net_autoencoder import AutoEncoder
from datasets.LLVIPDataset import LLVIPDataset

# ======================
# 1️⃣ 设备与模型加载
# ======================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

AE_ir = AutoEncoder().to(device)
AE_vi = AutoEncoder().to(device)

AE_ir.load_state_dict(torch.load("weights/server/AE_IR_final_hybrid.pth", map_location=device))
AE_vi.load_state_dict(torch.load("weights/server/AE_VI_final_hybrid.pth", map_location=device))

AE_ir.eval()
AE_vi.eval()

# ======================
# 2️⃣ 加载测试集
# ======================
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset_base_path = r"E:\workspace\python_work\dataSet\LLVIP"
test_ir_path = os.path.join(dataset_base_path, "infrared", "test")
test_vi_path = os.path.join(dataset_base_path, "visible", "test")

testset = LLVIPDataset(root_ir=test_ir_path, root_vi=test_vi_path, transform=transform)
testloader = DataLoader(testset, batch_size=4, shuffle=True)

# ======================
# 3️⃣ 推理与显示
# ======================
ir, vi = next(iter(testloader))
with torch.no_grad():
    recon_ir = AE_ir(ir.to(device)).cpu()
    recon_vi = AE_vi(vi.to(device)).cpu()

def show_pair(original, recon, title):
    plt.figure(figsize=(8,3))
    for i in range(4):
        plt.subplot(2,4,i+1)
        plt.imshow(original[i][0], cmap='gray'); plt.axis('off')
        if i==0: plt.title(f'{title} Original')
        plt.subplot(2,4,i+5)
        plt.imshow(recon[i][0], cmap='gray'); plt.axis('off')
        if i==0: plt.title(f'{title} Reconstructed')
    plt.show()

show_pair(ir, recon_ir, "IR")
show_pair(vi, recon_vi, "VI")
