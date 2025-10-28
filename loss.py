import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
# ==========================
# 3️⃣ SSIM + L1 混合损失
# ==========================
class SSIM_L1_Loss(nn.Module):
    def __init__(self, alpha=0.5):
        """
        alpha: SSIM权重 (0~1)，越大越关注结构
        """
        super().__init__()
        self.alpha = alpha
        self.l1 = nn.L1Loss()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0)

    def forward(self, recon, target):
        loss_l1 = self.l1(recon, target)
        loss_ssim = 1 - self.ssim(recon, target)
        return loss_l1 + self.alpha * loss_ssim

