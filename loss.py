import torch
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure
import torch.nn.functional as F
import torchvision.models as models
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


# =============== 梯度计算函数 ===============
def gradient_loss(pred, target):
    """使用Sobel算子计算梯度差异"""
    sobel_x = torch.tensor([[1, 0, -1],
                            [2, 0, -2],
                            [1, 0, -1]], dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)
    sobel_y = torch.tensor([[1, 2, 1],
                            [0, 0, 0],
                            [-1, -2, -1]], dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)

    grad_pred_x = F.conv2d(pred, sobel_x, padding=1)
    grad_pred_y = F.conv2d(pred, sobel_y, padding=1)
    grad_target_x = F.conv2d(target, sobel_x, padding=1)
    grad_target_y = F.conv2d(target, sobel_y, padding=1)

    grad_diff_x = torch.abs(grad_pred_x - grad_target_x)
    grad_diff_y = torch.abs(grad_pred_y - grad_target_y)

    return torch.mean(grad_diff_x + grad_diff_y)


# =============== 特征损失（VGG感知） ===============
class VGGFeatureLoss(nn.Module):
    def __init__(self, layer_ids=[3, 8, 15], weight=1.0):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.layers = nn.Sequential(*list(vgg.children())[:max(layer_ids)+1])
        self.layer_ids = layer_ids
        self.weight = weight
        for p in self.layers.parameters():
            p.requires_grad = False

    def forward(self, pred, target):
        # 将模型移动到与输入相同的设备
        device = pred.device
        self.layers = self.layers.to(device)

        # VGG期望3通道输入，因此重复灰度图
        pred_3ch = pred.repeat(1, 3, 1, 1)
        target_3ch = target.repeat(1, 3, 1, 1)
        features_pred, features_target = [], []
        x_pred, x_target = pred_3ch, target_3ch
        for i, layer in enumerate(self.layers):
            x_pred = layer(x_pred)
            x_target = layer(x_target)
            if i in self.layer_ids:
                features_pred.append(x_pred)
                features_target.append(x_target)
        loss = 0.0
        for fp, ft in zip(features_pred, features_target):
            loss += F.l1_loss(fp, ft)
        return loss * self.weight


# =============== 多损失函数模块 ===============
class MultiLoss(nn.Module):
    def __init__(self, w_pix=1.0, w_gra=1.0, w_mean=0.5, w_fea=0.1):
        super().__init__()
        self.w_pix = w_pix
        self.w_gra = w_gra
        self.w_mean = w_mean
        self.w_fea = w_fea

        self.l1 = nn.L1Loss()
        self.vgg_loss = VGGFeatureLoss()

    def forward(self, pred, target):
        # 像素级损失
        loss_pix = self.l1(pred, target)

        # 梯度损失
        loss_gra = gradient_loss(pred, target)

        # 均值损失
        loss_mean = torch.abs(torch.mean(pred) - torch.mean(target))

        # 特征损失
        loss_fea = self.vgg_loss(pred, target)

        # 加权总损失
        loss_total = (
            self.w_pix * loss_pix +
            self.w_gra * loss_gra +
            self.w_mean * loss_mean +
            self.w_fea * loss_fea
        )

        return loss_total, loss_pix, loss_gra, loss_mean, loss_fea


