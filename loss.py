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


# =============== 多损失函数模块 (编码器解码器的训练损失)===============
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


# =============== 可见光为主体、在红外高亮区域增强的融合损失 ===============
class VisMainIRHighlightLoss(nn.Module):
    """
    以可见光为主体、在红外高亮区域增强的融合损失

    pred: 融合图像
    ir  : 红外图像
    vi  : 可见光图像
    """
    def __init__(self,
                 w_pix_vis=1.0,   # 全局像素, 对齐 VIS
                 w_pix_ir=0.3,   # IR 高亮区域像素
                 w_gra_vis=1.0,  # 全局梯度, 对齐 VIS
                 w_gra_ir=0.7,   # IR 高亮区域梯度
                 w_mean_vis=0.2, # 全局亮度均值, 对齐 VIS
                 w_mean_ir=0.1,  # IR 高亮区域亮度
                 mask_gamma=1.0  # 掩膜非线性, >1 更强调高亮区域
                 ):
        super().__init__()
        self.w_pix_vis = w_pix_vis
        self.w_pix_ir  = w_pix_ir
        self.w_gra_vis = w_gra_vis
        self.w_gra_ir  = w_gra_ir
        self.w_mean_vis = w_mean_vis
        self.w_mean_ir  = w_mean_ir
        self.mask_gamma = mask_gamma

        self.l1 = nn.L1Loss()

    def _make_ir_mask(self, ir):
        """
        根据 IR 亮度构造掩膜:
        - 先归一化到 [0,1]
        - 再做 gamma 提升高亮区域权重
        """
        # ir: [B,1,H,W] or [B,C,H,W]，如果是 C>1 就取平均
        if ir.size(1) > 1:
            x = ir.mean(dim=1, keepdim=True)
        else:
            x = ir

        x_min = x.amin(dim=[2,3], keepdim=True)
        x_max = x.amax(dim=[2,3], keepdim=True)
        mask = (x - x_min) / (x_max - x_min + 1e-6)  # 归一化到 0~1

        if self.mask_gamma != 1.0:
            mask = mask.clamp(0, 1) ** self.mask_gamma

        return mask  # [B,1,H,W] ∈ [0,1]

    def _masked_l1(self, pred, target, mask):
        """
        掩膜加权 L1，防止 mask 面积不同导致尺度变化:
        sum(|p-t| * m) / (sum(m) + eps)
        """
        diff = torch.abs(pred - target) * mask
        return diff.sum() / (mask.sum() + 1e-6)

    def forward(self, pred, ir, vi):
        # 1. 构造 IR 高亮掩膜
        mask = self._make_ir_mask(ir)           # [B,1,H,W]
        # 如果 pred/vi 多通道, 扩展 mask
        if pred.size(1) > 1:
            mask_exp = mask.expand(-1, pred.size(1), -1, -1)
        else:
            mask_exp = mask

        # 2. 全局像素损失 (主要对齐 VIS)
        loss_pix_vis = self.l1(pred, vi)

        # 3. IR 高亮区域像素损失
        loss_pix_ir = self._masked_l1(pred, ir, mask_exp)

        # 4. 全局梯度损失 (对齐 VIS)
        loss_gra_vis = gradient_loss(pred, vi)

        # 5. IR 高亮区域梯度损失
        #    先算梯度再乘 mask, 保证只在高亮区域对齐结构
        loss_gra_ir = gradient_loss(pred * mask_exp, ir * mask_exp)

        # 6. 亮度均值: 全局对齐 VIS, IR 高亮区域对齐 IR
        loss_mean_vis = torch.abs(pred.mean() - vi.mean())

        # 掩膜区域的亮度均值
        mean_pred_mask = (pred * mask_exp).sum() / (mask_exp.sum() + 1e-6)
        mean_ir_mask   = (ir   * mask_exp).sum() / (mask_exp.sum() + 1e-6)
        loss_mean_ir   = torch.abs(mean_pred_mask - mean_ir_mask)

        # 7. 加权组合
        loss_pix  = self.w_pix_vis  * loss_pix_vis  + self.w_pix_ir  * loss_pix_ir
        loss_gra  = self.w_gra_vis  * loss_gra_vis  + self.w_gra_ir  * loss_gra_ir
        loss_mean = self.w_mean_vis * loss_mean_vis + self.w_mean_ir * loss_mean_ir

        loss_total = loss_pix + loss_gra + loss_mean

        return loss_total, loss_pix, loss_gra, loss_mean

