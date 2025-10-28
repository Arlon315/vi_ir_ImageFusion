import os
from torch.utils.data import Dataset
from PIL import Image

# ===== 数据集定义 =====
class LLVIPDataset(Dataset):
    def __init__(self, root_ir, root_vi, transform=None):
        self.ir_paths = sorted([os.path.join(root_ir, f) for f in os.listdir(root_ir)])
        self.vi_paths = sorted([os.path.join(root_vi, f) for f in os.listdir(root_vi)])
        self.transform = transform

    def __len__(self):
        return len(self.ir_paths)

    def __getitem__(self, idx):
        ir = Image.open(self.ir_paths[idx]).convert("L")
        vi = Image.open(self.vi_paths[idx]).convert("L")
        if self.transform:
            ir = self.transform(ir)
            vi = self.transform(vi)
        return ir, vi

