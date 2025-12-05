import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ===========================
# 0. 경로 / 기본 설정
# ===========================
# MODEL_PATH = "/workspace/Soft-Prompt-Generation/outputs_baseline/multi-dg/TRIP/channel2/no_loss/office_home/b32_ep50/ViT-B16/a/seed_1/warmup_1/best_test.pt"
MODEL_PATH = "/workspace/Soft-Prompt-Generation/outputs_baseline/multi-dg/TRIP/channel2/pregate_norm/office_home/b32_ep50/ViT-B16/a/seed_1/warmup_1/best_test.pt"
DATA_ROOT  = "/workspace/data_spg/office_home_dg/images"

BATCH_SIZE = 64
IMG_SIZE   = 224
DOMAINS    = ["art", "clipart", "product", "real_world"]

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from: {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=device)

# 보통 TRIP에서는 torch.save(model, path) 많이 쓰니까,
# 일단 checkpoint가 곧바로 모델이라고 가정
model = checkpoint

model = model.to(device)
model.eval()

print(type(model))
print("Model loaded. dtype:", getattr(model, "dtype", None))


# ===========================
# 1. Dataset 정의
# ===========================
class OfficeHomeDG(Dataset):
    """
    data_spg/office_home_dg/images
      ├─ art/train/Alarm_Clock/xxx.png
      ├─ clipart/train/Alarm_Clock/xxx.png
      ├─ product/train/Alarm_Clock/xxx.png
      └─ real_world/train/Alarm_Clock/xxx.png
    """

    def __init__(self, root, split="train", domains=None, transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        if domains is None:
            domains = ["art", "clipart", "product", "real_world"]
        self.domains = domains

        self.samples = []        # (path, class_id, domain_id)
        self.classes = None
        self.class_to_idx = None

        self._build_index()

    def _build_index(self):
        # 첫 도메인 기준으로 클래스 이름 리스트 생성
        first_dom = self.domains[0]
        first_root = os.path.join(self.root, first_dom, self.split)
        class_names = sorted(
            d for d in os.listdir(first_root)
            if os.path.isdir(os.path.join(first_root, d))
        )
        self.classes = class_names
        self.class_to_idx = {c: i for i, c in enumerate(class_names)}

        # 각 도메인/클래스의 모든 이미지 경로 수집
        for d_id, dom in enumerate(self.domains):
            dom_root = os.path.join(self.root, dom, self.split)
            for cls_name in self.classes:
                cls_dir = os.path.join(dom_root, cls_name)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in os.listdir(cls_dir):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                        path = os.path.join(cls_dir, fname)
                        cls_id = self.class_to_idx[cls_name]
                        self.samples.append((path, cls_id, d_id))

        print(f"[OfficeHomeDG] split={self.split}, "
              f"domains={self.domains}, num_samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls_id, dom_id = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, cls_id, dom_id


# 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])

dataset = OfficeHomeDG(DATA_ROOT, split="train", domains=DOMAINS, transform=transform)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ===========================
# 3. pre-gate 전/후 이미지 feature 추출
# ===========================
@torch.no_grad()
def encode_raw_and_pregate_feats(m, data_loader, device):
    """
    m.image_encoder: ViT 등 visual backbone
    m.pre_gate: ChannelWisePreGate (img_dim -> img_dim)

    반환:
      raw_feats:      [N, D]  # image_encoder 출력 정규화 후
      pregate_feats:  [N, D]  # pre_gate 통과 후 정규화
      all_cls_ids:    [N]
      all_dom_ids:    [N]
    """
    m.eval()

    raw_list = []
    pregate_list = []
    all_cls_ids = []
    all_dom_ids = []

    for imgs, cls_ids, dom_ids in data_loader:
        # CustomCLIP이 forward에서 쓰는 dtype 따라감 (보통 fp16)
        dtype = m.dtype if hasattr(m, "dtype") else torch.float32
        imgs = imgs.to(device=device, dtype=dtype)
        cls_ids = cls_ids.to(device)
        dom_ids = dom_ids.to(device)

        # 1) image encoder
        img_feat = m.image_encoder(imgs)        # [B, D]
        img_feat = img_feat.float()             # 후처리는 float32로
        img_feat_norm = F.normalize(img_feat, dim=-1)   # [B, D]
        raw_list.append(img_feat_norm.cpu())

        # 2) pre-gate 통과
        # forward에서 하는 것처럼 float32로 넣어줌
        pre_in = img_feat_norm.to(torch.float32)
        pre_out = m.pre_gate(pre_in)            # [B, D]
        pre_out_norm = F.normalize(pre_out, dim=-1)
        pregate_list.append(pre_out_norm.cpu())

        all_cls_ids.append(cls_ids.cpu())
        all_dom_ids.append(dom_ids.cpu())

    raw_feats = torch.cat(raw_list, dim=0)         # [N, D]
    pregate_feats = torch.cat(pregate_list, dim=0) # [N, D]
    all_cls_ids = torch.cat(all_cls_ids, dim=0).long()
    all_dom_ids = torch.cat(all_dom_ids, dim=0).long()

    print("raw_feats shape:", raw_feats.shape)
    print("pregate_feats shape:", pregate_feats.shape)
    return raw_feats, pregate_feats, all_cls_ids, all_dom_ids


raw_feats, pregate_feats, cls_ids, dom_ids = encode_raw_and_pregate_feats(
    model, loader, device
)
# ===========================
# 4. 채널별 도메인 / 클래스 std 계산 함수
# ===========================
def compute_channel_std(feats, cls_ids, dom_ids):
    """
    feats:    [N, D]
    cls_ids:  [N]
    dom_ids:  [N]

    반환:
      domain_std: [D]  # 채널별 domain sensitivity
      class_std:  [D]  # 채널별 class sensitivity
    """
    N, d = feats.shape
    num_domains = int(dom_ids.max().item() + 1)
    num_classes = int(cls_ids.max().item() + 1)

    print("num_domains:", num_domains)
    print("num_classes:", num_classes)
    print("feat_dim:", d)

    # (a) 도메인별 평균 → 채널 std
    domain_means = torch.zeros(num_domains, d)
    for di in range(num_domains):
        idx = (dom_ids == di)
        if idx.any():
            domain_means[di] = feats[idx].mean(dim=0)
        else:
            print(f"[warn] domain {di} has no samples")
    domain_std = domain_means.std(dim=0).numpy()    # [d]

    # (b) 클래스별 평균 → 채널 std
    class_means = torch.zeros(num_classes, d)
    for ci in range(num_classes):
        idx = (cls_ids == ci)
        if idx.any():
            class_means[ci] = feats[idx].mean(dim=0)
        else:
            print(f"[warn] class {ci} has no samples")
    class_std = class_means.std(dim=0).numpy()      # [d]

    return domain_std, class_std


raw_domain_std, raw_class_std = compute_channel_std(raw_feats, cls_ids, dom_ids)
pg_domain_std, pg_class_std   = compute_channel_std(pregate_feats, cls_ids, dom_ids)
# ===========================
# 5-1. Domain sensitivity 히스토그램 (pre-gate 전/후)
# ===========================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(raw_domain_std, bins=40, alpha=0.7)
plt.xlabel("Std across domains")
plt.ylabel("Num channels")
plt.title("Before pre-gate (image_encoder)")

plt.subplot(1, 2, 2)
plt.hist(pg_domain_std, bins=40, alpha=0.7)
plt.xlabel("Std across domains")
plt.ylabel("Num channels")
plt.title("After pre-gate (ChannelWisePreGate)")

plt.tight_layout()
plt.savefig("domain_sensitivity_raw_vs_pregate.png", dpi=300, bbox_inches="tight")
print("Saved: domain_sensitivity_raw_vs_pregate.png")
# ===========================
# 5-2. Class sensitivity 히스토그램 (pre-gate 전/후)
# ===========================
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(raw_class_std, bins=40, alpha=0.7)
plt.xlabel("Std across classes")
plt.ylabel("Num channels")
plt.title("Before pre-gate (image_encoder)")

plt.subplot(1, 2, 2)
plt.hist(pg_class_std, bins=40, alpha=0.7)
plt.xlabel("Std across classes")
plt.ylabel("Num channels")
plt.title("After pre-gate (ChannelWisePreGate)")

plt.tight_layout()
plt.savefig("class_sensitivity_raw_vs_pregate.png", dpi=300, bbox_inches="tight")
print("Saved: class_sensitivity_raw_vs_pregate.png")
