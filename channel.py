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
MODEL_PATH = "/workspace/Soft-Prompt-Generation/outputs_baseline/multi-dg/TRIP/test/office_home/b32_ep50/ViT-B16/a/seed_1/warmup_1/best_test.pt"
DATA_ROOT  = "/workspace/data_spg/office_home_dg/images"

BATCH_SIZE = 64
IMG_SIZE   = 224
DOMAINS    = ["art", "clipart", "product", "real_world"]

device = "cuda" if torch.cuda.is_available() else "cpu"


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
# 2. 모델 로드
# ===========================
print(f"Loading model from: {MODEL_PATH}")
try:
    model = torch.load(MODEL_PATH, map_location=device)
    model = model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    raise SystemExit


# ===========================
# 3. 프롬프트별 텍스트 임베딩 추출
# ===========================
@torch.no_grad()
def get_text_features_per_prompt(m):
    """
    return: list of [num_classes, dim], 길이는 K(프롬프트 개수)
    """
    text_feats_list = []
    tokenized = m.tokenized_prompts  # TRIP 구조 기준

    for pl in m.prompt_learner:          # ModuleList
        prompts = pl()                   # [n_cls, n_ctx, dim]
        text_feats_k = m.text_encoder(prompts, tokenized)  # [n_cls, dim]
        text_feats_k = F.normalize(text_feats_k, dim=-1)
        text_feats_list.append(text_feats_k.cpu())         # CPU에 저장

    return text_feats_list


text_feats_list = get_text_features_per_prompt(model)
K = len(text_feats_list)
num_classes, feat_dim = text_feats_list[0].shape
print(f"#prompts (K): {K}, text_feats shape per prompt: {text_feats_list[0].shape}")


# ===========================
# 4. 이미지 임베딩만 한 번 계산
# ===========================
@torch.no_grad()
def encode_images(m, data_loader, device):
    all_img_feats = []
    all_cls_ids   = []
    all_dom_ids   = []

    for imgs, cls_ids, dom_ids in data_loader:
        imgs    = imgs.to(device=device, dtype=torch.float16)  # weight가 fp16이라 가정
        cls_ids = cls_ids.to(device)

        img_feats = m.image_encoder(imgs)        # [B, dim], fp16
        img_feats = img_feats.float()            # 이후 연산용으로 float32
        img_feats = F.normalize(img_feats, dim=-1)

        all_img_feats.append(img_feats.cpu())
        all_cls_ids.append(cls_ids.cpu())
        all_dom_ids.append(dom_ids.cpu())

    all_img_feats = torch.cat(all_img_feats, dim=0)   # [N, dim]
    all_cls_ids   = torch.cat(all_cls_ids,   dim=0).long()
    all_dom_ids   = torch.cat(all_dom_ids,   dim=0).long()

    return all_img_feats, all_cls_ids, all_dom_ids


all_img_feats, cls_ids, dom_ids = encode_images(model, loader, device)
N, d = all_img_feats.shape
print("all_img_feats:", all_img_feats.shape)


# ===========================
# 5. 프롬프트별 domain/class std 계산
# ===========================
num_domains = int(dom_ids.max().item() + 1)
num_classes = int(cls_ids.max().item() + 1)

domain_std_list = []   # 길이 K, 각 원소 shape: [d]
class_std_list  = []

for k in range(K):
    text_feats_k = text_feats_list[k].float()        # [num_classes, d]
    sample_text_k = text_feats_k[cls_ids]            # [N, d]

    channel_scores_k = all_img_feats * sample_text_k # [N, d]

    # --- 도메인 std ---
    domain_means_k = torch.zeros(num_domains, d)
    for di in range(num_domains):
        idx = (dom_ids == di)
        domain_means_k[di] = channel_scores_k[idx].mean(dim=0)
    domain_std_k = domain_means_k.std(dim=0).numpy()

    # --- 클래스 std ---
    class_means_k = torch.zeros(num_classes, d)
    for ci in range(num_classes):
        idx = (cls_ids == ci)
        class_means_k[ci] = channel_scores_k[idx].mean(dim=0)
    class_std_k = class_means_k.std(dim=0).numpy()

    domain_std_list.append(domain_std_k)
    class_std_list.append(class_std_k)

    print(f"[Prompt {k}] domain_std shape: {domain_std_k.shape}, class_std shape: {class_std_k.shape}")


# ===========================
# 6. 프롬프트별 히스토그램 시각화
# ===========================
plt.figure(figsize=(10, 4))

# 색/라벨
colors = ["C0", "C1", "C2", "C3", "C4"]  # 프롬프트가 5개 넘지 않는다고 가정

# (a) 도메인 std
plt.subplot(1, 2, 1)
for k in range(K):
    plt.hist(domain_std_list[k],
             bins=40,
             alpha=0.4,
             label=f"Prompt {k}",
             histtype="stepfilled")
plt.xlabel("Standard deviation")
plt.ylabel("Number of channels")
plt.title("Magnitude of Domain Standard Deviations")
plt.legend()

# (b) 클래스 std
plt.subplot(1, 2, 2)
for k in range(K):
    plt.hist(class_std_list[k],
             bins=40,
             alpha=0.4,
             label=f"Prompt {k}",
             histtype="stepfilled")
plt.xlabel("Standard deviation")
plt.ylabel("Number of channels")
plt.title("Magnitude of Class Standard Deviations")
plt.legend()

plt.tight_layout()
plt.savefig("channel_std_hist_per_prompt.png", dpi=300, bbox_inches="tight")
print("Saved figure as channel_std_hist_per_prompt.png")
