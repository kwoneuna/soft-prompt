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
# ⚠️ 실제 best_test.pt 경로로 수정
MODEL_PATH = "/workspace/Soft-Prompt-Generation/outputs_baseline/multi-dg/CoOp/office_home/b32_ep50/ViT-B16/a/seed_1/warmup_1/best_test.pt"

# ⚠️ 실제 OfficeHome DG 이미지 루트로 수정
DATA_ROOT = "/workspace/data_spg/office_home_dg/images"

BATCH_SIZE = 64
IMG_SIZE = 224
DOMAINS = ["art", "clipart", "product", "real_world"]  # 사용할 도메인

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


# 이미지 전처리 (CLIP ViT-B/16과 비슷하게)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])

dataset = OfficeHomeDG(DATA_ROOT, split="train", domains=DOMAINS, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)


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
# 3. 텍스트(프롬프트) 임베딩 추출
# ===========================
@torch.no_grad()
def get_text_features(m):
    """
    - 멀티 프롬프트: m.prompt_learner가 ModuleList[PromptLearner] 인 경우
        -> 각 PromptLearner에서 나온 텍스트 임베딩을 평균
    - CoOp(싱글 프롬프트): m.prompt_learner가 단일 PromptLearner 인 경우
        -> 그대로 한 번만 text_encoder에 넣어서 사용
    """
    text_feats = None

    # 1) 멀티 프롬프트 (기존 TRIP / SPG 쪽 구조)
    if isinstance(m.prompt_learner, torch.nn.ModuleList) or isinstance(m.prompt_learner, list):
        text_feat_list = []
        for pl in m.prompt_learner:
            prompts = pl()  # [n_cls, n_ctx, dim]
            tokenized = m.tokenized_prompts
            text_feats_k = m.text_encoder(prompts, tokenized)  # [n_cls, dim]
            text_feat_list.append(text_feats_k)

        # [K, n_cls, dim] -> 평균해서 [n_cls, dim]
        text_feats = torch.stack(text_feat_list, dim=0).mean(dim=0)

    # 2) CoOp 스타일: prompt_learner 하나만 있는 경우
    else:
        # CoOp 구현에서 pl()이 [n_cls, n_ctx, dim]을 리턴한다고 가정
        prompts = m.prompt_learner()      # [n_cls, n_ctx, dim]
        tokenized = m.tokenized_prompts   # [n_cls, n_ctx]
        text_feats = m.text_encoder(prompts, tokenized)  # [n_cls, dim]

    text_feats = F.normalize(text_feats, dim=-1)
    return text_feats.cpu()

text_feats = get_text_features(model)
num_classes, feat_dim = text_feats.shape
print("text_feats:", text_feats.shape)


# ===========================
# 4. 이미지 임베딩 + channel_scores 계산
# ===========================
@torch.no_grad()
def encode_images_and_scores(m, data_loader, text_feats_cpu, device):
    all_img_feats = []
    all_cls_ids = []
    all_dom_ids = []

    for imgs, cls_ids, dom_ids in data_loader:
        imgs = imgs.to(device)
        cls_ids = cls_ids.to(device)

        # ⚠️ CustomCLIP 구현에 맞게 이미지 인코더 이름 수정 필요
        # 예시 1) m.image_encoder(imgs)
        # 예시 2) m.visual(imgs)
        # 예시 3) m.encode_image(imgs)
        imgs = imgs.to(device=device, dtype=torch.float16)
        img_feats = m.image_encoder(imgs)        # [B, dim]
        img_feats = F.normalize(img_feats, dim=-1)

        all_img_feats.append(img_feats.cpu())
        all_cls_ids.append(cls_ids.cpu())
        all_dom_ids.append(dom_ids.cpu())

    all_img_feats = torch.cat(all_img_feats, dim=0)   # [N, dim]
    all_cls_ids = torch.cat(all_cls_ids, dim=0).long()
    all_dom_ids = torch.cat(all_dom_ids, dim=0).long()

    # 각 샘플에 해당하는 텍스트 임베딩 선택 후 element-wise product
    sample_text = text_feats_cpu[all_cls_ids]         # [N, dim]
    channel_scores = all_img_feats * sample_text      # [N, dim]

    return channel_scores, all_cls_ids, all_dom_ids


channel_scores, cls_ids, dom_ids = encode_images_and_scores(
    model, loader, text_feats, device
)

print("channel_scores:", channel_scores.shape)


# ===========================
# 5. 도메인 / 클래스별 채널 표준편차 계산
# ===========================
N, d = channel_scores.shape
num_domains = int(dom_ids.max().item() + 1)
num_classes = int(cls_ids.max().item() + 1)

# (a) 도메인별 평균 → std
domain_means = torch.zeros(num_domains, d)
for di in range(num_domains):
    idx = (dom_ids == di)
    domain_means[di] = channel_scores[idx].mean(dim=0)
domain_std = domain_means.std(dim=0).numpy()   # [d]

# (b) 클래스별 평균 → std
class_means = torch.zeros(num_classes, d)
for ci in range(num_classes):
    idx = (cls_ids == ci)
    class_means[ci] = channel_scores[idx].mean(dim=0)
class_std = class_means.std(dim=0).numpy()     # [d]

print("domain_std shape:", domain_std.shape)
print("class_std shape:", class_std.shape)


# ===========================
# 6. 히스토그램 시각화
# ===========================
plt.figure(figsize=(10, 4))

# (a) 도메인 표준편차
plt.subplot(1, 2, 1)
plt.hist(domain_std, bins=40, alpha=0.7)
plt.xlabel("Standard deviation")
plt.ylabel("Number of channels")
plt.title("Magnitude of Domain Standard Deviations")

# (b) 클래스 표준편차
plt.subplot(1, 2, 2)
plt.hist(class_std, bins=40, alpha=0.7)
plt.xlabel("Standard deviation")
plt.ylabel("Number of channels")
plt.title("Magnitude of Class Standard Deviations")

plt.tight_layout()

# plt.show()  # ❌ 서버에서는 창이 안 뜸
plt.savefig("coop_channel_std_hist.png", dpi=300, bbox_inches="tight")  # ✅ 파일로 저장
print("Saved figure as channel_std_hist.png")
