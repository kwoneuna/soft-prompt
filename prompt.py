import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# ⚠️ [경고]: CustomCLIP 클래스 정의가 현재 환경에 로드되어 있어야 합니다.
# from trainers_baseline.trip import CustomCLIP 
# (클래스가 로드되었다고 가정하고 진행합니다.)

# --- 설정 및 경로 ---
# ⚠️ 실제 경로로 수정하세요.
# path = '/workspace/Soft-Prompt-Generation/outputs_baseline/multi-dg/TRIP/test/vlcs/b32_ep50/ViT-B16/l/seed_1/warmup_1/best_test.pt'
path = '/workspace/Soft-Prompt-Generation/outputs_baseline/multi-dg/TRIP/test/office_home/b32_ep50/ViT-B16/a/seed_1/warmup_1/best_test.pt'
# path = '/workspace/Soft-Prompt-Generation/outputs_baseline/multi-dg/TRIP/test/pacs/b32_ep50/ViT-B16/s/seed_1/warmup_1/best_test.pt'
colors = ["red", "blue", "green"] # K=3 프롬프트용 색상

# --- 1) 모델 로드 및 벡터 추출 ---
print(f"Loading model from: {path}")

try:
    # GPU에서 학습했더라도 CPU로 로드 (best_test.pt는 보통 전체 모델 객체를 저장)
    model = torch.load(path, map_location="cpu")
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure the CustomCLIP class and its dependencies are defined or imported.")
    exit()

all_ctx = []
labels = []

# Prompt ctx 추출 (K=3 Prompt Learner)
for i, pl in enumerate(model.prompt_learner):
    # .ctx는 nn.Parameter이므로 detach().cpu()로 NumPy 변환 준비
    ctx = pl.ctx.detach().cpu() 

    # CSC=True → shape: [n_cls, n_ctx, 512] -> flatten to [n_cls * n_ctx, 512]
    # CSC=False → shape: [n_ctx, 512] -> 그대로 사용
    if ctx.dim() == 3:  
        n_ctx_tokens = ctx.size(0) * ctx.size(1)
        ctx = ctx.view(n_ctx_tokens, -1)
    else:
        n_ctx_tokens = ctx.size(0)

    all_ctx.append(ctx)
    labels += [i] * n_ctx_tokens # Prompt ID (0, 1, 2)를 라벨로 저장

all_ctx = torch.cat(all_ctx, dim=0).numpy() # [N, 512] NumPy 변환
labels = np.array(labels)

n_samples = all_ctx.shape[0]
print(f"Extracted context vectors. Total samples (N): {n_samples}")

# ----------------------------------------
# ## 1. PCA (Principal Component Analysis) 3D 시각화
# ----------------------------------------

# # --- PCA 3D ---
# print("Applying PCA to 3D...")
# pca = PCA(n_components=3)
# feat_3d = pca.fit_transform(all_ctx) # [N, 3]

# # --- Plot 3D PCA ---
# fig = plt.figure(figsize=(10, 8))
# ax = fig.add_subplot(111, projection="3d")

# for i in range(len(model.prompt_learner)):
#     mask = (labels == i)
#     ax.scatter(
#         feat_3d[mask, 0],
#         feat_3d[mask, 1],
#         feat_3d[mask, 2],
#         s=50,
#         alpha=0.8,
#         color=colors[i],
#         label=f"Prompt Learner {i}"
#     )

# ax.set_title("Prompt Context Vector Distribution (3D PCA)")
# ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
# ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
# ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]*100:.1f}%)")
# ax.legend()
# plt.savefig("prompt_pca_3d.png")
# plt.close(fig)
# print("Saved 3D PCA plot to prompt_pca_3d.png")

# ----------------------------------------
# ## 2. T-SNE (t-SNE) 2D 시각화
# ----------------------------------------

# --- Perplexity 설정 및 T-SNE 2D ---
print("Applying T-SNE...")

# Perplexity는 N_samples보다 작아야 함 (최대 N_samples - 1)
chosen_perplexity = min(30, max(1, n_samples - 1))
print(f"Using Perplexity: {chosen_perplexity}")

if n_samples > 1:
    tsne = TSNE(
        n_components=2, 
        random_state=42, 
        perplexity=chosen_perplexity,
        n_iter=1000
    )
    feat_2d = tsne.fit_transform(all_ctx)
    
    # --- Plot 2D T-SNE ---
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)

    for i in range(len(model.prompt_learner)):
        mask = (labels == i)
        ax.scatter(
            feat_2d[mask, 0],
            feat_2d[mask, 1],
            s=50,
            alpha=0.8,
            color=colors[i],
            label=f"Prompt Learner {i}"
        )

    ax.set_title(f"Prompt Context Vector Distribution (2D t-SNE, P={chosen_perplexity})")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend()
    plt.savefig("prompt_tsne_2d.png")
    plt.close(fig)
    print("Saved 2D t-SNE plot to prompt_tsne_2d.png")
else:
    print("Warning: Insufficient data points for T-SNE visualization.")
    
    # --- 1. 통계량 계산 ---
unique_prompts = np.unique(labels)
prompt_stats = {}

for p_id in unique_prompts:
    mask = (labels == p_id)
    feats = feat_2d[mask]
    
    # 평균 (중심점)
    mean = np.mean(feats, axis=0)
    # 공분산 행렬 (분포의 크기와 방향)
    cov = np.cov(feats, rowvar=False) 
    
    prompt_stats[p_id] = {'mean': mean, 'cov': cov, 'count': feats.shape[0]}

# --- 2. 2D T-SNE 시각화 (개선 버전) ---
fig, ax = plt.subplots(figsize=(10, 10))

# 1) 개별 토큰 (약한 시각화)
for p_id in unique_prompts:
    mask = (labels == p_id)
    ax.scatter(
        feat_2d[mask, 0],
        feat_2d[mask, 1],
        s=25, # 작은 크기
        alpha=0.4, # 투명도 높여 배경처럼 보이게
        color=colors[p_id],
        label=f"P{p_id} Tokens" if p_id == unique_prompts[0] else None # 범례 한 번만 표시
    )

# 2) 군집 타원 및 중심점 (강한 시각화)
confidence = 0.95 # 95% 신뢰 타원
for p_id, stats in prompt_stats.items():
    
    mean = stats['mean']
    cov = stats['cov']
    
    # 2-a. 타원 생성 및 플롯 (분포 범위 시각화)
    # 통계적으로 유의미한 분산을 시각화합니다.
    lambda_, v = np.linalg.eig(cov)
    lambda_ = np.sqrt(lambda_)
    
    # 카이제곱 분포를 이용한 신뢰 구간 계산 (2D, 95%)
    # chi^2(2) for 95% confidence is approx 5.991
    scale_factor = np.sqrt(5.991) 
    
    ellipse = Ellipse(
        xy=mean, 
        width=lambda_[0] * scale_factor * 2,
        height=lambda_[1] * scale_factor * 2,
        angle=np.rad2deg(np.arctan2(v[1, 0], v[0, 0])),
        alpha=0.15,
        facecolor=colors[p_id],
        edgecolor=colors[p_id],
        linewidth=2,
        linestyle='--'
    )
    ax.add_patch(ellipse)

    # 2-b. 중심점 플롯 및 라벨 추가
    ax.scatter(
        mean[0], 
        mean[1], 
        marker='*', # 별 모양으로 강조
        s=300, 
        color='black',
        edgecolor=colors[p_id],
        linewidth=2,
        zorder=3,
        label=f"P{p_id} Center"
    )
    
    # 2-c. 중심점 라벨 텍스트 추가
    ax.text(
        mean[0] + 2, mean[1], # 텍스트 위치 오프셋
        f"P{p_id} Avg.",
        fontsize=12,
        weight='bold',
        color=colors[p_id]
    )

# 3) CLIP Embedding Space Center (참조점)
ax.scatter(0, 0, marker='x', s=200, color='gray', zorder=1, label="Space Center (0,0)")

ax.set_title(f"TRIP Prompt Vector Distribution (2D t-SNE | P={chosen_perplexity})", fontsize=14, weight='bold')
ax.set_xlabel("t-SNE 1", fontsize=12)
ax.set_ylabel("t-SNE 2", fontsize=12)
ax.legend(loc='upper right', fontsize=10)
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_aspect('equal', adjustable='box') # 비율 유지
plt.savefig("office_prompt_tsne_enhanced.png")
plt.close(fig)
print("Saved ENHANCED 2D t-SNE plot to prompt_tsne_enhanced.png")