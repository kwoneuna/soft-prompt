import torch
import torch.nn.functional as F

# 시뮬레이션 설정
B = 1  # 배치 크기
C = 2  # 클래스 개수 (Class 0, Class 1)
K = 3  # 프롬프트 개수
logit_scale = torch.tensor(10.0) # CLIP의 logit_scale (실제 값은 exp(10) = 22026.46)

# 1. 시뮬레이션 로짓 (Logit) 설정
# 정답 클래스: Class 1
logits_P1 = torch.tensor([[10.0, 20.0]]) # P1: Class 1에 압도적으로 높은 점수 (정답)
logits_P2 = torch.tensor([[15.0, 15.0]]) # P2: 구분이 모호함 (중립)
logits_P3 = torch.tensor([[25.0, 5.0]])  # P3: Class 0에 높은 점수 (오답/노이즈)
all_logits = torch.stack([logits_P1, logits_P2, logits_P3], dim=0) # [K, B, C] -> [3, 1, 2]

# 2. 시뮬레이션 Alignment Score 설정 (이미지 특징과 프롬프트 특징 간의 평균 유사도)
# P3은 부적합하다고 가정하고 낮은 스코어 부여
# Score S: [B, K] -> [1, 3]
avg_similarity = torch.tensor([[0.5, 0.4, 0.2]])

# ----------------- 단순 평균 (Baseline) 계산 -----------------
# (이 경우 K=3이므로 1/3로 평균)
baseline_logits = all_logits.mean(dim=0)
baseline_pred = baseline_logits.argmax(dim=1)
print(f"1. P1 로짓: {logits_P1.tolist()}")
print(f"   P3 로짓: {logits_P3.tolist()}")
print(f"2. 단순 평균 로짓 (P1+P2+P3)/3: {baseline_logits.tolist()}")
print(f"   단순 평균 예측: Class {baseline_pred.item()}") 
print("-" * 40)
# (10+15+25)/3 = 16.67, (20+15+5)/3 = 13.33 -> Class 0 예측 (오답)

# ----------------- 동적 가중 평균 (Dynamic Weighted Averaging) 계산 -----------------
# 1. Softmax 가중치 계산
# logit_scale을 곱하여 스코어의 차이를 증폭
alpha_raw = avg_similarity * logit_scale.detach()
alpha = F.softmax(alpha_raw, dim=1) # [B, K] -> [1, 3]

# 2. 가중 로짓 계산
all_logits_permuted = all_logits.permute(1, 0, 2) # [1, 3, 2]
weighted_logits = all_logits_permuted * alpha.unsqueeze(2) # [1, 3, 2] * [1, 3, 1]
final_logits = weighted_logits.sum(dim=1) # [1, 2]

dynamic_pred = final_logits.argmax(dim=1)

print(f"3. Alignment Score (S): {avg_similarity.tolist()}")
print(f"   Softmax 가중치 (α): {alpha.tolist()}")
print(f"4. 동적 가중 평균 로짓 (α1*P1 + α2*P2 + α3*P3): {final_logits.tolist()}")
print(f"   동적 가중 평균 예측: Class {dynamic_pred.item()}")