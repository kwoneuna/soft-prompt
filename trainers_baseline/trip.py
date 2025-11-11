
import os
import time
import datetime
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.utils.tensorboard import SummaryWriter

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from utils.templates import SELECT_TEMPLATES
from trainers_baseline.basedg import *
from utils.clip_part import *

_tokenizer = _Tokenizer()


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model,template_idx=None):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        # n_ctx = 4
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        if template_idx is not None and template_idx < len(SELECT_TEMPLATES):
            chosen_template = SELECT_TEMPLATES[template_idx]
        else:
            # 기본값 (일반 CoOp 설정)
            chosen_template = "a photo of a {}."
        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            prompt = prompt.to('cuda')
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
                nn.init.normal_(ctx_vectors, std=0.02)
                prompt_prefix = " ".join(["X"] * n_ctx)
        print(f'Initial template: "{chosen_template}"')
        self.ctx = nn.Parameter(ctx_vectors)
        classnames = [name.replace("_", " ") for name in classnames]

        prompts = [chosen_template.format(name) for name in classnames]
        prompts_with_ctx = [prompt_prefix + " " + p for p in prompts]
        name_lens = [len(_tokenizer.encode(p)) - n_ctx for p in prompts_with_ctx]
        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts_with_ctx])
        tokenized_prompts = tokenized_prompts.to('cuda') 
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # ClassName, EOS 등
        print(f"Number of context words (tokens): {n_ctx}")


        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        else:
            raise ValueError

        return prompts
    @staticmethod
    def encode_template_features(classnames, clip_model, device):
        n_cls = len(classnames)
        classnames = [name.replace("_", " ") for name in classnames]
        
        all_features = []
        
        with torch.no_grad():
            for template in SELECT_TEMPLATES:
                prompts = [template.format(name) for name in classnames]
                tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
                
                # Context (X)가 없는 순수한 템플릿의 특징을 CLIP의 기본 텍스트 인코더로 추출
                text_features = clip_model.encode_text(tokenized_prompts)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                all_features.append(text_features.cpu()) 
        
        # [Num_Templates, n_cls, D]
        return torch.stack(all_features, dim=0)

def _softplus_pos(x, eps=1e-6):
    # 양수 보장 (분산 안정화)
    return F.softplus(x) + eps

def mahalanobis_diag(x, mu, var_diag):
    """
    x:      (B, D)   - 이미지 특징
    mu:     (B, D)   - 비교할 텍스트 특징(프롬프트 k의 각 샘플별 타깃 클래스 텍스트)
    var_diag:(D,)    - 프롬프트 k의 대각 공분산(분산)
    return: (B,)     - 샘플별 마할라노비스 거리
    """
    inv_var = 1.0 / var_diag  # (D,)
    diff = x - mu             # (B, D)
    # (x-mu)^T Sigma^{-1} (x-mu)  with diagonal Sigma: sum( diff^2 * inv_var )
    dist = torch.sum(diff * diff * inv_var, dim=1)
    return torch.sqrt(torch.clamp(dist, min=1e-8))
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model,best_idx):
        super().__init__()
        self.num_selected_prompts = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = clip_model.to(self.device)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        
        self.prompt_learner = nn.ModuleList([
            PromptLearner(cfg, classnames, self.clip_model,best_idx[i]) 
            for i in range(self.num_selected_prompts)
        ])
        self.tokenized_prompts = self.prompt_learner[0].tokenized_prompts 
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner[0]) 
        
        D = clip_model.ln_final.weight.shape[0]
        self.num_classes = len(classnames)
        K, C = self.num_selected_prompts, self.num_classes
        self.var_eps = float(getattr(cfg.TRAINER.COOP, "VAR_EPS", 1e-5)) 
        self.min_var = float(getattr(cfg.TRAINER.COOP, "VAR_MIN", 1e-6))
        self.routing_gamma = float(getattr(cfg.TRAINER.COOP, "ROUTING_GAMMA", 1.0))
        self.alpha_pbeb = float(getattr(cfg.TRAINER.COOP, "ALPHA_PBEB", 0.2))  # PB-EB 보정 강도
        self.beta_var = float(getattr(cfg.TRAINER.COOP, "BETA_VAR", 0.5))      # 분산 정규화 계수

    def forward(self, image, label=None):
        """
        Forward pass:
        - 배치 기반 Gaussian 추정
        - PAC-Bayes Empirical Bernstein 보정
        - Softmax 가중합 앙상블
        """
        # 1️⃣ 이미지 특징
        image = image.to(self.device, dtype=self.dtype)
        img_feat = self.image_encoder(image)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # [B, D]
        B, D = img_feat.shape
        logit_scale = self.logit_scale.exp()

        # 2️⃣ 프롬프트별 텍스트 특징 및 로짓
        text_feats_per_prompt, logits_per_prompt = [], []
        for pl in self.prompt_learner:
            prompts = pl()
            tfeat = self.text_encoder(prompts, self.tokenized_prompts)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
            text_feats_per_prompt.append(tfeat)
            logits = logit_scale * img_feat @ tfeat.t()  # [B, C]
            logits_per_prompt.append(logits)

        # 3️⃣ 평가 모드일 때는 평균만 반환
        stacked_logits = torch.stack(logits_per_prompt, dim=0)  # [K, B, C]
        avg_logits = stacked_logits.mean(dim=0)
        if label is None:
            return avg_logits

        # 4️⃣ 배치 기반 Gaussian 추정
        label_idx = label.to(self.device).long().view(-1)
        K, C = self.num_selected_prompts, self.num_classes
        mu_stack = torch.stack(text_feats_per_prompt, dim=0)  # [K, C, D]

        prompt_gaussians = [{} for _ in range(K)]
        class_masks = {c: (label_idx == c) for c in range(C)}
        for c in range(C):
            mask_c = class_masks[c]
            if mask_c.sum() == 0:
                continue
            feats_c = img_feat[mask_c]
            for k in range(K):
                mu = mu_stack[k, c]
                var = (feats_c - mu).pow(2).mean(dim=0) + self.var_eps
                var = torch.clamp(var, min=self.min_var)
                prompt_gaussians[k][c] = (mu, var)

        # 5️⃣ PBEB 로그확률 계산
        nleep_scores = torch.full((B, K), -float("inf"), device=self.device, dtype=self.dtype)
        var_loss_accum = []

        for k in range(K):
            for c, (mu, var) in prompt_gaussians[k].items():
                mask_c = class_masks[c]
                if mask_c.sum() == 0:
                    continue
                x = img_feat[mask_c]
                adjusted_var = var + self.alpha_pbeb * var.var(dim=0, keepdim=True)
                adjusted_var = torch.clamp(adjusted_var, min=self.min_var)
                log_prob = -0.5 * (((x - mu) ** 2) / adjusted_var + adjusted_var.log()).sum(dim=1)
                nleep_scores[mask_c, k] = log_prob
                var_loss_accum.append(var.mean())

        # 6️⃣ Softmax 가중치 및 가중합 로짓
        weights = F.softmax(self.routing_gamma * nleep_scores, dim=1)  # [B, K]
        weighted_logits = torch.zeros(B, C, device=self.device, dtype=self.dtype)
        for k in range(K):
            w_k = weights[:, k].unsqueeze(1)
            weighted_logits += w_k * logits_per_prompt[k]

        # 7️⃣ Trainer가 쓸 수 있도록 정규화 항 저장
        self.loss_var_reg = (torch.stack(var_loss_accum).mean()
                             if var_loss_accum else torch.tensor(0.0, device=self.device))
       
        return weighted_logits

    # def forward(self, image, zeroshot_logit=None):
    #     self.image_features = self.image_encoder(image.type(self.dtype))
    #     self.image_features = self.image_features / self.image_features.norm(dim=-1, keepdim=True)

    #     self.logits = []
    #     logit_scale = self.logit_scale.exp()

    #     for prompt_learner in self.prompt_learner:
    #         prompts = prompt_learner()
    #         text_feat = self.text_encoder(prompts, self.tokenized_prompts)
    #         text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    #         logit = logit_scale * self.image_features @ text_feat.t()
    #         self.logits.append(logit)

    #     final_logits = torch.stack(self.logits, dim=0).mean(dim=0)

    #     return final_logits
    # CustomCLIP 내부에 추가
   

    
    def diag_gaussian_logprob(self,x, mu, var, eps=1e-8):
        # ... (기존 diag_gaussian_logprob 로직 유지) ...
        var = var.clamp_min(eps)
        diff2 = (x - mu).pow(2)
        term = diff2 / var + var.log()
        return -0.5 * term.sum(dim=-1)

   
    # def forward(self, image, label=None):
    #     # 1) 이미지 특징
    #     img_feat = self.image_encoder(image.type(self.dtype))
    #     img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # [B, D]
    #     B, D = img_feat.shape
    #     logit_scale = self.logit_scale.exp()

    #     # 2) 프롬프트별 텍스트 특징 + 프롬프트별 로짓
    #     text_feats_per_prompt = []
    #     logits_per_prompt = []
    #     for pl in self.prompt_learner:
    #         prompts = pl()
    #         tfeat = self.text_encoder(prompts, self.tokenized_prompts)  # [C, D]
    #         tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
    #         text_feats_per_prompt.append(tfeat)
    #         logits = logit_scale * img_feat @ tfeat.t()                  # [B, C]
    #         logits_per_prompt.append(logits)

    #     # 3) 임시 앙상블(평균) 로짓
    #     stacked_logits = torch.stack(logits_per_prompt, dim=0)  # [K, B, C]
    #     avg_logits = stacked_logits.mean(dim=0)                 # [B, C]

    #     # 4) 라우팅에 사용할 클래스 선택
    #     #    - 학습 중(label 제공): 정답 사용
    #     #    - 평가 중(label 없음): 임시 예측 사용
    #     if label is not None:
    #         if not torch.is_tensor(label):
    #             label_idx = torch.as_tensor(label, device=img_feat.device)
    #         else:
    #             label_idx = label.to(img_feat.device)
    #         label_idx = label_idx.long().view(-1)  # [B]

    #         distances = []  # [K, B]
    #         for k in range(self.num_selected_prompts):
    #             tfeat_k = text_feats_per_prompt[k]              # [C, D]
    #             # 각 샘플의 "정답 클래스" 임베딩을 선택: [B, D]
    #             mu_k = tfeat_k[label_idx]                       # indexing by true class
    #             var_k = _softplus_pos(self.prompt_variance[k])  # (D,)
    #             d_k = mahalanobis_diag(img_feat, mu_k, var_k)   # (B,)
    #             distances.append(d_k)

    #         distances = torch.stack(distances, dim=0)           # [K, B]

    #         # [K, B] -> [B, K], 작은 거리일수록 가중↑
    #         weights = F.softmax(-self.routing_gamma * distances.t(), dim=1)  # [B, K]

    #         weighted_logits = 0
    #         for k in range(self.num_selected_prompts):
    #             w_k = weights[:, k].unsqueeze(1)                # [B,1]
    #             weighted_logits = weighted_logits + w_k * logits_per_prompt[k]
    #     else:
    #         weighted_logits = avg_logits
        

    #     return weighted_logits
    # def forward(self, image, label=None):
    #     self.image_features = self.image_encoder(image.type(self.dtype))
    #     self.image_features = self.image_features / self.image_features.norm(dim=-1, keepdim=True)

    #     individual_logits = [] # 개별 프롬프트의 Logits 저장
    #     mahalanobis_scores = [] 
    #     logit_scale = self.logit_scale.exp()

    #     for prompt_learner in self.prompt_learner:
    #         prompts = prompt_learner()
    #         text_feat = self.text_encoder(prompts, self.tokenized_prompts)
    #         text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    #         logit = logit_scale * self.image_features @ text_feat.t()
    #         individual_logits.append(logit)

    #     # 1. 앙상블 (Ensemble)
    #     stacked_logits = torch.stack(individual_logits, dim=0) # [num_prompts, batch_size, n_cls]
        
    #     final_logits = stacked_logits.mean(dim=0)

    #     return final_logits
 

@torch.no_grad()
def build_text_features_per_template(clip_model, classnames, templates, device):
    text_features_per_t = []
    for t in templates:
        # 각 클래스 이름을 템플릿에 끼워넣어 전체 프롬프트 리스트 생성
        texts = [t.format(c.replace('_', ' ')) for c in classnames]
        tokenized = clip.tokenize(texts).to(device)
        # CLIP 텍스트 인코더
        text_emb = clip_model.encode_text(tokenized)
        text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        text_features_per_t.append(text_emb)
    return text_features_per_t
    
@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]
   
    def _ensure_writer(self, initialize=False):
        # 이미 writer가 초기화되었으면 바로 반환 (싱글톤 패턴)
        if hasattr(self, 'writer') and self.writer is not None and not initialize:
            return self.writer

        # dassl의 output_dir을 활용하여 고유한 TensorBoard 경로 생성
        # self.output_dir은 dassl에서 run마다 고유하게 설정됩니다.
        if not hasattr(self, 'output_dir'):
            # self.output_dir이 설정되지 않은 경우를 대비 (매우 드물겠지만)
            tb_dir = os.path.join('/workspace/Soft-Prompt-Generation/', "tensorboard/vlcs/fallback_run")
        else:
            # output_dir 아래에 tensorboard 폴더를 만들어 이어서 기록
            # 이미지에서 'ana' 폴더를 사용한 것을 반영하여 os.path.join을 사용합니다.
            tb_dir = os.path.join(self.output_dir, 'ana', "tensorboard_log") 
            # 'tensorboard_log' 폴더를 하나 더 만들어 run 폴더 자체는 생성하지 않도록 합니다.

        os.makedirs(tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_dir)
        print(f"[TensorBoard] Initialized writer in: {self.writer.log_dir}")
        return self.writer

    def _log_routing_scalars_tb(self, global_step, image, label, tag="train"):
        # 큰 배치는 일부만
        max_b = getattr(self, "viz_max_batch", 64)
        image = image[:max_b]
        label = label[:max_b]

        S = self.model.routing_debug_scalars(image, label)

        w = self._ensure_writer()
        w.add_scalar(f"{tag}/maha_minDist_mean", S["min_dist_mean"].item(), global_step)
        w.add_scalar(f"{tag}/maha_maxW_mean",    S["max_w_mean"].item(),    global_step)
        w.add_scalar(f"{tag}/maha_entropy_mean", S["entropy_mean"].item(),  global_step)

        # 선택 지표들
        for k in range(S["util"].numel()):
            w.add_scalar(f"{tag}/prompt_utilization/p{k}", S["util"][k].item(), global_step)
        for k in range(S["acc_per_prompt"].numel()):
            w.add_scalar(f"{tag}/acc_per_prompt/p{k}", S["acc_per_prompt"][k].item(), global_step)

        w.add_scalar(f"{tag}/acc_top_prompt", S["acc_top_prompt"].item(), global_step)
        w.add_scalar(f"{tag}/weighted_margin_mean", S["weighted_margin_mean"].item(), global_step)

    def kd_loss_with_class_graph(self, logits, labels, T_k=2.0, lam=0.3):
        """
        logits: [B, C]
        labels: [B]
        self.S_class: [C, C]  (사전 계산된 클래스 유사도)
        """
        with torch.no_grad():
            q = self.S_class[labels]              # [B, C] soft target 분포
        log_p = F.log_softmax(logits / T_k, dim=1)
        kd = F.kl_div(log_p, q, reduction='batchmean') * (T_k * T_k)
        ce = F.cross_entropy(logits, labels)
        return (1 - lam) * ce + lam * kd
    @torch.no_grad()
    def select_best_template_prefix(self, clip_model, dataloader, classnames, templates, device, top_k=3):
        """
        여러 text template 후보 중에서 top-k 성능이 좋은 템플릿 인덱스를 선택합니다.
        기준은 (이미지-텍스트 유사도 평균).

        Args:
            clip_model: CLIP 모델 (이미지, 텍스트 인코더 포함)
            dataloader: 학습 이미지 데이터로더
            classnames: 클래스 이름 리스트
            templates: 후보 템플릿 문자열 리스트
            device: torch.device
            top_k: 상위 몇 개 템플릿을 선택할지 (default: 3)

        Returns:
            topk_template_indices: 선택된 템플릿 인덱스 리스트
        """
        print(f"Selecting the top-{top_k} template indices...")

        # 1️⃣ 이미지 특징 추출
        clip_model.eval()
        image_features_list = []
        for batch in tqdm(dataloader, desc="Extracting Image Features"):
            image = batch["img"].to(device)
            with torch.no_grad():
                features = clip_model.visual(image.type(clip_model.dtype))
                features = features / features.norm(dim=-1, keepdim=True)
                image_features_list.append(features)
        image_features = torch.cat(image_features_list, dim=0)  # [N, D]

        # 2️⃣ 텍스트 특징 추출
        text_features_per_t = build_text_features_per_template(
            clip_model, classnames, templates, device
        )  # list of [C, D] 텐서

        # 3️⃣ 모든 템플릿별 유사도 평균 계산
        template_scores = []
        for i, text_features in enumerate(text_features_per_t):
            similarity_matrix = image_features @ text_features.t()  # [N, C]
            avg_max_similarity = similarity_matrix.max(dim=1)[0].mean().item()
            template_scores.append((avg_max_similarity, i))

        # 4️⃣ 점수 기준 정렬 후 Top-K 선택
        template_scores.sort(key=lambda x: x[0], reverse=True)
        topk_results = template_scores[:top_k]
        topk_template_indices = [index for score, index in topk_results]

        # 5️⃣ 결과 출력
        print(f"\n--- Top-{top_k} Template Selection Complete ---")
        for score, index in topk_results:
            print(f"Index {index:2d} | Template: '{templates[index]}' | Similarity: {score:.4f}")
        print(f"Final Selected Indices: {topk_template_indices}\n")

        return topk_template_indices
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
                                                 # 저장
        if torch.cuda.is_available() and cfg.USE_CUDA:
            if len(cfg.GPU) == 1:
                self.device = torch.device("cuda:{}".format(cfg.GPU))
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.best_test_result = -np.inf
        self.best_val_test_result = -np.inf

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        clip_model.to(self.device)
        with torch.no_grad():
            texts = [f"a photo of a {c.replace('_', ' ')}." for c in classnames]
            tok = clip.tokenize(texts).to(self.device)
            text_emb = clip_model.encode_text(tok)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)   # [C, D]

            tau = 0.07
            S_logits = text_emb @ text_emb.t() / tau                    # [C, C]
            S = torch.softmax(S_logits, dim=1)
            self.text_bank = text_emb
            self.S_class = S  
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
        templates = SELECT_TEMPLATES # utils.templates에서 가져온 Template 리스트
        train_loader = self.dm.train_loader_x # 학습 데이터 로더 접근 가정
        clip_model.to('cuda')
        best_prefix = self.select_best_template_prefix(
             clip_model, train_loader, classnames, templates, self.device
        )
       
        clip_model = load_clip_to_cpu(cfg)

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model,best_prefix)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        self.clip_model = clip_model
        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner.parameters(), cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self._global_step = 0

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

    def forward_backward(self, batch):
        image, label,domain = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
 
            output = self.model(image,label) # output에는 weighted_logits가 반환
            
            loss_var_reg_loss = self.model.loss_var_reg # New

            lambda_loss_var = 0.1 # New
            # Cross-Entropy Loss
            ce_loss = F.cross_entropy(output, label)
            
            # [Modified] 최종 Loss = CE + λ_var * Var_Reg + λ_util * Util_Reg
            total_loss = (
                ce_loss 
                + lambda_loss_var * loss_var_reg_loss
            )
            
            self.model_backward_and_update(total_loss)
            loss = total_loss # 최종 Loss를 summary에 사용
            
        loss_summary = {
            "loss": loss.item(),
            "ce_loss": ce_loss.item(),
            "acc": compute_accuracy(output, label)[0].item(), 
            "loss_var_reg": loss_var_reg_loss.item(), # New Loss Variance Reg 로깅
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            
        LOG_FREQ = 50 
        if self._global_step % LOG_FREQ == 0:
            
            w = self._ensure_writer()
            # [Modified] TensorBoard 로깅 추가
            w.add_scalar("train/loss_var_reg_loss", loss_var_reg_loss.item(), self._global_step) # New Loss Var Reg 로깅
      
        
        self._global_step += 1
        return loss_summary
    
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        domain = batch['domain']
        input = input.to(self.device)
        label = label.to(self.device)
        domain = domain.to(self.device)

        return input, label, domain
    
    def after_train(self):
        print("Finish the whole training")

        do_test = not self.cfg.TEST.NO_TEST
      
        # Show elapsed time
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")

        # Close writer
        self.close_writer()

    @torch.no_grad()
    def test(self, split=None):
        """A generic testing pipeline."""
        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"  # in case val_loader is None
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")
       
        for batch_idx, batch in enumerate(tqdm(data_loader)):
            input, label = self.parse_batch_test(batch)
            output = self.model(input)
            self.evaluator.process(output, label)

        results = self.evaluator.evaluate()

        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        return list(results.values())[0]
    
    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = ((self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0 
                                if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False)

        curr_result = self.test('val')
        is_best = curr_result > self.best_result
        if is_best:
            self.best_result = curr_result
            self.best_epoch = self.epoch

            self.save_model(self.epoch, self.output_dir, model_name="model-best.pth.tar")
            best_val_dir = os.path.join(self.output_dir, 'best_val.pt')
            torch.save(self.model, best_val_dir)
        
        curr_test_result = self.test('test')
        
        if is_best:
            self.best_val_test_result = curr_test_result
        
        is_test_best = curr_test_result > self.best_test_result
        if is_test_best:
            self.best_test_result = curr_test_result
            self.best_test_epoch = self.epoch
            self.save_model(self.epoch, self.output_dir, model_name="model-best-test.pth.tar")
            best_test_dir = os.path.join(self.output_dir, 'best_test.pt')
            torch.save(self.model, best_test_dir)
                
        try:
            print('******* Domain {} best val acc:      {:.1f}%, epoch: {} *******'.format(self.cfg.TARGET_DOMAIN, self.best_result, self.best_epoch+1))
            if do_test:
                print('******* Domain {} best val test acc: {:.1f}%, epoch: {} *******'.format(self.cfg.TARGET_DOMAIN, self.best_val_test_result, self.best_epoch+1))
                print('******* Domain {} best test acc:     {:.1f}%, epoch: {} *******'.format(self.cfg.TARGET_DOMAIN, self.best_test_result, self.best_test_epoch+1))
        
        except:
            try:
                print('******* Domain {} best val acc:      {:.1f}%, epoch: {} *******'.format(self.cfg.SOURCE_DOMAIN, self.best_result, self.best_epoch+1))
                if do_test:
                    print('******* Domain {} best val test acc: {:.1f}%, epoch: {} *******'.format(self.cfg.SOURCE_DOMAIN, self.best_val_test_result, self.best_epoch+1))
                    print('******* Domain {} best test acc:     {:.1f}%, epoch: {} *******'.format(self.cfg.SOURCE_DOMAIN, self.best_test_result, self.best_test_epoch+1))
            except:
                print('******* Domain {} best val acc:      {:.1f}%, epoch: {} *******'.format(self.cfg.SOURCE_DATASET, self.best_result, self.best_epoch+1))
                if do_test:
                    print('******* Domain {} best val test acc: {:.1f}%, epoch: {} *******'.format(self.cfg.TARGET_DATASETS, self.best_val_test_result, self.best_epoch+1))
                    print('******* Domain {} best test acc:     {:.1f}%, epoch: {} *******'.format(self.cfg.TARGET_DATASETS, self.best_test_result, self.best_test_epoch+1))
        
        
        n_iter = self.epoch
        self.write_scalar("train/val_acc", curr_result, n_iter)
        
        self.set_model_mode("train")
        if self.cfg.SAVE_MODEL and (meet_checkpoint_freq or last_epoch):
            self.save_model(self.epoch, self.output_dir)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
            