import os
import os.path as osp
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


# ----------------------------
# Prompt Learner (CoOp-style)
# ----------------------------
class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model, template_idx=None):
        super().__init__()

        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]

        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, \
            f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        # 템플릿 선택
        # if template_idx is not None and template_idx < len(SELECT_TEMPLATES):
        #     chosen_template = SELECT_TEMPLATES[template_idx]
        # else:
        chosen_template = "a photo of a {}."

        # context 초기화
        if ctx_init:
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init).to("cuda")
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1:1 + n_ctx, :]
            prompt_prefix = ctx_init
        else:
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            # nn.init.normal_(ctx_vectors, std=0.02)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial template: "{chosen_template}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)
        classnames = [name.replace("_", " ") for name in classnames]

        prompts = [chosen_template.format(name) for name in classnames]
        prompts_with_ctx = [prompt_prefix + " " + p for p in prompts]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts_with_ctx])
        tokenized_prompts = tokenized_prompts.to("cuda")
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # [SOS] 토큰 임베딩
        self.register_buffer("token_prefix", embedding[:, :1, :])
        # suffix: (classname + 나머지 + EOS)
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        name_lens = [len(_tokenizer.encode(name)) for name in classnames]

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            # generic context를 class마다 broadcast
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        if self.class_token_position == "end":
            # [SOS] + ctx + classname...
            prompts = torch.cat([prefix, ctx, suffix], dim=1)

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts_all = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]

                prefix_i = prefix[i:i + 1]
                class_i = suffix[i:i + 1, :name_len]
                suffix_i = suffix[i:i + 1, name_len:]
                ctx_i_half1 = ctx[i:i + 1, :half_n_ctx]
                ctx_i_half2 = ctx[i:i + 1, half_n_ctx:]

                p = torch.cat(
                    [prefix_i, ctx_i_half1, class_i, ctx_i_half2, suffix_i],
                    dim=1
                )
                prompts_all.append(p)
            prompts = torch.cat(prompts_all, dim=0)

        elif self.class_token_position == "front":
            prompts_all = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]

                prefix_i = prefix[i:i + 1]
                class_i = suffix[i:i + 1, :name_len]
                suffix_i = suffix[i:i + 1, name_len:]
                ctx_i = ctx[i:i + 1]

                p = torch.cat([prefix_i, class_i, ctx_i, suffix_i], dim=1)
                prompts_all.append(p)
            prompts = torch.cat(prompts_all, dim=0)

        else:
            raise ValueError

        return prompts

    @staticmethod
    @torch.no_grad()
    def encode_template_features(classnames, clip_model, device):
        classnames = [name.replace("_", " ") for name in classnames]
        all_features = []
        for template in SELECT_TEMPLATES:
            prompts = [template.format(name) for name in classnames]
            tokenized = torch.cat([clip.tokenize(p) for p in prompts]).to(device)
            text_features = clip_model.encode_text(tokenized)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            all_features.append(text_features.cpu())
        return torch.stack(all_features, dim=0)  # [num_templates, n_cls, D]


# ----------------------------
# Custom CLIP: PAC-Bayes + Mean-field Routing
# ----------------------------
class CustomCLIP(nn.Module):
    """
    - 여러 PromptLearner (K개)
    - text feature = prior, image feature = posterior처럼 보고
      PAC-Bayes 스타일로 image feature를 prior와의 KL-like term로 rescale
    - mean-field variance는 전부 image feature 기반으로 추정 (batch 통계)
    - NLEEP-style sleep score로 prompt routing (EMA 없음)
    """
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()

        self.num_selected_prompts = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_model = clip_model.to(self.device)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.current_selection = None # 현재 배치에서 선택된 인덱스 저장 용도
        self.prompt_learner = nn.ModuleList([
            PromptLearner(cfg, classnames, self.clip_model)
            for i in range(self.num_selected_prompts)
        ])

        self.tokenized_prompts = self.prompt_learner[0].tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner[0])

        self.num_classes = len(classnames)
        self.K = self.num_selected_prompts
        self.C = self.num_classes

        # mean-field / routing hyperparams
        self.var_eps = float(getattr(cfg.TRAINER.COOP, "VAR_EPS", 1e-6))
        self.default_var = float(getattr(cfg.TRAINER.COOP, "MEAN_FIELD_VAR_INIT", 1e-2))
        self.routing_gamma = float(getattr(cfg.TRAINER.COOP, "ROUTING_GAMMA", 0.5))
        # pac-bayes regularization strength
        self.pac_lambda = float(getattr(cfg.TRAINER.COOP, "PAC_LAMBDA", 0.001))## small hyper parameter 

        # for logging
        self.mf_sqdist = None

    @torch.no_grad()
    def predict_per_prompt(self, image):
        """
        평가 용: 입력 이미지를 받아 프롬프트별 로짓을 stack하여 반환.
        Returns:
            logits_stack: [K, B, C]  (forward의 내부 스케일링/정규화 그대로 적용된 로짓)
        """
        img_feat = self.image_encoder(image.type(self.dtype))
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # [B,D]
        logits_per_prompt, _ = self._compute_logits_and_sqdist(img_feat)  # list[K] of [B,C]
        logits_stack = torch.stack(logits_per_prompt, dim=0)  # [K,B,C]
        return logits_stack
    # ---- 내부 유틸: PAC-Bayes style image feature 조정 + sqdist 계산 ----
    def _compute_logits_and_sqdist(self, img_feat):
        """
        img_feat: [B, D], 이미 L2 normalized 상태라고 가정
        Returns:
          logits_per_prompt: list[K] of [B, C]
          mf_sqdist: [K, B, C]  = mean_d (x - t)^2  (조정 전 posterior 기준)
        """
        B, D = img_feat.shape
        logit_scale = self.logit_scale.exp()

        logits_per_prompt = []
        mf_sqdist_list = []

        # image feature mean-field 통계 (posterior 쪽)
        # (PAC-Bayes scaling에 사용)
        f_mean = img_feat.mean(dim=-1, keepdim=True)          # [B,1]
        f_msq = (img_feat ** 2).mean(dim=-1, keepdim=True)    # [B,1]

        for pl in self.prompt_learner:
            # text prior (per prompt)
            prompts = pl()
            tfeat = self.text_encoder(prompts, self.tokenized_prompts)   # [C, D]
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)

            # mean-field squared distance: posterior vs prior mean (조정 기준)
            diff = img_feat.unsqueeze(1) - tfeat.unsqueeze(0)            # [B, C, D]
            sqdist = (diff ** 2).mean(dim=-1)                            # [B, C]
            mf_sqdist_list.append(sqdist)

            # prior variance (per class, scalar, mean-field)
            t_mean = tfeat.mean(dim=-1, keepdim=True)                    # [C,1]
            prior_var = (tfeat - t_mean).pow(2).mean(dim=-1, keepdim=True)  # [C,1]
            prior_var = prior_var + self.var_eps                         # [C,1]

            # PAC-Bayes style KL-like term:
            #   KL_like(b) ~ mean_c [ sqdist(b,c) / prior_var(c) ]
            kl_like = (sqdist / prior_var.squeeze(-1)).mean(dim=1, keepdim=True)  # [B,1]

            # posterior norm 기준 + KL regularization으로 scale 결정
            # mismatch가 큰 이미지일수록 scale 줄어듦
            scale = f_msq / (f_msq + self.pac_lambda * kl_like)     # [B,1]
            scale = torch.clamp(scale, min=0.0, max=1.0)
        
            adj_feat = img_feat * scale                                           # [B,D]

            # 최종 logits (prompt k에 대해)
            logits = logit_scale * (adj_feat @ tfeat.t())                         # [B, C]
            logits_per_prompt.append(logits)

        mf_sqdist = torch.stack(mf_sqdist_list, dim=0)  # [K, B, C]
        return logits_per_prompt, mf_sqdist

    # ---- Sleep score 기반 routing 디버그 ----
    @torch.no_grad()
    def routing_debug_scalars(self, image, label, routing_gamma=None):
        if routing_gamma is None:
            routing_gamma = float(self.routing_gamma)

        # image feature (posterior)
        img_feat = self.image_encoder(image.type(self.dtype))
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # [B,D]
        B, D = img_feat.shape
        label_idx = label.to(img_feat.device).long().view(-1)      # [B]

        logits_per_prompt, mf_sqdist = self._compute_logits_and_sqdist(img_feat)
        self.mf_sqdist = mf_sqdist.detach()

        K, C = self.K, self.C

        # image mean-field variance (class-wise, per prompt)
        # var_{k,c} = mean_{b: y_b=c} mf_sqdist_{k,b,c}
        var_all = torch.full((K, C), self.default_var,
                             dtype=img_feat.dtype, device=img_feat.device)
        for c in range(C):
            mask = (label_idx == c)
            if mask.any():
                # [K, Nc]
                sq_c = mf_sqdist[:, mask, c]
                v_c = sq_c.mean(dim=1) + self.var_eps
                var_all[:, c] = v_c

        # sleep score용 positive class distance
        # pos_sqdist_{k,b} = mf_sqdist_{k,b,y_b}
        pos_sqdist = mf_sqdist[:, torch.arange(B), label_idx]          # [K,B]
        pos_var = var_all[:, label_idx]                                # [K,B]

        energy = pos_sqdist / (pos_var + self.var_eps) + \
                 (pos_var + self.var_eps).log()                        # [K,B]
        sleep_score = (-0.5 * energy).t()                              # [B,K]

        weights = F.softmax(routing_gamma * sleep_score, dim=1)        # [B,K]

        # metrics
        max_w_mean = weights.max(dim=1).values.mean()
        entropy = -(weights.clamp_min(1e-9)
                    * weights.clamp_min(1e-9).log()).sum(dim=1)
        entropy_mean = entropy.mean()

        top_prompt = weights.argmax(dim=1)                             # [B]
        util = torch.bincount(top_prompt, minlength=K).float() / B     # [K]

        # per-prompt acc
        acc_per_prompt = []
        for k in range(K):
            pred_k = logits_per_prompt[k].argmax(dim=1)
            acc_k = (pred_k == label_idx).float().mean()
            acc_per_prompt.append(acc_k)
        acc_per_prompt = torch.stack(acc_per_prompt, dim=0)

        # top-weight prompt acc
        logits_stack = torch.stack(logits_per_prompt, dim=0)           # [K,B,C]
        chosen_logits = logits_stack.permute(1, 0, 2)[
            torch.arange(B), top_prompt, :
        ]                                                              # [B,C]
        pred_top = chosen_logits.argmax(dim=1)
        acc_top_prompt = (pred_top == label_idx).float().mean()

        # weighted ensemble margin
        weighted_logits = (weights.unsqueeze(-1) *
                           torch.stack(logits_per_prompt, dim=1)
                           ).sum(dim=1)                                # [B,C]
        top2 = torch.topk(weighted_logits, k=2, dim=1).values
        weighted_margin_mean = (top2[:, 0] - top2[:, 1]).mean()

        # min mean-field pos distance
        min_dist_mean = pos_sqdist.min(dim=0).values.mean()

        return {
            "min_dist_mean": min_dist_mean,
            "max_w_mean": max_w_mean,
            "entropy_mean": entropy_mean,
            "util": util,
            "acc_per_prompt": acc_per_prompt,
            "acc_top_prompt": acc_top_prompt,
            "weighted_margin_mean": weighted_margin_mean,
        }
    @torch.no_grad()
    def compute_prompt_text_features(self):
        """
        현재 각 PromptLearner의 프롬프트로부터 텍스트 임베딩을 구해 반환.
        Returns:
            list[K] of [C, D] (L2-normalized)
        """
        tfeats = []
        for pl in self.prompt_learner:
            prompts = pl()  # [C, L, dim]
            tfeat = self.text_encoder(prompts, self.tokenized_prompts)  # [C, D]
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
            tfeats.append(tfeat.detach())
        return tfeats

    @torch.no_grad()
    def compute_inter_prompt_similarity(self, reduce: str = "mean"):
        """
        프롬프트 k, l 쌍별 텍스트 임베딩 유사도(코사인).
        같은 클래스 c에서 나온 임베딩끼리 비교하여:
        - reduce='mean'  -> 클래스 평균 KxK 행렬
        - reduce='none'  -> 클래스별 KxK(텐서 shape: [C, K, K])
        """
        tfeats = self.compute_prompt_text_features()  # list[K] of [C, D]
        K = len(tfeats)
        C = tfeats[0].shape[0]
        T = torch.stack(tfeats, dim=0)               # [K, C, D]
        # 클래스별 KxK: for each c, sim[k,l] = dot(T[k,c], T[l,c])
        Tperm = T.permute(1, 0, 2).contiguous()      # [C, K, D]
        sim_per_class = torch.matmul(Tperm, Tperm.transpose(1, 2))  # [C, K, K]
        if reduce == "mean":
            return sim_per_class.mean(dim=0)         # [K, K]
        return sim_per_class                         # [C, K, K]
    
    def forward(self, image, label=None):
        # 1) image posterior feature
        img_feat = self.image_encoder(image.type(self.dtype))
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)      # [B,D]
        B, D = img_feat.shape

        # 2) logits, mf_sqdist (per-prompt)
        logits_per_prompt, mf_sqdist = self._compute_logits_and_sqdist(img_feat)
        # mf_sqdist: [K,B,C] 가정
        self.mf_sqdist = mf_sqdist.detach()

        K, C = self.K, self.C

        # 3) 평가 모드: 단순 평균 앙상블
        stacked_logits = torch.stack(logits_per_prompt, dim=0)         # [K,B,C]
        if label is None:
            # Step 1: 각 프롬프트의 예측 클래스를 찾습니다.
            # pred_idx: [K, B] - 프롬프트 k가 샘플 b에 대해 예측한 클래스 인덱스
            pred_idx = stacked_logits.argmax(dim=-1)                   # [K, B]
            mf_sqdist_T = mf_sqdist.permute(1, 0, 2)                   # [B, K, C]
            pred_idx_T = pred_idx.t()
            
            min_dist = torch.gather(
                mf_sqdist_T, dim=2,
                index=pred_idx_T.unsqueeze(-1)
            ).squeeze(-1)                                              # [B, K]
            
            distance_energy = -min_dist
            weights = F.softmax(self.routing_gamma * distance_energy, dim=1) # [B, K]

            # Step 5: 가중 앙상블 (Weighted Ensemble)
            weighted_logits = (weights.unsqueeze(-1) *
                               torch.stack(logits_per_prompt, dim=1)
                               ).sum(dim=1)                                # [B,C]

            return weighted_logits


        # -------------------------------
        # 4) 학습 모드: 프롬프트별 부트스트랩 샘플링으로 var 추정 & routing
        # -------------------------------
        label_idx = label.to(img_feat.device).long().view(-1)          # [B]

        # (a) 프롬프트별로 중복 허용 부트스트랩 인덱스 생성
        #    self.bootstrap_routing이 False면 원배치로 계산(기존과 동일)
        use_bootstrap = getattr(self, "bootstrap_routing", True) and self.training
        if use_bootstrap:
            # 각 프롬프트 k마다 길이 B의 인덱스 샘플을 생성 (중복 허용)
            # idx_kb[k, b] = 원배치에서 선택된 샘플 인덱스
            idx_kb = torch.randint(low=0, high=B, size=(K, B), device=img_feat.device)
        else:
            idx_kb = torch.arange(B, device=img_feat.device).unsqueeze(0).expand(K, B)  # [K,B]

        # (b) 부트스트랩된 배치의 레이블 인덱스
        label_idx_kb = label_idx[idx_kb]                                              # [K,B]

        # (c) mf_sqdist를 부트스트랩 인덱스에 맞춰 gather
        #     mf_sqdist: [K,B,C], idx_kb: [K,B] -> gathered_sq: [K,B,C]
        gathered_sq = torch.gather(
            mf_sqdist, dim=1,
            index=idx_kb.unsqueeze(-1).expand(K, B, C)
        )

        # (d) 클래스별 분산(var_all[k,c])을 프롬프트 k의 부트스트랩 샘플로 추정
        #     v_{k,c} = mean_{b: y_{k,b}=c} gathered_sq[k,b,c]
        var_all = torch.full((K, C), self.default_var,
                            dtype=img_feat.dtype, device=img_feat.device)
        var_eps = getattr(self, "var_eps", 1e-6)
        #prompt k가 클래스 c를 표현할때의 불확실성 (분산)
        for c in range(C):
            mask_kb = (label_idx_kb == c)                                             # [K,B]
            # 합과 개수를 한 번에 계산
            # gathered_sq[:, :, c]: [K,B]
            cls_sq = gathered_sq[:, :, c]
            sum_sq = (cls_sq * mask_kb.to(cls_sq.dtype)).sum(dim=1)                   # [K]
            cnt = mask_kb.sum(dim=1).clamp_min(0)                                     # [K]
            # cnt > 0 인 위치에만 업데이트
            has_any = cnt > 0
            v_c = torch.zeros(K, dtype=img_feat.dtype, device=img_feat.device)
            # 안전한 평균 (cnt가 0인 곳은 쓰지 않음)
            v_c[has_any] = (sum_sq[has_any] / cnt[has_any].to(sum_sq.dtype)) + var_eps
            var_all[:, c] = torch.where(has_any, v_c, var_all[:, c])

       #각 샘플 b에 대해 각 프롬프트가 클래스를 얼마나 잘 설명하는지 측정
       #샘플b가 잘 설명할수록 점수 높아짐
       #가우시안 확률분포의 log likelihood에서 상수항 제외
       
        pos_sqdist = mf_sqdist[:, torch.arange(B, device=img_feat.device), label_idx]  # [K,B]
        pos_var = var_all[:, label_idx]                                               # [K,B]

        energy = pos_sqdist / (pos_var + var_eps) + (pos_var + var_eps).log()         # [K,B]
        sleep_score = (-0.5 * energy).t()                                             # [B,K]

        # 프롬프트별 가중치
        weights = F.softmax(self.routing_gamma * sleep_score, dim=1)                  # [B,K]

        # 5) weighted ensemble (출력 정렬 유지)
        weighted_logits = 0.0
        for k in range(K):
            w_k = weights[:, k].unsqueeze(1)                                          # [B,1]
            weighted_logits = weighted_logits + w_k * logits_per_prompt[k]            # [B,C]
        correct_mask = torch.zeros_like(mf_sqdist, dtype=torch.bool).scatter_(
            2, label_idx.unsqueeze(0).unsqueeze(-1).expand(K, B, 1), 
            True
        )
        negative_mask = ~correct_mask
        mf_sqdist_masked = mf_sqdist.masked_fill(correct_mask, float('inf'))
        neg_sqdist, _ = mf_sqdist_masked.min(dim=-1)
        margin = 0.2  # 마진 값
        dist_gap = neg_sqdist - pos_sqdist # [K, B]
        distance_penalty_raw = torch.relu(margin - dist_gap) 

        # 프롬프트별 가중치(weights)를 적용한 페널티 (라우팅 결과를 이용)
        # weights: [B, K] -> weights.t(): [K, B]
        distance_penalty = (weights.t() * distance_penalty_raw).sum() / B
        return weighted_logits, distance_penalty


    # ---- Forward ----
    # def forward(self, image, label=None):
    #     # 1) image posterior feature
    #     img_feat = self.image_encoder(image.type(self.dtype))
    #     img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)      # [B,D]
    #     B, D = img_feat.shape

    #     # 2) PAC-Bayes style 조정 + prompt별 logits, mf_sqdist
    #     logits_per_prompt, mf_sqdist = self._compute_logits_and_sqdist(img_feat)
    #     self.mf_sqdist = mf_sqdist.detach()

    #     K, C = self.K, self.C

    #     # 3) 평가 모드: 단순 평균 앙상블
    #     stacked_logits = torch.stack(logits_per_prompt, dim=0)         # [K,B,C]
    #     avg_logits = stacked_logits.mean(dim=0)                        # [B,C]
    #     if label is None:
    #         return avg_logits

    #     # 4) 학습 모드: image mean-field variance 기반 sleep-score routing
    #     label_idx = label.to(img_feat.device).long().view(-1)          # [B]

    #     # var_{k,c}: mean_{b: y_b=c} mf_sqdist_{k,b,c}
    #     var_all = torch.full((K, C), self.default_var,
    #                          dtype=img_feat.dtype, device=img_feat.device)
    #     for c in range(C):
    #         mask = (label_idx == c)
    #         if mask.any():
    #             sq_c = mf_sqdist[:, mask, c]                           # [K,Nc]
    #             v_c = sq_c.mean(dim=1) + self.var_eps
    #             var_all[:, c] = v_c

    #     # pos sqdist / var
    #     pos_sqdist = mf_sqdist[:, torch.arange(B), label_idx]          # [K,B]
    #     pos_var = var_all[:, label_idx]                                # [K,B]

    #     energy = pos_sqdist / (pos_var + self.var_eps) + \
    #              (pos_var + self.var_eps).log()                        # [K,B]
    #     sleep_score = (-0.5 * energy).t()                              # [B,K]

    #     weights = F.softmax(self.routing_gamma * sleep_score, dim=1)   # [B,K]

    #     # 5) weighted ensemble
    #     weighted_logits = 0.0
    #     for k in range(K):
    #         w_k = weights[:, k].unsqueeze(1)                           # [B,1]
    #         weighted_logits = weighted_logits + w_k * logits_per_prompt[k]

    #     return weighted_logits
    
    


# ----------------------------
# Utility: Build template text features
# ----------------------------
@torch.no_grad()
def build_text_features_per_template(clip_model, classnames, templates, device):
    feats = []
    for t in templates:
        texts = [t.format(c.replace("_", " ")) for c in classnames]
        tokenized = clip.tokenize(texts).to(device)
        emb = clip_model.encode_text(tokenized)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        feats.append(emb)
    return feats


# ----------------------------
# Trainer
# ----------------------------
@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """CoOp with PAC-Bayes Mean-field Multi-Prompt Routing."""

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]
    # --- class CustomCLIP(nn.Module) 내부에 추가 ---
    
    # ---- TensorBoard helpers ----
    def _ensure_writer(self, initialize=False):
        if hasattr(self, "writer") and self.writer is not None and not initialize:
            return self.writer

        if not hasattr(self, "output_dir"):
            tb_dir = os.path.join(
                "/workspace/Soft-Prompt-Generation/",
                "tensorboard/vlcs/fallback_run"
            )
        else:
            tb_dir = os.path.join(self.output_dir, "ana", "tensorboard_log")

        os.makedirs(tb_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tb_dir)
        print(f"[TensorBoard] Initialized writer in: {self.writer.log_dir}")
        return self.writer
# --- class CoOp(TrainerX) 내부, 다른 TB 헬퍼들과 나란히 추가 ---
    def _log_prompt_similarity_tb(self, global_step, tag="train"):
        """
        프롬프트-프롬프트 유사도(클래스 평균 KxK) + 요약 지표를 TensorBoard에 기록.
        """
        w = self._ensure_writer()
        sim_mat = self.model.compute_inter_prompt_similarity(reduce="mean")  # [K, K]
        K = sim_mat.shape[0]

        # 개별 pair 스칼라
        for k in range(K):
            for l in range(K):
                w.add_scalar(f"{tag}/prompt_sim/p{k}_p{l}", sim_mat[k, l].item(), global_step)

        # 오프대각 요약 (mean/min) 및 히스토그램
        mask_off = ~torch.eye(K, dtype=torch.bool, device=sim_mat.device)
        offdiag = sim_mat[mask_off]
        if offdiag.numel() > 0:
            w.add_scalar(f"{tag}/prompt_sim_offdiag/mean", offdiag.mean().item(), global_step)
            w.add_scalar(f"{tag}/prompt_sim_offdiag/min", offdiag.min().item(), global_step)
            w.add_histogram(f"{tag}/prompt_sim_offdiag/hist", offdiag.detach().cpu().numpy(), global_step)

    def _log_routing_scalars_tb(self, global_step, image, label, tag="train"):
        max_b = getattr(self, "viz_max_batch", 64)
        image = image[:max_b]
        label = label[:max_b]

        S = self.model.routing_debug_scalars(image, label)

        w = self._ensure_writer()
        w.add_scalar(f"{tag}/mf_minDist_mean", S["min_dist_mean"].item(), global_step)
        w.add_scalar(f"{tag}/mf_maxW_mean", S["max_w_mean"].item(), global_step)
        w.add_scalar(f"{tag}/mf_entropy_mean", S["entropy_mean"].item(), global_step)

        for k in range(S["util"].numel()):
            w.add_scalar(f"{tag}/prompt_utilization/p{k}", S["util"][k].item(), global_step)
        for k in range(S["acc_per_prompt"].numel()):
            w.add_scalar(f"{tag}/acc_per_prompt/p{k}", S["acc_per_prompt"][k].item(), global_step)

        w.add_scalar(f"{tag}/acc_top_prompt", S["acc_top_prompt"].item(), global_step)
        w.add_scalar(f"{tag}/weighted_margin_mean", S["weighted_margin_mean"].item(), global_step)

        if self.model.mf_sqdist is not None:
            w.add_scalar(
                f"{tag}/mf_sqdist_mean_all",
                self.model.mf_sqdist.mean().item(),
                global_step
            )

    # ---- KD (옵션) ----
    def kd_loss_with_class_graph(self, logits, labels, T_k=2.0, lam=0.3):
        with torch.no_grad():
            q = self.S_class[labels]  # [B, C]
        log_p = F.log_softmax(logits / T_k, dim=1)
        kd = F.kl_div(log_p, q, reduction="batchmean") * (T_k * T_k)
        ce = F.cross_entropy(logits, labels)
        return (1 - lam) * ce + lam * kd

    # ---- Template selection ----
    @torch.no_grad()
    def select_best_template_prefix(self, clip_model, dataloader,
                                    classnames, templates, device, top_k=3):
        print(f"Selecting the top-{top_k} template indices...")

        clip_model.eval()
        image_features_list = []
        for batch in tqdm(dataloader, desc="Extracting Image Features"):
            image = batch["img"].to(device)
            with torch.no_grad():
                feat = clip_model.visual(image.type(clip_model.dtype))
                feat = feat / feat.norm(dim=-1, keepdim=True)
                image_features_list.append(feat)

        image_features = torch.cat(image_features_list, dim=0)  # [N, D]

        text_features_per_t = build_text_features_per_template(
            clip_model, classnames, templates, device
        )

        template_scores = []
        for i, text_features in enumerate(text_features_per_t):
            sim = image_features @ text_features.t()   # [N, C]
            avg_max = sim.max(dim=1)[0].mean().item()
            template_scores.append((avg_max, i))

        template_scores.sort(key=lambda x: x[0], reverse=True)
        topk = template_scores[:top_k]
        topk_indices = [idx for _, idx in topk]

        print("\n--- Top-{} Template Selection Complete ---".format(top_k))
        for score, idx in topk:
            print(f"Index {idx:2d} | Template: '{templates[idx]}' | Similarity: {score:.4f}")
        print(f"Final Selected Indices: {topk_indices}\n")

        return topk_indices

    # ---- Build model ----
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        # device 설정
        if torch.cuda.is_available() and cfg.USE_CUDA:
            if isinstance(cfg.GPU, int) or (isinstance(cfg.GPU, str) and cfg.GPU.isdigit()):
                self.device = torch.device(f"cuda:{cfg.GPU}")
            else:
                self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.best_test_result = -np.inf
        self.best_val_test_result = -np.inf

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg).to(self.device)

        # class similarity graph (KD용)
        with torch.no_grad():
            texts = [f"a photo of a {c.replace('_', ' ')}." for c in classnames]
            tok = clip.tokenize(texts).to(self.device)
            text_emb = clip_model.encode_text(tok)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)   # [C, D]

            tau = 0.07
            S_logits = text_emb @ text_emb.t() / tau
            S = torch.softmax(S_logits, dim=1)
            self.text_bank = text_emb
            self.S_class = S

        if cfg.TRAINER.COOP.PREC in ["fp32", "amp"]:
            clip_model.float()

        # # 템플릿 선택
        # templates = SELECT_TEMPLATES
        # train_loader = self.dm.train_loader_x
        # clip_model_for_sel = clip_model.to(self.device)
        # best_prefix = self.select_best_template_prefix(
        #     clip_model_for_sel, train_loader, classnames, templates, self.device
        # )

        # backbone용 CLIP 재로딩 (학습 안정)
        clip_model = load_clip_to_cpu(cfg)

        print("Building CustomCLIP (PAC-Bayes mean-field routing)")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in image/text encoder (only prompts train)")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        # prompt 초기 weights
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.clip_model = clip_model
        self.model.to(self.device)

        # optimizer: prompt_learner만
        self.optim = build_optimizer(self.model.prompt_learner.parameters(), cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        self.register_model("prompt_learner", self.model.prompt_learner,
                            self.optim, self.sched)

        self._global_step = 0
        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

    # ---- Train step ----
    def forward_backward(self, batch):
        image, label, domain = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.COOP.PREC

        if prec == "amp":
            with autocast():
                output = self.model(image, label)
                loss = F.cross_entropy(output, label)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output,penalty = self.model(image, label)
            loss = F.cross_entropy(output, label) + 0.01*penalty
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        LOG_FREQ = 50
        if self._global_step % LOG_FREQ == 0:
            self._log_routing_scalars_tb(self._global_step, image, label, tag="train")
            self._log_prompt_similarity_tb(self._global_step, tag="train")  # ← 추가

        self._global_step += 1
        return loss_summary

    # ---- Batch parsing ----
    def parse_batch_train(self, batch):
        x = batch["img"].to(self.device)
        y = batch["label"].to(self.device)
        d = batch["domain"].to(self.device)
        return x, y, d

    def parse_batch_test(self, batch):
        x = batch["img"].to(self.device)
        y = batch["label"].to(self.device)
        return x, y

    # ---- After train ----
    def after_train(self):
        print("Finish the whole training")
        elapsed = round(time.time() - self.time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print(f"Elapsed: {elapsed}")
        self.close_writer()
   
    # ---- Test ----
    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.model.eval()
        self.evaluator.reset()

        if split is None:
            split = self.cfg.TEST.SPLIT

        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader

        print(f"Evaluate on the *{split}* set")

        # --- 프롬프트별 정확도 누적 준비 ---
        K = self.model.K
        correct_per_prompt = torch.zeros(K, device=self.device, dtype=torch.long)
        total_samples = 0

        for _, batch in enumerate(tqdm(data_loader)):
            x, y = self.parse_batch_test(batch)

            # (A) 기존 evaluator(앙상블 출력)
            out = self.model(x)
            self.evaluator.process(out, y)

            # (B) 프롬프트별 로짓 -> 프롬프트별 예측
            lp = self.model.predict_per_prompt(x)        # [K,B,C]
            preds_k = lp.argmax(dim=-1)                  # [K,B]
            # 배치별 누적
            total_samples += y.size(0)
            # 각 k 프롬프트의 정답 개수 더하기
            for k in range(K):
                correct_per_prompt[k] += (preds_k[k] == y).sum()

        # 기존 총괄 지표
        results = self.evaluator.evaluate()
        for k, v in results.items():
            tag = f"{split}/{k}"
            self.write_scalar(tag, v, self.epoch)

        # --- 프롬프트별 정확도 계산/로깅 ---
        acc_per_prompt = (correct_per_prompt.float() / float(total_samples) * 100.0).tolist()

        # TensorBoard에 기록
        w = self._ensure_writer()
        for k in range(K):
            w.add_scalar(f"{split}/acc_per_prompt/p{k}", acc_per_prompt[k], self.epoch)

        # 콘솔 출력
        pretty = " | ".join([f"p{k}: {acc_per_prompt[k]:.2f}%" for k in range(K)])
        print(f"[{split}] per-prompt acc -> {pretty}")

        return list(results.values())[0]
        # for _, batch in enumerate(tqdm(data_loader)):
        #     x, y = self.parse_batch_test(batch)
        #     out = self.model(x)
        #     self.evaluator.process(out, y)

        # results = self.evaluator.evaluate()
        # for k, v in results.items():
        #     tag = f"{split}/{k}"
        #     self.write_scalar(tag, v, self.epoch)

        # return list(results.values())[0]

    # ---- After each epoch: track best models ----
    def after_epoch(self):
        last_epoch = (self.epoch + 1) == self.max_epoch
        do_test = not self.cfg.TEST.NO_TEST
        meet_checkpoint_freq = (
            (self.epoch + 1) % self.cfg.TRAIN.CHECKPOINT_FREQ == 0
            if self.cfg.TRAIN.CHECKPOINT_FREQ > 0 else False
        )

        curr_val = self.test("val")
        is_best = curr_val > self.best_result
        if is_best:
            self.best_result = curr_val
            self.best_epoch = self.epoch
            self.save_model(self.epoch, self.output_dir, "model-best.pth.tar")
            torch.save(self.model, os.path.join(self.output_dir, "best_val.pt"))

        curr_test = self.test("test")
        if is_best:
            self.best_val_test_result = curr_test

        is_test_best = curr_test > self.best_test_result
        if is_test_best:
            self.best_test_result = curr_test
            self.best_test_epoch = self.epoch
            self.save_model(self.epoch, self.output_dir, "model-best-test.pth.tar")
            torch.save(self.model, os.path.join(self.output_dir, "best_test.pt"))

        # logging of best
        try:
            print(
                '******* Domain {} best val acc:      {:.1f}%, epoch: {} *******'
                .format(self.cfg.TARGET_DOMAIN, self.best_result, self.best_epoch + 1)
            )
            if do_test:
                print(
                    '******* Domain {} best val test acc: {:.1f}%, epoch: {} *******'
                    .format(self.cfg.TARGET_DOMAIN,
                            self.best_val_test_result, self.best_epoch + 1)
                )
                print(
                    '******* Domain {} best test acc:     {:.1f}%, epoch: {} *******'
                    .format(self.cfg.TARGET_DOMAIN,
                            self.best_test_result, self.best_test_epoch + 1)
                )
        except:
            try:
                print(
                    '******* Domain {} best val acc:      {:.1f}%, epoch: {} *******'
                    .format(self.cfg.SOURCE_DOMAIN, self.best_result, self.best_epoch + 1)
                )
                if do_test:
                    print(
                        '******* Domain {} best val test acc: {:.1f}%, epoch: {} *******'
                        .format(self.cfg.SOURCE_DOMAIN,
                                self.best_val_test_result, self.best_epoch + 1)
                    )
                    print(
                        '******* Domain {} best test acc:     {:.1f}%, epoch: {} *******'
                        .format(self.cfg.SOURCE_DOMAIN,
                                self.best_test_result, self.best_test_epoch + 1)
                    )
            except:
                print(
                    '******* Domain {} best val acc:      {:.1f}%, epoch: {} *******'
                    .format(self.cfg.SOURCE_DATASET,
                            self.best_result, self.best_epoch + 1)
                )
                if do_test:
                    print(
                        '******* Domain {} best val test acc: {:.1f}%, epoch: {} *******'
                        .format(self.cfg.TARGET_DATASETS,
                                self.best_val_test_result, self.best_epoch + 1)
                    )
                    print(
                        '******* Domain {} best test acc:     {:.1f}%, epoch: {} *******'
                        .format(self.cfg.TARGET_DATASETS,
                                self.best_test_result, self.best_test_epoch + 1)
                    )

        self.write_scalar("train/val_acc", curr_val, self.epoch)
        self.set_model_mode("train")

        if self.cfg.SAVE_MODEL and (meet_checkpoint_freq or last_epoch):
            self.save_model(self.epoch, self.output_dir)

    # ---- Load model ----
    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()
        model_file = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"

        for name in names:
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError(f'Model not found at "{model_path}"')

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            ep = checkpoint["epoch"]

            state_dict.pop("token_prefix", None)
            state_dict.pop("token_suffix", None)

            print(f'Loading weights to {name} from "{model_path}" (epoch = {ep})')
            self._models[name].load_state_dict(state_dict, strict=False)
