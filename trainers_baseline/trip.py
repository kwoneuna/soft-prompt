97.6
import os
import time
import datetime
import numpy as np
from tqdm import tqdm
from itertools import chain

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
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.TRIP.N_CTX
        ctx_init = cfg.TRAINER.TRIP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.TRIP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized

        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])

        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
            
        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.TRIP.CLASS_TOKEN_POSITION

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
class FeatureCovAlign(nn.Module):
    """
    x' = x @ C_I^{-1/2} @ C_T^{1/2}
    - 이미지 공분산을 whitening, 텍스트 공분산으로 recoloring
    - 학습 시 배치 통계로 러닝 평균 갱신, 평가 시 러닝 평균만 사용
    """
    def __init__(self, dim, momentum=0.99, shrink=0.05, eps=1e-5, track_mean=False):
        super().__init__()
        self.dim = dim
        self.momentum = momentum
        self.shrink = shrink
        self.eps = eps
        self.track_mean = track_mean

        I = torch.eye(dim)
        self.register_buffer("running_Ci_invsqrt", I.clone())
        self.register_buffer("running_Ct_sqrt",    I.clone())
        self.register_buffer("running_mean_img",   torch.zeros(dim))
        self.register_buffer("initialized",        torch.tensor(0, dtype=torch.uint8))

    @torch.no_grad()
    def _cov_shrink(self, X: torch.Tensor):
        # X: [N, D]
        N, D = X.shape
        Xm = X - X.mean(dim=0, keepdim=True)
        C = (Xm.t() @ Xm) / max(N - 1, 1)
        C = (1 - self.shrink) * C + self.shrink * torch.eye(D, device=X.device, dtype=X.dtype)
        C = 0.5 * (C + C.transpose(0, 1))  # 수치 대칭화
        return C

    @torch.no_grad()
    def _psd_eigh_sqrt_invsqrt(self, C: torch.Tensor):
        # 고유분해 기반 sqrt / invsqrt
        evals, evecs = torch.linalg.eigh(C)
        evals = torch.clamp(evals, min=self.eps)
        sqrtD = torch.diag_embed(torch.sqrt(evals))
        invsqrtD = torch.diag_embed(torch.rsqrt(evals))
        C_sqrt = evecs @ sqrtD @ evecs.t()
        C_invsqrt = evecs @ invsqrtD @ evecs.t()
        return C_sqrt, C_invsqrt

    @torch.no_grad()
    def update(self, img_feat: torch.Tensor, txt_feat: torch.Tensor):
        # img_feat: [B, D], txt_feat: [M, D] (예: K*C 개 텍스트 임베딩 풀)
        X = img_feat.detach().to(torch.float32)
        T = txt_feat.detach().to(torch.float32)

        Ci = self._cov_shrink(X)
        Ct = self._cov_shrink(T)

        Ct_sqrt, Ci_invsqrt = self._psd_eigh_sqrt_invsqrt(Ct)

        m = self.momentum
        if self.initialized.item() == 0:
            self.running_Ci_invsqrt.copy_(Ci_invsqrt)
            self.running_Ct_sqrt.copy_(Ct_sqrt)
            if self.track_mean:
                self.running_mean_img.copy_(X.mean(dim=0))
            self.initialized.fill_(1)
        else:
            self.running_Ci_invsqrt.copy_(m * self.running_Ci_invsqrt + (1 - m) * Ci_invsqrt)
            self.running_Ct_sqrt.copy_(   m * self.running_Ct_sqrt    + (1 - m) * Ct_sqrt)
            if self.track_mean:
                self.running_mean_img.copy_(m * self.running_mean_img + (1 - m) * X.mean(dim=0))

    def transform(self, img_feat: torch.Tensor):
        X = img_feat.to(torch.float32)
        if self.track_mean:
            X = X - self.running_mean_img.to(X.dtype)
        X = X @ self.running_Ci_invsqrt   # whitening
        X = X @ self.running_Ct_sqrt      # recoloring
        return X.to(img_feat.dtype)

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.num_selected_prompts = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.num_classes = len(classnames)

        self.K = self.num_selected_prompts
        self.C = self.num_classes
        self.prompt_learner = nn.ModuleList([
            PromptLearner(cfg, classnames, clip_model)
            for i in range(self.num_selected_prompts)
        ])
        self.tokenized_prompts = self.prompt_learner[0].tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner[0])
    # --- Gating head: image-only -> K logits ---
        # D_text = self.text_encoder.text_projection.shape[1] if hasattr(self.text_encoder, "text_projection") else None
        self.gate = nn.Linear(self.image_encoder.output_dim if hasattr(self.image_encoder, "output_dim") else self.prompt_learner[0].ctx.shape[-1],  # 대개 CLIP visual dim
                        self.K, bias=True)
        D_vis = 512  # 안전한 기본값

        self.align_enable   = True
        self.align_momentum = 0.99
        self.align_shrink   = 0.05
        self.align_eps      = 1e-5
        self.align_track_mu = False
        self.align_update_every = 1  # 매 step 업데이트 간격

        self.align = FeatureCovAlign(
            dim=D_vis, momentum=self.align_momentum, shrink=self.align_shrink,
            eps=self.align_eps, track_mean=self.align_track_mu
        )
        self.register_buffer("_align_step", torch.tensor(0, dtype=torch.long))

        # --- 게이팅/지식증류 하이퍼 ---
        self.lambda_kd = float(getattr(cfg.TRAINER.TRIP, "LAMBDA_KD", 1.0))    # teacher vs student KL
        self.lambda_div = float(getattr(cfg.TRAINER.TRIP, "LAMBDA_DIV", 0.01)) # diversity(학생 분포에 적용)
        self.tau_gate   = float(getattr(cfg.TRAINER.TRIP, "TAU_GATE", 1.0))    # softmax 온도(학생)
        self.var_eps = float(getattr(cfg.TRAINER.TRIP, "VAR_EPS", 1e-5))
        self.min_var = float(getattr(cfg.TRAINER.TRIP, "VAR_MIN", 1e-6))
        self.alpha_pbeb = float(getattr(cfg.TRAINER.TRIP, "ALPHA_PBEB", 0.2))
        self.beta_var = float(getattr(cfg.TRAINER.TRIP, "BETA_VAR", 0.5))
        self.lambda_div = float(getattr(cfg.TRAINER.TRIP, "LAMBDA_DIV", 0.01))

        self.bootstrap_ratio = float(getattr(cfg.TRAINER.TRIP, "BOOTSTRAP_RATIO", 0.7))
    def forward(self, image, label=None):
        image = image.to(self.device, dtype=self.dtype)
        logit_scale = self.logit_scale.exp()

        # img_feat = self.image_encoder(image)
        # img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # [B, D]
        # B, D = img_feat.shape

        # # 2) 프롬프트별 텍스트 특징 및 로짓
        # text_feats_per_prompt, logits_per_prompt = [], []
        # for pl in self.prompt_learner:
        #     prompts = pl()
        #     tfeat = self.text_encoder(prompts, self.tokenized_prompts)
        #     tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
        #     text_feats_per_prompt.append(tfeat)
        #     logits = logit_scale * img_feat @ tfeat.t()
        #     logits_per_prompt.append(logits)
           # --- (A) 텍스트 임베딩 먼저 계산 (정렬용 텍스트 공분산 풀) ---
        text_feats_per_prompt = []
        for pl in self.prompt_learner:
            prompts = pl()
            tfeat = self.text_encoder(prompts, self.tokenized_prompts)  # [C, D]
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
            text_feats_per_prompt.append(tfeat)

        with torch.no_grad():
            # [K, C, D] -> [K*C, D]
            T_pool = torch.stack(text_feats_per_prompt, dim=0).reshape(-1, text_feats_per_prompt[0].shape[-1])

        # --- (B) 이미지 임베딩 ---
        img_feat = self.image_encoder(image)  # [B, D]

        # --- (C) 공분산 정렬 (학습 중 주기적으로 update, 항상 transform 적용) ---
        if self.align_enable:
            if self.training:
                self._align_step += 1
                if int(self._align_step.item()) % max(self.align_update_every, 1) == 0:
                    self.align.update(img_feat, T_pool)  # 러닝 평균 갱신
            img_feat = self.align.transform(img_feat)

        # --- (D) CLIP 관례의 L2 정규화 ---
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # [B, D]
        B, D = img_feat.shape

        # --- (E) 프롬프트별 로짓 ---
        logits_per_prompt = []
        for tfeat in text_feats_per_prompt:
            logits = logit_scale * img_feat @ tfeat.t()  # [B, C]
            logits_per_prompt.append(logits)

      

        # ---------- 평가 모드: 학생 게이트만 사용 ----------
        if label is None:
            # student weights from image-only gate
            gate_logits = self.gate(img_feat.to(torch.float32)) / max(self.tau_gate, 1e-6)  # [B,K] float32
            w_student32 = F.softmax(gate_logits, dim=1)                                     # [B,K]
            w_student = w_student32.to(self.dtype)
            weighted_logits = (w_student.unsqueeze(-1) * torch.stack(logits_per_prompt, dim=1)).sum(dim=1)
            return weighted_logits

        # ---------- 학습 모드 ----------
        label_idx = label.to(self.device).long().view(-1)
        K, C = self.num_selected_prompts, self.num_classes
        mu_stack = torch.stack(text_feats_per_prompt, dim=0)  # [K,C,D]

        # (a) 부트스트랩+PBEB로 teacher 분산 추정 (네가 쓴 코드 그대로)
        R = int(getattr(self, "bootstrap_reps", 8))
        boot_ratio = float(getattr(self, "bootstrap_ratio", 0.7))
        var_eps = float(getattr(self, "var_eps", 1e-6))
        min_var = float(getattr(self, "min_var", 1e-6))
        prompt_gaussians = [{} for _ in range(K)]
        class_masks = {c: (label_idx == c) for c in range(C)}

        for c in range(C):
            mask_c = class_masks[c]
            if mask_c.sum() == 0:
                continue
            feats_c = img_feat[mask_c]    # [Nc,D]
            Nc = feats_c.size(0)
            n_boot = max(1, int(round(Nc * boot_ratio)))
            for k in range(K):
                mu = mu_stack[k, c]
                if Nc < 2:
                    var_mean = torch.full_like(mu, getattr(self, "default_var", 1e-2))
                    var_disp = torch.zeros_like(mu)
                else:
                    idx = torch.randint(0, Nc, (R, n_boot), device=self.device)
                    feats_boot = feats_c[idx]                                  # [R,n_boot,D]
                    var_r = ((feats_boot - mu) ** 2).mean(dim=1) + var_eps     # [R,D]
                    var_mean = var_r.mean(dim=0)                                # [D]
                    var_disp = var_r.var(dim=0, unbiased=False)                 # [D]
                prompt_gaussians[k][c] = (mu, var_mean, var_disp)

        # (b) teacher 라우팅 점수 NLEEP (float32) → softmax
        nleep_scores = torch.full((B, K), -1e9, device=self.device, dtype=torch.float32)
        for k in range(K):
            for c, (mu, var_mean, var_disp) in prompt_gaussians[k].items():
                mask_c = class_masks[c]
                if mask_c.sum() == 0:
                    continue
                x = img_feat[mask_c].to(torch.float32)
                mu32 = mu.to(torch.float32)
                adjusted_var = (var_mean + self.alpha_pbeb * var_disp).to(torch.float32)
                adjusted_var = torch.clamp(adjusted_var, min=min_var)
                diff = x - mu32
                log_prob = -0.5 * ((diff * diff) / adjusted_var + adjusted_var.log()).sum(dim=1)
                nleep_scores[mask_c, k] = log_prob
        row_has_val = torch.isfinite(nleep_scores).any(dim=1)
        if not row_has_val.all():
            nleep_scores[~row_has_val] = 0.0
        weights_teacher32 = F.softmax(nleep_scores, dim=1)     # [B,K] float32
        # weights_teacher = weights_teacher32.to(self.dtype)
        
        gate_logits = self.gate(img_feat.to(torch.float32)) / max(self.tau_gate, 1e-6)  # [B,K] float32
        weights_student32 = F.softmax(gate_logits, dim=1)
        weights_student = weights_student32.to(self.dtype)

        # (d) 학생 분포로 앙상블 출력
        weighted_logits = (weights_student.unsqueeze(-1) *
                        torch.stack(logits_per_prompt, dim=1)).sum(dim=1)  # [B,C]

        # (e) 지식증류 KL(student || teacher) + diversity(student)
        #     - KL은 float32에서 batchmean으로, teacher는 detach
        kd = F.kl_div(
            F.log_softmax(gate_logits, dim=1),              # log P_student
            weights_teacher32.detach(),                     #   Q_teacher
            reduction="batchmean"
        )
        # diversity: KL(w_student || U) 상수항 제외 → Σ w log w
        eps = 1e-6 if weights_student.dtype == torch.float16 else 1e-12
        # w_safe = weights_student.clamp_min(eps)
        # self.loss_div = (w_safe * (w_safe + eps).log()).sum(dim=1).mean()

        # 모델 안에서 합치지 않고, 트레이너에서 조합해서 쓰기 쉽게 노출
        self.loss_gate_kd = kd
        # weighted_logits만 반환 (트레이너에서 CE + 람다*kd + 람다*div 더하기)
        return weighted_logits

    # def forward(self, image, label=None):
    #     # 1) 이미지 특징
    #     image = image.to(self.device, dtype=self.dtype)
    #     img_feat = self.image_encoder(image)
    #     img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # [B, D]
    #     B, D = img_feat.shape
    #     logit_scale = self.logit_scale.exp()

    #     # 2) 프롬프트별 텍스트 특징 및 로짓
    #     text_feats_per_prompt, logits_per_prompt = [], []
    #     for pl in self.prompt_learner:
    #         prompts = pl()
    #         tfeat = self.text_encoder(prompts, self.tokenized_prompts)
    #         tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
    #         text_feats_per_prompt.append(tfeat)
    #         logits = logit_scale * img_feat @ tfeat.t()
    #         logits_per_prompt.append(logits)

    #     # 3) 평가 모드: 평균 앙상블(그대로 유지)
    #     stacked_logits = torch.stack(logits_per_prompt, dim=0)  # [K, B, C]
    #     avg_logits = stacked_logits.mean(dim=0)
    #     if label is None:
    #         return avg_logits

    #     # -------------------------------
    #     # 4) 학습 모드: 부트스트랩 + PBEB + 라우팅
    #     # -------------------------------
    #     label_idx = label.to(self.device).long().view(-1)  # [B]
    #     K, C = self.num_selected_prompts, self.num_classes
    #     mu_stack = torch.stack(text_feats_per_prompt, dim=0)  # [K, C, D]

    #     # 하이퍼 (없으면 기본값 사용)
    #     R = int(getattr(self, "bootstrap_reps", 8))                # 부트스트랩 반복 횟수
    #     boot_ratio = float(getattr(self, "bootstrap_ratio", 0.7))  # 각 부트 크기 비율
    #     var_eps = float(getattr(self, "var_eps", 1e-6))
    #     min_var = float(getattr(self, "min_var", 1e-6))

    #     prompt_gaussians = [{} for _ in range(K)]
    #     class_masks = {c: (label_idx == c) for c in range(C)}

    #     # (a) 클래스별로, 프롬프트 k마다 R회 부트스트랩으로 분산 추정
    #     for c in range(C):
    #         mask_c = class_masks[c]
    #         if mask_c.sum() == 0:
    #             continue
    #         feats_c = img_feat[mask_c]         # [Nc, D]
    #         Nc = feats_c.size(0)
    #         n_boot = max(1, int(round(Nc * boot_ratio)))

    #         for k in range(K):
    #             mu = mu_stack[k, c]            # [D]

    #             if Nc < 2:
    #                 # 표본이 너무 적으면 안전 폴백
    #                 var_mean = torch.full_like(mu, getattr(self, "default_var", 1e-2))
    #                 var_disp = torch.zeros_like(mu)
    #             else:
    #                 # [R, n_boot] 부트스트랩 인덱스(중복 허용)
    #                 idx = torch.randint(0, Nc, (R, n_boot), device=self.device)
    #                 # [R, n_boot, D]
    #                 feats_boot = feats_c[idx]
    #                 # 각 리샘플 r에 대해 분산 추정 (diag) : mean_b (x - mu)^2
    #                 var_r = ((feats_boot - mu) ** 2).mean(dim=1) + var_eps   # [R, D]
    #                 var_mean = var_r.mean(dim=0)                              # [D]
    #                 var_disp = var_r.var(dim=0, unbiased=False)               # [D]

    #             prompt_gaussians[k][c] = (mu, var_mean, var_disp)

    #     # (b) NLEEP 점수 (float32로 안정화)
    #     nleep_scores = torch.full((B, K), -1e9, device=self.device, dtype=torch.float32)
    #     var_loss_accum = []

    #     for k in range(K):
    #         for c, (mu, var_mean, var_disp) in prompt_gaussians[k].items():
    #             mask_c = class_masks[c]
    #             if mask_c.sum() == 0:
    #                 continue

    #             x = img_feat[mask_c].to(torch.float32)   # [Nc', D] 안정화
    #             mu32 = mu.to(torch.float32)
    #             # PBEB 보정: var_mean + alpha * (리샘플 간 분산)
    #             adjusted_var = (var_mean + self.alpha_pbeb * var_disp).to(torch.float32)
    #             adjusted_var = torch.clamp(adjusted_var, min=min_var)

    #             # log N(x|mu, diag(adjusted_var)) up to const
    #             diff = x - mu32
    #             log_prob = -0.5 * ((diff * diff) / adjusted_var + adjusted_var.log()).sum(dim=1)
    #             nleep_scores[mask_c, k] = log_prob
    #             var_loss_accum.append(var_mean.mean())

    #     row_has_val = torch.isfinite(nleep_scores).any(dim=1)
    #     if not row_has_val.all():
    #         nleep_scores[~row_has_val] = 0.0

    #     weights32 = F.softmax(nleep_scores, dim=1)   # [B, K] float32
    #     weights = weights32.to(self.dtype)

    #     # (d) div loss (KL(w||U) 상수항 제거) — fp16 안전 eps
    #     eps = 1e-6 if weights.dtype == torch.float16 else 1e-12
    #     w_safe = weights.clamp_min(eps)
    #     self.loss_div = (w_safe * (w_safe + eps).log()).sum(dim=1).mean()

    #     # (e) 가중 앙상블
    #     weighted_logits = torch.zeros(B, C, device=self.device, dtype=self.dtype)
    #     for k in range(K):
    #         w_k = weights[:, k].unsqueeze(1)  # [B,1]
    #         weighted_logits = weighted_logits + w_k * logits_per_prompt[k]

    #     return weighted_logits


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
   

    

@TRAINER_REGISTRY.register()
class TRIP(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.TRIP.PREC in ["fp16", "fp32", "amp"]
   
    def _ensure_writer(self, initialize=False):
        # 이미 writer가 초기화되었으면 바로 반환 (싱글톤 패턴)
        if hasattr(self, 'writer') and self.writer is not None and not initialize:
            return self.writer
        if not hasattr(self, 'output_dir'):
            tb_dir = os.path.join('/workspace/Soft-Prompt-Generation/', "tensorboard/vlcs/fallback_run")
        else:
            tb_dir = os.path.join(self.output_dir, 'ana') 
      
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
        
        
        if cfg.TRAINER.TRIP.PREC == "fp32" or cfg.TRAINER.TRIP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()
      
        clip_model = load_clip_to_cpu(cfg)

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if ("prompt_learner" not in name) and ("gate" not in name):
                param.requires_grad_(False)


        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        train_params = chain(self.model.prompt_learner.parameters(),
                     self.model.gate.parameters())
        self.optim = build_optimizer(train_params, cfg.OPTIM)
        # NOTE: only give prompt_learner to the optimizer
        # self.optim = build_optimizer(self.model.prompt_learner.parameters(), cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self._global_step = 0

        self.scaler = GradScaler() if cfg.TRAINER.TRIP.PREC == "amp" else None
        # 학습 가능한(gradient 있는) 파라미터 개수 확인
        print("=== Trainable Parameters by Module ===")
        total = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                print(f"{name:50s} {p.numel():,d}")
                total += p.numel()
        print(f"Total trainable params: {total:,d}")

    def forward_backward(self, batch):
        image, label,domain = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.TRIP.PREC
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
            
            # div_loss = self.model.loss_div # New
            kd_loss = self.model.loss_gate_kd
            lambda_loss_var = 0.1 # New
            # Cross-Entropy Loss
            ce_loss = F.cross_entropy(output, label)
            
            # [Modified] 최종 Loss = CE + λ_var * Var_Reg + λ_util * Util_Reg
            total_loss = (
                ce_loss + kd_loss
                # + lambda_loss_var * div_loss
            )
            
            self.model_backward_and_update(total_loss)
            loss = total_loss # 최종 Loss를 summary에 사용
            
        loss_summary = {
            "loss": loss.item(),
            "ce_loss": ce_loss.item(),
            "acc": compute_accuracy(output, label)[0].item(), 
            "kd_loss": kd_loss.item(), # New Loss Variance Reg 로깅
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
            
        LOG_FREQ = 50 
        if self._global_step % LOG_FREQ == 0:
            
            w = self._ensure_writer()
            # [Modified] TensorBoard 로깅 추가
            # w.add_scalar("train/div_loss", div_loss.item(), self._global_step) # New Loss Var Reg 로깅
      
        
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
            
