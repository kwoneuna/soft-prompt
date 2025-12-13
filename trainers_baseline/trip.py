import os
import time
import datetime
import numpy as np
from tqdm import tqdm
from itertools import chain
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast
from scipy.stats import pointbiserialr
from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
from torch.utils.tensorboard import SummaryWriter
import json
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

def relu_with_gelu_grad(x: torch.Tensor) -> torch.Tensor:
    """
    forward 값은 ReLU(x), gradient는 GELU(x)를 따르는 functional 버전.
    """
    relu_x = F.relu(x)
    gelu_x = F.gelu(x)
    return relu_x.detach() + gelu_x - gelu_x.detach()


def non_negative_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Non-negative Contrastive Loss 스타일의 Cross Entropy:

    1) logits 에 GELU-grad ReLU 적용 (값은 비음수, gradient는 GELU 기준)
    2) 그 위에 Cross Entropy 계산
    """
    logits_nn = relu_with_gelu_grad(logits)
    return F.cross_entropy(logits_nn, target, reduction=reduction)

class ZeroShotTextFeatures(nn.Module):
    def __init__(self, classnames, clip_model, template="a photo of a {}."):
        super().__init__()

        # ⚠️ CPU에서 half 안 됨 → device / dtype 강제 정리
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # CLIP 전체를 해당 device로 옮긴다 (idempotent)
        clip_model.to(device)

        # CPU라면 반드시 float32로 변환
        if device.type == "cpu":
            clip_model.float()

        # zero-shot용 prompt 만들기
        prompts = [template.format(c.replace("_", " ")) for c in classnames]
        tokenized = clip.tokenize(prompts).to(device)

        with torch.no_grad():
            text_feat = clip_model.encode_text(tokenized)  # 여기서 이제 CPU면 float32, GPU면 fp16/32
            text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

        # buffer는 float32로 저장해 두는 걸 추천 (나중에 필요하면 cast)
        self.register_buffer("zs_text_feat", text_feat.to(torch.float32))

    def forward(self):
        # [C, D] 반환 (기본적으로 float32)
        return self.zs_text_feat
    
class ReLUWithGELUGrad(nn.Module):
    def forward(self, x):
        # 값은 ReLU(x), 기울기는 GELU(x) 기준
        relu_x = F.relu(x)
        gelu_x = F.gelu(x)
        return relu_x.detach() + gelu_x - gelu_x.detach()

class ChannelWisePreGate(nn.Module):
    def __init__(self, dim, out_dim, reduction=4):
        super().__init__()
        hid = max(1, dim // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hid),
            ReLUWithGELUGrad(),          # 여기서 트릭 적용
            nn.Linear(hid, out_dim),
            nn.Sigmoid()                 # 0~1 
        )

    def forward(self, x):
        # x: [B, dim]
        w = self.mlp(x) 
        return w
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.num_selected_prompts = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.num_classes = len(classnames)
        proj_dim = int(getattr(cfg.TRAINER.TRIP, "PROJ_DIM", 512))
        self.proj_dim = proj_dim

        self.K = self.num_selected_prompts
        self.C = self.num_classes
        self.prompt_learner = nn.ModuleList([
            PromptLearner(cfg, classnames, clip_model)
            for _ in range(self.num_selected_prompts)
        ])
        self.tokenized_prompts = self.prompt_learner[0].tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner[0])
       
        self.tau_gate  = float(getattr(cfg.TRAINER.TRIP, "TAU_GATE", 1.0))    # softmax 온도(학생)
        self.var_eps = float(getattr(cfg.TRAINER.TRIP, "VAR_EPS", 1e-5))
        self.min_var = float(getattr(cfg.TRAINER.TRIP, "VAR_MIN", 1e-6))
        self.alpha_pbeb = float(getattr(cfg.TRAINER.TRIP, "ALPHA_PBEB", 0.2))
        self.beta_var = float(getattr(cfg.TRAINER.TRIP, "BETA_VAR", 0.5))        
        self.gate = ChannelWisePreGate(dim=512, out_dim=self.K, reduction=4)
        self.bootstrap_ratio = float(getattr(cfg.TRAINER.TRIP, "BOOTSTRAP_RATIO", 0.7))
        self.zs_text = ZeroShotTextFeatures(classnames, clip_model)
        self.lambda_oracle = 0.5
        self.lambda_zs_kd = 0.5
        self.lambda_kd = 1.0
        self.tau_zs = 0.1
        self.tau_oracle = 1.0
        self.alpha_oracle_mix = 0.5
        self.hparams = {
            "backbone": cfg.MODEL.BACKBONE.NAME,
            "num_prompts": self.num_selected_prompts,
            "proj_dim": self.proj_dim,
            "tau_gate": self.tau_gate,
            "tau_zs": self.tau_zs,
            "var_eps": self.var_eps,
            "min_var": self.min_var,
            "alpha_pbeb": self.alpha_pbeb,
            "beta_var": self.beta_var,
            "bootstrap_ratio": self.bootstrap_ratio,
            "lambda_oracle": self.lambda_oracle,
            "lambda_zs_kd": self.lambda_zs_kd,
            "lambda_kd": self.lambda_kd,
            "prec": cfg.TRAINER.TRIP.PREC,  # precision 정보 등 추가 가능
            "batch_size": cfg.DATALOADER.TRAIN_X.BATCH_SIZE,
            "lr": cfg.OPTIM.LR,
            "tau_oracle":self.tau_oracle,
        }
                # ---------------------------
        # (1) EMA stats (prompt x class x feat_dim)
        # (2) mu-mix: mu = alpha*mu_text + (1-alpha)*mu_img_ema
        # (5) speed: will vectorize NLEEP scoring (done in forward)
        # ---------------------------
        self.bootstrap_reps = int(getattr(cfg.TRAINER.TRIP, "BOOTSTRAP_REPS", 8))

        self.ema_m = float(getattr(cfg.TRAINER.TRIP, "EMA_M", 0.99))                 # 0.95~0.99
        self.mu_mix_alpha = float(getattr(cfg.TRAINER.TRIP, "MU_MIX_ALPHA", 0.7))    # 텍스트 중심 비율
        self.var_shrink = float(getattr(cfg.TRAINER.TRIP, "VAR_SHRINK_LAMBDA", 0.1)) # 0이면 off

        # CLIP feature dim 추정 (대부분 512)
        feat_dim = int(getattr(clip_model, "embed_dim", 512))
        self.feat_dim = feat_dim

        # EMA buffers: 모델 .to(device) 따라 이동됨
        self.register_buffer("ema_mu", torch.zeros(self.K, self.C, feat_dim))
        self.register_buffer("ema_var_mean", torch.ones(self.K, self.C, feat_dim) * 1e-2)
        self.register_buffer("ema_var_disp", torch.zeros(self.K, self.C, feat_dim))

        # 이미지 중심 EMA (class x feat_dim)
        self.register_buffer("ema_img_mu", torch.zeros(self.C, feat_dim))

        # EMA 초기화 여부
        self.register_buffer("ema_inited", torch.tensor(False))

        # hparams에 기록(선택)
        self.hparams.update({
            "ema_m": self.ema_m,
            "mu_mix_alpha": self.mu_mix_alpha,
            "var_shrink_lambda": self.var_shrink,
            "bootstrap_reps": self.bootstrap_reps,
        })

    def get_entropy(self,probs):
            return -(probs * (probs + 1e-8).log()).sum(dim=1, keepdim=True)
    def forward(self, image, label=None):
        image = image.to(self.device, dtype=self.dtype)
        logit_scale = self.logit_scale.exp()

        
        img_feat = self.image_encoder(image)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # [B, D]
        B, D = img_feat.shape

        text_feats_per_prompt, logits_per_prompt = [], []
        for pl in self.prompt_learner:
            prompts = pl()
            tfeat = self.text_encoder(prompts, self.tokenized_prompts)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)     # [C,D]
            text_feats_per_prompt.append(tfeat)

            logits = logit_scale * img_feat @ tfeat.t()          # [B,C]
            logits_per_prompt.append(logits)

        # [B,K,C]로 한 번 쌓아두기 (label 유무와 상관없이 사용)
        logits_stack = torch.stack(logits_per_prompt, dim=1)     # [B,K,C]
        if label is None:
            gate_logits = self.gate(img_feat.to(torch.float32)) / self.tau_gate  # [B,K] float32
            w_student32 = F.softmax(gate_logits, dim=1)                                     # [B,K]
            w_student = w_student32.to(self.dtype)
            weighted_logits = (w_student.unsqueeze(-1) * logits_stack).sum(dim=1)  
                        # --- debug stats (no label) ---
            with torch.no_grad():
                gate_prob = w_student32  # [B,K]
                gate_ent = -(gate_prob * (gate_prob + 1e-8).log()).sum(dim=1)  # [B]
                usage = gate_prob.mean(dim=0)  # [K]

                self.debug_stats = {
                    "gate/entropy": gate_ent.mean().item(),
                    "gate/usage_max": usage.max().item(),
                    "gate/usage_min": usage.min().item(),
                    "gate/usage_std": usage.std(unbiased=False).item(),
                }
            return weighted_logits
          
        # ======================================================
        # 2) label 있는 경우: bootstrap + EMA teacher + NLEEP routing
        # ======================================================
        label_idx = label.to(self.device).long().view(-1)
        K, C = self.num_selected_prompts, self.num_classes
        mu_stack = torch.stack(text_feats_per_prompt, dim=0)  # [K,C,D]

        R = int(getattr(self, "bootstrap_reps", 8))
        boot_ratio = float(getattr(self, "bootstrap_ratio", 0.7))
        var_eps = float(getattr(self, "var_eps", 1e-6))

        # 클래스별 마스크 (Boolean)
        class_masks = {c: (label_idx == c) for c in range(C)}

        # ---- 배치 통계 텐서
        batch_mu       = torch.zeros(K, C, D, device=self.device)
        batch_var_mean = torch.ones(K, C, D, device=self.device) * 1e-2
        batch_var_disp = torch.zeros(K, C, D, device=self.device)

        # ---- (2) 이미지 중심 배치 평균 -> EMA 업데이트용
        batch_img_mu = torch.zeros(C, D, device=self.device)
        valid_cls = torch.zeros(C, device=self.device, dtype=torch.bool)

        for c in range(C):
            mask_c = class_masks[c]
            if mask_c.sum() == 0:
                continue

            valid_cls[c] = True
            feats_c = img_feat[mask_c].detach()  # [Nc, D]
            Nc = feats_c.size(0)
            batch_img_mu[c] = feats_c.mean(dim=0)

            n_boot = max(1, int(round(Nc * boot_ratio)))

            for k in range(K):
                mu_text = mu_stack[k, c]  # [D]

                # (2) mu-mix: 텍스트 중심 + (EMA 이미지 중심) 또는 배치 이미지 중심
                # EMA 이미지 중심이 아직 안정적이지 않으면 배치 중심을 먼저 사용
                # (ema_inited=False인 초기에는 batch_img_mu가 더 안전)
                if bool(self.ema_inited.item()):
                    mu_img = self.ema_img_mu[c]
                else:
                    mu_img = batch_img_mu[c]

                mu_mix = self.mu_mix_alpha * mu_text + (1.0 - self.mu_mix_alpha) * mu_img

                if Nc < 2:
                    batch_mu[k, c] = mu_mix
                    batch_var_mean[k, c] = torch.full_like(mu_mix, 1e-2)
                    batch_var_disp[k, c] = torch.zeros_like(mu_mix)
                else:
                    idx = torch.randint(0, Nc, (R, n_boot), device=self.device)  # [R, n_boot]
                    feats_boot = feats_c[idx]                                     # [R, n_boot, D]

                    # (2) mu_mix 기준 분산 추정
                    var_r = ((feats_boot - mu_mix) ** 2).mean(dim=1) + var_eps    # [R, D]

                    batch_mu[k, c]       = mu_mix
                    batch_var_mean[k, c] = var_r.mean(dim=0)
                    batch_var_disp[k, c] = var_r.var(dim=0, unbiased=False)

        # ======================================================
        # (1) EMA 업데이트 (no_grad)
        # ======================================================
        with torch.no_grad():
            if not bool(self.ema_inited.item()):
                # 초기에는 valid class만 채우고 inited=True
                if valid_cls.any():
                    self.ema_img_mu[valid_cls] = batch_img_mu[valid_cls].to(self.ema_img_mu.dtype)
                    self.ema_mu[:, valid_cls, :] = batch_mu[:, valid_cls, :].to(self.ema_mu.dtype)
                    self.ema_var_mean[:, valid_cls, :] = batch_var_mean[:, valid_cls, :].to(self.ema_var_mean.dtype)
                    self.ema_var_disp[:, valid_cls, :] = batch_var_disp[:, valid_cls, :].to(self.ema_var_disp.dtype)
                self.ema_inited.fill_(True)
            else:
                m = self.ema_m
                if valid_cls.any():
                    # img center EMA
                    self.ema_img_mu[valid_cls] = (
                        m * self.ema_img_mu[valid_cls] + (1.0 - m) * batch_img_mu[valid_cls]
                    ).to(self.ema_img_mu.dtype)

                    # prompt/class stats EMA
                    vc = valid_cls.view(1, C, 1)  # [1,C,1] broadcast
                    self.ema_mu = torch.where(
                        vc, (m * self.ema_mu + (1.0 - m) * batch_mu.to(self.ema_mu.dtype)), self.ema_mu
                    )
                    self.ema_var_mean = torch.where(
                        vc, (m * self.ema_var_mean + (1.0 - m) * batch_var_mean.to(self.ema_var_mean.dtype)), self.ema_var_mean
                    )
                    self.ema_var_disp = torch.where(
                        vc, (m * self.ema_var_disp + (1.0 - m) * batch_var_disp.to(self.ema_var_disp.dtype)), self.ema_var_disp
                    )

        # ======================================================
        # (5) NLEEP score 계산 벡터화: class 루프만, prompt 루프 제거
        # ======================================================
        nleep_scores = torch.full((B, K), -1e9, device=self.device, dtype=torch.float32)

        # [D] 형태로 전역 분산 prior (EMA var_mean 기반)
        global_var = self.ema_var_mean.to(torch.float32).mean(dim=(0, 1))  # [D]

        for c in range(C):
            mask_c = class_masks[c]
            if mask_c.sum() == 0:
                continue

            x = img_feat[mask_c].to(torch.float32)  # [Nc, D]

            mu = self.ema_mu[:, c, :].to(torch.float32)              # [K, D]
            var_mean = self.ema_var_mean[:, c, :].to(torch.float32)  # [K, D]
            var_disp = self.ema_var_disp[:, c, :].to(torch.float32)  # [K, D]

            adjusted_var = var_mean + self.alpha_pbeb * var_disp

            # (2) shrinkage (optional): 폭주 방지
            if self.var_shrink > 0:
                adjusted_var = (1.0 - self.var_shrink) * adjusted_var + self.var_shrink * global_var.unsqueeze(0)

            adjusted_var = torch.clamp(adjusted_var, min=self.min_var)  # [K, D]

            diff = x.unsqueeze(1) - mu.unsqueeze(0)                     # [Nc, K, D]
            log_prob = -0.5 * (
                (diff * diff) / adjusted_var.unsqueeze(0) + adjusted_var.log().unsqueeze(0)
            ).sum(dim=-1)                                               # [Nc, K]

            nleep_scores[mask_c] = log_prob

        row_has_val = torch.isfinite(nleep_scores).any(dim=1)
        if not row_has_val.all():
            nleep_scores[~row_has_val] = 0.0

        weights_teacher32 = F.softmax(nleep_scores, dim=1)  # [B, K] float32
        gate_logits = self.gate(img_feat.to(torch.float32)) / max(self.tau_gate, 1e-6)  # [B,K] float32

        tau_zs = self.tau_zs
        zs_text_feat = self.zs_text()  
        img_feat=img_feat.to(torch.float32)# [C, D]
        zs_logits = logit_scale * img_feat @ zs_text_feat.t()  # [B, C]
     
        teacher_T = zs_logits.detach() / tau_zs
        kd_zs_per_prompt = []
        for k in range(self.K):
            logits_k = logits_per_prompt[k].to(torch.float32) # [B, C]
            student_T_k = logits_k / tau_zs
            kd_k = F.kl_div(
                F.log_softmax(student_T_k, dim=1),        # log P_student_k(y|x)
                F.softmax(teacher_T, dim=1),                        # Q_target (정답 레이블로 보정됨)
                reduction="batchmean"
            ) * (tau_zs ** 2)
            kd_zs_per_prompt.append(kd_k)

        kd_zs = torch.stack(kd_zs_per_prompt).mean()
       
     
        self.loss_zs_kd = kd_zs
        ce_per_prompt = []
        for k in range(self.K):
            logits_k = logits_per_prompt[k].to(torch.float32)  # [B, C]
            ce_k = non_negative_cross_entropy(
                logits_k,
                label_idx,
                reduction="none"
            )  # [B]
            ce_per_prompt.append(ce_k)
        ce_per_prompt = torch.stack(ce_per_prompt, dim=1)  # [B, K]
        # weights_teacher = weights_teacher32.to(self.dtype)
        weights_teacher = weights_teacher32.detach().to(self.dtype)

        loss_teacher = (weights_teacher*ce_per_prompt).sum(dim=1).mean()  # scalar
        self.loss_teacher = loss_teacher

        weighted_logits = (weights_teacher.unsqueeze(-1) *  torch.stack(logits_per_prompt, dim=1)).sum(dim=1) # [B,C]
   
        correct_class_logits = torch.gather(
            logits_stack, 
            dim=2, 
            index=label_idx.view(-1, 1, 1).expand(-1, self.K, 1)
        ).squeeze(2)  # [B, K]
        
        tau_oracle = self.tau_oracle# 하이퍼파라미터 (상황에 맞춰 조절)
        oracle_weights = F.softmax(correct_class_logits / tau_oracle, dim=1).detach() # [B, K]

        loss_oracle = F.kl_div(
            F.log_softmax(gate_logits, dim=1), # Student Log P
            oracle_weights,                    # Target Q (정답 기반)
            reduction="batchmean"
        )
        self.loss_oracle = loss_oracle
        kd = F.kl_div(
            F.log_softmax(gate_logits, dim=1),  # log P_student
            weights_teacher32.detach(),         # Q_teacher
            reduction="batchmean"
        )
        self.loss_gate_kd = kd
         # --- debug stats (with label) ---
        with torch.no_grad():
            # gate stats
            gate_prob = F.softmax(gate_logits, dim=1)  # [B,K]
            gate_ent = -(gate_prob * (gate_prob + 1e-8).log()).sum(dim=1)  # [B]
            gate_usage = gate_prob.mean(dim=0)  # [K]

            # teacher stats
            teacher_prob = weights_teacher32  # [B,K]
            teacher_ent = -(teacher_prob * (teacher_prob + 1e-8).log()).sum(dim=1)  # [B]
            teacher_usage = teacher_prob.mean(dim=0)  # [K]

            # nleep scale sanity (row-wise max 중심)
            nleep_row_max = nleep_scores.max(dim=1).values  # [B]
            nleep_row_min = nleep_scores.min(dim=1).values  # [B]

            self.debug_stats = {
                "gate/entropy": gate_ent.mean().item(),
                "gate/usage_max": gate_usage.max().item(),
                "gate/usage_min": gate_usage.min().item(),
                "gate/usage_std": gate_usage.std(unbiased=False).item(),

                "teacher/entropy": teacher_ent.mean().item(),
                "teacher/usage_max": teacher_usage.max().item(),
                "teacher/usage_min": teacher_usage.min().item(),
                "teacher/usage_std": teacher_usage.std(unbiased=False).item(),

                "nleep/row_max_mean": nleep_row_max.mean().item(),
                "nleep/row_max_std": nleep_row_max.std(unbiased=False).item(),
                "nleep/row_min_mean": nleep_row_min.mean().item(),
            }
        return weighted_logits
   
@TRAINER_REGISTRY.register()
class TRIP(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.TRIP.PREC in ["fp16", "fp32", "amp"]
   
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
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
            if ("prompt_learner" not in name) and ("gate" not in name) : # <-- 1단계: visual_feature_learner를 제외
                param.requires_grad_(False)

        # for name, param in self.model.visual_proj.named_parameters():
        #      param.requires_grad_(True)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        train_params = chain(
            self.model.prompt_learner.parameters(), # 이건 이미 iterable (generator)
            self.model.gate.parameters(),           # 이것도 iterable
                       # <--- [중요] 이렇게 리스트로 감싸야 합니다!
        )
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
        self.save_hyperparameters()
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
            # kd_loss = self.model.lambda_kd*self.model.loss_gate_kd
            teacher_loss = self.model.loss_teacher
            # teacher_loss = non_negative_cross_entropy(output,label)
            oracle_loss = self.model.lambda_oracle*self.model.loss_oracle
            zs_kd_loss =  self.model.lambda_zs_kd*self.model.loss_zs_kd
            kd_loss = self.model.loss_gate_kd*self.model.lambda_kd
            #loss_flag
            total_loss = (
                teacher_loss +  oracle_loss+zs_kd_loss +kd_loss
            )
            loss = total_loss # 최종 Loss를 summary에 사용
            self.model_backward_and_update(total_loss)
        loss_summary = {
            "loss": loss.item(),
            'teacher_loss': teacher_loss.item(),
            "loss_zs_kd": self.model.loss_zs_kd.item(),
            'loss_oracle': self.model.loss_oracle.item(),
            'kd_loss': self.model.loss_gate_kd.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        if hasattr(self.model, "debug_stats") and isinstance(self.model.debug_stats, dict):
            loss_summary.update(self.model.debug_stats)


        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
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
    def save_hyperparameters(self):
        """모델의 하이퍼파라미터를 output_dir에 JSON 파일로 저장"""
        if not hasattr(self, 'output_dir'):
            print("Warning: output_dir is not defined yet.")
            return

        save_path = os.path.join(self.output_dir, "hyperparameters.json")
        
        # 모델에 저장해둔 hparams 가져오기
        params_to_save = self.model.hparams
        
        try:
            with open(save_path, "w") as f:
                # indent=4로 설정하면 사람이 읽기 편하게 들여쓰기 됨
                json.dump(params_to_save, f, indent=4)
            print(f"[Info] Hyperparameters saved to: {save_path}")
        except Exception as e:
            print(f"[Error] Failed to save hyperparameters: {e}")
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
            