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
# class ChannelWisePreGate(nn.Module):
#     def __init__(self, dim, reduction=4):
#         super().__init__()
#         hid = max(1, dim // reduction)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, hid),
#             nn.ReLU(),
#             nn.Linear(hid, dim),
#             nn.Sigmoid()  # 0~1 채널별 gate
#         )

#     def forward(self, x):
#         # x: [B, D]
#         w = self.mlp(x)        # [B, D]
#         return w
# class ChannelWisePreGate(nn.Module):
#     def __init__(self, dim, out_dim, reduction=4):
#         super().__init__()
#         hid = max(1, dim // reduction)
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, hid),
#             nn.ReLU(),
#             nn.Linear(hid, out_dim),
#             nn.Sigmoid()  # 0~1 범위 gate
#         )

#     def forward(self, x):
#         # x: [B, dim]
#         w = self.mlp(x)        # [B, out_dim]
#         return w

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
            for i in range(self.num_selected_prompts)
        ])
        self.tokenized_prompts = self.prompt_learner[0].tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner[0])
        # --- Gating head: image-only -> K logits ---
        # self.gate = nn.Linear(self.image_encoder.output_dim if hasattr(self.image_encoder, "output_dim") else self.prompt_learner[0].ctx.shape[-1],  # 대개 CLIP visual dim
        #                          self.K, bias=True)
        self.lambda_kd = float(getattr(cfg.TRAINER.TRIP, "LAMBDA_KD", 1.0))    # teacher vs student KL
        self.tau_gate  = float(getattr(cfg.TRAINER.TRIP, "TAU_GATE", 1.0))    # softmax 온도(학생)
        self.var_eps = float(getattr(cfg.TRAINER.TRIP, "VAR_EPS", 1e-5))
        self.min_var = float(getattr(cfg.TRAINER.TRIP, "VAR_MIN", 1e-6))
        self.alpha_pbeb = float(getattr(cfg.TRAINER.TRIP, "ALPHA_PBEB", 0.2))
        self.beta_var = float(getattr(cfg.TRAINER.TRIP, "BETA_VAR", 0.5))
        
        D = 512
        self.gate = ChannelWisePreGate(dim=512, out_dim=self.K, reduction=4)
      
        self.lambda_neighbor = float(
            getattr(cfg.TRAINER.TRIP, "LAMBDA_NEIGHBOR", 1.0)
        )
        self.lambda_align = float(getattr(cfg.TRAINER.TRIP, "LAMBDA_ALIGN", 0.1)) # Alignment Loss (새 특징 vs 텍스트 특징)
        self.lambda_reg = float(getattr(cfg.TRAINER.TRIP, "LAMBDA_REG", 0.1))      # Regularization Loss (새 특징 vs 원래 특징)
        # self.gate = nn.Linear(self.image_encoder.output_dim, self.K, bias=True)
        self.bootstrap_ratio = float(getattr(cfg.TRAINER.TRIP, "BOOTSTRAP_RATIO", 0.7))
        D = 512
        self.register_buffer(
            "ema_mu", 
            torch.zeros(self.K, self.C, D, dtype=self.dtype)
        )
        self.register_buffer(
            "ema_var_mean", 
            torch.ones(self.K, self.C, D, dtype=self.dtype) * 1e-2
        )
        self.register_buffer(
            "ema_var_disp", 
            torch.zeros(self.K, self.C, D, dtype=self.dtype)
        )
        self.zs_text = ZeroShotTextFeatures(classnames, clip_model)

        self.ema_momentum = float(getattr(cfg.TRAINER.TRIP, "EMA_MOMENTUM", 0.99))
        self.use_ema_teacher = False  # fla
        
        self.lambda_zs_kd = float(
            getattr(cfg.TRAINER.TRIP, "LAMBDA_ZS_KD", 1.0)
        )
        self.tau_zs = float(
            getattr(cfg.TRAINER.TRIP, "TAU_ZS", 1.0)    # class-logit KD 온도
        )
        # ---- teacher prompt (EMA) 세팅 ----
        self.prompt_ema_beta = float(
            getattr(cfg.TRAINER.TRIP, "PROMPT_EMA_BETA", 0.99)
        )

    def forward(self, image, label=None):
        image = image.to(self.device, dtype=self.dtype)
        logit_scale = self.logit_scale.exp()

        
        img_feat = self.image_encoder(image)
        n_img = img_feat

        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)  # [B, D]
        B, D = img_feat.shape
        # 2) 프롬프트별 텍스트 특징 및 로짓
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
            gate_logits = self.gate(img_feat.to(torch.float32)) / max(self.tau_gate, 1e-6)  # [B,K] float32
            w_student32 = F.softmax(gate_logits, dim=1)                                     # [B,K]
            w_student = w_student32.to(self.dtype)

            weighted_logits = (w_student.unsqueeze(-1) * logits_stack).sum(dim=1)  
            zs_text_feat = self.zs_text() # Zero-Shot 텍스트 특징 가져오기
            img_feat=img_feat.to(torch.float32)# [C, D]
            zs_logits = logit_scale * img_feat @ zs_text_feat.t() # [B, C]# [B,C]
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
        class_masks = {c: (label_idx == c) for c in range(C)}

        # 배치 단위 teacher 통계 (초기값)
        batch_mu       = torch.zeros(self.K, self.C, D, device=self.device)
        batch_var_mean = torch.ones(self.K, self.C, D, device=self.device) * 1e-2
        batch_var_disp = torch.zeros(self.K, self.C, D, device=self.device)

        # ------------------------------------------------------
        # (a) 이미지 label 기반 bootstrap으로 클래스별 분산 추정
        # ------------------------------------------------------
        for c in range(C):
            mask_c = class_masks[c]
            if mask_c.sum() == 0:
                continue

            feats_c = img_feat[mask_c].detach()    # [Nc, D]
            Nc = feats_c.size(0)
            n_boot = max(1, int(round(Nc * boot_ratio)))

            for k in range(K):
                mu_text = mu_stack[k, c]  # CLIP 텍스트 특징 [D]

                if Nc < 2:
                    # 데이터가 너무 적으면 default variance
                    batch_mu[k, c] = mu_text
                    batch_var_mean[k, c] = torch.full_like(mu_text, 1e-2)
                    batch_var_disp[k, c] = torch.zeros_like(mu_text)
                else:
                    # Bootstrap estimation
                    idx = torch.randint(0, Nc, (R, n_boot), device=self.device)  # [R, n_boot]
                    feats_boot = feats_c[idx]                                     # [R, n_boot, D]
                    var_r = ((feats_boot - mu_text) ** 2).mean(dim=1) + var_eps  # [R,D]

                    batch_mu[k, c]       = mu_text
                    batch_var_mean[k, c] = var_r.mean(dim=0)                     # [D]
                    batch_var_disp[k, c] = var_r.var(dim=0, unbiased=False)      # [D]

        # ------------------------------------------------------
        # (b) EMA update (teacher)
        # ------------------------------------------------------
        if self.use_ema_teacher:
            m = self.ema_momentum
            # 수정: batch_mu 등을 더할 때 .detach()를 사용해야 함
            self.ema_mu        = m * self.ema_mu        + (1 - m) * batch_mu.detach()
            self.ema_var_mean = m * self.ema_var_mean + (1 - m) * batch_var_mean.detach()
            self.ema_var_disp = m * self.ema_var_disp + (1 - m) * batch_var_disp.detach()
        else:
            # 수정: 여기도 detach 필요
            self.ema_mu        = batch_mu.detach()
            self.ema_var_mean = batch_var_mean.detach()
            self.ema_var_disp = batch_var_disp.detach()
        # ------------------------------------------------------
        # (c) Teacher routing (EMA NLEEP)
        # ------------------------------------------------------
        nleep_scores = torch.full((B, K), -1e9, device=self.device, dtype=torch.float32)

        for k in range(K):
            for c in range(C):
                mask_c = class_masks[c]
                if mask_c.sum() == 0:
                    continue

                mu       = self.ema_mu[k, c].to(torch.float32)        # [D]
                var_mean = self.ema_var_mean[k, c].to(torch.float32)  # [D]
                var_disp = self.ema_var_disp[k, c].to(torch.float32)  # [D]

                adjusted_var = var_mean + self.alpha_pbeb * var_disp
                adjusted_var = torch.clamp(adjusted_var, min=self.min_var)

                x = img_feat[mask_c].to(torch.float32)  # [Nc,D]
                diff = x - mu
                log_prob = -0.5 * (
                    (diff * diff) / adjusted_var + adjusted_var.log()
                ).sum(dim=1)                           # [Nc]

                nleep_scores[mask_c, k] = log_prob

        row_has_val = torch.isfinite(nleep_scores).any(dim=1)
        if not row_has_val.all():
            nleep_scores[~row_has_val] = 0.0

        # Teacher weight (라벨 + NLEEP 기반)
        weights_teacher32 = F.softmax(nleep_scores, dim=1)   # [B,K] float32

        # ------------------------------------------------------
        # (d) Student gate + KD (그대로 유지)
        # ------------------------------------------------------
        gate_logits = self.gate(img_feat.to(torch.float32)) / max(self.tau_gate, 1e-6)  # [B,K] float32
        weights_student32 = F.softmax(gate_logits, dim=1)                                # [B,K]
        weights_student = weights_student32.to(self.dtype)
        weighted_logits = (weights_student.unsqueeze(-1) *  torch.stack(logits_per_prompt, dim=1)).sum(dim=1) # [B,C]
        kd = F.kl_div(
            F.log_softmax(gate_logits, dim=1),  # log P_student
            weights_teacher32.detach(),         # Q_teacher
            reduction="batchmean"
        )
        self.loss_gate_kd = kd
        tau_zs = max(self.tau_zs, 1e-6)
        zs_text_feat = self.zs_text()  
        img_feat=img_feat.to(torch.float32)# [C, D]
        zs_logits = logit_scale * img_feat @ zs_text_feat.t()  # [B, C]
     
        teacher_T = zs_logits.detach() / tau_zs
        kd_zs_per_prompt = []
        for k in range(self.K):
            # k번째 프롬프트의 로짓 (Student)
            logits_k = logits_per_prompt[k].to(torch.float32) # [B, C]
            student_T_k = logits_k / tau_zs
            # 프롬프트 k의 로짓 분포 vs 하이브리드 Teacher Target 분포 간의 KD
            kd_k = F.kl_div(
                F.log_softmax(student_T_k, dim=1),        # log P_student_k(y|x)
                F.softmax(teacher_T, dim=1),                        # Q_target (정답 레이블로 보정됨)
                reduction="batchmean"
            ) * (tau_zs ** 2)
            kd_zs_per_prompt.append(kd_k)

        # 2. 모든 프롬프트별 KD 손실의 평균을 최종 kd_zs 손실로 사용
        kd_zs = torch.stack(kd_zs_per_prompt).mean()
        
        ##추가한거
        # ce_per_prompt = []
        # for k in range(self.K):
        #     # 각 prompt k의 CE: [B]
        #     ce_k = F.cross_entropy(
        #         logits_per_prompt[k].to(torch.float32),  # [B, C]
        #         label_idx,                               # [B]
        #         reduction="none"
        #     )
        #     ce_per_prompt.append(ce_k)
        # ce_per_prompt = torch.stack(ce_per_prompt, dim=1)

        # [B, K]
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


        # NLEEP teacher weight로 가중합 → teacher가 자기 "책임분" 만큼만 업데이트
        # weights_teacher32: [B, K]
        loss_teacher = (weights_teacher32 * ce_per_prompt).sum(dim=1).mean()  # scalar
        self.loss_teacher = loss_teacher
        self.loss_zs_kd = kd_zs
        # correct_class_logits = torch.gather(
        #     logits_stack, 
        #     dim=2, 
        #     index=label_idx.view(-1, 1, 1).expand(-1, self.K, 1)
        # ).squeeze(2)  # [B, K]

        # 2) 정답 기반 oracle 분포 (여러 prompt 동시에 클 수 있음)
        # tau_oracle = max(getattr(self, "tau_oracle", 0.1), 1e-6)
        # p_oracle = F.softmax(correct_class_logits / tau_oracle, dim=1)  # [B, K]

        # # 3) NLEEP teacher 분포와 oracle 분포를 섞어서 최종 target 만들기
        # alpha_mix = getattr(self, "alpha_oracle_mix", 0.9)  # 0~1 사이 값
        # # target = normalize( (1-alpha)*teacher + alpha*oracle )
        # mix = (1.0 - alpha_mix) * weights_teacher32 + alpha_mix * p_oracle
        # target_oracle = mix / (mix.sum(dim=1, keepdim=True) + 1e-8)  # [B, K]

        # # 4) gate가 이 soft 분포를 따라가도록 KL loss
        # loss_oracle = F.kl_div(
        #     F.log_softmax(gate_logits, dim=1),   # student 분포
        #     target_oracle.detach(),              # 정답+거리 섞인 soft 타겟
        #     reduction="batchmean"
        # )
        # self.loss_oracle = loss_oracle
        
        correct_class_logits = torch.gather(
            logits_stack, 
            dim=2, 
            index=label_idx.view(-1, 1, 1).expand(-1, self.K, -1)
        ).squeeze(2) # [B, K]
        
        # best_prompt_idx = correct_class_logits.argmax(dim=1)  # [B], 0 ~ K-1

        # loss_oracle = F.non_negative_cross_entropy(gate_logits, best_prompt_idx)
        # self.loss_oracle = loss_oracle

        # tau_oracle = 0.1 # 하이퍼파라미터 (상황에 맞춰 조절)
        tau_oracle = 0.1# 하이퍼파라미터 (상황에 맞춰 조절)
        oracle_weights = F.softmax(correct_class_logits / tau_oracle, dim=1).detach() # [B, K]

        # 3. Student Gate가 Oracle 분포를 따르도록 Loss 계산
        # gate_logits: [B, K] (Student의 게이트 출력)
        loss_oracle = F.kl_div(
            F.log_softmax(gate_logits, dim=1), # Student Log P
            oracle_weights,                    # Target Q (정답 기반)
            reduction="batchmean"
        )
        self.loss_oracle = loss_oracle
      
        return weighted_logits
   
    # def forward(self, image, label=None):
    #     image = image.to(self.device, dtype=self.dtype)
    #     logit_scale = self.logit_scale.exp()

    #     img_feat = self.image_encoder(image)
    #     img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True) # [B, D]
        
    #     B, D = img_feat.shape

    #     # 2) 프롬프트별 텍스트 특징 및 로짓
    #     text_feats_per_prompt, logits_per_prompt = [], []
    #     for pl in self.prompt_learner:
    #         prompts = pl()
    #         tfeat = self.text_encoder(prompts, self.tokenized_prompts)
    #         tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
    #         text_feats_per_prompt.append(tfeat)
    #         logits = logit_scale * img_feat @ tfeat.t()    # [B, C]
             
    #         logits_per_prompt.append(logits)
        
    #     if label is None:
    #         # student weights from image-only gate
    #         gate_logits = self.gate(img_feat.to(torch.float32)) / max(self.tau_gate, 1e-6)  # [B,K] float32
    #         w_student32 = F.softmax(gate_logits, dim=1)                                   # [B,K]
    #         w_student = w_student32.to(self.dtype)
    #         weighted_logits = (w_student.unsqueeze(-1) * torch.stack(logits_per_prompt, dim=1)).sum(dim=1)
    #         return weighted_logits

        
    #     label_idx = label.to(self.device).long().view(-1)
    #     K, C = self.num_selected_prompts, self.num_classes
    #     mu_stack = torch.stack(text_feats_per_prompt, dim=0)  # [K,C,D]

    #     R = int(getattr(self, "bootstrap_reps", 8))
    #     boot_ratio = float(getattr(self, "bootstrap_ratio", 0.7))
    #     var_eps = float(getattr(self, "var_eps", 1e-6))
    #     class_masks = {c: (label_idx == c) for c in range(C)}

        
    #     batch_mu = torch.zeros(self.K, self.C, D, device=self.device)
    #     batch_var_mean = torch.ones(self.K, self.C, D, device=self.device) * 1e-2
    #     batch_var_disp = torch.zeros(self.K, self.C, D, device=self.device)

    #     for c in range(C):
    #         mask_c = class_masks[c]
    #         if mask_c.sum() == 0:
    #             continue

    #         feats_c = img_feat[mask_c].detach()    # [Nc, D]
    #         Nc = feats_c.size(0)
    #         n_boot = max(1, int(round(Nc * boot_ratio)))

    #         for k in range(K):

    #             mu_text = mu_stack[k, c]  # CLIP 텍스트 특징

    #             if Nc < 2:
    #                 # default variance
    #                 batch_mu[k, c] = mu_text
    #                 batch_var_mean[k, c] = torch.full_like(mu_text, 1e-2)
    #                 batch_var_disp[k, c] = torch.zeros_like(mu_text)
    #             else:
    #                 # Bootstrap estimation
    #                 idx = torch.randint(0, Nc, (R, n_boot), device=self.device)
    #                 feats_boot = feats_c[idx]
    #                 var_r = ((feats_boot - mu_text) ** 2).mean(dim=1) + var_eps

    #                 batch_mu[k, c] = mu_text
    #                 batch_var_mean[k, c] = var_r.mean(dim=0)
    #                 batch_var_disp[k, c] = var_r.var(dim=0, unbiased=False)

    #     # ============================
    #     # 2) EMA update
    #     # ============================
    #     if self.use_ema_teacher:
    #         m = self.ema_momentum
    #         self.ema_mu        = m * self.ema_mu        + (1 - m) * batch_mu
    #         self.ema_var_mean = m * self.ema_var_mean + (1 - m) * batch_var_mean
    #         self.ema_var_disp = m * self.ema_var_disp + (1 - m) * batch_var_disp
    #     else:
    #         self.ema_mu        = batch_mu
    #         self.ema_var_mean = batch_var_mean
    #         self.ema_var_disp = batch_var_disp

    #     # ============================
    #     # Teacher routing (EMA NLEEP) - 🌟 분산 최대화 로직 적용 🌟
    #     # ============================
    #     nleep_scores = torch.full((B, K), -1e9, device=self.device, dtype=torch.float32)

    #     for k in range(K):
    #         for c in range(C):
    #             mask_c = class_masks[c]
    #             if mask_c.sum() == 0:
    #                 continue

    #             # teacher statistics (EMA)
    #             mu       = self.ema_mu[k, c].to(torch.float32)
    #             var_mean = self.ema_var_mean[k, c].to(torch.float32)
    #             var_disp = self.ema_var_disp[k, c].to(torch.float32)

    #             adjusted_var = var_mean + self.alpha_pbeb * var_disp
    #             adjusted_var = torch.clamp(adjusted_var, min=self.min_var)

    #             x = img_feat[mask_c].to(torch.float32)
    #             diff = x - mu
    #             log_prob = -0.5 * ((diff * diff) / adjusted_var + adjusted_var.log()).sum(dim=1) # <--- [수정]
    #             nleep_scores[mask_c, k] = log_prob

    #     row_has_val = torch.isfinite(nleep_scores).any(dim=1)
    #     if not row_has_val.all():
    #         nleep_scores[~row_has_val] = 0.0

    #     # (a) teacher weight (NLEEP 기반)
    #     weights_teacher32 = F.softmax(nleep_scores, dim=1)      # [B,K] float32

    #     # (b) student gate
    #     gate_logits = self.gate(img_feat.to(torch.float32)) / max(self.tau_gate, 1e-6)  # [B,K] float32
    #     weights_student32 = F.softmax(gate_logits, dim=1)
    #     weights_student = weights_student32.to(self.dtype)

    #     weighted_logits = (weights_student.unsqueeze(-1) *
    #                          torch.stack(logits_per_prompt, dim=1)).sum(dim=1)  # [B,C]
    #     kd = F.kl_div(
    #         F.log_softmax(gate_logits, dim=1),           # log P_student
    #         weights_teacher32.detach(),                  # Q_teacher
    #         reduction="batchmean"
    #     )
        

    #     self.loss_gate_kd = kd
    #     return weighted_logits

@TRAINER_REGISTRY.register()
class TRIP(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.TRIP.PREC in ["fp16", "fp32", "amp"]
    @torch.no_grad()
    def analyze_confidence_vs_distribution(self, split="test", n_bins=10):
        """
        confidence vs accuracy, distribution score vs accuracy
        를 한 번에 분석하고, correlation + reliability plot 을 저장하는 함수.

        사용 예시:
            trainer.analyze_confidence_vs_distribution(split="val")
            trainer.analyze_confidence_vs_distribution(split="test")
        """
        self.set_model_mode("eval")
        self.model.eval()

        # 1) 어떤 데이터셋을 볼지 선택
        if split == "val" and self.val_loader is not None:
            data_loader = self.val_loader
        else:
            split = "test"
            data_loader = self.test_loader

        all_conf = []
        all_dist_score = []
        all_correct = []

        print(f"[Analysis] Collecting scores on *{split}* set...")
        for batch in tqdm(data_loader, desc=f"Analyze {split}"):
            # TrainerX에 정의된 parse_batch_test 사용 (이미 test()에서 쓰고 있음)
            inputs, labels = self.parse_batch_test(batch)

            # CustomCLIP.forward_analysis 호출
            logits, w_student, nleep_scores, conf_weighted, pred = \
                self.model.forward_analysis(inputs)

            # numpy로 변환
            conf_np = conf_weighted.cpu().numpy()                         # [B]
            # 분포 기반 score: 각 샘플마다 가장 높은 NLEEP score 하나만 사용
            dist_np = nleep_scores.max(dim=1).values.cpu().numpy()        # [B]

            correct_np = (pred.cpu().numpy() == labels.cpu().numpy()).astype(np.float32)

            all_conf.append(conf_np)
            all_dist_score.append(dist_np)
            all_correct.append(correct_np)

        conf = np.concatenate(all_conf)          # [N]
        dist = np.concatenate(all_dist_score)    # [N]
        correct = np.concatenate(all_correct)    # [N], 0/1

        # 2) point-biserial correlation (이분형 정답 vs 연속 score)
        r_conf, p_conf = pointbiserialr(correct, conf)
        r_dist, p_dist = pointbiserialr(correct, dist)

        print("=== Correlation with correctness (point-biserial) ===")
        print(f"- Baseline confidence : r = {r_conf:.4f}, p = {p_conf:.2e}")
        print(f"- Dist. NLEEP score   : r = {r_dist:.4f}, p = {p_dist:.2e}")
        print("  (r이 클수록 'score ↑ → 정답일 확률 ↑' 관계가 더 강함)")

        # 3) Reliability plot 그리기 (score bin 별 실제 accuracy)
        save_dir = os.path.join(self.output_dir, "analysis")
        os.makedirs(save_dir, exist_ok=True)

        def reliability_plot(score, name):
            # 분포 기반 score는 log값이라 그대로 쓰면 범위가 안 예쁠 수 있음 → 0~1로 정규화
            s = score.copy()
            s = (s - s.min()) / (s.max() - s.min() + 1e-8)

            bins = np.linspace(0.0, 1.0, n_bins + 1)
            mids, accs = [], []
            for i in range(n_bins):
                lo, hi = bins[i], bins[i + 1]
                mask = (s >= lo) & (s < hi)
                if mask.sum() == 0:
                    continue
                mids.append((lo + hi) / 2.0)
                accs.append(correct[mask].mean())

            plt.figure()
            plt.plot(mids, accs, marker="o")
            plt.plot([0, 1], [0, 1], linestyle="--")  # y=x 이상적 calibration 라인
            plt.xlabel(f"{name} (normalized)")
            plt.ylabel("Empirical accuracy")
            plt.title(f"Reliability of {name}")
            plt.grid(True)

            save_path = os.path.join(save_dir, f"reliability_{name.replace(' ', '_')}.png")
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            print(f"  - Saved reliability plot for {name} at: {save_path}")

        reliability_plot(conf, "confidence")
        reliability_plot(dist, "dist_score")

      

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
            
            kd_loss = self.model.loss_gate_kd
            ce_loss = F.cross_entropy(output, label)
            teacher_loss = self.model.loss_teacher
            #loss_flag
            total_loss = (
                teacher_loss + kd_loss+ 0.5*self.model.loss_oracle + 0.0*self.model.loss_zs_kd
            )
            loss = total_loss # 최종 Loss를 summary에 사용


            self.model_backward_and_update(total_loss)
            
        loss_summary = {
            "loss": loss.item(),
            "ce_loss": ce_loss.item(),
            'teacher_loss':teacher_loss.item(),
            "loss_zs_kd":self.model.loss_zs_kd.item(),
            'loss_oracle':self.model.loss_oracle.item(),
            # 'loss_pre_gate_align':self.model.loss_pre_gate_align.item(),
            "acc": compute_accuracy(output, label)[0].item(), 
            "kd_loss": kd_loss.item(), # New Loss Variance Reg 로깅
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        
        self._global_step += 1
        return loss_summary
    def _log_prompt_stats(self):
        """각 prompt의 ctx variance / prompt 간 cosine similarity 로그"""

        w = self._ensure_writer()  # TensorBoard writer
        epoch = self.epoch

        with torch.no_grad():
            ctx_vecs = []

            # K개 prompt learner 순회
            for k, pl in enumerate(self.model.prompt_learner):
                # pl.ctx: [n_ctx, dim] 또는 [n_cls, n_ctx, dim]
                ctx = pl.ctx.detach().float()

                # 클래스별 context인 경우 [n_cls, n_ctx, dim] → flatten
                ctx_flat = ctx.view(-1)

                # variance (scalar)
                var_k = ctx_flat.var().item()
                w.add_scalar(f"prompt/ctx_var/p{k}", var_k, epoch)
                # 혹은 self.write_scalar(f"prompt/ctx_var/p{k}", var_k, epoch) 써도 됨

                # cosine similarity 계산용 (정규화 벡터)
                ctx_norm = F.normalize(ctx_flat, dim=0)
                ctx_vecs.append(ctx_norm)

            # prompt 간 cosine similarity (K > 1일 때만)
            if len(ctx_vecs) > 1:
                ctx_mat = torch.stack(ctx_vecs, dim=0)  # [K, D_total]
                sim = (ctx_mat @ ctx_mat.t()).cpu()     # [K, K]

                K = sim.size(0)
                for i in range(K):
                    for j in range(i + 1, K):
                        val = sim[i, j].item()
                        w.add_scalar(f"prompt/cos_sim/p{i}_p{j}", val, epoch)
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
    
        self._log_prompt_stats()

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
            