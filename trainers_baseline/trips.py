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


class CrossAttentionModule(nn.Module):
    def __init__(self, feature_dim, hidden_dim, scale_factor=None):
        super().__init__()
        self.feature_dim = feature_dim
        # Q: Image Feature (img_feat)
        self.W_Q = nn.Linear(feature_dim, hidden_dim, bias=False)
        # K/V: Text Feature (tfeat)
        self.W_K = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.W_V = nn.Linear(feature_dim, feature_dim, bias=False) # Output dim can be feature_dim
        
        # Scaling Factor (Transformer default: 1/sqrt(d_k))
        self.scale = scale_factor if scale_factor is not None else (hidden_dim ** -0.5)

    def forward(self, img_feat, text_feat):
        # img_feat (Query): [B, D]
        # text_feat (Key/Value): [C, D]
        
        Q = self.W_Q(img_feat) # [B, d_h]
        K = self.W_K(text_feat) # [C, d_h]
        V = self.W_V(text_feat) # [C, D]

        # 1. Attention Score (Scaled Dot-Product)
        # attn_scores: [B, d_h] @ [d_h, C] -> [B, C]
        attn_scores = (Q @ K.T) * self.scale
        
        # 2. Softmax (Attention Weights)
        # attn_weights: [B, C]
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        # 3. Weighted Sum (Output Feature)
        # out: [B, C] @ [C, D] -> [B, D]
        cross_feat = attn_weights @ V
        
        return cross_feat # 이 특징이 img_feat에 더해질 '보강 정보'가 됩니다.
class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.num_selected_prompts = 3
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.num_classes = len(classnames)
        self.queue_size = int(getattr(cfg.TRAINER.TRIP, "QUEUE_SIZE", 4096))
        proj_dim = int(getattr(cfg.TRAINER.TRIP, "PROJ_DIM", 512))
        self.proj_dim = proj_dim
        self.register_buffer("feature_queue",
            torch.randn(self.queue_size, self.proj_dim)
        )
        self.feature_queue = F.normalize(self.feature_queue, dim=1)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
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
        self.gate = nn.Linear(self.image_encoder.output_dim if hasattr(self.image_encoder, "output_dim") else self.prompt_learner[0].ctx.shape[-1],  # 대개 CLIP visual dim
                                 self.K, bias=True)
        self.lambda_kd = float(getattr(cfg.TRAINER.TRIP, "LAMBDA_KD", 1.0))    # teacher vs student KL
        self.tau_gate  = float(getattr(cfg.TRAINER.TRIP, "TAU_GATE", 1.0))    # softmax 온도(학생)
        self.var_eps = float(getattr(cfg.TRAINER.TRIP, "VAR_EPS", 1e-5))
        self.min_var = float(getattr(cfg.TRAINER.TRIP, "VAR_MIN", 1e-6))
        self.alpha_pbeb = float(getattr(cfg.TRAINER.TRIP, "ALPHA_PBEB", 0.2))
        self.beta_var = float(getattr(cfg.TRAINER.TRIP, "BETA_VAR", 0.5))
        
        D = 512
        
        self.visual_proj = nn.ModuleList([
            nn.Linear(self.image_encoder.output_dim, proj_dim, bias=False)
            for _ in range(self.K)
        ])
        self.lambda_neighbor = float(
            getattr(cfg.TRAINER.TRIP, "LAMBDA_NEIGHBOR", 1.0)
        )
        self.lambda_align = float(getattr(cfg.TRAINER.TRIP, "LAMBDA_ALIGN", 0.1)) # Alignment Loss (새 특징 vs 텍스트 특징)
        self.lambda_reg = float(getattr(cfg.TRAINER.TRIP, "LAMBDA_REG", 0.1))      # Regularization Loss (새 특징 vs 원래 특징)
        self.gate = nn.Linear(self.image_encoder.output_dim, self.K, bias=True)
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
        self.ema_momentum = float(getattr(cfg.TRAINER.TRIP, "EMA_MOMENTUM", 0.99))
        self.use_ema_teacher = True  # fla

   
    def forward(self, image, label=None,channel_mask=None):
        image = image.to(self.device, dtype=self.dtype)
        logit_scale = self.logit_scale.exp()

        img_feat = self.image_encoder(image)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True) # [B, D]
        proj_visual_feats = []
        for k in range(self.K):
            z = self.visual_proj[k](img_feat.float())    # [B, proj_dim]
            z = F.normalize(z, dim=-1)                   # 정규화
            proj_visual_feats.append(z)
        with torch.no_grad():
            z_all = proj_visual_feats[0]  # 예: prompt 0의 projection을 큐에 사용
            self._dequeue_and_enqueue(z_all)
        B, D = img_feat.shape

        # 2) 프롬프트별 텍스트 특징 및 로짓
        text_feats_per_prompt, logits_per_prompt = [], []
        for pl in self.prompt_learner:
            prompts = pl()
            tfeat = self.text_encoder(prompts, self.tokenized_prompts)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
            text_feats_per_prompt.append(tfeat)
            logits = logit_scale * img_feat @ tfeat.t()    # [B, C]
            logits_per_prompt.append(logits)
        
        if label is None:
            # student weights from image-only gate
            gate_logits = self.gate(img_feat.to(torch.float32)) / max(self.tau_gate, 1e-6)  # [B,K] float32
            w_student32 = F.softmax(gate_logits, dim=1)                                   # [B,K]
            w_student = w_student32.to(self.dtype)
            weighted_logits = (w_student.unsqueeze(-1) * torch.stack(logits_per_prompt, dim=1)).sum(dim=1)
            return weighted_logits

        # ============================
        # 아래부터 label 있는 학습 경로
        # ============================
        label_idx = label.to(self.device).long().view(-1)
        K, C = self.num_selected_prompts, self.num_classes
        mu_stack = torch.stack(text_feats_per_prompt, dim=0)  # [K,C,D]

        R = int(getattr(self, "bootstrap_reps", 8))
        boot_ratio = float(getattr(self, "bootstrap_ratio", 0.7))
        var_eps = float(getattr(self, "var_eps", 1e-6))
        class_masks = {c: (label_idx == c) for c in range(C)}

        # ============================
        # 1) Batch-based estimation (EMA 업데이트에 사용)
        # ============================
        batch_mu = torch.zeros(self.K, self.C, D, device=self.device)
        batch_var_mean = torch.ones(self.K, self.C, D, device=self.device) * 1e-2
        batch_var_disp = torch.zeros(self.K, self.C, D, device=self.device)

        for c in range(C):
            mask_c = class_masks[c]
            if mask_c.sum() == 0:
                continue

            feats_c = img_feat[mask_c].detach()    # [Nc, D]
            Nc = feats_c.size(0)
            n_boot = max(1, int(round(Nc * boot_ratio)))

            for k in range(K):

                mu_text = mu_stack[k, c]  # CLIP 텍스트 특징

                if Nc < 2:
                    # default variance
                    batch_mu[k, c] = mu_text
                    batch_var_mean[k, c] = torch.full_like(mu_text, 1e-2)
                    batch_var_disp[k, c] = torch.zeros_like(mu_text)
                else:
                    # Bootstrap estimation
                    idx = torch.randint(0, Nc, (R, n_boot), device=self.device)
                    feats_boot = feats_c[idx]
                    var_r = ((feats_boot - mu_text) ** 2).mean(dim=1) + var_eps

                    batch_mu[k, c] = mu_text
                    batch_var_mean[k, c] = var_r.mean(dim=0)
                    batch_var_disp[k, c] = var_r.var(dim=0, unbiased=False)

        # ============================
        # 2) EMA update
        # ============================
        if self.use_ema_teacher:
            m = self.ema_momentum
            self.ema_mu        = m * self.ema_mu        + (1 - m) * batch_mu
            self.ema_var_mean = m * self.ema_var_mean + (1 - m) * batch_var_mean
            self.ema_var_disp = m * self.ema_var_disp + (1 - m) * batch_var_disp
        else:
            self.ema_mu        = batch_mu
            self.ema_var_mean = batch_var_mean
            self.ema_var_disp = batch_var_disp

        # ============================
        # Teacher routing (EMA NLEEP) - 🌟 분산 최대화 로직 적용 🌟
        # ============================
        nleep_scores = torch.full((B, K), -1e9, device=self.device, dtype=torch.float32)

        for k in range(K):
            for c in range(C):
                mask_c = class_masks[c]
                if mask_c.sum() == 0:
                    continue

                # teacher statistics (EMA)
                mu       = self.ema_mu[k, c].to(torch.float32)
                var_mean = self.ema_var_mean[k, c].to(torch.float32)
                var_disp = self.ema_var_disp[k, c].to(torch.float32)

                adjusted_var = var_mean + self.alpha_pbeb * var_disp
                adjusted_var = torch.clamp(adjusted_var, min=self.min_var)

                x = img_feat[mask_c].to(torch.float32)
                diff = x - mu
                
                # NLEEP 공식에서 분산 항 (adjusted_var.log())의 부호를 반전합니다.
                # 이는 분산이 클수록 log_prob(점수)가 높아지도록 유도합니다.
                log_prob = -0.5 * ((diff * diff) / adjusted_var - adjusted_var.log()).sum(dim=1) # <--- [변경]

                nleep_scores[mask_c, k] = log_prob

        row_has_val = torch.isfinite(nleep_scores).any(dim=1)
        if not row_has_val.all():
            nleep_scores[~row_has_val] = 0.0

        # (a) teacher weight (NLEEP 기반)
        weights_teacher32 = F.softmax(nleep_scores, dim=1)      # [B,K] float32

        # (b) student gate
        gate_logits = self.gate(img_feat.to(torch.float32)) / max(self.tau_gate, 1e-6)  # [B,K] float32
        weights_student32 = F.softmax(gate_logits, dim=1)
        weights_student = weights_student32.to(self.dtype)

        # (c) per-prompt 로짓 가중합
        weighted_logits = (weights_student.unsqueeze(-1) *
                             torch.stack(logits_per_prompt, dim=1)).sum(dim=1)  # [B,C]

        # ============================
        # 3) Gate KD loss (teacher 분포 ↔ student gate)
        # ============================
        kd = F.kl_div(
            F.log_softmax(gate_logits, dim=1),           # log P_student
            weights_teacher32.detach(),                  # Q_teacher
            reduction="batchmean"
        )
        self.loss_gate_kd = kd
        
        # 🌟 [변경] visual_loss (Variance Floor) 제거 및 0으로 설정 🌟
        # std = img_feat.std(dim=0) # per-dimension std
        # var_floor = F.relu(0.2 - std).mean() 
        self.visual_loss = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        
        self.spread_loss = self.class_spread_loss(img_feat, label_idx, min_norm=0.7)
        
        
        per_prompt_loss = []
        for logits in logits_per_prompt:
            # [B] : 각 샘플에 대한 CE
            loss_k = F.cross_entropy(logits, label_idx, reduction="none")
            per_prompt_loss.append(loss_k)

        per_prompt_loss = torch.stack(per_prompt_loss, dim=1)  # [B, K]

        # teacher가 중요하게 보는 prompt일수록 더 잘 맞추도록 압박
        loss_prompt_align = (weights_teacher32.detach() * per_prompt_loss).sum(dim=1).mean()
        self.loss_prompt_align = loss_prompt_align

        # forward는 여전히 logits만 반환하고,
        # 실제 total loss는 바깥 training loop에서 조합해서 쓰면 됨
        return weighted_logits

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
            if ("prompt_learner" not in name) and ("gate" not in name): # <-- 1단계: visual_feature_learner를 제외
                param.requires_grad_(False)

        for name, param in self.model.visual_proj.named_parameters():
             param.requires_grad_(True)
        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        self.model.to(self.device)
        train_params = chain(self.model.prompt_learner.parameters(),
                             self.model.gate.parameters(),
                            self.model.visual_proj.parameters())
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
            per_loss = self.model.loss_prompt_align
            spred = self.model.spread_loss
            total_loss = (
                ce_loss + kd_loss + per_loss  + spred
                # + lambda_loss_var * div_loss
            )
         
            self.model_backward_and_update(total_loss)
            loss = total_loss # 최종 Loss를 summary에 사용
            
        loss_summary = {
            "loss": loss.item(),
            "ce_loss": ce_loss.item(),
            'per_loss':per_loss.item(),
            'spread':spred.item(),
            # 'loss_reg':self.model.loss_reg.item(),
            # 'loss_align':self.model.loss_align.item(),
            "acc": compute_accuracy(output, label)[0].item(), 
            "kd_loss": kd_loss.item(), # New Loss Variance Reg 로깅
        }

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
            
