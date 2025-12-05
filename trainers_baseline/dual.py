import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, count_num_param
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
import itertools

from trainers_baseline.basedg import *
from utils.clip_part import *
from utils.templates import CUSTOM_TEMPLATES
from utils.visual_prompt import *

_tokenizer = _Tokenizer()

class text_PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        # n_ctx = cfg.TRAINER.COOP.N_CTX
        n_ctx = 4
        ctx_init = cfg.TRAINER.DUAL.CTX_INIT
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
            if cfg.TRAINER.DUAL.CSC:
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
        self.class_token_position = cfg.TRAINER.DUAL.CLASS_TOKEN_POSITION

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

class VisualPromptLearner(Base_PromptLearner):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        dtype = clip_model.dtype
        self.num_tokens = cfg.TRAINER.DUAL.NUM_TOKENS    # number of prompted tokens
        prompt_dim = cfg.MODEL.HIDDEN_SIZE
        self.prompt_dropout = nn.Dropout(cfg.TRAINER.DUAL.DROPOUT)
        self.location = cfg.TRAINER.DUAL.LOCATION
        self.deep_layer = cfg.TRAINER.DUAL.DEEP_LAYERS
        
        self.vctx = None
        self.deep_vctx = None
        if not cfg.TRAINER.DUAL.ENABLE_CONV:
            if cfg.TRAINER.DUAL.VP:
                vctx_vectors = torch.empty(self.num_tokens, prompt_dim)
                nn.init.normal_(vctx_vectors, std=0.02)
                self.vctx = nn.Parameter(vctx_vectors)
            
            if cfg.TRAINER.DUAL.V_DEEP:  
                if self.deep_layer == None:
                    deep_vctx_vectors = torch.empty(cfg.MODEL.NUM_LAYER - 1, self.num_tokens, prompt_dim)
                    nn.init.normal_(deep_vctx_vectors, std=0.02)
                else:
                    deep_ctx_vectors = torch.empty(self.deep_layer[1] - self.deep_layer[0] + 1, self.num_tokens, prompt_dim)
                    nn.init.normal_(deep_ctx_vectors, std=0.02)
                self.deep_vctx = nn.Parameter(deep_vctx_vectors)
        
        else:
            if cfg.TRAINER.DUAL.TYPE == "random":
                random_prompter = RandomPatchPrompter(cfg)
                self.prompter = random_prompter
            elif cfg.TRAINER.DUAL.TYPE == "fix":
                fix_prompter = FixedPatchPrompter(cfg)
                self.prompter = fix_prompter
            elif cfg.TRAINER.DUAL.TYPE == "pad":
                pad_prompter = PadPrompter(cfg)
                self.prompter = pad_prompter
            else:
                raise ValueError('Conv VPT type is wrong!')
        
        prompt_prefix = CUSTOM_TEMPLATES[cfg.DATASET.NAME]
        prompts = [prompt_prefix.format(c.replace("_", " ")) for c in classnames]
        tokenized_prompts = clip.tokenize(prompts)
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)
        self.ctx = embedding.to(torch.device("cuda:{}".format(cfg.GPU))) 
        self.tokenized_prompts = tokenized_prompts
                
    def forward(self, image=None):
        if image == None:
            return self.ctx, None, self.vctx, self.deep_vctx
        else:
            return self.ctx, None, self.vctx, self.deep_vctx, self.prompter(image)


class CustomCLIP(Base_CustomCLIP):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__(cfg, classnames, clip_model)
        self.visual_prompt_learner = VisualPromptLearner(cfg, classnames, clip_model)
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        # 1. 텍스트 프롬프트 학습기 (CoOp 스타일)
        self.num_prompt = 3
        
        self.prompt_learner = nn.ModuleList([
            text_PromptLearner(cfg, classnames, clip_model)
            for i in range(self.num_prompt)
        ])
        self.tokenized_prompts = self.prompt_learner[0].tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner[0])
        # self.text_prompt_learner = text_PromptLearner(cfg, classnames, clip_model)
        # self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts
        # self.text_encoder = TextEncoder(cfg, clip_model, self.text_prompt_learner)

        if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT':
            self.image_encoder = ImageEncoder_Trans(cfg, clip_model, self.visual_prompt_learner)
        else:  # RN50, RN101
            self.image_encoder = ImageEncoder_Conv(cfg, clip_model, self.visual_prompt_learner)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.gate = nn.Linear(512, self.num_prompt, bias=True)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.prompts = torch.tensor(0)
        self.token_prompts = torch.tensor(0)
        D = 512
        self.C = len(classnames)
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.conv = cfg.TRAINER.DUAL.ENABLE_CONV
        self.register_buffer(
            "ema_mu", 
            torch.zeros(self.num_prompt, self.C, D, dtype=self.dtype)
        )
        self.register_buffer(
            "ema_var_mean", 
            torch.ones(self.num_prompt, self.C, D, dtype=self.dtype) * 1e-2
        )
        self.register_buffer(
            "ema_var_disp", 
            torch.zeros(self.num_prompt, self.C, D, dtype=self.dtype)
        )
        self.ema_momentum = float(getattr(cfg.TRAINER.DUAL, "EMA_MOMENTUM", 0.99))
        self.use_ema_teacher = True

    def forward(self, image, label=None):
        if self.conv:
            _, deep_ctx, vctx, deep_vctx, image = self.visual_prompt_learner(image)
        else:
            _, deep_ctx, vctx, deep_vctx = self.visual_prompt_learner()
        
        # --- 2. K개의 텍스트 특징 및 로짓 계산 ---
        text_feats_per_prompt, logits_per_prompt = [], []
        # if prompt:
        #         image_features = self.image_encoder(image.type(self.dtype), None, None) 
        # else:   
        image_features = self.image_encoder(image.type(self.dtype), vctx, deep_vctx)    
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()

        # self.prompt_learner는 K개의 PromptLearner를 포함하는 nn.ModuleList
        for pl in self.prompt_learner:
            prompts = pl()
            tfeat = self.text_encoder(prompts, self.tokenized_prompts)
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
            text_feats_per_prompt.append(tfeat)
            
            # 로짓 계산: logit_k = logit_scale * img_feat @ tfeat.t()
            logits_k = logit_scale * image_features @ tfeat.t()  # [B, C]
            logits_per_prompt.append(logits_k)
        if label is None:
            gate_logits = self.gate(image_features.float()) / max(1.0, 1e-6)
            w_student32 = F.softmax(gate_logits, dim=1)
            w_student = w_student32.to(self.dtype)

            weighted_logits = (w_student.unsqueeze(-1)
                            * torch.stack(logits_per_prompt, dim=1)).sum(dim=1)
            return weighted_logits

            # --------------------------
            # 4) 라벨 있음 → Teacher (EMA NLEEP)
            # --------------------------

        label_idx = label.to(self.device).long().view(-1)
        C = self.C
        K = self.num_prompt
        B, D = image_features.shape

        mu_stack = torch.stack(text_feats_per_prompt, dim=0)

        R = int(getattr(self, "bootstrap_reps", 8))
        boot_ratio = float(getattr(self, "bootstrap_ratio", 0.7))
        var_eps = float(getattr(self, "var_eps", 1e-6))

        class_masks = {c: (label_idx == c) for c in range(C)}

        batch_mu = torch.zeros(K, C, D, device=self.device)
        batch_var_mean = torch.ones(K, C, D, device=self.device) * 1e-2
        batch_var_disp = torch.zeros(K, C, D, device=self.device)

        # ---- batch estimation (bootstrap) ----
        for c in range(C):
            mask_c = class_masks[c]
            if mask_c.sum() == 0:
                continue

            feats_c = image_features[mask_c].detach()
            Nc = feats_c.size(0)
            n_boot = max(1, int(round(Nc * boot_ratio)))

            for k in range(K):
                mu_text = mu_stack[k, c]

                if Nc < 2:
                    batch_mu[k, c] = mu_text
                    batch_var_mean[k, c] = torch.full_like(mu_text, 1e-2)
                    batch_var_disp[k, c] = torch.zeros_like(mu_text)
                else:
                    idx = torch.randint(0, Nc, (R, n_boot), device=self.device)
                    feats_boot = feats_c[idx]
                    var_r = ((feats_boot - mu_text) ** 2).mean(dim=1) + var_eps

                    batch_mu[k, c] = mu_text
                    batch_var_mean[k, c] = var_r.mean(dim=0)
                    batch_var_disp[k, c] = var_r.var(dim=0, unbiased=False)

        # ---- EMA update ----
        if self.use_ema_teacher:
            m = self.ema_momentum
            self.ema_mu       = m * self.ema_mu       + (1-m) * batch_mu
            self.ema_var_mean = m * self.ema_var_mean + (1-m) * batch_var_mean
            self.ema_var_disp = m * self.ema_var_disp + (1-m) * batch_var_disp
        else:
            self.ema_mu = batch_mu
            self.ema_var_mean = batch_var_mean
            self.ema_var_disp = batch_var_disp

        # --------------------------------
        # Teacher routing (Gaussian likelihood)
        # --------------------------------
        nleep_scores = torch.full((B, K), -1e9,
                                device=self.device, dtype=torch.float32)

        for k in range(K):
            for c in range(C):
                mask_c = class_masks[c]
                if mask_c.sum() == 0:
                    continue

                mu       = self.ema_mu[k, c].float()
                var_mean = self.ema_var_mean[k, c].float()
                var_disp = self.ema_var_disp[k, c].float()

                adjusted_var = var_mean + 0.2 * var_disp
                adjusted_var = torch.clamp(adjusted_var, min=1e-6)
                x = image_features[mask_c].float()
                diff = x - mu

                log_prob = -0.5 * (
                    (diff * diff) / adjusted_var + adjusted_var.log()).sum(dim=1)

                nleep_scores[mask_c, k] = log_prob

        if not torch.isfinite(nleep_scores).any(dim=1).all():
            nleep_scores[~torch.isfinite(nleep_scores).any(dim=1)] = 0.0

        weights_teacher32 = F.softmax(nleep_scores, dim=1)

        # --------------------------------
        # Student gate (KD)
        # --------------------------------
        gate_logits = self.gate(image_features.float()) / max(1.0, 1e-6)
        weights_student32 = F.softmax(gate_logits, dim=1)
        weights_student = weights_student32.to(self.dtype)

        weighted_logits = (
            weights_student.unsqueeze(-1)
            * torch.stack(logits_per_prompt, dim=1)
        ).sum(dim=1)

        # KD loss
        self.loss_gate_kd = F.kl_div(
            F.log_softmax(gate_logits, dim=1),
            weights_teacher32.detach(),
            reduction="batchmean"
        )

        # Prompt alignment loss
        per_prompt_loss = []
        for logits in logits_per_prompt:
            loss_k = F.cross_entropy(logits, label_idx, reduction="none")
            per_prompt_loss.append(loss_k)

        per_prompt_loss = torch.stack(per_prompt_loss, dim=1)

        self.loss_prompt_align = (
            weights_teacher32.detach() * per_prompt_loss
        ).sum(dim=1).mean()

        return weighted_logits


    # def forward(self, image, prompt=False):
    #     if self.conv:
    #         _, deep_ctx, vctx, deep_vctx, image = self.visual_prompt_learner(image)
    #     else:
    #         _, deep_ctx, vctx, deep_vctx = self.visual_prompt_learner()
        
    #     prompts = self.text_prompt_learner()
    #     self.prompts = prompts
    #     self.token_prompts = self.tokenized_prompts
    #     self.text_features = self.text_encoder(prompts, self.tokenized_prompts)

    #     # text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_ctx) 
    #     text_features = self.text_features
    #     text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    #     if prompt:
    #         image_features = self.image_encoder(image.type(self.dtype), None, None) 
    #     else:   
    #         image_features = self.image_encoder(image.type(self.dtype), vctx, deep_vctx)    
    #     image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    #     logit_scale = self.logit_scale.exp()
    #     logits = logit_scale * image_features @ text_features.t()

    #     return logits
    

@TRAINER_REGISTRY.register()
class Dual(BaseDG):
    '''Visual Prompt Tuning (VPT)
    
    Adapt from Visual Prompt Tuning
    https://arxiv.org/pdf/2203.12119.pdf
    '''
    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames
        optim_params = []
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

        if cfg.TRAINER.DUAL.PREC == "fp32" or cfg.TRAINER.DUAL.PREC == "amp":
            clip_model.float()  # CLIP's default precision is fp16

        print("Building custom CLIP...")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder...")
        
        train_keywords = [
            "prompt_learner",
            "visual_prompt_learner",
            "gate",
        ]

        for name, param in self.model.named_parameters():
            if any(k in name for k in train_keywords):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)


        trainable_param_groups = []

        # text prompt learners
        trainable_param_groups.append(self.model.prompt_learner.parameters())
        # visual prompt learner
        trainable_param_groups.append(self.model.visual_prompt_learner.parameters())
        # gate
        trainable_param_groups.append(self.model.gate.parameters())
        optim_params = itertools.chain(*trainable_param_groups)        
        self.optim = build_optimizer(optim_params, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("model", self.model, self.optim, self.sched)
        
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print("=== Trainable Parameters by Module ===")
        total = 0
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                print(f"{name:50s} {p.numel():,d}")
                total += p.numel()
        print(f"Total trainable params: {total:,d}")
        self.model.to(self.device)
        
        self.scaler = GradScaler() if cfg.TRAINER.DUAL.PREC == "amp" else None

    def forward_backward(self, batch):
        images, labels = self.parse_batch_train(batch)
        prec = self.cfg.TRAINER.DUAL.PREC
        
        if prec == "amp":
            with autocast():
                output = self.model(images)
                loss = F.cross_entropy(output, labels)
            self.optim.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(images, labels)
            ce_loss = F.cross_entropy(output, labels)
            teacher_loss = self.model.loss_prompt_align
            kd = self.model.loss_gate_kd
            loss = ce_loss+teacher_loss+kd
          
            self.model_backward_and_update(loss)
                
        loss_summary = {
            "loss": loss.item(),
            'kd':kd.item(),
            'teacher':teacher_loss.item(),
            'ce':ce_loss.item(),
            "acc": compute_accuracy(output, labels)[0].item(), # 불변 로짓 기준 정확도 보고
        }
            # output = self.model(images)
            # loss = F.cross_entropy(output, labels)
            # self.model_backward_and_update(loss)
            
        # loss_summary = {
        #     "loss": loss.item(),
        #     "acc": compute_accuracy(output, labels)[0].item(),
        # }
        
        # if (self.batch_idx + 1) == self.num_batches:
        #     self.update_lr()

        return loss_summary
