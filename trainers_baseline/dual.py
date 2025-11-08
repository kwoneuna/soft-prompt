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
        self.num_domains = 3
        
        self.domain_text_prompt_learners = nn.ModuleList([
            text_PromptLearner(cfg, classnames, clip_model) for _ in range(self.num_domains)
        ])
        # 2. 도메인 불변 텍스트 프롬프트 (1개)
        self.invariant_text_prompt_learner = text_PromptLearner(cfg, classnames, clip_model)

        # 3. 도메인 특화 시각 프롬프트 (N개) - VPT 스타일
        self.domain_visual_prompt_learners = nn.ModuleList([
            VisualPromptLearner(cfg, classnames, clip_model) for _ in range(self.num_domains)
        ])
        
        # 4. 도메인별 이미지 어댑터 (Adapter for Image Features)
        ctx_dim = clip_model.ln_final.weight.shape[0]
        self.image_adapters = nn.ModuleList([
            nn.Linear(ctx_dim, ctx_dim) for _ in range(self.num_domains)
        ])
        for adapter in self.image_adapters:
            nn.init.constant_(adapter.weight, 0.)
            nn.init.constant_(adapter.bias, 0.)
        self.tokenized_prompts = self.domain_text_prompt_learners[0].tokenized_prompts
        self.text_encoder = TextEncoder(cfg, clip_model, self.domain_text_prompt_learners[0])
        # self.text_prompt_learner = text_PromptLearner(cfg, classnames, clip_model)
        # self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts
        # self.text_encoder = TextEncoder(cfg, clip_model, self.text_prompt_learner)

        if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT':
            self.image_encoder = ImageEncoder_Trans(cfg, clip_model, self.visual_prompt_learner)
        else:  # RN50, RN101
            self.image_encoder = ImageEncoder_Conv(cfg, clip_model, self.visual_prompt_learner)
            
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.prompts = torch.tensor(0)
        self.token_prompts = torch.tensor(0)
        self.ctx_dim = clip_model.ln_final.weight.shape[0]
        self.conv = cfg.TRAINER.DUAL.ENABLE_CONV

    def forward(self, image, domain=None, test_mode=False):
        # 1. CLIP Image Feature 계산 (frozen encoder)
        B = image.size(0)
        dtype = self.dtype
        # 2. 도메인별 Visual Prompt 텐서와 Image Feature Adapter 적용
        logit_scale = self.logit_scale.exp()
        adapted_image_features = torch.zeros(
            B, 
            self.ctx_dim, # <- 이 부분이 수정됨
            dtype=dtype, 
            device=image.device
        )
        domain_logits = torch.zeros(B, self.tokenized_prompts.size(0), device=image.device, dtype=dtype)
            
        # 2-1. 불변 텍스트 프롬프트 특징 계산
        inv_text_prompts = self.invariant_text_prompt_learner()
        inv_text_feat = self.text_encoder(inv_text_prompts, self.tokenized_prompts)
        inv_text_feat = inv_text_feat / inv_text_feat.norm(dim=-1, keepdim=True)
        
        domain_text_feats = []
        for p_learner in self.domain_text_prompt_learners:
            tfeat = self.text_encoder(p_learner(), self.tokenized_prompts)
            # 👇 여기! L2 normalize만 해주고 모양은 [num_classes, 512] 그대로 둔다
            tfeat = tfeat / tfeat.norm(dim=-1, keepdim=True)
            domain_text_feats.append(tfeat)


        if test_mode:
            image_features = self.image_encoder(image.type(self.dtype), None, None) # 기본 인코딩
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logit_scale = self.logit_scale.exp()
            return logit_scale * image_features @ inv_text_feat.t()
        
        # 훈련 모드: Domain-Specific Adaptation 적용
        for d_idx in range(self.num_domains):
            mask = (domain == d_idx+1)
            if mask.any():
                img_d = image[mask].type(self.dtype)
                
                vpl_d = self.domain_visual_prompt_learners[d_idx]
                
                if self.conv:
                    _, deep_ctx, vctx, deep_vctx, img_d = vpl_d(img_d)
                    raw_feat_d = self.image_encoder(img_d, None, None)
                else:
                    _, deep_ctx, vctx, deep_vctx = vpl_d()
                    raw_feat_d = self.image_encoder(img_d, vctx, deep_vctx) 
                raw_feat_d = raw_feat_d.to(dtype)

                # ⚠️ adapter는 보통 fp32로 만들어졌으니까, 여기서 맞춰줌
                adapted_feat_d = self.image_adapters[d_idx](raw_feat_d.float()).to(dtype) + raw_feat_d
                adapted_image_features[mask] = adapted_feat_d

                text_feat_d = domain_text_feats[d_idx]  # [C, D]
                logits_d = logit_scale * adapted_feat_d @ text_feat_d.t()
                domain_logits[mask] = logits_d
        adapted_image_features = adapted_image_features / adapted_image_features.norm(dim=-1, keepdim=True)
        inv_logits = logit_scale * adapted_image_features @ inv_text_feat.t()

        return inv_logits, domain_logits, inv_text_feat, domain_text_feats

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

        # return logits
    

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
        
        for p in self.model.parameters():
            p.requires_grad_(False)

        optim_params = []
        
            # 1. 도메인 특화 텍스트 프롬프트 (N개) - 파라미터 수집만
        for p_learner in self.model.domain_text_prompt_learners:
            for p in p_learner.parameters(): 
                p.requires_grad_(True)
            optim_params += list(p_learner.parameters())
            
        # 2. 도메인 불변 텍스트 프롬프트 (1개) - 파라미터 수집만
        for p in self.model.invariant_text_prompt_learner.parameters(): 
            p.requires_grad_(True)
        optim_params += list(self.model.invariant_text_prompt_learner.parameters())

        # 3. 도메인 특화 시각 프롬프트 (N개) - 파라미터 수집만
        for vpl in self.model.domain_visual_prompt_learners:
            for p in vpl.parameters(): 
                p.requires_grad_(True)
            optim_params += list(vpl.parameters())
            
        # 4. 도메인별 이미지 어댑터 (N개) - 파라미터 수집만
        for adapter in self.model.image_adapters:
            for p in adapter.parameters(): 
                p.requires_grad_(True)
            optim_params += list(adapter.parameters())

        # --- 🌟 해결책 적용: 옵티마이저 및 스케줄러를 먼저 빌드합니다. ---
        self.optim = build_optimizer(optim_params, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        
        # 5. 이제 등록 (Register)을 수행합니다.
        # 등록은 optim_params 리스트 순서와 관계 없이, 원하는 모듈 이름으로 수행하면 됩니다.

        # 5-1. 텍스트 프롬프트 등록
        for i, p_learner in enumerate(self.model.domain_text_prompt_learners):
            self.register_model(f"dom_text_prompt_{i}", p_learner, self.optim, self.sched)
        self.register_model("inv_text_prompt", self.model.invariant_text_prompt_learner, self.optim, self.sched)

        # 5-2. 시각 프롬프트 등록
        for i, vpl in enumerate(self.model.domain_visual_prompt_learners):
            self.register_model(f"dom_visual_prompt_{i}", vpl, self.optim, self.sched)
            
        # 5-3. 어댑터 등록
        for i, adapter in enumerate(self.model.image_adapters):
            self.register_model(f"image_adapter_{i}", adapter, self.optim, self.sched)
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {sorted(enabled)}")
        total_trainable_params = sum(p.numel() for p in optim_params)
        print(f"✅ Total trainable parameters: {total_trainable_params:,}")
        self.model.to(self.device)
        
        self.scaler = GradScaler() if cfg.TRAINER.DUAL.PREC == "amp" else None

    def forward_backward(self, batch):
        images, labels,domain = self.parse_batch_train(batch)
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
            inv_logits, dom_logits, inv_tfeat, dom_tfeats = self.model(images, domain)

            # 2. Classification Loss
            ce_inv = F.cross_entropy(inv_logits, labels)
            ce_dom = F.cross_entropy(dom_logits, labels)

            # 3. KL Regularization: 불변이 각 도메인과 비슷해지도록
            kl = 0
            for tfeat in dom_tfeats:
                # KL-Div는 Logit에 Softmax를 적용한 분포에 대해 계산하는 것이 일반적
                # 여기서는 텍스트 특징에 Softmax를 적용한 분포를 사용
                p = F.log_softmax(inv_tfeat, dim=-1)
                q = F.softmax(tfeat, dim=-1)
                kl = kl + F.kl_div(p, q, reduction="batchmean")
                
            kl = kl / len(dom_tfeats)
            
            # 4. Total Loss
            loss = ce_inv + ce_dom + 0.1 * kl

            self.model_backward_and_update(loss)
                
        loss_summary = {
            "loss": loss.item(),
            "loss_inv": ce_inv.item(),
            "loss_dom": ce_dom.item(),
            "loss_kl": kl.item(),
            "acc": compute_accuracy(inv_logits, labels)[0].item(), # 불변 로짓 기준 정확도 보고
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
