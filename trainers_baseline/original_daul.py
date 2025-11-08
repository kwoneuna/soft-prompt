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
        self.text_prompt_learner = text_PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.text_prompt_learner.tokenized_prompts
        self.text_encoder = TextEncoder(cfg, clip_model, self.text_prompt_learner)

        if cfg.MODEL.BACKBONE.NAME.split('-')[0] == 'ViT':
            self.image_encoder = ImageEncoder_Trans(cfg, clip_model, self.visual_prompt_learner)
        else:  # RN50, RN101
            self.image_encoder = ImageEncoder_Conv(cfg, clip_model, self.visual_prompt_learner)
            
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.prompts = torch.tensor(0)
        self.token_prompts = torch.tensor(0)

        self.conv = cfg.TRAINER.DUAL.ENABLE_CONV

    def forward(self, image, prompt=False):
        if self.conv:
            _, deep_ctx, vctx, deep_vctx, image = self.visual_prompt_learner(image)
        else:
            _, deep_ctx, vctx, deep_vctx = self.visual_prompt_learner()
        
        prompts = self.text_prompt_learner()
        self.prompts = prompts
        self.token_prompts = self.tokenized_prompts
        self.text_features = self.text_encoder(prompts, self.tokenized_prompts)

        # text_features = self.text_encoder(prompts, self.tokenized_prompts, deep_ctx) 
        text_features = self.text_features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        if prompt:
            image_features = self.image_encoder(image.type(self.dtype), None, None) 
        else:   
            image_features = self.image_encoder(image.type(self.dtype), vctx, deep_vctx)    
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ text_features.t()

        return logits
    

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
        # for name, param in self.model.named_parameters():
        #     if "visual_prompt_learner" not in name and "text_prompt_learner" not in name:
        #         param.requires_grad_(False)
        #     elif "visual_prompt_learner" in name or "text_prompt_learner" in name:
        #         param.requires_grad_(True) # 프롬프트 파라미터는 학습 허용
        #         optim_params.append(param)
        for p in self.model.parameters():
            p.requires_grad_(False)

        # 필요한 모듈만 켭니다
        for p in self.model.visual_prompt_learner.parameters():
            p.requires_grad_(True)
        for p in self.model.text_prompt_learner.parameters():
            p.requires_grad_(True)
        # 3) optimizer 파라미터 목록을 "모듈 기준"으로 만든다
        optim_params += list(self.model.image_encoder.prompt_learner.parameters())
        optim_params += list(self.model.text_prompt_learner.parameters())

        # 4) 최종 체크(디버그)
        print("Trainable params:",
            [n for n, p in self.model.named_parameters() if p.requires_grad])
        print("len(optim_params) =", len(optim_params))
        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {sorted(enabled)}")
        # print("# params: {:,}".format(count_num_param(self.model.prompt_learner)))

        # if cfg.MODEL.INIT_WEIGHTS:
        #     load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)
        num_vpt_params = count_num_param(self.model.visual_prompt_learner)
        num_coop_params = count_num_param(self.model.text_prompt_learner)
        print(f"# visual prompt params: {num_vpt_params:,}")
        print(f"# text prompt params: {num_coop_params:,}")
        print(f"# total prompt params: {num_vpt_params + num_coop_params:,}")
        self.model.to(self.device)
        
        # NOTE: only give prompt_learner to the optimizer
        # self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.optim = build_optimizer(optim_params, cfg.OPTIM) # 수정된 코드: 두 프롬프트의 파라미터를 모두 전달
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        # 여러 학습기를 등록할 경우:
        self.register_model("visual_prompt_learner", self.model.visual_prompt_learner, self.optim, self.sched)
        self.register_model("text_prompt_learner", self.model.text_prompt_learner, self.optim, self.sched)
        # 단일 파라미터 그룹을 등록할 경우:
        # self.register_model("dual_prompt_learners", [self.model.visual_prompt_learner, self.model.text_prompt_learner], self.optim, self.sched)       
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
            output = self.model(images)
            loss = F.cross_entropy(output, labels)
            self.model_backward_and_update(loss)
            
        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, labels)[0].item(),
        }
        
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary
