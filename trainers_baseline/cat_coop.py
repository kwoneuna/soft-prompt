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

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from trainers_baseline.basedg import *
from utils.clip_part import *

_tokenizer = _Tokenizer()


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
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
            if cfg.TRAINER.COOP.CSC:
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

class ChannelWiseTransform(nn.Module):
    """cwT: f' = gamma ⊙ f + beta (per-channel scale & shift)"""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        # x: (..., dim)
        return x * self.gamma + self.beta


class CATHead(nn.Module):
    """
    DePT의 Channel Adjusted Transfer (CAT) head.
    - 이미지/텍스트 feature에 각각 cwT 적용 (논문 v1에서 shared cwT보다 성능 좋다고 보고). :contentReference[oaicite:1]{index=1}
    - 그 뒤 같은 Linear classifier에 넣어서 class logits 생성.
    """
    def __init__(self, dim, n_cls, use_two_cwt: bool = True):
        super().__init__()
        self.use_two_cwt = use_two_cwt
        if use_two_cwt:
            self.cwt_img = ChannelWiseTransform(dim)
            self.cwt_txt = ChannelWiseTransform(dim)
        else:
            self.cwt_shared = ChannelWiseTransform(dim)

        self.fc = nn.Linear(dim, n_cls)

    def forward(self, img_feat, txt_feat=None):
        """
        img_feat: (B, dim)
        txt_feat: (B, dim) 또는 None
        return:
            logits_img: (B, C)
            logits_txt: (B, C) 또는 None
        """
        if self.use_two_cwt:
            img_t = self.cwt_img(img_feat)
            txt_t = self.cwt_txt(txt_feat) if txt_feat is not None else None
        else:
            img_t = self.cwt_shared(img_feat)
            txt_t = self.cwt_shared(txt_feat) if txt_feat is not None else None

        logits_img = self.fc(img_t)
        logits_txt = self.fc(txt_t) if txt_t is not None else None
        return logits_img, logits_txt

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(cfg, clip_model, self.prompt_learner)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.image_features = torch.tensor(0)
        self.text_features = torch.tensor(0)
        self.prompts = torch.tensor(0)
        self.token_prompts = torch.tensor(0)
        self.logit = -1
        # === DePT: CAT head 추가 ===
        feat_dim = clip_model.ln_final.weight.shape[0]  # CLIP text feature dim (image도 동일)
        n_cls = len(classnames)
        self.cat_head = CATHead(dim=feat_dim, n_cls=n_cls, use_two_cwt=True)

        self.lambda_cat = 0.7
        self.use_text_in_cat = True
    
    def forward(self, image):
        """기본 ITM head만 사용하는 기존 CoOp 스타일 forward (호환용)."""
        logits_itm, _, _ = self.encode_image_text(image)
        return logits_itm

    def encode_image_text(self, image):
        """ITM head용 image/text feature 및 logits 계산 (기존 forward 내용)."""
        # image: (B, C, H, W)
        self.image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        self.prompts = prompts
        self.token_prompts = self.tokenized_prompts
        self.text_features = self.text_encoder(prompts, self.tokenized_prompts)

        # 정규화 (CLIP 스타일)
        self.image_features = self.image_features / self.image_features.norm(dim=-1, keepdim=True)
        self.text_features = self.text_features / self.text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()
        self.logit = logit_scale
        logits_itm = logit_scale * self.image_features @ self.text_features.t()  # (B, C)

        return logits_itm, self.image_features, self.text_features
    def forward_dept(
        self,
        image,
        label=None,
        lambda_cat=None,
        use_cat=True,
        use_fusion=True,
        use_text_in_cat=None,
        return_loss=False,
    ):
        """
        DePT forward.
        - label이 None이면 pure inference 모드 (보통 test).
        - label이 있으면 dual-head loss L = γ L_CAT + (1-γ)L_ITM 계산.
        """
        if lambda_cat is None:
            lambda_cat = self.lambda_cat
        if use_text_in_cat is None:
            use_text_in_cat = self.use_text_in_cat

        logits_itm, img_feat, txt_feat_all = self.encode_image_text(image)  # (B,C), (B,D), (C,D)

        # --------- 학습이 아니거나 CAT 안 쓸 때: 그냥 ITM만 ---------
        if (label is None) or (not use_cat):
            if not use_fusion:
                if return_loss:
                    return logits_itm, None, {}
                return logits_itm

            # (label이 없으면 CAT로부터 base-specific 정보도 쓸 수 없음)
            if return_loss:
                return logits_itm, None, {}
            return logits_itm

        # --------- CAT head로 L_CAT 계산 ---------
        # label: (B,)
        # 각 샘플의 GT 클래스 텍스트 feature 가져오기
        if use_text_in_cat:
            # txt_feat_all: (C, D) -> (B, D)
            txt_feat_per_sample = txt_feat_all[label]
        else:
            txt_feat_per_sample = None

        # CAT head에서 이미지/텍스트 각각 logits 계산
        logits_cat_img, logits_cat_txt = self.cat_head(img_feat, txt_feat_per_sample)

        # L_ITM: 원래 ITM head 분류 손실
        loss_itm = F.cross_entropy(logits_itm, label)

        # L_CAT: 이미지 + (옵션) 텍스트 feature 둘 다 사용
        if logits_cat_txt is not None:
            logits_cat = torch.cat([logits_cat_img, logits_cat_txt], dim=0)   # (2B, C)
            labels_cat = torch.cat([label, label], dim=0)
        else:
            logits_cat = logits_cat_img
            labels_cat = label

        loss_cat = F.cross_entropy(logits_cat, labels_cat)

        loss = lambda_cat * loss_cat + (1.0 - lambda_cat) * loss_itm

        # --------- Test-time fusion용 logits (base task에서 사용) ---------
        if use_fusion:
            # 논문 Eq.(8): p = γ P_CAT + (1-γ) P_ITM
            probs_itm = F.softmax(logits_itm, dim=-1)
            probs_cat_img = F.softmax(logits_cat_img, dim=-1)
            probs = lambda_cat * probs_cat_img + (1.0 - lambda_cat) * probs_itm
            logits_fused = torch.log(probs + 1e-8)  # 그냥 log-prob로 만들어 CE에 넣을 수 있게
        else:
            logits_fused = logits_itm

        if return_loss:
            extra = {
                "loss_itm": loss_itm.detach(),
                "loss_cat": loss_cat.detach(),
            }
            return logits_fused, loss, extra

        return logits_fused

    
    
@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

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
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in CLIP encoders (image/text), keep prompt & CAT head trainable")
        for name, param in self.model.named_parameters():
            if ("prompt_learner" in name) or ("cat_head" in name):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        params = list(self.model.prompt_learner.parameters()) + list(self.model.cat_head.parameters())
        self.optim = build_optimizer(params, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)
        self.register_model("cat_head", self.model.cat_head, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        # device_count = torch.cuda.device_count()
        # if device_count > 1:
        #     print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
        #     self.model = nn.DataParallel(self.model)
    
    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        use_dept = True
        prec = self.cfg.TRAINER.COOP.PREC

        if use_dept:
            lambda_cat = 0.7
            use_fusion = True
            if prec == "amp":
                with autocast():
                    output, loss, extra = self.model.forward_dept(
                        image,
                        label=label,
                        lambda_cat=lambda_cat,
                        use_cat=True,
                        use_fusion=use_fusion,
                        return_loss=True,
                    )
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                output, loss, extra = self.model.forward_dept(
                    image,
                    label=label,
                    lambda_cat=lambda_cat,
                    use_cat=True,
                    use_fusion=use_fusion,
                    return_loss=True,
                )
                self.model_backward_and_update(loss)

            loss_summary = {
                "loss": loss.item(),
                "loss_itm": extra.get("loss_itm", loss).item(),
                "loss_cat": extra.get("loss_cat", loss).item(),
                "acc": compute_accuracy(output, label)[0].item(),
            }

        else:
            # DePT 끈 상태: 기존 CoOp 학습
            if prec == "amp":
                with autocast():
                    output = self.model(image)
                    loss = F.cross_entropy(output, label)
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
            else:
                output = self.model(image)
                loss = F.cross_entropy(output, label)
                self.model_backward_and_update(loss)

            loss_summary = {
                "loss": loss.item(),
                "acc": compute_accuracy(output, label)[0].item(),
            }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    
    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label
    
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

            use_dept = True
            if use_dept:
                # base/new 구분해서 fusion 여부 결정
                if split == "val":
                    use_fusion = True
                else:  # split == "test" -> new task
                    use_fusion = False

                # test에서는 label이 없으므로 loss 계산 없이 forward_dept 사용
                output = self.model.forward_dept(
                    input,
                    label=None,
                    lambda_cat=0.7,
                    use_cat=True,
                    use_fusion=use_fusion,
                    return_loss=False,
                )
            else:
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
            