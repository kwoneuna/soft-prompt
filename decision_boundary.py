# 0) 경로
DATASET_YAML = "configs/datasets/multi_source/office_home.yaml"
TRAINER_YAML = "configs/trainers/BASELINE/b32_ep50.yaml"   # ← COOP용으로 변경!

from yacs.config import CfgNode as CN
from dassl.config import get_cfg_default
from dassl.engine import build_trainer  # ← 핵심: 이 한 줄이면 충분

cfg = get_cfg_default()
cfg.set_new_allowed(True)   # 새 키 허용
cfg.defrost()

# dataset → trainer 순서로 병합
cfg.merge_from_file(DATASET_YAML)
cfg.merge_from_file(TRAINER_YAML)

# 런타임 오버라이드
cfg.MODEL.BACKBONE.NAME = "ViT-B/16"
cfg.OUTPUT_DIR = "outputs_baseline/baseline/COOP/office_home/b32_ep50/ViT-B16/r/seed_1/warmup_1"

DOMAIN_MAP = {"a":"Art","c":"Clipart","p":"Product","r":"RealWorld"}
target = "r"
cfg.DATASET.SOURCE_DOMAINS = [v for k,v in DOMAIN_MAP.items() if k!=target]
cfg.DATASET.TARGET_DOMAINS = [DOMAIN_MAP[target]]

import torch
cfg.USE_CUDA = torch.cuda.is_available()
cfg.GPU = [0]

cfg.TEST.SPLIT = "test"
cfg.TEST.NO_TEST = False

# 필요시 transform 기본값 보강
cfg.INPUT.RRCROP_SCALE = cfg.INPUT.get("RRCROP_SCALE", (0.08, 1.0))
cfg.INPUT.RRCROP_RATIO = cfg.INPUT.get("RRCROP_RATIO", (0.75, 1.3333))

# COOP 섹션이 없다면 기본값 넣기 (대소문자 주의: "COOP")
if "TRAINER" not in cfg: cfg.TRAINER = CN()
if "COOP" not in cfg.TRAINER: cfg.TRAINER.COOP = CN()
cfg.TRAINER.COOP.PREC = cfg.TRAINER.COOP.get("PREC", "fp32")

cfg.freeze()

# 트레이너 불러오기 & 평가
cfg.defrost()
cfg.TRAINER.NAME = "CoOp"  # 트레이너 이름 지정 (대소문자 정확히)
cfg.freeze()

trainer = build_trainer(cfg)  # ← 레지스트리 접근 없이 생성
trainer.build_model()
trainer = CoOp(cfg)
trainer.build_model()

import os, torch
ckpt = os.path.join(cfg.OUTPUT_DIR, "best_val.pt")
if not os.path.exists(ckpt):
    alt = os.path.join(cfg.OUTPUT_DIR, "best_test.pt")
    ckpt = alt if os.path.exists(alt) else ckpt

trainer.model = torch.load(ckpt, map_location="cuda:0" if cfg.USE_CUDA else "cpu").to(trainer.device)
trainer.model.eval()

acc = trainer.test("test")
print(f"[TEST] accuracy: {acc:.4f}")
