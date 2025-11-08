#!/bin/bash

source activate spg

# custom config
DATA='/workspace/data_spg'   # ******* your data path *******
CFG=b32_ep50

BACKBONE=$1 # backbone name
TRAINER=$2
GPU=$3

# bash scripts/baseline/zsclip.sh RN50 CLIP_ZS 0
# bash scripts/baseline/zsclip.sh ViT-B/16 CLIP_ZS 0

# bash scripts/baseline/zsclip.sh RN50 CLIP_LR 0
# bash scripts/baseline/zsclip.sh ViT-B/16 CLIP_LR 0
DATASET=pacs

# 돌릴 target domain 조합을 정의 (원하는 조합만 남겨도 됨)
combos=(
#   "a" "c" "p" "s"                # 단일
  "a c" "a p" "a s" "c p" "c s" "p s"   # 2개 조합
  "a c p" "a c s" "a p s" "c p s"     # 3개 조합 (원하면 주석 해제)
#   "a c p s"                           # 4개 전부 (원하면 주석 해제)
)

for SEED in 1; do
  for WARMUP in 1; do
    for combo in "${combos[@]}"; do
      # "a c" → ("a" "c")
      IFS=' ' read -r -a DOMS <<< "$combo"

      # 폴더명용 키: a_c
      DOM_KEY=$(IFS=_; echo "${DOMS[*]}")

      DIR=outputs_baseline/T-dg/${TRAINER}/${DATASET}/${CFG}/${BACKBONE//\//}/${DOM_KEY}/seed_${SEED}/warmup_${WARMUP}

      if [ -d "$DIR" ]; then
        echo "Results are available in ${DIR}, so skip this job"
      else
        echo "Run this job and save the output to ${DIR}"

        python train_baseline.py \
          --gpu "${GPU}" \
          --backbone "${BACKBONE}" \
          --target-domains "${DOMS[@]}" \
          --root "${DATA}" \
          --trainer "${TRAINER}" \
          --dataset-config-file "configs/datasets/multi_source/${DATASET}.yaml" \
          --config-file "configs/trainers/BASELINE/${CFG}.yaml" \
          --output-dir "${DIR}" \
          --seed "${SEED}" \
          --warmup_epoch "${WARMUP}" \
          --eval-only
      fi
    done
  done
done
