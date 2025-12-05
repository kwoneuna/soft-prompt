import os
import torch
import clip
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# CLIP Zero-shot Feature Extraction 및 거리 계산 코드
# -------------------------------------------------------------

def load_clip(device="cuda" if torch.cuda.is_available() else "cpu"):
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def get_image_paths(root_dir):
    image_paths = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(root, f))
    return image_paths


def extract_features(image_paths, model, preprocess, device):
    features = []
    for p in tqdm(image_paths, desc="Extracting features"):
        try:
            img = preprocess(Image.open(p).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model.encode_image(img)
                feat /= feat.norm(dim=-1, keepdim=True)
            features.append(feat.cpu().numpy()[0])
        except Exception as e:
            print(f"[Error] {p}: {e}")
    return np.array(features)


def compute_distances(feats_A, feats_B):
    # Cosine distance = 1 - cosine similarity
    sims = feats_A @ feats_B.T
    dists = 1 - sims
    return dists


def compute_distance_matrix(feats_domain_class_dict):
    results = {}
    keys = list(feats_domain_class_dict.keys())
    for key1 in keys:
        for key2 in keys:
            feats1 = feats_domain_class_dict[key1]
            feats2 = feats_domain_class_dict[key2]
            d = compute_distances(feats1, feats2)
            results[(key1, key2)] = d.flatten()  # 모든 거리 저장
    return results

def compute_average_dist(feats_domain_class_dict):
    results = {}
    keys = list(feats_domain_class_dict.keys())

    for i, key1 in enumerate(keys):
        for j, key2 in enumerate(keys):
            feats1 = feats_domain_class_dict[key1]
            feats2 = feats_domain_class_dict[key2]
            d = compute_distances(feats1, feats2)
            avg = d.mean()
            results[(key1, key2)] = float(avg)
    return results


# -------------------------------------------------------------
# 사용 예시 (사용자가 데이터 폴더 맞게 수정)
# -------------------------------------------------------------
def plot_histograms(distance_matrix):
    for k, dist_array in distance_matrix.items():
        plt.figure(figsize=(6,4))
        plt.hist(dist_array, bins=40)
        plt.title(f"Histogram: {k}")
        plt.xlabel("Cosine Distance")
        plt.ylabel("Frequency")
        plt.tight_layout()
        # 파일명에 key 정보를 포함 (도메인_클래스 형태)
        name1 = f"{k[0][0]}_{k[0][1]}"
        name2 = f"{k[1][0]}_{k[1][1]}"
        save_name = f"histogram_{name1}__{name2}.png"

        plt.savefig(save_name)
        plt.close()   # 그래프 메모리 해제        # plt.show()


if __name__ == "__main__":

    # 예시 폴더 구조
    # /workspace/data/office_home/art/Alarm_Clock
    # /workspace/data/office_home/product/Alarm_Clock
    # /workspace/data/office_home/art/Chair
    # /workspace/data/office_home/product/Chair

    # base_dir = "/workspace/data/office_home"
    # domains = ["art", "product", "clipart", "real_world"]  # 필요시 수정
    # classes = ["Alarm_Clock", "Chair"]  # 필요시 수정

    
    base_dir = "/workspace/data/VLCS"
    domains = ["Caltech101", "LabelMe", "SUN09", "VOC2007"]  # 필요시 수정
    classes = ["bird", "person"]  # 필요시 수정

    model, preprocess, device = load_clip()

    feats_dict = {}

    for d in domains:
        for c in classes:
            data_path = os.path.join(base_dir, d, c)
            if not os.path.isdir(data_path):
                continue

            image_paths = get_image_paths(data_path)
            if len(image_paths) == 0:
                continue

            feats = extract_features(image_paths, model, preprocess, device)
            feats_dict[(d, c)] = feats
            print(f"Loaded {len(feats)} features for {(d, c)}")
 
    # 거리 계산 (평균)
    results = compute_average_dist(feats_dict)
    # distance_matrix = compute_distance_matrix(feats_dict)
    # plot_histograms(distance_matrix)
    print("\n===== 평균 거리 결과 =====")
    for k, v in results.items():
        print(k, "->", v)
