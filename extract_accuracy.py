import re
import os
import pandas as pd

def extract_log_info(base_path, log_file_path):
    """
    로그 파일 경로에서 Method/Dataset/Domain 정보를,
    로그 파일 내용에서 모든 Test Acc 중 최대값과 해당 Epoch을 추출합니다.
    """
    # 1. 초기값 설정
    method_name = "N/A"
    dataset_name = "N/A"
    domain_index = "N/A"

    # Best Test Acc 및 Epoch을 찾기 위한 초기값
    max_test_acc = -1.0  # 최대 정확도를 찾기 위해 -1로 초기화
    best_epoch = "N/A"

    # 2. 경로에서 Method, Dataset, Domain Index 추출
    try:
        relative_path = os.path.relpath(log_file_path, base_path)
        log_parts = relative_path.split(os.sep)

        if len(log_parts) >= 2:
            method_name = log_parts[0]  # 예: Dual
            dataset_name = log_parts[2] # 예: vlcs, terra_incognita

        try:
            vit_index = log_parts.index('ViT-B16')
            if len(log_parts) > vit_index + 1:
                domain_index = log_parts[vit_index + 1]
        except ValueError:
            pass

    except Exception as e:
        print(f"❌ 경로 분석 오류: {log_file_path} - {e}")

    # 3. 로그 파일에서 모든 test acc와 epoch 쌍을 추출하고 최대값 찾기
    try:
        with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()

        # 예: "test acc: 72.34 %, epoch: 10"
        regex = r"test\s+acc:\s*(\d+\.\d+)\s*%\s*,\s*epoch:\s*(\d+)"
        all_matches = re.findall(regex, log_content)

        if all_matches:
            for acc_str, epoch_str in all_matches:
                current_acc = float(acc_str)
                current_epoch = int(epoch_str)

                if current_acc > max_test_acc:
                    max_test_acc = current_acc
                    best_epoch = current_epoch

            best_test_acc_result = max_test_acc
        else:
            best_test_acc_result = "N/A"
            best_epoch = "N/A"

    except FileNotFoundError:
        print(f"⚠️ 경고: 파일이 존재하지 않습니다: {log_file_path}")
        best_test_acc_result = "File Not Found"
        best_epoch = "N/A"
    except Exception as e:
        print(f"❌ 로그 내용 분석 오류: {log_file_path} - {e}")
        best_test_acc_result = "Error"
        best_epoch = "N/A"

    return {
        "Method Name": method_name,
        "Dataset Name": dataset_name,
        "Domain Index": domain_index,
        "Best Test Acc (%)": best_test_acc_result,
        "Epoch": best_epoch,
        "Log File Path": log_file_path
    }

def find_and_process_logs(base_path, output_csv_filename="icml2.csv"):
    """
    기준 경로에서 모든 log.txt 파일을 찾아 정보를 추출하고 CSV로 저장합니다.
    기존 파일이 있으면 데이터를 추가(append)합니다.
    """
    all_results = []
    log_count = 0

    print(f"🔍 '{base_path}' 경로 아래에서 log.txt 파일을 검색하고, 최대 Test Acc를 추출합니다...")
    for root, dirs, files in os.walk(base_path):
        if "log.txt" in files:
            log_file_path = os.path.join(root, "log.txt")
            info = extract_log_info(base_path, log_file_path)
            all_results.append(info)
            log_count += 1

    if not all_results:
        print("\n🚫 log.txt 파일을 찾지 못했습니다. 경로를 확인해주세요.")
        return

    # Pandas DataFrame 생성
    df = pd.DataFrame(all_results)

    # 기존 파일 존재 여부에 따른 저장 모드
    if os.path.exists(output_csv_filename):
        df.to_csv(output_csv_filename, mode='a', header=False, index=False, encoding='utf-8')
        print(f"\n📝 기존 CSV 파일에 새로운 {log_count}개 로그 데이터를 **추가**했습니다.")
    else:
        df.to_csv(output_csv_filename, index=False, encoding='utf-8')
        print(f"\n✅ 새로운 CSV 파일 '{output_csv_filename}'을 생성했습니다.")

    print("=" * 60)
    print(f"총 {log_count}개의 로그 파일에서 최대 정확도를 추출했습니다.")
    print("=" * 60)

    print("\n--- 추출된 데이터 미리보기 (상위 5개) ---")
    print(df.head())

# --- 스크립트 실행 ---

BASE_SEARCH_PATH = "/workspace/Soft-Prompt-Generation/icml/multi-dg/tuning/"

if not os.path.isdir(BASE_SEARCH_PATH):
    print(f"🚨 오류: 기준 경로가 존재하지 않습니다: {BASE_SEARCH_PATH}")
    print("스크립트를 실행하기 전에 해당 경로에 실제 로그 파일이 존재하는지 확인해주세요.")
else:
    find_and_process_logs(BASE_SEARCH_PATH)
