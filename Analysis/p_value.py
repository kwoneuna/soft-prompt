import pandas as pd
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# 파일 이름 및 컬럼 설정
variance_file_name = 'terra_2_variance.csv'
accuracy_file_name = 'terra_2_test_acc.csv'
merge_key_column = 'Step'
variance_column = 'Value'
accuracy_column = 'Value'

# 1. 두 CSV 파일 불러오기
try:
    df_var = pd.read_csv(variance_file_name)
    df_acc = pd.read_csv(accuracy_file_name)
except FileNotFoundError as e:
    # 에러 처리 코드
    pass

# 2. 병합을 위해 컬럼 이름 변경 및 데이터프레임 병합
df_var_processed = df_var[[merge_key_column, variance_column]].rename(columns={variance_column: 'Prompt_Variance'})
df_acc_processed = df_acc[[merge_key_column, accuracy_column]].rename(columns={accuracy_column: 'Test_Accuracy'})

merged_df = pd.merge(
    df_var_processed,
    df_acc_processed,
    on=merge_key_column,
    how='inner'
)

# 3. Pearson 상관분석 수행
data_var = merged_df['Prompt_Variance'].dropna()
data_acc = merged_df['Test_Accuracy'].dropna()

correlation_coefficient, p_value = pearsonr(data_var, data_acc)

# 4. 병합된 데이터를 CSV 파일로 저장
# merged_df.to_csv('merge_pacs_s.csv', index=False)

# 5. 상관 분석 결과 출력
# ...

# 6. 산점도(Scatter Plot) 시각화 및 저장
plt.figure(figsize=(8, 6))

sns.regplot(
    x='Prompt_Variance',
    y='Test_Accuracy',
    data=merged_df,
    scatter_kws={'alpha': 0.7, 's': 80},
    line_kws={'color': 'red', 'linestyle': '--'}
)

plt.title(f'Prompt Variance vs. Test Domain Accuracy\n(R = {correlation_coefficient:.4f}, P-value = {p_value:.2e})', fontsize=14)
plt.xlabel('Prompt Variance', fontsize=12)
plt.ylabel('Test Domain Accuracy (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

plot_filename = 'terra_2_variance_accuracy_correlation_scatter_new.png'
plt.savefig(plot_filename, bbox_inches='tight')