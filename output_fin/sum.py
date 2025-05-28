import pandas as pd

# 파일 불러오기
df_eval = pd.read_csv("merged_stats_fin_fin.csv")
df_ivd = pd.read_csv("summary_swir_stats.csv")

# 클러스터 ID가 두 데이터프레임 모두에 있을 경우 df_ivd 쪽 제거
if 'cluster_id' in df_ivd.columns:
    df_ivd = df_ivd.drop(columns=["cluster_id"])

# file_id 기준 병합 (공통 file_id만 유지)
merged_df = pd.merge(df_eval, df_ivd, on="file_id", how="inner")

# file_id 중복 제거 (첫 번째만 유지)
merged_df = merged_df.drop_duplicates(subset="file_id", keep="first")

# 결과 저장
merged_df.to_csv("merged_stats_fin_fin_fin.csv", index=False)

print("✅ 병합 완료: merged_stats_fin_fin_fin.csv 생성됨")
