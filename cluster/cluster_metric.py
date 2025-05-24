import os
import pandas as pd
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

def evaluate_cluster_quality(csv_folder, output_txt_path):
    report_lines = []

    for filename in sorted(os.listdir(csv_folder)):
        if not filename.endswith(".csv"):
            continue

        filepath = os.path.join(csv_folder, filename)
        df = pd.read_csv(filepath)

        if 'cluster_id' not in df.columns or 'reviewText' not in df.columns:
            report_lines.append(f"[{filename}] -> ❌ cluster_id 또는 reviewText 컬럼 없음\n")
            continue

        if df['cluster_id'].nunique() < 2:
            report_lines.append(f"[{filename}] -> ❌ 클러스터 수가 2개 미만이라 평가 생략\n")
            continue

        # 임시 벡터 (텍스트 길이 기반)
        review_lengths = df['reviewText'].fillna("").str.len().values.reshape(-1, 1)
        vectors = StandardScaler().fit_transform(review_lengths)
        labels = df['cluster_id'].values

        try:
            sil_score = silhouette_score(vectors, labels)
            ch_score = calinski_harabasz_score(vectors, labels)
            db_score = davies_bouldin_score(vectors, labels)

            cluster_count = df['cluster_id'].nunique()
            total_reviews = len(df)

            report = (
                f"[{filename}]\n"
                f"  - 클러스터 수:             {cluster_count}\n"
                f"  - 리뷰 수:                 {total_reviews}\n"
                f"  - Silhouette Score:        {sil_score:.4f}\n"
                f"  - Calinski-Harabasz Score: {ch_score:.2f}\n"
                f"  - Davies-Bouldin Score:    {db_score:.4f}\n"
            )

            # 🔍 HDBSCAN이면 노이즈 정보 추가
            if "hdbscan" in filename.lower():
                noise_count = np.sum(labels == -1)
                noise_percent = (noise_count / total_reviews) * 100
                report += (
                    f"  - 노이즈 수:               {noise_count}\n"
                    f"  - 노이즈 퍼센트:           {noise_percent:.2f}%\n"
                )

            report_lines.append(report + "\n")

        except Exception as e:
            report_lines.append(f"[{filename}] -> ❌ 평가 실패: {str(e)}\n")

    # 결과 저장
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.writelines(report_lines)

    print(f"📄 클러스터 품질 평가 완료! 결과 저장 위치: {output_txt_path}")

# 실행 예시
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_folder = os.path.join(project_root, "results2")
    output_txt = os.path.join(project_root, "cluster/results2/emb2_cluster_metrics_report.txt")

    evaluate_cluster_quality(csv_folder, output_txt)
