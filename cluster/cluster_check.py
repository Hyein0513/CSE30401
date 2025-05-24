# hdbscan이 제대로 되었는지 확인하기 위해서 
# 각 csv 마다 클러스터 수, 노이즈 비율, 2차원 시각화, 클러스터 안에 있는 리뷰, 노이즈 데이터 예시 출력해서 확인
# 정성적으로 클러스터 확인해서 제일 적절하게 클러스터링 된 파라메터 값을 찾기 


import os
import pandas as pd

def analyze_clusters_to_txt(cluster_dir, output_txt_path, examples_per_cluster=2):
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for filename in sorted(os.listdir(cluster_dir)):
            if not filename.endswith(".csv") or not filename.startswith("cluster_"):
                continue

            filepath = os.path.join(cluster_dir, filename)
            df = pd.read_csv(filepath)

            f.write(f"\n📁 파일: {filename}\n")
            f.write(f"총 리뷰 수: {len(df)}\n")

            cluster_counts = df['cluster_id'].value_counts().sort_index()

            for cluster_id, count in cluster_counts.items():
                f.write(f"\n🔹 클러스터 {cluster_id} — 리뷰 수: {count}\n")

                examples = df[df['cluster_id'] == cluster_id]['reviewText'].head(examples_per_cluster).tolist()
                for i, ex in enumerate(examples):
                    preview = ex.strip().replace('\n', ' ')[:200]
                    f.write(f"    예시 {i+1}: {preview}...\n")

                if cluster_id == -1:
                    f.write("⚠️ 노이즈 \n")
        
if __name__ == "__main__":
    cluster_dir = "./results" 

    # 📄 출력 파일 경로
    output_txt_path = os.path.join("..", "cluster", "cluster_report.txt")

    analyze_clusters_to_txt(cluster_dir, output_txt_path)

    print(f"\n✅ 리포트 저장 완료: {output_txt_path}")
