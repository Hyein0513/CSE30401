import os
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from tqdm import tqdm

# 요약 생성 함수
def centroid_based_summary_and_used_sentences(reviews, embedding_model, top_k=3):
    if not reviews:
        raise ValueError("Empty review list")
    embeddings = embedding_model.encode(reviews, convert_to_numpy=True)
    centroid = embeddings.mean(axis=0)
    similarities = cosine_similarity([centroid], embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]
    selected_sentences = [reviews[i] for i in top_indices]
    selected_sentences_sorted = sorted(selected_sentences, key=lambda s: reviews.index(s))
    summary = " ".join(selected_sentences_sorted)
    return summary, selected_sentences_sorted

# 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.abspath(os.path.join(script_dir, "../../cluster/results/cluster"))
output_dir = os.path.join(script_dir, "summary_1_Centroid")
os.makedirs(output_dir, exist_ok=True)
fail_log_path = os.path.join(output_dir, "fail_log.txt")

# 모델 로드
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 실패 로그 초기화
with open(fail_log_path, "w", encoding="utf-8") as log_file:
    log_file.write("요약 실패 파일 목록:\n")

# input 파일 리스트
input_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

# 진행바 적용
for file_name in tqdm(input_files, desc="Centroid 요약 진행 중"):
    input_path = os.path.join(input_dir, file_name)

    try:
        df = pd.read_csv(input_path)

        if 'reviewText' not in df.columns or 'cluster_id' not in df.columns:
            raise ValueError("Required columns not found")

        summaries = []

        # cluster_id별 그룹 처리
        grouped = df.groupby('cluster_id')
        for cluster_id, group in grouped:
            reviews = group['reviewText'].dropna().tolist()
            if len(reviews) == 0:
                continue

            summary, used_reviews = centroid_based_summary_and_used_sentences(reviews, embedding_model, top_k=3)

            summaries.append({
                "cluster_id": cluster_id,
                "summary": summary,
                "used_reviews": used_reviews
            })

        # 결과 저장
        out_df = pd.DataFrame(summaries)
        output_file_name = file_name.replace(".csv", "_Centroid.csv")
        output_path = os.path.join(output_dir, output_file_name)
        out_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    except Exception as e:
        with open(fail_log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"{file_name}: {str(e)}\n")
        print(f"❌ 실패: {file_name} → {e}")
