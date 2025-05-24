# 파라메터 조정으로 클러스터에 노이즈를 없애기 위해서 hdbscan 만 코드를 다시 짜고
import os
import numpy as np
import pandas as pd
import hdbscan
from sklearn.preprocessing import StandardScaler
import umap

def run_hdbscan(embeddings, min_cluster_size, min_samples=None):
    # 1. 정규화
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    # 2. UMAP으로 차원 축소 (차원 수는 조절 가능)
    reducer = umap.UMAP(n_components=10, random_state=42)
    embeddings_umap = reducer.fit_transform(embeddings_scaled)

    # 3. HDBSCAN 클러스터링
    model = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples if min_samples else min_cluster_size,
        metric='euclidean'
    )
    return model.fit_predict(embeddings_umap)

def cluster_and_save(embedding_path, review_path, model_tag, min_cluster_size, output_dir):
    print(f"📥 임베딩 로딩 중: {embedding_path}")
    embeddings = np.load(embedding_path)

    print(f"📥 리뷰 데이터 로딩 중: {review_path}")
    df = pd.read_csv(review_path)
    assert len(embeddings) == len(df), "❌ 리뷰 수와 임베딩 수가 다릅니다."

    print(f"🔍 HDBSCAN 클러스터링 중 (min_cluster_size={min_cluster_size})...")
    labels = run_hdbscan(embeddings, min_cluster_size=min_cluster_size)
    df['cluster_id'] = labels

    filename = f"cluster_{model_tag}_hdbscan_{min_cluster_size}.csv"
    output_path = os.path.join(output_dir, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[['reviewText', 'cluster_id']].to_csv(output_path, index=False)

    actual_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_count = np.sum(labels == -1)
    noise_percent = (noise_count / len(df)) * 100
    print(f"✅ 저장 완료: {filename} | 유효 클러스터 수: {actual_clusters} | 노이즈: {noise_count} | 노이즈 퍼센트: {noise_percent:.2f}%")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    embed_dir = os.path.join(project_root, "embed/results2")
    review_path = os.path.join(project_root, "amazon", "amazon_reviews_pre.csv")
    output_dir = os.path.join(project_root, "results2")


    # models = {
    #     "minilm": "embeddings_sentence-transformers_all-MiniLM-L6-v2.npy",
    #     "bge": "embeddings_BAAI_bge-small-en.npy",
    #     "e5": "embeddings_intfloat_e5-small-v2.npy"
    # }

    models = {
    "minilm": "embeddings_sentence-transformers_all-MiniLM-L6-v2.npy",
    "mpnet": "embeddings_sentence-transformers_all-mpnet-base-v2.npy",
    "paraphrase": "embeddings_sentence-transformers_paraphrase-MiniLM-L12-v2.npy",
    "t5": "embeddings_sentence-transformers_sentence-t5-base.npy"
}


    hdbscan_min_sizes = [10, 15, 20, 25]  # 원하는 실험값들로 조정 가능

    for model_tag, filename in models.items():
        embedding_path = os.path.join(embed_dir, filename)
        for m in hdbscan_min_sizes:
            cluster_and_save(
                embedding_path, review_path, model_tag,
                min_cluster_size=m,
                output_dir=output_dir
            )
