# 총 리뷰 수가 5000여개(4914개)이므로 클러스터 수를 정해줘야함.
# KMeans 는 10, 15, 20 개 선정 -> 클러스터 하나당 500 / 330 / 250 개 
# HDBSCAN은 10, 20, 30 으로 선정 -> min 클러스터 수 -> 알고리즘이 알아서 적정 크기 설정
#   : min값을  너무 크게 설정해서 대부분의 값이 노이즈로 분류됨...
#   : 고차원에는 HDBSCAN이 작동 잘 안함 그래서 UMAP붙여서 저차원으로 투영하고 클러스터링 함 


import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def run_kmeans(embeddings, n_clusters):
    model = KMeans(n_clusters=n_clusters, random_state=42)
    return model.fit_predict(embeddings)

def cluster_and_save(embedding_path, review_path, model_tag, n_clusters, output_dir):
    print(f"📥 임베딩 로딩 중: {embedding_path}")
    embeddings = np.load(embedding_path)

    print(f"📥 리뷰 데이터 로딩 중: {review_path}")
    df = pd.read_csv(review_path)
    assert len(embeddings) == len(df), "❌ 리뷰 수와 임베딩 수가 다릅니다."

    labels = run_kmeans(embeddings, n_clusters=n_clusters)
    df['cluster_id'] = labels

    filename = f"cluster_{model_tag}_kmeans_{n_clusters}.csv"
    output_path = os.path.join(output_dir, filename)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df[['reviewText', 'cluster_id']].to_csv(output_path, index=False)

    actual_clusters = len(set(labels))
    print(f"✅ 저장 완료: {filename} | 클러스터 수: {actual_clusters}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    embed_dir = os.path.join(project_root, "embed/results2")
    review_path = os.path.join(project_root, "amazon", "amazon_reviews_pre.csv")
    output_dir = os.path.join(project_root, "results2")

    # models = {
    #    "minilm": "embeddings_sentence-transformers_all-MiniLM-L6-v2.npy",
    #    "bge": "embeddings_BAAI_bge-small-en.npy",
    #    "e5": "embeddings_intfloat_e5-small-v2.npy"
    # }

    models = {
    "minilm": "embeddings_sentence-transformers_all-MiniLM-L6-v2.npy",
    "mpnet": "embeddings_sentence-transformers_all-mpnet-base-v2.npy",
    "paraphrase": "embeddings_sentence-transformers_paraphrase-MiniLM-L12-v2.npy",
    "t5": "embeddings_sentence-transformers_sentence-t5-base.npy"
}


    kmeans_n_clusters = [10, 15, 20]

    for model_tag, filename in models.items():
        embedding_path = os.path.join(embed_dir, filename)

        for n in kmeans_n_clusters:
            cluster_and_save(
                embedding_path, review_path, model_tag,
                n_clusters=n,
                output_dir=output_dir
            )
