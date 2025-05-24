import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.preprocessing import normalize

def visualize_clusters(csv_path, embedding_path, output_image_path, n_neighbors=50, random_state=42):
    df = pd.read_csv(csv_path)
    embeddings = np.load(embedding_path)

    if len(df) != len(embeddings):
        print(f"❗ 크기 불일치: {csv_path}")
        return

    embeddings = normalize(embeddings)

    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=random_state)
    umap_embeddings = reducer.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))
    unique_labels = sorted(df['cluster_id'].unique())
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))

    for label, color in zip(unique_labels, colors):
        indices = df['cluster_id'] == label
        label_name = f"Cluster {label}" if label != -1 else "Noise"
        plt.scatter(umap_embeddings[indices, 0],
                    umap_embeddings[indices, 1],
                    c=[color], label=label_name, alpha=0.6, s=10)

    plt.title(os.path.basename(csv_path))
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    plt.savefig(output_image_path, dpi=300)
    plt.close()
    print(f"✅ 저장 완료: {output_image_path}")

if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    embed_dir = os.path.join(project_root, "embed/results2")
    result_dir = os.path.join(project_root, "results2")
    vis_dir = os.path.join(project_root, "cluster/results2/vis")

    
    # base_dir = os.path.dirname(os.path.abspath(__file__))  # cluster/
    # result_dir = os.path.join(base_dir, "results")
    # embed_dir = os.path.join(base_dir, "..", "embed")
    # vis_dir = os.path.join(base_dir, "vis_result")

    # model_map = {
    #     "minilm": "embeddings_sentence-transformers_all-MiniLM-L6-v2.npy",
    #     "bge": "embeddings_BAAI_bge-small-en.npy",
    #     "e5": "embeddings_intfloat_e5-small-v2.npy"
    # }

    model_map = {
        "minilm": "embeddings_sentence-transformers_all-MiniLM-L6-v2.npy",
        "mpnet": "embeddings_sentence-transformers_all-mpnet-base-v2.npy",
        "paraphrase": "embeddings_sentence-transformers_paraphrase-MiniLM-L12-v2.npy",
        "t5": "embeddings_sentence-transformers_sentence-t5-base.npy"
    }


    for fname in os.listdir(result_dir):
        if not fname.endswith(".csv") or not fname.startswith("cluster_"):
            continue

        model_key = None
        for key in model_map:
            if f"_{key}_" in fname:
                model_key = key
                break

        if model_key is None:
            print(f"❗ 모델 구분 실패: {fname}")
            continue

        csv_path = os.path.join(result_dir, fname)
        embedding_path = os.path.join(embed_dir, model_map[model_key])
        output_path = os.path.join(vis_dir, fname.replace(".csv", ".png"))

        visualize_clusters(csv_path, embedding_path, output_path)
