# 임베딩 성능이 안나와서 그런가 싶어서 
# 새로운 임베딩 시도 
# fㅣ뷰 데이터에 적합한 all-mpnet-base-v2 ,paraphrase-MiniLM-L12-v2,sentence-t5-base

import pandas as pd
import numpy as np
import os
from sentence_transformers import SentenceTransformer

def embed_reviews(input_path, output_dir, model_name):
    print(f"📥 데이터 불러오는 중: {input_path}")
    df = pd.read_csv(input_path)

    if 'reviewText' not in df.columns:
        raise ValueError("리뷰 텍스트 컬럼('reviewText')이 필요합니다.")

    print(f"📦 모델 로딩 중: {model_name}")
    model = SentenceTransformer(model_name)

    print("🔢 리뷰 임베딩 중...")
    embeddings = model.encode(
        df['reviewText'].tolist(),
        batch_size=32,
        show_progress_bar=True
    )

    model_tag = model_name.replace("/", "_")
    output_path = os.path.join(output_dir, f"embeddings_{model_tag}.npy")

    os.makedirs(output_dir, exist_ok=True)
    np.save(output_path, embeddings)

    print(f"✅ 저장 완료: {output_path}")
    print(f"🧾 총 {len(embeddings)}개 벡터")

if __name__ == "__main__":
    # 경로 설정
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(project_root, "amazon", "amazon_reviews_pre.csv")
    output_dir = os.path.join(project_root, "embed/result2")

    # ✅ 리뷰에 적합한 모델 리스트
    model_names = [
        "sentence-transformers/all-MiniLM-L6-v2",         
        "sentence-transformers/paraphrase-MiniLM-L12-v2", 
        "sentence-transformers/all-mpnet-base-v2",        
        "sentence-transformers/sentence-t5-base"          
    ]

    for model_name in model_names:
        print("\n🚀 시작:", model_name)
        embed_reviews(input_file, output_dir, model_name)

