import os
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import numpy as np
from sentence_transformers import SentenceTransformer

# ===== T5 요약 함수 =====
def t5_summarize(text, model, tokenizer, max_input=512, max_output=75):
    input_text = "summarize: " + text.strip().replace("\n", " ")
    tokens = tokenizer.encode(input_text, add_special_tokens=True)

    if len(tokens) > max_input:
        print(f"\u26a0\ufe0f 입력 길이 {len(tokens)} > {max_input} → 스킵됨")
        return "SKIPPED_TOO_LONG"

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_input,
        padding="max_length"
    ).to(model.device)

    summary_ids = model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=max_output,
        min_length=20,
        num_beams=2,
        length_penalty=2.0,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ===== 긴 텍스트 청크 분할 =====
def split_text_into_chunks(reviews, tokenizer, token_limit=510):
    chunks = []
    current_chunk = []
    current_tokens = 0

    for review in reviews:
        tokens = tokenizer.encode(review, add_special_tokens=False)
        if len(tokens) > token_limit:
            continue

        i = 0
        while i < len(tokens):
            remaining = token_limit - current_tokens
            slice_tokens = tokens[i:i + remaining]
            if not slice_tokens:
                break

            chunk_piece = tokenizer.decode(slice_tokens, skip_special_tokens=True)
            current_chunk.append(chunk_piece)
            current_tokens += len(slice_tokens)
            i += remaining

            if current_tokens >= token_limit:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_tokens = 0

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# ===== 중심 기반 리뷰 선택 =====
def select_top_k_by_center(reviews, embeddings, k=30):
    center = np.mean(embeddings, axis=0, keepdims=True)
    sims = cosine_similarity(center, embeddings)[0]
    top_indices = sims.argsort()[::-1][:k]
    return [reviews[i] for i in top_indices]

# ===== 클러스터 요약 함수 =====
def summarize_cluster_by_center(reviews, model, tokenizer, embedder, cluster_id=None):
    try:
        embeddings = embedder.encode(reviews, convert_to_numpy=True, normalize_embeddings=True)

        if len(reviews) > 30:
            selected_reviews = select_top_k_by_center(reviews, embeddings, k=30)
        else:
            selected_reviews = reviews

        chunks = split_text_into_chunks(selected_reviews, tokenizer)
        if not chunks:
            return "SKIPPED_ALL_LONG", selected_reviews

        chunk_summaries = []
        for chunk in chunks:
            summary = t5_summarize(chunk, model, tokenizer)
            if summary == "SKIPPED_TOO_LONG":
                continue
            chunk_summaries.append(summary)

        combined = " ".join(chunk_summaries)

        tokens = tokenizer.encode(combined, add_special_tokens=True)
        if len(tokens) > 512:
            tokens = tokens[:512]
            combined = tokenizer.decode(tokens, skip_special_tokens=True)

        final_summary = t5_summarize(combined, model, tokenizer)
        return final_summary, selected_reviews

    except Exception as e:
        print(f"\u274c [cluster_id={cluster_id}] 전체 요약 실패: {e}")
        return "SUMMARY_FAILED", []

# ===== 전체 실행 파이프라인 =====
def run_pipeline():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.abspath(os.path.join(script_dir, "../../cluster/results/cluster"))
    output_dir = os.path.join(script_dir, "summary_1_T5_center")
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model = model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    input_files = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    total_clusters = 0
    for file_name in input_files:
        try:
            df = pd.read_csv(os.path.join(input_dir, file_name))
            if 'cluster_id' in df.columns:
                total_clusters += df['cluster_id'].nunique()
        except:
            continue

    with tqdm(total=total_clusters, desc="T5 중심 요약 진행 중") as pbar:
        for file_name in input_files:
            input_path = os.path.join(input_dir, file_name)
            try:
                df = pd.read_csv(input_path)
                if 'reviewText' not in df.columns or 'cluster_id' not in df.columns:
                    print(f"\u26a0\ufe0f 필수 컬럼 없음: {file_name}")
                    continue

                summary_rows = []
                grouped = df.groupby("cluster_id")

                for cluster_id, group in grouped:
                    reviews = group['reviewText'].dropna().tolist()
                    if not reviews:
                        pbar.update(1)
                        continue

                    summary, used_reviews = summarize_cluster_by_center(
                        reviews, model, tokenizer, embedder, cluster_id=cluster_id
                    )

                    summary_rows.append({
                        "cluster_id": cluster_id,
                        "summary": summary,
                        "original_reviews": used_reviews  # ✅ 실제 사용된 리뷰만 저장
                    })
                    pbar.update(1)

                out_df = pd.DataFrame(summary_rows)
                out_file = os.path.join(output_dir, file_name.replace(".csv", "_T5.csv"))
                out_df.to_csv(out_file, index=False, encoding="utf-8-sig")

            except Exception as e:
                print(f"\u274c 파일 처리 실패: {file_name} → {e}")

# ===== 실행 =====
if __name__ == "__main__":
    run_pipeline()
