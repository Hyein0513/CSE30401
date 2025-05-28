# 클러스터별로 요약된걸 다 모아서 5000개의 리뷰를 하나의 요약으로 모아내는 코드 

import os
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util

# === 기본 설정 ===
BASE_PATH = './'
SUMMARY_MODELS = {
    'T5-small': {'type': 't5'},
    'KeyBERT': {'type': 'keybert'},
    'Centroid': {'type': 'centroid'}
}

# === 모델 초기화 ===
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
tokenizer = AutoTokenizer.from_pretrained("t5-small")
t5_model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
summarizer = pipeline("summarization", model=t5_model, tokenizer=tokenizer)

# === 재귀 요약 (T5 전용) ===
def summarize_recursive(texts, max_tokens=512):
    input_text = " ".join(texts)
    token_count = len(tokenizer.encode(input_text))
    if token_count <= max_tokens:
        return summarizer(input_text, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
    else:
        chunks, chunk = [], []
        for t in texts:
            chunk.append(t)
            if len(tokenizer.encode(" ".join(chunk))) > max_tokens:
                chunk.pop()
                chunks.append(" ".join(chunk))
                chunk = [t]
        if chunk:
            chunks.append(" ".join(chunk))
        summaries = [summarize_recursive([c]) for c in chunks]
        return summarize_recursive(summaries)

# === T5 방식 요약 ===
def generate_t5_summary(file_path):
    df = pd.read_csv(file_path)
    summaries = df['summary'].dropna().tolist()
    if not summaries:
        return ""
    embeddings = embedding_model.encode(summaries, convert_to_tensor=True)
    centroid = embeddings.mean(dim=0)
    similarities = util.pytorch_cos_sim(centroid, embeddings)[0]
    top_indices = similarities.topk(min(30, len(summaries))).indices
    top_summaries = [summaries[i] for i in top_indices]
    return summarize_recursive(top_summaries)

# === KeyBERT 방식 요약: 요약된 문장들을 그대로 연결 ===
def generate_keybert_summary(file_path):
    df = pd.read_csv(file_path)
    # 가능한 요약 컬럼 후보들
    candidate_columns = [col for col in df.columns if any(
        keyword in col.lower() for keyword in ['summary', 'represent', 'keyword'])]
    
    if not candidate_columns:
        raise ValueError(f"No valid summary-like column found in {file_path}")
    
    # 첫 번째 유효한 컬럼 사용
    summaries = df[candidate_columns[0]].dropna().tolist()
    return " ".join(summaries)


# === Centroid 방식 요약: 중심에 가장 가까운 문장 1개 선택 ===
def generate_centroid_summary(file_path):
    df = pd.read_csv(file_path)
    summaries = df['summary'].dropna().tolist()
    if not summaries:
        return ""
    embeddings = embedding_model.encode(summaries, convert_to_tensor=True)
    centroid = embeddings.mean(dim=0)
    similarities = util.pytorch_cos_sim(centroid, embeddings)[0]
    best_idx = similarities.argmax().item()
    return summaries[best_idx]

# === 모델별 폴더 탐색 및 요약 처리 ===
def process_model(model_name, model_info):
    for summary_level in ['summary_1_' + model_name, 'summary_2_' + model_name]:
        model_dir = os.path.join(BASE_PATH, model_name, summary_level)
        if not os.path.isdir(model_dir):
            continue

        result = []
        for fname in tqdm(sorted(os.listdir(model_dir))):
            if not fname.endswith('.csv'):
                continue
            path = os.path.join(model_dir, fname)
            try:
                if model_info['type'] == 't5':
                    summary = generate_t5_summary(path)
                elif model_info['type'] == 'keybert':
                    summary = generate_keybert_summary(path)
                elif model_info['type'] == 'centroid':
                    summary = generate_centroid_summary(path)
                else:
                    summary = "Unsupported model"
                result.append({'file_name': fname, 'overall_summary': summary})
            except Exception as e:
                print(f"❌ Failed to process {fname}: {e}")

        df_summary = pd.DataFrame(result)
        output_path = os.path.join(model_dir, f'summary_overall_{model_name}.csv')
        df_summary.to_csv(output_path, index=False)
        print(f"✅ Saved: {output_path}")

# === 메인 실행 ===
if __name__ == '__main__':
    for model_name, model_info in SUMMARY_MODELS.items():
        process_model(model_name, model_info)
