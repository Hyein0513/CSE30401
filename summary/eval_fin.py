# ../cluster/results/cluster/에 있는 클러스터링된 리뷰 CSV들을 순회
# 각 클러스터에 대해 T5, Centroid, KeyBERT 요약 결과와 비교하여 평가
# 5가지 평가 지표: uswir, srnr, buysumm_fixed, buysumm_ref, ivd
# 10개 파일 단위로 평가 결과 CSV 중간 저장:
# 클러스터 단위 평가 결과 → summary_evaluation_results_10.csv, ...
# 파일 단위 평균 결과 → summary_evaluation_stats_10.csv, ...
# 전체 완료 시 최종 결과 2개 저장:
# summary_evaluation_results.csv, summary_evaluation_stats.csv



import os
import glob
import pandas as pd
import numpy as np
import re
import spacy
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from keybert import KeyBERT
from tqdm import tqdm

# === 설정 ===
INPUT_DIR = '../cluster/results2/cluster'
SUMMARY_DIRS = {
    't5': 'T5-small/summary_2_T5_center',
    'centroid': 'Centroid/summary_2_Centroid',
    'keybert': 'KeyBERT/summary_2_KeyBERT'
}
SUMMARY_TYPES = ['t5', 'centroid', 'keybert']
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
TOP_K_KEYWORDS = 10

OUTPUT_FILE = 'summary2_evaluation_results.csv'
STATS_FILE = 'summary2_evaluation_stats.csv'

DECISIONAL_ATTRIBUTES = [
    'battery', 'screen', 'camera', 'weight', 'performance',
    'price', 'design', 'build', 'sound', 'storage'
]
OPINION_LIST = [
    'good', 'bad', 'great', 'poor', 'excellent', 'weak', 'strong',
    'heavy', 'light', 'slow', 'fast', 'sharp', 'blurry', 'clear'
]

# === 모델 로드 ===
model = SentenceTransformer(MODEL_NAME)
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
kw_model = KeyBERT(model)
nlp = spacy.load("en_core_web_sm")

# === 평가 지표 ===
def compute_uswir(summary, reviews):
    if not summary or not reviews:
        return np.nan
    emb_summary = model.encode(summary, convert_to_tensor=True)
    emb_reviews = model.encode(reviews, convert_to_tensor=True)
    sims = util.cos_sim(emb_summary, emb_reviews)
    return sims.mean().item()

def compute_srnr(summary):
    if not summary:
        return np.nan
    sentences = [s.strip() for s in re.split(r'[.!?]', summary) if s.strip()]
    if len(sentences) < 2:
        return 0.0
    embeddings = model.encode(sentences, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
    np.fill_diagonal(sim_matrix, np.nan)
    return np.nanmean(sim_matrix)

def compute_buysumm_fixed(summary, attribute_list=None):
    if not summary or not summary.strip():
        return np.nan
    if attribute_list is None:
        attribute_list = DECISIONAL_ATTRIBUTES
    included = sum(attr in summary.lower() for attr in attribute_list)
    return included / len(attribute_list) if attribute_list else 0.0

def compute_buysumm_ref(summary, ref_keywords):
    if not summary or not summary.strip() or not ref_keywords:
        return np.nan
    summary = summary.lower()
    matched = sum(1 for kw in ref_keywords if kw in summary)
    return matched / len(ref_keywords) if ref_keywords else 0.0

def extract_aspect_opinion_units(text):
    doc = nlp(text)
    units = []
    for token in doc:
        if token.dep_ in ("amod", "acomp") and token.head.pos_ == "NOUN":
            units.append((token.head.text, token.text))
        elif token.dep_ == "attr" and token.pos_ == "ADJ":
            for child in token.head.children:
                if child.dep_ == "nsubj":
                    units.append((child.text, token.text))
    return units

def compute_ivd(summary):
    if not summary or not summary.strip():
        return np.nan
    sentences = [s.strip() for s in re.split(r'[.!?]', summary) if s.strip()]
    if len(sentences) == 0:
        return np.nan
    units = extract_aspect_opinion_units(summary)
    return len(units) / len(sentences)

# === 데이터 로딩 ===
def load_reviews_by_cluster(csv_file):
    df = pd.read_csv(csv_file)
    return df.groupby('cluster_id')['reviewText'].apply(list).to_dict()

def load_summary_file(path, summary_type):
    df = pd.read_csv(path)
    if summary_type == 'keybert':
        for col in df.columns:
            if 'summary' in col.lower() or 'keyword' in col.lower():
                df.rename(columns={col: 'summary'}, inplace=True)
                break
    return df.set_index('cluster_id')['summary'].to_dict()

# === 평가 함수 ===
def evaluate_model(summary_path, summary_type, cluster_reviews, keybert_keywords=None):
    summary_dict = load_summary_file(summary_path, summary_type)
    results = []
    for cluster_id, summary in summary_dict.items():
        reviews = cluster_reviews.get(cluster_id, [])
        ref_keywords = keybert_keywords.get(cluster_id, []) if keybert_keywords else []
        result = {
            'cluster_id': cluster_id,
            f'uswir_{summary_type}': compute_uswir(summary, reviews),
            f'srnr_{summary_type}': compute_srnr(summary),
            f'buysumm_fixed_{summary_type}': compute_buysumm_fixed(summary),
            f'buysumm_ref_{summary_type}': compute_buysumm_ref(summary, ref_keywords) if summary_type != 'keybert' else np.nan,
            f'ivd_{summary_type}': compute_ivd(summary)
        }
        results.append(result)
    return pd.DataFrame(results)

# === 메인 루프 ===
def main():
    all_results = []
    file_counter = 0
    suffix_map = {
        't5': 'T5',
        'centroid': 'Centroid',
        'keybert': 'keyBERT'
    }

    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, '*.csv')))

    for input_file in tqdm(input_files, desc='Processing input files'):
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        cluster_reviews = load_reviews_by_cluster(input_file)

        keybert_path = os.path.join(SUMMARY_DIRS['keybert'], f"{base_filename}_keyBERT.csv")
        keybert_dict = load_summary_file(keybert_path, 'keybert')
        keybert_keywords = {
            cid: [w.lower() for w in re.findall(r'\b\w+\b', summary)]
            for cid, summary in keybert_dict.items()
        }

        model_results = []
        for summary_type in SUMMARY_TYPES:
            summary_dir = SUMMARY_DIRS[summary_type]
            summary_suffix = suffix_map[summary_type]
            summary_file = os.path.join(summary_dir, f"{base_filename}_{summary_suffix}.csv")

            if not os.path.exists(summary_file):
                print(f"⚠️ Not found: {summary_file}")
                continue
            print(f"✅ Found: {summary_file}")

            df_model = evaluate_model(summary_file, summary_type, cluster_reviews, keybert_keywords)
            df_model['file_id'] = base_filename
            model_results.append(df_model)

        if model_results:
            merged = model_results[0]
            for df in model_results[1:]:
                merged = pd.merge(merged, df, on=['cluster_id', 'file_id'], how='outer')
            all_results.append(merged)
            file_counter += 1

        if file_counter > 0 and file_counter % 10 == 0:
            interim_df = pd.concat(all_results, ignore_index=True)
            interim_df.to_csv(f'summary_evaluation_results_{file_counter}.csv', index=False)
            stats_df = interim_df.groupby('file_id').mean(numeric_only=True).reset_index()
            stats_df.to_csv(f'summary_evaluation_stats_{file_counter}.csv', index=False)
            print(f"📝 Interim results saved at {file_counter} files")

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        stats_df = final_df.groupby('file_id').mean(numeric_only=True).reset_index()
        stats_df.to_csv(STATS_FILE, index=False)
        print("✅ Final evaluation results saved.")
    else:
        print("❌ No evaluation results found.")

if __name__ == '__main__':
    main()
