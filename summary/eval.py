import os
import glob
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from keybert import KeyBERT
from tqdm import tqdm

# 전역 설정
INPUT_DIR = '../cluster/results/cluster'
SUMMARY_DIRS = {
    't5': 'T5-small/summary_1_T5_center',
    'centroid': 'Centroid/summary_1_Centroid',
    'keybert': 'KeyBERT/summary_1_KeyBERT'
}
SUMMARY_TYPES = ['t5', 'centroid', 'keybert']
MODEL_NAME = 'paraphrase-MiniLM-L6-v2'
OUTPUT_FILE = 'summary_evaluation_results.csv'
STATS_FILE = 'summary_evaluation_stats.csv'
TOP_K_KEYWORDS = 10

# 모델 로드
model = SentenceTransformer(MODEL_NAME)
sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
kw_model = KeyBERT(model)

# ------------------- 지표 계산 -------------------

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
    sentences = [s.strip() for s in summary.split('.') if s.strip()]
    if len(sentences) < 2:
        return 0.0
    embeddings = model.encode(sentences, convert_to_tensor=True)
    sim_matrix = util.cos_sim(embeddings, embeddings).cpu().numpy()
    np.fill_diagonal(sim_matrix, np.nan)
    return np.nanmean(sim_matrix)

def compute_kcs(summary, reviews):
    if not summary or not reviews:
        return np.nan
    doc = " ".join(reviews)
    keywords = kw_model.extract_keywords(doc, top_n=TOP_K_KEYWORDS, stop_words='english')
    keywords = [kw[0].lower() for kw in keywords]
    count = sum(1 for kw in keywords if kw in summary.lower())
    return count / len(keywords) if keywords else 0.0

def compute_srs(summary, reviews):
    if not summary or not reviews:
        return np.nan
    emb_reviews = model.encode(reviews)
    cluster_mean = np.mean(emb_reviews, axis=0)
    emb_summary = model.encode(summary)
    return cosine_similarity([emb_summary], [cluster_mean])[0][0]

def compute_sentiment(text):
    if not text.strip():
        return np.nan
    result = sentiment_analyzer(text[:512])[0]
    score = result['score']
    return score if result['label'] == 'POSITIVE' else 1 - score

def compute_scs(summary, reviews):
    if not summary or not reviews:
        return np.nan
    review_scores = [compute_sentiment(r) for r in reviews if r.strip()]
    if not review_scores:
        return np.nan
    summary_score = compute_sentiment(summary)
    return 1 - abs(np.mean(review_scores) - summary_score)

# ------------------- 데이터 로딩 -------------------

def load_reviews_by_cluster(csv_file):
    df = pd.read_csv(csv_file)
    cluster_dict = df.groupby('cluster_id')['reviewText'].apply(list).to_dict()
    return cluster_dict

def load_summary_file(path, summary_type):
    df = pd.read_csv(path)
    if summary_type == 'keybert':
        df.rename(columns={'summary_keywords': 'summary'}, inplace=True)
    return df.set_index('cluster_id')['summary'].to_dict()

# ------------------- 평가 루틴 -------------------

def evaluate_model(summary_path, summary_type, cluster_reviews):
    summary_dict = load_summary_file(summary_path, summary_type)
    results = []
    desc_text = f"Evaluating {summary_type} summaries: {os.path.basename(summary_path)}"
    for cluster_id, summary in tqdm(summary_dict.items(), desc=desc_text, leave=False):
        reviews = cluster_reviews.get(cluster_id, [])
        result = {
            'cluster_id': cluster_id,
            f'uswir_{summary_type}': compute_uswir(summary, reviews),
            f'srnr_{summary_type}': compute_srnr(summary),
            f'kcs_{summary_type}': compute_kcs(summary, reviews),
            f'srs_{summary_type}': compute_srs(summary, reviews),
            f'scs_{summary_type}': compute_scs(summary, reviews)
        }
        results.append(result)
    return pd.DataFrame(results)

# ------------------- 메인 루프 -------------------
def main():
    all_results = []

    suffix_map = {
        't5': 'T5',
        'centroid': 'Centroid',
        'keybert': 'keyBERT'
    }

    for input_file in tqdm(glob.glob(os.path.join(INPUT_DIR, '*.csv')), desc='Processing input files'):
        base_filename = os.path.splitext(os.path.basename(input_file))[0]
        cluster_reviews = load_reviews_by_cluster(input_file)

        model_results = []
        for summary_type in SUMMARY_TYPES:
            summary_dir = SUMMARY_DIRS[summary_type]
            summary_suffix = suffix_map[summary_type]
            summary_file = os.path.join(summary_dir, f"{base_filename}_{summary_suffix}.csv")

            if not os.path.exists(summary_file):
                print(f"⚠️ Not found: {summary_file}")
                continue
            print(f"✅ Found: {summary_file}")

            df_model = evaluate_model(summary_file, summary_type, cluster_reviews)
            df_model['file_id'] = base_filename
            model_results.append(df_model)

        if model_results:
            merged = model_results[0]
            for df in model_results[1:]:
                merged = pd.merge(merged, df, on=['cluster_id', 'file_id'], how='outer')
            all_results.append(merged)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"✅ Evaluation complete. Results saved to {OUTPUT_FILE}")

        # 통계 저장
        melted = final_df.melt(id_vars=['cluster_id', 'file_id'], var_name='metric', value_name='value')
        melted[['metric_name', 'summary_type']] = melted['metric'].str.extract(r'(\w+)_([\w\d]+)')
        stats_df = (
            melted.groupby(['file_id', 'summary_type', 'metric_name'])['value']
            .agg(['mean', 'std'])
            .reset_index()
            .pivot(index=['file_id', 'summary_type'], columns='metric_name')
        )
        stats_df.columns = [f'{stat}_{metric}' for metric, stat in stats_df.columns]
        stats_df.reset_index(inplace=True)
        stats_df.to_csv(STATS_FILE, index=False)
        print(f"📊 Statistical summary saved to {STATS_FILE}")
    else:
        print("❌ No evaluation results found.")

if __name__ == '__main__':
    main()
