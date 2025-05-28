import os, glob, logging
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jensenshannon

# 콘솔 로그 최소화
logging.getLogger("transformers").setLevel(logging.ERROR)

# === 설정 ===
INPUT_DIR = '../cluster/results2/cluster'
SUMMARY_DIRS = {
    't5': 'T5-small/summary_2_T5_center',
    'centroid': 'Centroid/summary_2_Centroid',
    'keybert': 'KeyBERT/summary_2_KeyBERT',
}
SUMMARY_TYPES = ['t5', 'centroid', 'keybert']
OUTPUT_FILE = 'summary2_nonref_eval.csv'
STATS_FILE = 'summary2_nonref_stats.csv'

# === 평가 함수 ===
def compute_tfidf_cosine(source, summary):
    vectorizer = TfidfVectorizer(stop_words='english')
    try:
        tfidf = vectorizer.fit_transform([source, summary])
        return cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    except:
        return np.nan

def compute_js_divergence(source, summary):
    def get_distribution(text):
        tokens = text.lower().split()
        counts = Counter(tokens)
        total = sum(counts.values())
        vocab = set(counts.keys())
        return counts, total, vocab

    c1, t1, v1 = get_distribution(source)
    c2, t2, v2 = get_distribution(summary)
    vocab = list(v1.union(v2))

    p = np.array([c1.get(word, 0)/t1 for word in vocab])
    q = np.array([c2.get(word, 0)/t2 for word in vocab])
    try:
        return jensenshannon(p, q)
    except:
        return np.nan

# === 평가 실행 ===
def load_summary_dict(path, s_type):
    df = pd.read_csv(path)
    if s_type == 'keybert':
        for col in df.columns:
            if 'summary' in col.lower() or 'keyword' in col.lower():
                df = df.rename(columns={col: 'summary'})
                break
    return df.set_index('cluster_id')['summary'].to_dict()

def load_source_text_dict(path):
    df = pd.read_csv(path)
    return df.set_index('cluster_id')['reviewText'].to_dict()

def evaluate_one_file(summary_path, s_type, source_texts, file_id):
    rows = []
    s_dict = load_summary_dict(summary_path, s_type)

    for cid, summ in tqdm(s_dict.items(), desc=f'{os.path.basename(summary_path)} [{s_type}]', leave=False):
        source = source_texts.get(cid, "")
        if not source or not summ or not isinstance(source, str) or not isinstance(summ, str):
            rows.append({
                'cluster_id': cid,
                'file_id': file_id,
                f'tfidf_sim_{s_type}': np.nan,
                f'js_div_{s_type}': np.nan,
            })
            continue
        tfidf_sim = compute_tfidf_cosine(source, summ)
        js_div = compute_js_divergence(source, summ)
        rows.append({
            'cluster_id': cid,
            'file_id': file_id,
            f'tfidf_sim_{s_type}': tfidf_sim,
            f'js_div_{s_type}': js_div,
        })
    return pd.DataFrame(rows)

def main():
    all_results = []
    suf = {'t5': 'T5', 'centroid': 'Centroid', 'keybert': 'keyBERT'}

    for in_file in tqdm(sorted(glob.glob(os.path.join(INPUT_DIR, '*.csv'))), desc='Input CSV', unit='file'):
        base = os.path.splitext(os.path.basename(in_file))[0]
        source_texts = load_source_text_dict(in_file)
        dfs = []

        for t in SUMMARY_TYPES:
            s_file = os.path.join(SUMMARY_DIRS[t], f'{base}_{suf[t]}.csv')
            if os.path.exists(s_file):
                try:
                    df = evaluate_one_file(s_file, t, source_texts, base)
                    dfs.append(df)
                except Exception as e:
                    print(f'⚠️ Error processing {s_file}: {e}')

        if dfs:
            merged = dfs[0]
            for d in dfs[1:]:
                merged = pd.merge(merged, d, on=['cluster_id', 'file_id'], how='outer')
            all_results.append(merged)

    if all_results:
        final = pd.concat(all_results, ignore_index=True)

        # ▶ 파일별 평균값 통계 테이블 생성
        avg = final.groupby('file_id').mean(numeric_only=True).reset_index()
        avg = avg.sort_values('file_id')
        avg.to_csv(STATS_FILE, index=False)

        print(f'📊 Saved summary statistics to: {STATS_FILE}')
    else:
        print('❌ No results.')

if __name__ == '__main__':
    main()
