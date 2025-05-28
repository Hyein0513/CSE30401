import os
import glob
import logging
import pandas as pd
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# 콘솔 로그 최소화
logging.getLogger("transformers").setLevel(logging.ERROR)

# === 설정 ===
INPUT_DIR = '../cluster/cluster_fin'
SUMMARY_DIRS = {
    't5': 'T5-small/summary_T5_center',
    'centroid': 'Centroid/summary_Centroid',
    'keybert': 'KeyBERT/summary_KeyBERT',
}
SUMMARY_TYPES = ['t5', 'centroid', 'keybert']
STATS_FILE = 'summary_swir_stats.csv'

# === SE-tiny 모델 로드 ===
model_se_tiny = SentenceTransformer('all-MiniLM-L6-v2')

def compute_se_tiny_score(source, summary):
    try:
        embeddings = model_se_tiny.encode([source, summary], convert_to_tensor=True)
        sim = cosine_similarity([embeddings[0].cpu().numpy()], [embeddings[1].cpu().numpy()])[0][0]
        return sim
    except:
        return np.nan

# === 데이터 로딩 함수 ===
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

# === 한 파일 평가 ===
def evaluate_one_file(summary_path, s_type, source_texts, file_id):
    rows = []
    s_dict = load_summary_dict(summary_path, s_type)

    for cid, summ in tqdm(s_dict.items(), desc=f'{os.path.basename(summary_path)} [{s_type}]', leave=False):
        source = source_texts.get(cid, "")
        if not source or not summ or not isinstance(source, str) or not isinstance(summ, str):
            rows.append({
                'cluster_id': cid,
                'file_id': file_id,
                f'se_tiny_sim_{s_type}': np.nan,
            })
            continue
        se_sim = compute_se_tiny_score(source, summ)
        rows.append({
            'cluster_id': cid,
            'file_id': file_id,
            f'se_tiny_sim_{s_type}': se_sim,
        })
    return pd.DataFrame(rows)

# === 메인 평가 루프 ===
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

        # ▶ 평균 점수만 CSV로 저장
        avg = final.groupby('file_id').mean(numeric_only=True).reset_index()
        avg = avg.sort_values('file_id')
        avg.to_csv(STATS_FILE, index=False)

        print(f'✅ 평균 점수 파일 저장 완료: {STATS_FILE}')
    else:
        print('❌ 평가 결과 없음.')

if __name__ == '__main__':
    main()
