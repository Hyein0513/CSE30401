# -----------------------  summary_ivd_aste.py  ----------------------- #
import os, glob, warnings, contextlib, io, logging
import pandas as pd
import numpy as np
from tqdm import tqdm

import nltk
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.tokenize import sent_tokenize
from pyabsa import ASTECheckpointManager

# ▣ 콘솔 노이즈 최소화 ▣
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("pyabsa").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)

# === 설정 ===
INPUT_DIR = '../cluster/results/cluster'
SUMMARY_DIRS = {
    't5'      : 'T5-small/summary_1_T5_center',
    'centroid': 'Centroid/summary_1_Centroid',
    'keybert' : 'KeyBERT/summary_1_KeyBERT',
}
SUMMARY_TYPES = ['t5', 'centroid', 'keybert']
OUTPUT_FILE = 'summary1_ivd_results.csv'
STATS_FILE  = 'summary1_ivd_stats.csv'

# === ASTE 모델 (조용히) ===
triplet_extractor = ASTECheckpointManager.get_aspect_sentiment_triplet_extractor(
    checkpoint='english'
)

# ------------------------------------------------------------------ #
def extract_triplets(text: str):
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
             contextlib.redirect_stderr(io.StringIO()):
            preds = triplet_extractor.predict(
                [text],
                print_result=False,
                save_result=False,
                ignore_error=True
            )
        return preds[0].get('Triplets', []) if preds else []
    except Exception as e:
        raise RuntimeError(f"Triplet extraction failed: {e}")


def ivd_stats(summary: str, file_id: str, cluster_id: str, error_log: list):
    if not summary or not summary.strip():
        return np.nan, 0, 0

    sents = [s.strip() for s in sent_tokenize(summary) if s.strip()]
    sents = [s for s in sents if len(s.split()) >= 3 and len(s) <= 300 and any(c.isalnum() for c in s)]

    if not sents:
        return np.nan, 0, 0

    pairs, valid_sent_count = 0, 0
    for s in sents:
        try:
            triplets = extract_triplets(s)
            pairs += len(triplets)
            valid_sent_count += 1
        except Exception as e:
            error_log.append({
                'file_id': file_id,
                'cluster_id': cluster_id,
                'sentence': s,
                'error': str(e)
            })
    if valid_sent_count == 0:
        return np.nan, 0, 0
    return pairs / valid_sent_count, pairs, valid_sent_count

# ------------------------------------------------------------------ #

def load_summary_dict(path, s_type):
    df = pd.read_csv(path)
    if s_type == 'keybert':
        for col in df.columns:
            if 'summary' in col.lower() or 'keyword' in col.lower():
                df = df.rename(columns={col: 'summary'})
                break
    return df.set_index('cluster_id')['summary'].to_dict()

def evaluate_one_file(summary_path, s_type, file_id, error_log):
    rows = []
    s_dict = load_summary_dict(summary_path, s_type)

    for cid, summ in tqdm(
            s_dict.items(),
            desc=f'{os.path.basename(summary_path)} [{s_type}]',
            leave=False):
        ivd, pairs, sents = ivd_stats(summ, file_id, cid, error_log)
        rows.append({
            'cluster_id'          : cid,
            f'ivd_{s_type}'       : ivd,
            f'pairs_{s_type}'     : pairs,
            f'sentences_{s_type}' : sents,
        })
    return pd.DataFrame(rows)
def main():
    all_results = []
    error_log = []  # 🔧 에러 누적 리스트
    processed = 0
    suf = {'t5': 'T5', 'centroid': 'Centroid', 'keybert': 'keyBERT'}

    for in_file in tqdm(
            sorted(glob.glob(os.path.join(INPUT_DIR, '*.csv'))),
            desc='Input CSV', unit='file'):
        
        base = os.path.splitext(os.path.basename(in_file))[0]
        dfs = []

        for t in SUMMARY_TYPES:
            s_file = os.path.join(SUMMARY_DIRS[t], f'{base}_{suf[t]}.csv')
            if os.path.exists(s_file):
                try:
                    df = evaluate_one_file(s_file, t, base, error_log)
                    df['file_id'] = base
                    dfs.append(df)
                except Exception as e:
                    error_log.append({
                        'file_id': base,
                        'cluster_id': 'N/A',
                        'sentence': f'[FILE LEVEL ERROR] {s_file}',
                        'error': str(e)
                    })

        if dfs:
            merged = dfs[0]
            for d in dfs[1:]:
                merged = pd.merge(merged, d, on=['cluster_id', 'file_id'], how='outer')
            all_results.append(merged)
            processed += 1

        # 🔧 10개 단위로 결과 및 에러 로그 저장
        if processed and processed % 10 == 0:
            tmp = pd.concat(all_results, ignore_index=True)
            tmp.to_csv(f'summary1_ivd_results_{processed}.csv', index=False)
            avg = tmp.groupby('file_id').mean(numeric_only=True).reset_index()
            avg.to_csv(f'summary1_ivd_stats_{processed}.csv', index=False)
            pd.DataFrame(error_log).to_csv(f'summary1_ivd_errors_{processed}.csv', index=False)

    if all_results:
        final = pd.concat(all_results, ignore_index=True)
        final.to_csv(OUTPUT_FILE, index=False)
        avg = final.groupby('file_id').mean(numeric_only=True).reset_index()
        avg.to_csv(STATS_FILE, index=False)
        pd.DataFrame(error_log).to_csv('summary1_ivd_errors_final.csv', index=False)
        print('✅ Finished – results and errors saved.')
    else:
        print('❌ Nothing processed.')


if __name__ == '__main__':
    main()
# -------------------------------------------------------------------- #
