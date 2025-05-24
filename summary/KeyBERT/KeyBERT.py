import os
import pandas as pd
from keybert import KeyBERT
from collections import defaultdict

# ✅ 현재 스크립트 위치 기준으로 경로 설정
script_dir = os.path.dirname(os.path.abspath(__file__))
input_dir = os.path.abspath(os.path.join(script_dir, "../../cluster/results/cluster"))
output_dir = os.path.join(script_dir, "summary_1_KeyBERT")
os.makedirs(output_dir, exist_ok=True)

# ✅ KeyBERT 모델 로딩
print("🚀 KeyBERT 모델 로딩 중...")
kw_model = KeyBERT()

# ✅ 입력 폴더 내 모든 csv 파일 처리
for filename in os.listdir(input_dir):
    if not filename.endswith(".csv"):
        continue

    input_path = os.path.join(input_dir, filename)
    print(f"\n📂 처리 중: {filename}")
    df = pd.read_csv(input_path)

    # ✅ 클러스터별 리뷰 모으기
    clusters = defaultdict(list)
    for _, row in df.iterrows():
        cid = row["cluster_id"]
        if cid != -1:
            clusters[cid].append(str(row["reviewText"]))

    # ✅ 클러스터별 키워드 요약 생성
    summary_rows = []
    for i, (cid, reviews) in enumerate(clusters.items(), start=1):
        print(f"  🧠 클러스터 {cid} ({len(reviews)}개 리뷰) 요약 중... [{i}/{len(clusters)}]")
        full_text = " ".join(reviews)
        keywords = kw_model.extract_keywords(
            full_text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=5
        )
        keyword_list = [kw[0] for kw in keywords]
        summary_rows.append({
            "cluster_id": cid,
            "summary_keywords": ", ".join(keyword_list)
        })

    # ✅ 출력 파일 저장
    output_filename = filename.replace(".csv", "_keyBERT.csv")
    output_path = os.path.join(output_dir, output_filename)
    pd.DataFrame(summary_rows).to_csv(output_path, index=False)
    print(f"✅ 저장 완료: {output_path}")
