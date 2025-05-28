import pandas as pd
from transformers import T5Tokenizer

# ✅ 파일 경로 설정
csv_path = "../../amazon/amazon_reviews_pre.csv"  # 여기에 분석할 CSV 파일 경로 입력
output_csv = "over_token_limit_reviews.csv"

# ✅ tokenizer 로드
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# ✅ 데이터 로드
df = pd.read_csv(csv_path)

# ✅ 컬럼 확인
if 'reviewText' not in df.columns:
    raise ValueError("❌ 'reviewText' 컬럼이 없습니다.")

# ✅ 분석 시작
max_input = 512
over_limit_reviews = []

for idx, review in df['reviewText'].dropna().items():
    input_text = "summarize: " + str(review).strip().replace("\n", " ")
    tokens = tokenizer.encode(input_text, add_special_tokens=True)
    token_length = len(tokens)

    if token_length > max_input:
        over_limit_reviews.append({
            "index": idx,
            "token_length": token_length,
            "reviewText": review
        })

# ✅ 결과 요약
num_skipped = len(over_limit_reviews)
print(f"\n⚠️ 스킵될 리뷰 수: {num_skipped}개\n")

# ✅ 스킵된 리뷰 정보 출력
for item in over_limit_reviews:
    print(f"[index: {item['index']}] 토큰 수: {item['token_length']}개")
    print(f"리뷰 미리보기: {item['reviewText'][:100]}...\n")

# ✅ CSV 저장
if num_skipped > 0:
    result_df = pd.DataFrame(over_limit_reviews)
    result_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"📁 저장 완료 → {output_csv}")
else:
    print("✅ 스킵될 리뷰는 없습니다.")