import pandas as pd
import re
import os

def preprocess_amazon_reviews_extended(input_path, output_path):
    print("📥 1. 데이터 로딩 중...")
    df = pd.read_csv(input_path)

    # 2. 사용할 컬럼만 필터링 (존재하는 컬럼만 유지)
    expected_cols = ['reviewText', 'overall', 'reviewTime', 'reviewerName', 'helpful_yes', 'helpful_no', 'total_vote']
    existing_cols = [col for col in expected_cols if col in df.columns]
    df = df[existing_cols]
    print(f"✅ 포함된 컬럼: {existing_cols}")

    # 3. reviewText 비어있는 행 제거
    df = df[df['reviewText'].notna() & (df['reviewText'].str.strip() != '')]
    print(f"🧾 유효한 리뷰 수: {len(df)}")

    # 4. reviewText 소문자화
    df['reviewText'] = df['reviewText'].str.lower()

    # 5. HTML 엔티티 제거
    df['reviewText'] = df['reviewText'].str.replace(r"&[a-z]+;", " ", regex=True)

    # 6. 특수문자 정리 (알파벳, 숫자, .,!?’ 만 유지)
    df['reviewText'] = df['reviewText'].apply(lambda x: re.sub(r"[^a-z0-9\s.,!?']", ' ', x))

    # 7. 공백 정리
    df['reviewText'] = df['reviewText'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # 8. 저장
    df.to_csv(output_path, index=False)
    print(f"✅ 전처리 완료! 저장 위치: {output_path}")
    print(f"🧾 최종 리뷰 수: {len(df)}개")

# 실행 예시
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    input_file = os.path.join(current_dir, "amazon_reviews.csv")
    output_file = os.path.join(current_dir, "amazon_reviews_pre.csv")
    preprocess_amazon_reviews_extended(input_file, output_file)
