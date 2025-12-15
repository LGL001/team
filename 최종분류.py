import pandas as pd
import joblib
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. 데이터 로드
df = pd.read_csv("dataset_v2.csv")

# 2. 라벨링 (IT=1, Non-IT=0)
df['label'] = df['category'].apply(lambda x: 1 if x == 'IT_Engineering' else 0)
X = df['text']
y = df['label']

print(f"📊 데이터 구성: IT({sum(y==1)}개) vs 비IT({sum(y==0)}개)")

# --- [전처리 함수] ---
# 불용어(Stopwords)는 글자 단위 학습에서는 큰 의미가 없어서 제거 로직을 단순화합니다.
def custom_preprocessor(text):
    """최소한의 노이즈만 제거"""
    # 1. 숫자 제거 (연도 등 노이즈 방지)
    text = re.sub(r'\d+', ' ', text)
    # 2. 특수문자 제거 (점, 쉼표 등)
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    # 3. 여러 공백을 하나로 줄임
    text = re.sub(r'\s+', ' ', text)
    return text

# --- [핵심 변경: 글자 단위 분석기] ---
# analyzer='char_wb': 단어 경계 안에서 글자 패턴을 찾음
# ngram_range=(2, 4): 2글자~4글자 덩어리를 학습 (예: '컴퓨', '퓨터', '프로그', '래망')
vectorizer = TfidfVectorizer(
    preprocessor=custom_preprocessor,
    analyzer='char_wb',
    ngram_range=(2, 4),
    min_df=1,            # 한 번이라도 나오면 무조건 학습
    max_features=10000   # 패턴을 넉넉하게 10000개까지 기억
)

print("⚙️ 텍스트를 글자 조각(Character N-grams)으로 변환 중...")
X_vec = vectorizer.fit_transform(X)

# 3. 모델 학습 (강력한 규제 적용)
model = LogisticRegression(class_weight='balanced', C=10.0, random_state=42, max_iter=2000)
model.fit(X_vec, y)

# 저장
joblib.dump(model, 'major_predictor_model.pkl')
joblib.dump(vectorizer, 'major_vectorizer.pkl')

print("-" * 30)
print("✅ 모델 재학습 완료! (오타가 있어도 문맥을 파악합니다)")

# --- [검증: 이제 오타도 인식하는지 확인] ---
# 이제는 단어가 아니라 '패턴'이 있는지 확인해야 합니다.
vocab = vectorizer.vocabulary_
test_patterns = ['파이', '이썬', 'Py', 'yt', 'th', 'ho', 'on', '그래', '래망', 'POT']

print("\n🔍 [글자 조각 인식 테스트]")
for pattern in test_patterns:
    if pattern in vocab:
        print(f"🆗 '{pattern}' -> 패턴 학습됨!")
    else:
        print(f"❌ '{pattern}' -> 없음")

# --- [어떤 패턴이 IT 점수를 올렸을까?] ---
coefficients = model.coef_[0]
feature_names = vectorizer.get_feature_names_out()
sorted_idx = coefficients.argsort()

print("\n🔑 [IT 합격 핵심 글자 패턴 TOP 20]")
# 글자 단위라 결과가 '컴퓨', '퓨터' 처럼 보일 겁니다. 이게 정상입니다!
top_keywords = [feature_names[i] for i in sorted_idx[-20:]]
print(top_keywords)