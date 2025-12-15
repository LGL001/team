import streamlit as st
import pandas as pd
import joblib
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------------------------------------------------------
# 1. 설정 및 전처리 함수 정의 (모델 학습때랑 똑같아야 함!)
# -----------------------------------------------------------------------------

# 불용어 처리는 글자 단위라 크게 중요하진 않지만 형태는 유지
def custom_preprocessor(text):
    text = re.sub(r'\d+', ' ', text)  # 숫자 제거
    text = re.sub(r'[^\w\s가-힣]', ' ', text)  # 특수문자 제거
    text = re.sub(r'\s+', ' ', text)  # 공백 정리
    return text


# Streamlit은 속도를 위해 캐싱(@st.cache_resource)을 사용합니다.
@st.cache_resource
def load_prediction_model():
    """학과 예측 모델 로드"""
    try:
        model = joblib.load('major_predictor_model.pkl')
        vec = joblib.load('major_vectorizer.pkl')
        return model, vec
    except Exception as e:
        return None, None


@st.cache_resource
def load_recommendation_engine():
    """추천 시스템 데이터 및 벡터 로드 (시간이 좀 걸리니 캐싱 필수)"""
    try:
        df = pd.read_csv("dataset_v2.csv")
        # IT 계열 데이터만 사용 (족보)
        it_df = df[df['category'] == 'IT_Engineering']

        # 문장 단위로 쪼개기
        sentences = []
        for text in it_df['text']:
            # 문장 분리 (. 또는 줄바꿈 기준)
            splits = re.split(r'[.|\n]', str(text))
            for s in splits:
                s = s.strip()
                if len(s) > 15:  # 너무 짧은 문장은 제외
                    sentences.append(s)

        # 추천용 벡터화 (글자 단위)
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4))
        tfidf_matrix = vectorizer.fit_transform(sentences)

        return sentences, vectorizer, tfidf_matrix
    except Exception as e:
        return [], None, None


# -----------------------------------------------------------------------------
# 2. UI 구성 (여기가 웹사이트 화면 만드는 곳)
# -----------------------------------------------------------------------------

st.set_page_config(page_title="상명대 AI 입시 컨설턴트", page_icon="🎓", layout="wide")

# 사이드바
st.sidebar.title("🎓 AI 입시 컨설턴트")
st.sidebar.info("상명대학교 실제 합격생 데이터를 기반으로 분석합니다.")
menu = st.sidebar.radio("메뉴 선택", ["홈(Home)", "IT 적합도 진단", "세특 문장 추천기"])

# 모델 로딩
pred_model, pred_vec = load_prediction_model()
rec_sentences, rec_vec, rec_matrix = load_recommendation_engine()

# --- [메뉴 1] 홈 ---
if menu == "홈(Home)":
    st.title("🎓 상명대 생기부 AI 분석 솔루션")
    st.markdown("""
    ### 환영합니다! 👋
    이 서비스는 **자연어 처리(NLP)** 기술을 활용하여 여러분의 생활기록부를 분석해줍니다.

    #### 🔍 주요 기능
    1. **IT 적합도 진단**: 내가 쓴 세특이 컴퓨터과학과/IT계열에 얼마나 적합한지 점수로 알려줍니다.
    2. **세특 문장 추천**: 내 활동 키워드를 입력하면, **실제 합격생 선배들의 명문장**을 찾아줍니다.

    ---
    *Developed by Computer Science Dept. Student*
    """)

    col1, col2 = st.columns(2)
    with col1:
        st.success(f"📚 학습된 데이터: **{len(rec_sentences)}개**의 문장")
    with col2:
        st.info("🤖 적용된 AI 모델: **Logistic Regression & TF-IDF (Char-level)**")

# --- [메뉴 2] IT 적합도 진단 ---
elif menu == "IT 적합도 진단":
    st.header("💻 IT/컴퓨터공학 적합도 진단")
    st.write("작성하신 세특 내용이나 자기소개서 초안을 입력해보세요.")

    if pred_model is None:
        st.error("🚨 모델 파일이 없습니다! (major_predictor_model.pkl)")
    else:
        user_input = st.text_area("내용 입력", height=200, placeholder="예: 파이썬을 활용하여 데이터 분석 프로젝트를 진행함...")

        if st.button("진단하기"):
            if len(user_input) < 10:
                st.warning("내용이 너무 짧습니다. 10자 이상 입력해주세요.")
            else:
                # 예측
                with st.spinner("AI가 분석 중입니다..."):
                    vec_input = pred_vec.transform([user_input])
                    prob = pred_model.predict_proba(vec_input)[0]
                    score = prob[1] * 100  # IT 확률

                # 결과 시각화
                st.divider()
                st.subheader("📊 분석 결과")

                # 게이지 바
                st.progress(int(score))
                st.metric(label="IT 계열 적합도", value=f"{score:.1f}점")

                if score >= 85:
                    st.success("🏆 **[합격권]** 완벽합니다! 전공 관련 키워드가 풍부합니다.")
                    st.balloons()
                elif score >= 60:
                    st.info("✨ **[우수]** 좋습니다. 조금 더 구체적인 기술 용어를 추가해보세요.")
                else:
                    st.warning("🤔 **[노력 필요]** IT 관련 전문 용어(알고리즘, 언어 이름 등)가 부족합니다.")

# --- [메뉴 3] 세특 문장 추천기 ---
elif menu == "세특 문장 추천기":
    st.header("📝 합격생 족보(세특) 추천기")
    st.write("활동 키워드를 입력하면, 선배들이 실제로 썼던 **합격 문장**을 찾아드립니다.")

    if len(rec_sentences) == 0:
        st.error("🚨 데이터셋 로딩 실패 (dataset_v2.csv 확인 필요)")
    else:
        keyword = st.text_input("활동 키워드 입력", placeholder="예: 게임 제작, 데이터 분석, 동아리 활동")

        if st.button("합격 문장 검색"):
            with st.spinner("선배들의 생기부를 뒤지는 중..."):
                # 검색 로직
                query_vec = rec_vec.transform([keyword])
                similarities = cosine_similarity(query_vec, rec_matrix).flatten()

                # Top 3 추출
                top_indices = similarities.argsort()[-5:][::-1]  # 5개 뽑음

                st.divider()
                st.subheader(f"🔍 '{keyword}' 관련 추천 문장")

                count = 0
                for idx in top_indices:
                    sim_score = similarities[idx]
                    if sim_score > 0.15:  # 유사도 15% 이상만 표시
                        count += 1
                        rec_text = rec_sentences[idx]

                        # OCR 오타 안내
                        st.markdown(f"""
                        > **추천 {count}** (유사도 {sim_score * 100:.1f}%)  
                        > " {rec_text} "
                        """)

                if count == 0:
                    st.warning("비슷한 내용을 찾지 못했습니다. 키워드를 다르게 입력해보세요.")