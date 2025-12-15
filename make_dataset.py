import os
import pandas as pd

# 1. 파일 경로 (본인 경로로 확인!)
DATA_FOLDER = r"C:\Users\anfdp\Downloads\생기부_censored_txt (1)\생기부_censored_txt"

# 2. 대분류 매핑 (오타 및 변형 케이스 전격 추가된 최종 버전)
MAJOR_MAPPING = {
    # [IT/공학 계열]
    "컴퓨터과학과": "IT_Engineering",
    "컴퓨터과학전공": "IT_Engineering",
    "컴퓨터과확전공": "IT_Engineering",  # 오타 처리
    "휴먼AI공학전공": "IT_Engineering",
    "지능IOT": "IT_Engineering",
    "지능ioT융합": "IT_Engineering",
    "지능IoT융합": "IT_Engineering",
    "소프트웨어": "IT_Engineering",
    "자유전공(it계열)": "IT_Engineering",

    # [경영/경제 계열]
    "경영학부": "Business_Economics",
    "글로벌경영학과": "Business_Economics",
    "경제금융": "Business_Economics",
    "지적재산권전공": "Business_Economics",

    # [바이오/자연과학 계열]
    "생명공학과": "Bio_Science",
    "식품영양학과": "Bio_Science",

    # [교육/인문 계열]
    "교육학과": "Education",
    "자유전공학부(인문사회계열)": "Humanities_Social",

    # [기타/자율]
    "자율전공": "Undecided"
}


def parse_filename_info(filename):
    """파일명에서 [학과, 전형(수시/정시)] 정보를 추출"""

    # 1. 전형 구분
    admission_type = "Unknown"
    if "수시" in filename:
        admission_type = "Susi"
    elif "정시" in filename:
        admission_type = "Jeongsi"

    # 2. 학과 매핑
    final_category = "Unknown"
    original_major = "Unknown"

    for key_major, category in MAJOR_MAPPING.items():
        if key_major in filename:
            original_major = key_major
            final_category = category
            break

    if final_category == "Unknown":
        if "자율" in filename:
            final_category = "Undecided"
            original_major = "자율전공"

    return original_major, final_category, admission_type


def create_dataset():
    data_list = []

    if not os.path.exists(DATA_FOLDER):
        print(f"🚨 에러: 폴더 경로를 찾을 수 없습니다: {DATA_FOLDER}")
        return

    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith(".txt")]
    print(f"📂 총 {len(files)}개의 파일 분석 시작...")

    for file in files:
        file_path = os.path.join(DATA_FOLDER, file)

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        except:
            with open(file_path, "r", encoding="cp949") as f:
                text = f.read()

        if len(text) < 50:
            continue

        orig_major, category, admission = parse_filename_info(file)

        data_list.append({
            "filename": file,
            "original_major": orig_major,
            "category": category,
            "admission": admission,
            "text": text
        })

    df = pd.DataFrame(data_list)

    # 저장
    df.to_csv("dataset_v2.csv", index=False, encoding="utf-8-sig")
    print("-" * 30)
    print(f"✅ 'dataset_v2.csv' 생성 완료! ({len(df)}개 데이터)")
    print(df['category'].value_counts())


if __name__ == "__main__":
    create_dataset()