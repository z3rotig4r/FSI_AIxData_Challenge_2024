"""
utils.py
유틸리티 함수를 제공하는 모듈

Functions
---------
save_uploaded_file(uploaded_file, directory)
    업로드된 파일을 지정된 디렉토리에 저장
load_ground_truth(df)
    DataFrame에서 각 ETF 종목별로 ground truth를 로드하는 함수
prepare_questions_and_gts(ground_truth)
    평가에 사용할 질문 및 ground truth 데이터를 준비하는 함수
save_log_to_csv(file_path, question, rag_answer, refined_answer, customer_level, level_count)
    채팅 이력을 csv에 저장하는 함수
"""

import os
import csv

# Streamlit에 PDF 업로드 시 로컬 폴더에 저장하는 함수
def save_uploaded_file(uploaded_file, directory="./data"):
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return file_path

def load_ground_truth(df):
    ground_truth = {}

    # DataFrame의 각 행이 하나의 ETF에 해당한다고 가정
    for idx, row in df.iterrows():
        etf_name = row['상품명']
        ground_truth[etf_name] = {
            '투자등급': row['답변1 (투자등급)'],
            '기초지수': row['답변2 (기초지수)'],
            '투자포인트': row['답변3 (투자 포인트)'],
            '주요보유현황': row['답변4 (주요보유현황)'],
            '운용성과': row['답변5 (운용성과)']
        }
    return ground_truth

def prepare_questions_and_gts(ground_truth):
    questions = []
    gts = []

    for etf_name, gt_data in ground_truth.items():
        # ETF 종목명을 질문에 포함
        questions.append(f"{etf_name}의 투자등급은 몇 등급입니까?")
        questions.append(f"{etf_name}의 기초 지수는 어떻게 되나요?")
        questions.append(f"{etf_name}의 투자 포인트는 무엇인가요?")
        questions.append(f"{etf_name}의 주요 종목 현황은 어떻게 되나요?")
        questions.append(f"{etf_name}의 운용 성과는 어떻게 되나요?")

        # 해당하는 ground truth를 답변으로 추가
        gts.append(gt_data['투자등급'])
        gts.append(gt_data['기초지수'])
        gts.append(gt_data['투자포인트'])
        gts.append(gt_data['주요보유현황'])
        gts.append(gt_data['운용성과'])

    return questions, gts

def save_log_to_csv(file_path, question, rag_answer, refined_answer, customer_level, level_count):
    # CSV 파일이 없으면 헤더를 추가하여 생성
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, quoting=csv.QUOTE_ALL)
        
        # 파일이 처음 생성될 때 헤더를 추가
        if not file_exists:
            writer.writerow(['question', 'rag_answer', 'refined_answer', 'customer_level', 'level_count'])
        
        # 데이터 쓰기
        writer.writerow([question, rag_answer, refined_answer, customer_level, level_count])