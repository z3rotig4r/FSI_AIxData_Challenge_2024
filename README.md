# 실행방법
streamlit run app.py
# 프로젝트 구조
fsi_idea_poc/
│
├── app.py                # Streamlit 인터페이스 코드
├── chatbot.py            # PDFRAGChatbot 클래스 및 관련 로직
├── eval_rag.py           # RAGAS를 이용한 RAG 평가 클래스 EVALRAGBot 관련 로직
├── config.py             # 환경 변수 관리 및 설정
├── utils.py              # 유틸리티 함수들 (예: 파일 처리)
└── .env                  # 환경 변수 설정 파일
# 구현된 기능
1. PDF 기반으로 Client 질문에 대한 답변 생성 (RAG 이용) - 09.01. by z3rotig4r
2. PDF 파일을 여러 개 올릴 수 있도록 코드 수정 - 09.02. by z3rotig4r
3. 평가지표 반영 RAGAS 이용 w/ langsmith - 09.14. by z3rotig4r