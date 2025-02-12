"""
config.py
애플리케이션의 설정 값을 관리하는 모듈

Attributes
----------
OPENAI_API_KEY : str
    OpenAI API 호출에 사용되는 키
LANGSMITH_API_KEY : str
    LangSmith API 키
QUESTION_LIMIT : int
    질문 최대 횟수
PRJ_DIR : str
    프로젝트의 루트 디렉토리 경로
"""

from dotenv import load_dotenv
import os

# .env 파일을 로드합니다.
load_dotenv()

# 환경 변수 불러오기
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
LANGSMITH_API_KEY = os.environ.get('LANGSMITH_API_KEY')

# 질문 횟수 설정
QUESTION_LIMIT = 5

# 프로젝트 루트 폴더 설정
PRJ_DIR = 'C:/z3rotig4r/ai_project/fsi_idea_poc/'

# Langchain 관련 설정
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGCHAIN_PROJECT"] = "fsi_idea_poc"
os.environ["LANGCHAIN_API_KEY"] = LANGSMITH_API_KEY