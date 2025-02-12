from chatbot import PDFRAGChatbot
from config import OPENAI_API_KEY, LANGSMITH_API_KEY, PRJ_DIR
from utils import load_ground_truth, prepare_questions_and_gts

import os
import pandas as pd
from datasets import Dataset

import warnings
warnings.filterwarnings('ignore')

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from ragas import evaluate
# configure RunConfig
from ragas.run_config import RunConfig

from datasets import Dataset
from ragas.metrics import (
    faithfulness, answer_relevancy, context_precision, context_recall
)

from pydantic import ValidationError
import asyncio

class EvalRAGBot(PDFRAGChatbot):
    def __init__ (self, pdf_files, questions, gts, openai_api_key, prj_dir):
        super().__init__(prj_dir, openai_api_key)
        self.pdf_files = pdf_files
        self.questions = questions
        self.gts = gts

        self.answers = []
        self.contexts = []

        # vectorstore 및 langchain 파이프라인 초기화
        self.vectorstore = None
        self.qa_chain = None
        self.embeddings = self.embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002", 
            openai_api_key=self.openai_api_key)
        self.llm = None

        # 초기화 시 모든 과정을 처리
        self.setup_rag_pipeline()

    def setup_vectorstore(self):
        # PDF 읽고 Text Split
        doc_loaders = [PyMuPDFLoader(pdf) for pdf in self.pdf_files]
        documents = []
        for loader in doc_loaders:
            docs = loader.load()
            documents.extend(docs)

        text_splitter = CharacterTextSplitter(
            separator=r"[.!?]\s+",
            chunk_size=1500,
            chunk_overlap=300,
            length_function=len,
            is_separator_regex=True,
        )
        split_docs = text_splitter.split_documents(documents)

        # PDF split 된 문서로 ChromaDB 생성
        try:
            if not self.vectorstore:
                self.vectorstore = Chroma.from_documents(
                    documents=split_docs,
                    embedding=self.embeddings,
                    collection_name="collection_legal_pdf",
                    collection_metadata={'hnsw:space': 'ip'},
                    persist_directory=os.path.join(self.project_dir, "chroma_db")
                )
            else:
                self.vectorstore.add_documents(documents=docs)

            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        except Exception as e:
            print(f"벡터 스토어를 생성하는 중 오류가 발생했습니다: {e}")

    def setup_rag_pipeline(self):
        # OpenAI LLM 설정
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=500, openai_api_key=self.openai_api_key)
        # vectorstore 생성
        self.setup_vectorstore()
        # RAG 체인 생성 (질문과 문서를 결합하여 답변을 생성하는 체인)
        self.qa_chain = self.create_rag_chain()

    def create_rag_chain(self):
        """
        LCEL을 사용하여 RAG 체인을 생성하는 메서드.
        """
        # 문맥과 질문을 포함한 프롬프트 템플릿 설정
        template = '''다음 문맥을 기반으로 질문에 답변하세요:
        {context}

        질문: {question}
        '''
        prompt = ChatPromptTemplate.from_template(template)

        # 문서 포맷팅 함수 (검색된 문서들을 "\n\n"으로 구분하여 하나의 문맥으로 만듦)
        format_docs = lambda docs: "\n\n".join([d.page_content for d in docs])

        # LCEL을 사용한 RAG 체인 구성
        rag_chain = (
            {"context": self.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return rag_chain

    def generate_responses(self):
        """
        복수의 질문에 대한 복수의 답변을 생성하는 메서드.
        """
        for question in self.questions:
            # 질문마다 관련 문서 검색
            retrieved_docs = self.retriever.invoke(question)

            # 검색된 문서가 없으면 빈 결과 처리
            if not retrieved_docs:
                self.answers.append("검색된 문서가 없습니다.")
            else:
                # 검색된 문서를 기반으로 답변 생성             
                response = self.qa_chain.invoke(question)

                self.contexts.append([doc.page_content for doc in retrieved_docs]) 
                self.answers.append(response)
    
    def evaluate_rag_performance(self):

        self.generate_responses()

        data_samples = {
            'question': self.questions,
            'answer': self.answers,
            'ground_truth': self.gts,
            'contexts': self.contexts
        }
        dataset = Dataset.from_dict(data_samples)
        print(":-----Saving RAG Evaluation Dataset-----:")
        # rag 평가 데이터셋 저장
        dataset.to_csv('./rag_eval_dataset.csv')


        print("::-----Evaluation Start-----::")
        # RunConfig로 evaluate customizing
        # 병렬 처리 포기하고 평가 결과 누락되지 않도록
        results = evaluate(dataset,
                           run_config=RunConfig(
                               timeout = 240,
                               max_retries = 10,
                               max_wait = 180,
                               max_workers = 1,
                               exception_types = (ValidationError, TimeoutError, asyncio.TimeoutError),
                           ),
                           metrics=[faithfulness, 
                                    answer_relevancy, 
                                    context_precision, 
                                    context_recall])

        return results

def main():
    # QA 데이터셋 불러오기 및 가공
    df = pd.read_excel('./data/pdf_목록.xlsx', engine="openpyxl")

    # Ground Truth를 로드
    ground_truth = load_ground_truth(df)

    # ETF 종목명에 맞게 질문과 답변을 생성
    questions, gts = prepare_questions_and_gts(ground_truth)
    
    base_dir_temp = './data/간이투자설명서/'
    base_dir_monthly = './data/월간운용보고서'

    pdf_files_temp = [os.path.join(base_dir_temp, file) for file in os.listdir(base_dir_temp) if file.endswith('.pdf')]
    pdf_files_monthly = [os.path.join(base_dir_monthly, file) for file in os.listdir(base_dir_monthly) if file.endswith('.pdf')]

    pdf_files = pdf_files_temp + pdf_files_monthly

    eval_rag = EvalRAGBot(pdf_files, 
                          questions, 
                          gts, 
                          OPENAI_API_KEY,
                          PRJ_DIR,
                          )
    result = eval_rag.evaluate_rag_performance()
    print(result)
    # result => csv 저장
    result_df=result.to_pandas()
    result_df.to_csv('./ragas_evaluation_result.csv')
    

if __name__ == "__main__":
    main()