"""
chatbot.py
PDF 파일을 RAG 기반으로 처리하는 챗봇 모듈

Classes
-------
PDFRAGChatbot
    PDF 파일을 처리하고, 질문에 대한 응답을 생성하는 챗봇 클래스

Methods
-------
__init__(prj_dir, openai_api_key)
    챗봇을 초기화하고 필요한 모델과 설정을 준비
load_pdf(pdf_path)
    PDF 파일을 로드하고 처리
split_text(documents)
    로드된 텍스트를 분할하여 처리 가능한 형태로 변환
create_vectorstore(docs)
    텍스트 데이터를 벡터 스토어에 저장
setup_rag_chain()
    langchain을 활용해 rag chain을 생성
generate_response(query)
    검색된 정보를 바탕으로 응답을 생성
"""

import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class PDFRAGChatbot:
    def __init__(self, prj_dir, openai_api_key):
        self.project_dir = prj_dir
        self.openai_api_key = openai_api_key
        self.embeddings_model = OpenAIEmbeddings(model="text-embedding-ada-002", openai_api_key=self.openai_api_key)
        self.vectorstore = None  # Chroma 벡터 스토어 초기화
        self.retriever = None # retriever 초기화
        self.rag_chain = None  # RAG 체인 초기화

    def load_pdf(self, pdf_path):
        try:
            loader = PyMuPDFLoader(pdf_path)
            documents = loader.load()
            return documents
        except Exception as e:
            print(f"PDF 로드 중 오류 발생: {str(e)}")
            return []
        
    def split_text(self, documents):
        try:
            text_splitter = CharacterTextSplitter(
                separator=r"[.!?]\s+",
                chunk_size=1500,
                chunk_overlap=300,
                length_function=len,
                is_separator_regex=True,
            )
            return text_splitter.split_documents(documents)
        except Exception as e:
            print(f"텍스트 분할 중 오류 발생: {str(e)}")
            return []
        
    def create_vectorstore(self, docs):
        try:
            self.vectorstore = Chroma.from_documents(
                documents=docs,
                embedding=self.embeddings_model,
                collection_name="collection_legal_pdf",
                collection_metadata={'hnsw:space': 'ip'},
                persist_directory=os.path.join(self.project_dir, "chroma_db")
            )
            self.retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
            print("벡터 스토어가 성공적으로 생성되었습니다.")
        except Exception as e:
            print(f"벡터 스토어 생성 중 오류 발생: {str(e)}")

    def setup_rag_chain(self):
        try:
            template = '''다음 문맥을 기반으로 질문에 답변하세요:
            {context}

            질문: {query}
            '''
            prompt = ChatPromptTemplate.from_template(template)
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, max_tokens=1000, openai_api_key=self.openai_api_key)
            format_docs = lambda docs: "\n\n".join([d.page_content for d in docs])

            self.rag_chain = (
                {"context": self.retriever | format_docs, "query": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            print("RAG 체인이 성공적으로 설정되었습니다.")

        except Exception as e:
            print(f"RAG 체인 설정 중 오류 발생: {str(e)}")

    def generate_response(self, query):
        if self.rag_chain:
            try:
                result = self.rag_chain.invoke(query)
                return result
            except Exception as e:
                print(f"답변 생성 중 오류 발생: {str(e)}")
                return "답변 생성 중 오류가 발생했습니다."
        else:
            return "RAG 체인이 설정되지 않았습니다."