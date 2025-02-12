"""
app.py
Streamlit 애플리케이션의 진입점을 처리하는 모듈
챗봇 및 고객 분류기의 초기화를 포함하여, 파일 업로드 및 세션 관리 기능을 제공

Functions
---------
initialize_chatbot()
    챗봇을 세션 상태에서 초기화

initialize_classifier()
    고객 분류기를 세션 상태에서 초기화

initialize_conversation()
    대화 기록 및 Q&A 라운드를 세션 상태에서 초기화

handle_pdf_upload(uploaded_files)
    업로드된 PDF 파일을 처리하여 저장
"""

import streamlit as st
from config import OPENAI_API_KEY, PRJ_DIR, QUESTION_LIMIT
from utils import save_uploaded_file, save_log_to_csv
from chatbot import PDFRAGChatbot
from customer_classification import CustomerClassifier

import warnings
warnings.filterwarnings('ignore')

def initialize_chatbot():
    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = PDFRAGChatbot(PRJ_DIR, OPENAI_API_KEY)

def initialize_classifier():
    if 'classifier' not in st.session_state:
        st.session_state.classifier = CustomerClassifier(model_name="gpt-4o-mini")

def initialize_conversation():
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'qa_round' not in st.session_state:
        st.session_state.qa_round = 0

def handle_pdf_upload(uploaded_files):
    if uploaded_files:
        all_docs = []
        for uploaded_file in uploaded_files:
            file_path = save_uploaded_file(uploaded_file)
            data = st.session_state.chatbot.load_pdf(file_path)
            docs = st.session_state.chatbot.split_text(data)
            all_docs.extend(docs)
        
        if 'vectorstore_created' not in st.session_state:
            st.session_state.chatbot.create_vectorstore(all_docs)
            st.session_state.chatbot.setup_rag_chain()
            st.session_state.vectorstore_created = True
        
        st.write(f"모든 PDF 파일이 성공적으로 처리되었습니다. 총 {len(all_docs)} 페이지가 있습니다.")

def main():
    st.title('🦆 FSI Chatbot by. water_fight')

    initialize_chatbot()
    initialize_classifier()
    initialize_conversation()

    st.subheader('PDF 파일 업로드 (여러 개 선택 가능)')
    uploaded_files = st.file_uploader("PDF 파일을 업로드하세요.", type="pdf", accept_multiple_files=True)

    if uploaded_files:
        handle_pdf_upload(uploaded_files)

    if st.session_state.qa_round >= QUESTION_LIMIT:
        st.write(f"{QUESTION_LIMIT}회의 질문과 답변에 따른 고객의 금융 지식 수준은 {st.session_state.classifier.determine_final_level()}입니다.")
    else:
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "투자설명서를 읽고 궁금한 점을 질문하세요!"}]

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if query := st.chat_input(f"질문을 입력하세요."):
            st.session_state.messages.append({"role":"user", "content": query})
            st.chat_message("user").write(query)

            # Fetch the response and classify customer
            response = st.session_state.chatbot.generate_response(query)

            classification_result  = st.session_state.classifier.classify(query, response, st.session_state.conversation_history)
            refined_answer = classification_result["refined_answer"]
            customer_level = classification_result["customer_level"]

            print(f"질문: {query}\nRAG 답변: {response}\n가공된 답변: {refined_answer}\n해당 질문에 대한 고객의 금융 지식 수준: {customer_level}\n")

            # Update conversation history
            st.session_state.conversation_history.append({
                '질문': query,
                '가공된 답변': refined_answer,
                '고객 등급': customer_level
            })

            combined_response = f"위 질문으로 판단한 고객의 금융 지식 수준: {customer_level}\n\n답변: {refined_answer}"

            st.session_state.messages.append({"role": "assistant", "content": combined_response})
            st.chat_message("assistant").write(combined_response)

            # Chat logging at CSV
            save_log_to_csv('./chat_log.csv',query, response, refined_answer, customer_level, st.session_state.classifier.levels_count)

            st.session_state.qa_round += 1

if __name__ == "__main__":
    main()