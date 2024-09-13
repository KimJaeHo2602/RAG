from dotenv import load_dotenv
import os
import streamlit as st

from langchain_teddynote import logging
from langchain_community.document_loaders import PDFPlumberLoader


logging.langsmith("[Project] PDF Multu-turn RAG")

load_dotenv()

if not os.path.exists(".cache"):
    os.mkdir("cache")

if not os.path.exists(".cache/embeddings"):
    os.mkdir("cache/embeddings")

if not os.path.exists(".cache/files"):
    os.mkdir("cache/files")

st.title("PDF기반 멀티턴 QA")

# 처음 한번만 실행하기 위한 코드 - 리스트로 저장하는것은 순서대로 저장하는 특징이 있다
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 딕셔너리로 저장하는것은 키벨류 쌍으로 저장하는 특징이 있다.
if "store" not in st.session_state:
    st.session_state["store"] = {}

with st.sidebar:
    clear_btn = st.button("대화초기화")

    uploaded_file = st.file_uploader("PDF파일 업로드", type=["PDF"], accept_multiple_files=True)

    selected_model = st.selectbox(
        "model선택", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    )

    # 세션 ID를 지정하는 메뉴
    session_id = st.text_input("세션 ID를 입력해 주세요", "abc123")


# 이전 대화를 출력 - chat_message에는 role과 content가 들어가 있다.
def print_message():
    for chat_message in st.session_state["messages"]
    st.chat_message(chat_message.role).write(chat_message.content)

from langchain_core.messages import ChatMessage
# 새로운 메세지를 추가 - ChatMessage은 langchain_core.messages.chat에 있다.
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))

from langchain_community.chat_message_histories import ChatMessageHistory
# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_state(session_ids):
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]

# 파일을 캐시 저장
@st.cache_resource(show_spinner="업로드한 파일을 처리중입니다.")
def embed_file(file):
    file_content = file.read()
    file_path = f"./cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)