import os
import streamlit as st
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_teddynote.prompts import load_prompt
from langchain_core.messages import ChatMessage  # 메세지 추가
from langchain_community.chat_message_histories import (
    ChatMessageHistory,
)  # 대화기록 저장
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)  # 대화기록 저장
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory  # 대화기록 저장

load_dotenv()

logging.langsmith("[Project]Multi-turn")

# 캐시 디렉토리 생성 .을 찍는건 숨김표시 의미
if not os.path.exists("./.cache"):
    os.mkdir("./.cache")

if not os.path.exists("./.cache/files"):
    os.mkdir("./.cache/files")

if not os.path.exists("./.cache/embeddings"):
    os.mkdir("./.cache.embeddings")

st.title("대화 내용을 기억하는 챗봇")


if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    st.session_state["chain"] = {}

# ★★★ store를캐싱하는 코드. 세션 기록을 저장하는 딕셔너리
if "store" not in st.session_state:
    st.session_state["store"] = {}

with st.sidebar:
    clear_btn = st.button("대화 초기화")

    uploaded_file = st.file_uploader("pdf파일 업로드", type=["pdf"])

    selected_model = st.selectbox("모델선택", ["gpt-4o-mini", "gpt-4o"], index=0)

    system_prompt = st.text_area(
        "시스템 프롬프트", "프롬프트를 입력해주세요.", height=200
    )
    session_id = st.text_input("세션아이디를 입력하세요", "abc123")

# from langchain_core.messages import ChatMessage


# 이전대화 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메세지 추가
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


@st.cache_resource(show_spinner="파일을 업로드 중 입니다.")
def embed_file(file):

    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    ##### indexing
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    documents = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever()
    return retriever


# 세션 기록을 저장하는 딕셔너리
# store = {}
# from langchain_community.chat_message_histories import ChatMessageHistory


# ★★★세션 id를 기반으로 세션 기록을 가져오는 함수 store를 st.session_state["store"]로 바꿔줌
def get_session_history(session_ids):
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]


# from langchain_core.runnables.history import RunnableWithMessageHistory
def create_chain(retriever, model_name="gpt-4o-mini"):

    # prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 Question-Answering 챗봇입니다. 주어진 문서를 통해서 질문에 대한 답변을 제공해 주세요. 모르면 모른다고 답변해 주세요. 한국어로 대답해 주세요.",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "#Question:\n{question} #Context\n{context}"),
        ]
    )

    llm = ChatOpenAI(model=model_name)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    # ★★★
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    return chain_with_history


if uploaded_file:
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain


if clear_btn:
    st.session_state["messages"] = []


print_messages()


# 유저 질문
user_input = st.chat_input("궁금한 내용을 물어보세요")

# 경고메세지를 위함
warning_msg = st.empty()

# if "chain" not in st.session_state:
#     st.session_state["chain"] = create_chain(retriever, model_name=selected_model)

if user_input:
    # 체인생성
    chain = st.session_state["chain"]

    if chain:
        response = chain.stream(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}},
        )

        st.chat_message("user").write(user_input)

        with st.chat_message("assistant"):
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        add_message("user", user_input)
        add_message("assistant", ai_answer)

    else:
        warning_msg.error("파일을 업로드 해 주세요")
