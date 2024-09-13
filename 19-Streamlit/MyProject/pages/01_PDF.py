import streamlit as st
import os
from dotenv import load_dotenv

from langchain_teddynote import logging
from langchain_core.messages import ChatMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_teddynote.prompts import load_prompt
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

logging.langsmith("[Project] PDF RAG")
# api key 가져오기
load_dotenv()

# 캐시 디렉토리 생성 .은 숨김파일로 처리한다.
if not os.path.exists(".cache"):
    os.mkdir(".cache")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")


st.title("PDF기반 QA💬")


if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    st.session_state["chain"] = None

with st.sidebar:
    clear_btn = st.button("대화초기화")

    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    # selected_prompt = "prompts/pdf-rag.yaml"

    selected_model = st.selectbox(
        "model선택", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    )


def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시 저장.(시간이 오래걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리중입니다.")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    ############ indexing
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

    documents = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever()
    return retriever


# 체인 생성
def create_chain(retriever, model_name="gpt-4o-mini"):
    prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")

    llm = ChatOpenAI(model=model_name, temperature=0)

    output_parser = StrOutputParser()

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )
    return chain


if clear_btn:
    st.session_state["messages"] = []

if uploaded_file:
    # 파일 업로드 후 retriever생성. 작업시간이 오래걸릴 예정
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain

print_messages()

user_input = st.chat_input("메세지를 입력해 주세요")

# 빈 영역을 잡아주는 역할. 경고 메세지를 띄우기 위함
warning_msg = st.empty()

if user_input:

    # 체인생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.stream(user_input)

        with st.chat_message("assistant"):
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)

    # 파일을 업로드하는 경고메세지
    else:
        warning_msg.error("파일을 업로드 해 주세요")
