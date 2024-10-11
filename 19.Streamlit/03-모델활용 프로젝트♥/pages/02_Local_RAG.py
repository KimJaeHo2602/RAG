import os
import streamlit as st
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.messages.chat import ChatMessage
from langchain_teddynote.prompts import load_prompt
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from retriever import create_retriever

load_dotenv()

logging.langsmith("[Project] PDF RAG")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("Local model 기반 RAG💬")

# 처음 1번만 실행하기 위한 코드. - 대화기록을 저장하기 위한 코드로 생성한다.
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 아무런 파일을 업로드 하지 않는경우
if "chain" not in st.session_state:
    st.session_state["chain"] = None

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("PDF파일 업로드", type=["pdf"])

    # 모델 메뉴 선택
    selected_model = st.selectbox(
        "model선택", ["gpt-4o-mini", "Ollama-EEVE", "Ollama-Llama3.1"]
    )


# 이전 대화를 츌력 - chat_message에는 role과 content가 들어가 있다.
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메세지를 추가 - ChatMessage은 langchain_core.messages.chat에 있다.
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리중입니다.")
def embed_file(file):
    # 업로드한 파일을 캐시 디렉토리에 저장
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # create_retriever를 retriever로부터 가져온다.
    return create_retriever(file_path)


# retriever할때, 메타데이터를제외하고, page_content만 불러오기 위함이다. meta data를 이해 못하는 경우가 있기 때문이다.
def format_doc(document_list):
    return "\n\n".join([doc.page_content for doc in document_list])


# 체인 생성
def create_chain(retriever, model_name="gpt-4o-mini"):
    # 모델 이름이 gpt-4o-mini면,
    if model_name == "gpt-4o-mini":
        # 6단계: 프롬프트 생성
        prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")

        # 7단계 언어모델 생성
        llm = ChatOpenAI(model=model_name, temperature=0)

    # xionic = ChatOpenAI(
    #     model_name="xionic-1-72b-20240610",
    #     base_url="https://sionic.chat/v1/",
    #     api_key="934c4bbc-c384-4bea-af82-1450d7f8128d",
    # )
    # pdf-rag로 써도잘되네?
    elif model_name == "Ollama-EEVE":
        prompt = load_prompt("prompts/pdf-rag-ollama-EEVE.yaml", encoding="utf-8")
        llm = ChatOllama(model="EEVE-Korean-10.8b:latest", temperature=0)

    elif model_name == "Ollama-Llama3.1":
        prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")
        llm = ChatOllama(model="llama3.1", temperature=0)

    # 8단계 체인 생성
    chain = (
        {"context": retriever | format_doc, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# 파일이 업로드 되었을 때
if uploaded_file:
    # 파일 업로드 후 retriever생성. 시간이 오래걸릴 예정
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain

# 초기화 버튼이 눌렸을 때
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요.")

# 경고메세지를 뜨우기 위한 빈 영역
warning_msg = st.empty()

# 만약 사용자의 입력이 들어오면,
if user_input:
    # 체인 생성
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.stream(user_input)
        with st.chat_message("assistant"):
            # 빈 컨테이너를 만들어서, 여기에 토큰을 스트리밍
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화기록을 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)

    else:
        # 파일을업로드 하라는 경고 메세지
        warning_msg.error("파일을 업로드 해주세요")

# poetry shell로 가상환경 실행
# streamlit run .\02.Local_lag.py
