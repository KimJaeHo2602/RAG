import os
import streamlit as st
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages.chat import ChatMessage
from langchain_teddynote.prompts import load_prompt
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


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

st.title("PDF기반 QA💬")

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
    uploaded_files = st.file_uploader(
        "PDF파일 업로드", type=["pdf"], accept_multiple_files=True
    )

    # 모델 메뉴 선택
    selected_model = st.selectbox(
        "model선택", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    )


# 이전 대화를 츌력 - chat_message에는 role과 content가 들어가 있다.
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메세지를 추가 - ChatMessage은 langchain_core.messages.chat에 있다.
def add_messages(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 파일을 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 파일을 처리중입니다.")
def embed_files(files):

    all_documents = []

    for file in files:
        # 업로드한 파일을 캐시 디렉토리에 저장
        file_content = file.read()
        file_path = f"./.cache/files/{file.name}"
        with open(file_path, "wb") as f:
            f.write(file_content)

        # ------------------------------ indexing
        # 1단계: 문서 로드
        loader = PDFPlumberLoader(file_path)
        docs = loader.load()

        # 2단계: 문서 분할
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000, chunk_overlap=100
        )
        documents = text_splitter.split_documents(docs)

        all_documents.extend(documents)

    if not all_documents:
        st.error("PDF 파일에서 문서를 추출하지 못했습니다.")
        return None

    # 3단계: 임베딩 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 4단계: 벡터 스토어 생성
    vector_store = FAISS.from_documents(all_documents, embeddings)

    # 5단계: 검색기(Retriever) 생성
    retriever = vector_store.as_retriever()
    return retriever


# 체인 생성
def create_chain(retriever, model_name="gpt-4o-mini"):
    # 6단계: 프롬프트 생성
    prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")

    # 7단계 언어모델 생성
    llm = ChatOpenAI(model=model_name, temperature=0)

    # 8단계 체인 생성
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# 파일이 업로드 되었을 때
if uploaded_files:
    # 파일 업로드 후 retriever생성. 시간이 오래걸릴 예정
    retriever = embed_files(uploaded_files)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain

# 초기화 버튼이 눌렸을 때
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요.")

# 경고메세지를 띄우기 위한 빈 영역
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
        add_messages("user", user_input)
        add_messages("assistant", ai_answer)

    else:
        # 파일을업로드 하라는 경고 메세지
        warning_msg.error("파일을 업로드 해주세요")

# poetry shell로 가상환경 실행
# streamlit run PDF.py
