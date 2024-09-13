import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import ChatMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from langchain_teddynote.prompts import load_prompt

load_dotenv()


# 캐시 디렉토리 생성
if not os.path.exists("cache"):
    os.mkdir("cache")

# 파일 업로드 전용 폴더
if not os.path.exists("cache/files"):
    os.mkdir("cache/files")

if not os.path.exists("cache/embeddings"):
    os.mkdir("cache/embeddings")

#########
st.title("PDF기반 QA")

# 대화기록 저장을 위한 session_state 생성
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 아무런 파일을 업로드 하지 않은 경우
if "chain" not in st.session_state:
    st.session_state["chain"] = None


# 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메세지 추가
def add_messages(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("파일 업로드", type=["pdf"])

    # 모델 메뉴 선택
    selected_model = st.selectbox(
        "모델 선택", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    )


# 파일을 캐시 저장 + indexing
@st.cache_resource(show_spinner="업로드한 파일을 처리중입니다")
def embed_file(file):
    file_content = file.read()
    file_path = f"cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # ----------- Indexing
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    documents = text_splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vector_store = FAISS.from_documents(documents, embeddings)

    retriever = vector_store.as_retriever()
    return retriever


# 체인생성
def create_chain(retriever, model_name="gpt-4o-mini"):

    prompt = load_prompt("prompts/pdf-rag.yaml", encoding="utf-8")

    llm = ChatOpenAI(model=model_name)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


# 파일이 업로드 되었을 때,
if uploaded_file:
    retriever = embed_file(uploaded_file)
    chain = create_chain(retriever, model_name=selected_model)
    st.session_state["chain"] = chain

# 초기화 버튼이 눌렸을 때
if clear_btn:
    st.session_state["messages"] = []

# 이전대화기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요")

# 경고메세지를 띄우기 위한 빈 영역
warning_msg = st.empty()

# 만약 사용자의 입력이 들어오면,
if user_input:
    chain = st.session_state["chain"]

    if chain is not None:
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 스트리밍 호출
        response = chain.stream("assistant")
        with st.chat_message("assistant"):
            # 빈 컨테이너를 만들어, 이 컨테이너에 토큰을 스트리밍
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

        # 대화 기록을 저장
        add_messages("user", user_input)
        add_messages("assistant", ai_answer)

    else:
        warning_msg.error("파일을 업로드 해 주세요")

# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# if "chain" not in st.session_state:
#     st.session_state["chain"] = None

# def print_messages():
#     for chat_message in st.session_state["messages"]:
#         st.chat_message(chat_message.role).write(chat_message.content)

# def add_messages(role, content):
#     st.session_state["messages"].append(ChatMessage(role=role, content=content))

# @st.cache_resource(show_spinner="업로드한 파일을 처리중입니다.")
# def embed_file(file):
#     file_content = file.read()
#     file_path = f"cache/files/{file.name}"
#     with open(file_path, "wb") as f:
#         f.write(file_content)
# 'wb is write binary, rb is read binary, ab is add binary


# if "messages" not in st.session_state:
#     st.session_state["messages"] = []

# if "chain" not in st.session_state:
#     st.session_state["chain"] = None


# def print_messages():
#     for chat_message in st.session_state["messages"]:
#         st.chat_message(chat_message.role).write(chat_message.content)


# def add_messages(role, content):
#     st.session_state["messages"].append(ChatMessage(role=role, content=content))


# @st.cache_resource(show_spinner="업로드한 파일을 처리중입니다.")
# def embed_file(file):
#     file_content = file.read()
#     file_path = f"cache/files/{file.name}"
#     with open(file_path, "wb") as f:
#         f.write(file_content)
