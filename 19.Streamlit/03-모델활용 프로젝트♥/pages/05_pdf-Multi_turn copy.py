from dotenv import load_dotenv
import os
import streamlit as st
from operator import itemgetter

from langchain_teddynote import logging
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# kiwi
from kiwipiepy import Kiwi

kiwi = Kiwi()
# Ensemble Retriever
from langchain.retrievers import BM25Retriever, EnsembleRetriever

# reranker
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

# reorder
from langchain_community.document_transformers import LongContextReorder
from langchain_core.runnables import RunnableLambda

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

    uploaded_file = st.file_uploader("PDF파일 업로드", type=["PDF"])

    selected_model = st.selectbox(
        "model선택", ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    )

    # 세션 ID를 지정하는 메뉴
    session_id = st.text_input("세션 ID를 입력해 주세요", "abc123")


# 이전 대화를 출력 - chat_message에는 role과 content가 들어가 있다.
def print_message():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


from langchain_core.messages import ChatMessage


# 새로운 메세지를 추가 - ChatMessage은 langchain_core.messages.chat에 있다.
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


from langchain_community.chat_message_histories import ChatMessageHistory


# 세션 ID를 기반으로 세션 기록을 가져오는 함수
def get_session_history(session_ids):
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]


# Kiwi 함수 : kiwi tokenizer
def kiwi_tokenize(docs):
    return [token.form for token in kiwi.tokenize(docs)]


# 파일을 캐시 저장
@st.cache_resource(show_spinner="업로드한 파일을 처리중입니다.")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # indexing
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    documents = text_splitter.split_documents(docs)

    embedding = OpenAIEmbeddings(model="text-embedding-3-small")
    # chroma
    chroma_vector_store = Chroma.from_documents(documents, embedding)
    chroma_retriever = chroma_vector_store.as_retriever()
    # kiwi + bm25
    kiwi_vector_store = BM25Retriever.from_documents(
        documents, preprocess_func=kiwi_tokenize
    )
    # ensemble
    ensemble_retriever = EnsembleRetriever(
        retrievers=[chroma_retriever, kiwi_vector_store],
        weights=[0.5, 0.5],
        search_kwargs={"k": 10},
    )
    # reranker
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
    # 상위 3개 모델 선택
    compressor = CrossEncoderReranker(model=reranker_model, top_n=3)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )
    return compression_retriever


# Reorder
def reorder_documents(compression_retriever):
    # 재정렬
    reordering = LongContextReorder()
    reordered_docs = reordering.transform_documents(compression_retriever)
    return reordered_docs


def create_chain(compression_retriever, model_name="gpt-4o-mini"):

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
            당신은 Question-Answering 챗봇입니다. 주어진 문서를 통해서 질문에 대한 답변을 제공해 주세요. 모르면 모른다고 답변해 주세요. 한국어로 대답해 주세요
            """,
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            (
                "user",
                """
            #Question
            {question}
            #Context
            {context}
            """,
            ),
        ]
    )
    llm = ChatOpenAI(model=model_name, temperature=0)
    chain = (
        {
            "context": itemgetter("question")
            | compression_retriever
            | reorder_documents,
            "question": itemgetter("question"),
            "chat_history": itemgetter("chat_history"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    from langchain_core.runnables import RunnableWithMessageHistory

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )
    return chain_with_history


if uploaded_file:
    compression_retriever = embed_file(uploaded_file)
    chain = create_chain(compression_retriever, model_name=selected_model)
    st.session_state["chain"] = chain

if clear_btn:
    st.session_state["messages"] = []

# 이전 대화기록 출력
print_message()

user_input = st.chat_input("궁금한 내용을 물어보세요")
warning_msg = st.empty()

if user_input:
    chain = st.session_state["chain"]

    if chain is not None:
        response = chain.stream(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        # 사용자의 입력
        st.chat_message("user").write(user_input)
        # 빈 컨테이너를 만들어서 여기에 토큰을 스트리밍 출력한다.
        with st.chat_message("assistant"):
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token
                container.markdown(ai_answer)

            # 대화기록 저장
            add_message("user", user_input)
            add_message("assistant", ai_answer)

    else:
        warning_msg.error("pdf 파일을 업로드 해주세요.")
