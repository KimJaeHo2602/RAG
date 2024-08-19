import os
import streamlit as st
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages.chat import ChatMessage
from langchain_teddynote.prompts import load_prompt
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_teddynote.models import MultiModal
from langchain_teddynote.messages import stream_response
from langchain_core.output_parsers import StrOutputParser

from retriever import create_retriever

load_dotenv()

logging.langsmith("[Project] Multi Turn 챗봇")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("대화 내용을 기억하는 Multi-Turn 챗봇💬")

# 처음 1번만 실행하기 위한 코드. - 대화기록을 저장하기 위한 코드로 생성한다.
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 캐싱해서 저장하는 함수. # 세션 기록을 저장할 딕셔너리
if "store" not in st.session_state:
    st.session_state["store"] = {}


# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 모델 메뉴 선택
    selected_model = st.selectbox("model선택", ["gpt-4o-mini", "gpt-4o"])

    # 세션 ID를 지정하는 메뉴
    session_id = st.text_input("세션 ID를 입력하세요.", "abc123")


# 이전 대화를 츌력 - chat_message에는 role과 content가 들어가 있다.
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)
        # st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메세지를 추가 - ChatMessage은 langchain_core.messages.chat에 있다.
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 세션 기록을 저장할 딕셔너리
# 세션 ID를 기반으로 세션 기록을 가져오는 함수.
def get_session_history(session_ids):
    if session_ids not in st.session_state["store"]:
        st.session_state["store"][session_ids] = ChatMessageHistory()
    return st.session_state["store"][session_ids]


# 체인생성
def create_chain(model_name="gpt-4o-mini"):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 Question-Answering Chatbot입니다. 주어진 질문에 대한 답변을 제공해 주세요,",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "#Question:\n{question}"),
        ]
    )
    # model_name으로 넣어줘야 동적으로 사용가능.
    llm = ChatOpenAI(temperature=0, model=model_name)

    chain = prompt | llm | StrOutputParser()

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,  # 세션기록을 저장하는 함수,
        input_messages_key="question",  # 사용자의 질문이 템플릿 변수에 들어갈 key
        history_messages_key="chat_history",
    )

    return chain_with_history


# 초기화 버튼이 눌렸을 때
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요.")

# 경고메세지를 뜨우기 위한 빈 영역
warning_msg = st.empty()

if "chain" not in st.session_state:
    st.session_state["chain"] = create_chain(model_name=selected_model)

# 만약 사용자의 입력이 들어오면,
if user_input:
    chain = st.session_state["chain"]
    if chain:
        response = chain.stream(
            {"question": user_input},
            config={"configurable": {"session_id": session_id}},
        )
        # 사용자의 입력
        st.chat_message("user").write(user_input)

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
        # 이미지를 업로드 하라는 경고 메세지
        warning_msg.error("이미지를 업로드 해주세요")

# poetry shell로 가상환경 실행
# streamlit run .\02.Local_lag.py
