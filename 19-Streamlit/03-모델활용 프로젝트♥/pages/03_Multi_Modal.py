import os
import streamlit as st
from dotenv import load_dotenv
from langchain_teddynote import logging
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages.chat import ChatMessage
from langchain_teddynote.prompts import load_prompt
from langchain_teddynote.models import MultiModal
from langchain_teddynote.messages import stream_response

from retriever import create_retriever

load_dotenv()

logging.langsmith("[Project] 이미지 인식")

# 캐시 디렉토리 생성
if not os.path.exists(".cache"):
    os.mkdir(".cache")

# 파일 업로드 전용 폴더
if not os.path.exists(".cache/files"):
    os.mkdir(".cache/files")

if not os.path.exists(".cache/embeddings"):
    os.mkdir(".cache/embeddings")

st.title("이미지 인식 기반 챗봇💬")

# 처음 1번만 실행하기 위한 코드. - 대화기록을 저장하기 위한 코드로 생성한다.
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 탭을 생성
main_tab1, main_tab2 = st.tabs(["이미지", "대화내용"])


# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 파일 업로드
    uploaded_file = st.file_uploader("이미지파일 업로드", type=["jpg", "jpeg", "png"])

    # 모델 메뉴 선택
    selected_model = st.selectbox("model선택", ["gpt-4o-mini", "gpt-4o"])

    # 시스템 프롬프트 추가
    system_prompt = st.text_area(
        "시스템 프롬프트",
        "당신은 표(재무제표)를 해석하는 금융 AI어시스턴트 입니다. \n당신의 임무는 주어진 테이블 형식의 재무제표를 바탕으로 흥미로운 사실을 정리하여 친절하게 답변하는것 입니다.",
        height=200,
    )


# 이전 대화를 츌력 - chat_message에는 role과 content가 들어가 있다.
def print_messages():
    for chat_message in st.session_state["messages"]:
        main_tab2.chat_message(chat_message.role).write(chat_message.content)
        # st.chat_message(chat_message.role).write(chat_message.content)


# 새로운 메세지를 추가 - ChatMessage은 langchain_core.messages.chat에 있다.
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 이미지를 캐시 저장(시간이 오래 걸리는 작업을 처리할 예정)
@st.cache_resource(show_spinner="업로드한 이미지를 처리중입니다.")
def process_imagefile(file):
    # 업로드한 파일을 캐시 디렉토리에 저장
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"

    with open(file_path, "wb") as f:
        f.write(file_content)

    # create_retriever를 retriever로부터 가져온다.
    return file_path


# 이미지에 대한 대답 생성
def generate_answer(
    image_filepath, system_prompt, user_prompt, model_name="gpt-4o-mini"
):
    # model_name으로 넣어줘야 동적으로 사용가능.
    llm = ChatOpenAI(temperature=0, model=model_name)

    # 멀티모댈 객체 생성
    multimodal = MultiModal(llm, system_prompt=system_prompt, user_prompt=user_prompt)

    # 이미지 파일로 부터 질의(스트림 방식)
    answer = multimodal.stream(image_filepath)
    return answer


# 초기화 버튼이 눌렸을 때
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요.")

# 경고메세지를 뜨우기 위한 빈 영역
warning_msg = main_tab2.empty()

# 이미지가 업로드가 된다면
if uploaded_file:
    # 이미지 파일을 처리
    image_filepath = process_imagefile(uploaded_file)
    main_tab1.image(image_filepath)
    # st.image(image_filepath)

# 만약 사용자의 입력이 들어오면,
if user_input:
    # 파일이 업로드 되었는지 확인
    if uploaded_file:
        # process_imagefile 함수를 통해 file uploaded : 이미지 파일 처리
        image_filepath = process_imagefile(uploaded_file)
        # # 이미지 파일을 업로드 했다는 메세지 출력
        # warning_msg.success("이미지 파일을 업로드했습니다.")
        # generate_answer 함수사용 : 답변요청
        response = generate_answer(
            image_filepath, system_prompt, user_input, selected_model
        )

        # 사용자의 입력
        main_tab2.chat_message("user").write(user_input)

        with main_tab2.chat_message("assistant"):
            # 빈 컨테이너를 만들어서, 여기에 토큰을 스트리밍
            container = st.empty()

            ai_answer = ""
            for token in response:
                ai_answer += token.content
                container.markdown(ai_answer)

        # 대화기록을 저장
        add_message("user", user_input)
        add_message("assistant", ai_answer)

    else:
        # 이미지를 업로드 하라는 경고 메세지
        warning_msg.error("이미지를 업로드 해주세요")

# poetry shell로 가상환경 실행
# streamlit run .\02.Local_lag.py
