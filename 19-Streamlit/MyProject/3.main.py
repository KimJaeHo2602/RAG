import streamlit as st

from langchain_core.messages.chat import ChatMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain import hub

from dotenv import load_dotenv

# API_KEY 정보 로드
load_dotenv()

st.title("방이 Chat GPT Test")


# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화내용을 저장하는 기능
    st.session_state["messages"] = []

# 사이드바 생성
with st.sidebar:
    # 초기화 버튼 생성
    clear_btn = st.button("대화 초기화")

    # 셀렉트 박스 만들기. 기본모드 , SNS 게시글, 요약 3개가 출력됌.
    selected_prompt = st.selectbox(
        "프롬프트를 선택해 주세요",
        ("기본모드", "SNS 게시글", "요약"),
        index=0,
    )


# 이전 대화내용을 출력, 프린트 함수.
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)
        # 역할과, 내용만 출력


# 새로운 메세지를 추가하는 함수를 만듬.
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


# 체인생성 - 매개변수에 prompt추가
def create_chain(prompt_type):
    # prompt | llm | output_parser
    # prompt (기본모드)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 친절한 AI 어시스턴트 입니다. 다음의 질문에 간결하게 담변해 주세요.",
            ),  # 전역변수로, 지시사항
            ("user", "#Question:\n{question}"),  # 입력
        ]
    )

    # SNS 마케터 프롬프트
    if prompt_type == "SNS 게시글":
        prompt = load_prompt(
            "19-Streamlit/MyProject/prompts/sns.yaml", encoding="utf-8"
        )
        # load_prompt("prompts/sns.yaml", encoding="utf-8")

    # 19-Streamlit\MyProject\prompts\sns.yaml

    # 요약 prompt
    elif prompt_type == "요약":
        prompt = hub.pull("teddynote/chain-of-density-korean:946ed62d")

    # model
    # llm = ChatOpenAI(model_name="gpt-4o", temperature=0)
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # 출력 파서
    output_parser = StrOutputParser()

    # 체인 생성
    chain = prompt | llm | output_parser

    return chain


### 초기화 버튼이 눌리면, 빈 list를 만들어준다.
if clear_btn:
    st.session_state["messages"] = []

### 이전 대화 기록 출력
print_messages()


# 사용자 입력창
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 만약 사용자 입력이 들어오면..
# 저장, message의 경우 contrainer안에 담아주는 역할.
if user_input:
    # 사용자의 입력
    st.chat_message("user").write(user_input)  # 입력

    # chain을 생성
    chain = create_chain(selected_prompt)
    # ai_answer = chain.invoke({"question": user_input})

    ## 스트리밍 호출:한글자씩 출력하기 위한 stream이용
    response = chain.stream({"question": user_input})
    with st.chat_message("assistant"):  # 질문을
        # 빈 공간(컨테이너)을 만들어서, 여기에 토큰을 스트리밍 출력한다.
        container = st.empty()

        ai_answer = ""
        for token in response:  # token을 하나씩 받는다.
            ai_answer += token  # answer에다가 token을 누적시킨다.
            container.markdown(ai_answer)  # ai_answer를 markdown형식으로 저장.

    # 대화내용을 저장. 위의 st.session_state["messages"] = []를 받는다.
    add_message("user", user_input)
    add_message("assistant", ai_answer)
    # add_message("assistant", user_input)

## Terminal창에서 streamlit을 켜준다.
# streamlit run 19-Streamlit\MyProject\main.py
