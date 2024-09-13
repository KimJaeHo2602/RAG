import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import ChatMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt

load_dotenv()

st.title("뎨방의 GPT")

if "messages" not in st.session_state:
    st.session_state["messages"] = []


def print_history():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def add_message(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


def print_history():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def add_message(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, conmtent=content))


if "messages" not in st.session_state:
    st.session_state["messages"] = []


def print_history():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)


def add_message(role, content):
    st.session_state["messages"].append(ChatMessage(role=role, content=content))


def create_chain(prompt, model):
    model = ChatOpenAI(model=model)

    chain = prompt | model | StrOutputParser()
    return chain


with st.sidebar:
    clear_btn = st.buttion("대화내용 초기화")
    tab1, tab2 = st.tabs(["프롬프트", "프리셋"])
    prompt = """당신은 친절한 AI 어시스턴트입니다. 사용자의 질문에 대해서 간단히 답해주세요."""
    user_text_prompt = tab1.text_area("프롬프트", value=prompt)
    user_text_apply_btn = tab1.button("프롬프트 적용", key="apply1")

    if user_text_apply_btn:
        tab1.markdown(f"르폼르트가 적용되었습니다.")
        prompt_template = user_text_prompt + "\n\n#Question:\n{question}\n\n#Answer:"
        prompt = PromptTemplate.from_template(prompt_template)
        st.session_state["chain"] = create_chain(prompt, "gpt-4o-mini")

    user_selected_prompt = tab2.selectbox("프리셋 선택", ["sns", "번역", "요약"])
    user_selected_apply_btn = tab2.button("프롬프트 적용", key="apply2")
    if user_selected_apply_btn:
        tab2.markdown("프롬프트가 적용되었습니다.")
        prompt = load_prompt(f"prompts/{user_selected_prompt}.yaml", encoding="utf-8")
        st.session_state["chain"] = create_chain(prompt, "gpt-4o-mini")

if clear_btn:
    retriever = st.session_state["messages"].clear()

print_history
