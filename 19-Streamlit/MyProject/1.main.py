import streamlit as st
from langchain_core.messages.chat import ChatMessage

st.title("방이 Chat GPT Test")


# 처음 1번만 실행하기 위한 코드
if "messages" not in st.session_state:
    # 대화내용을 저장하는 기능
    st.session_state["messages"] = []


# 이전 대화내용을 출력, 프린트 함수.
# for role, message in st.session_state["messages"]:
#     st.chat_message(role).write(message)
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)
        # st.write(f"{chat_message.role}: {chat_message.comtent}")


# 새로운 메세지를 추가하는 함수를 만듬.
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


print_messages()


# 사용자 입력창
user_input = st.chat_input("궁금한 것을 물어보세요")

# 만약 사용자 입력이 들어오면..
# 저장, message의 경우 contrainer안에 담아주는 역할.
if user_input:
    # st.write(f"사용자 입력: {user_input}")
    st.chat_message("user").write(user_input)  # 입력
    st.chat_message("assistant").write(user_input)  # ai도 그대로 입력

    # 대화내용을 저장. 위의 st.session_state["messages"] = []를 받는다.
    add_message("user", user_input)
    add_message("assistant", user_input)
    # ChatMessage(role="user", content=user_input)
    # ChatMessage(role="assistant", content=user_input)
    # st.session_state["messages"].append(("user", user_input))
    # st.session_state["messages"].append(("assistant", user_input))

## Terminal창에서 streamlit을 켜준다.
# streamlit run .\19-Streamlit\MyProject\main.py
