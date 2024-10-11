import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import ChatMessage


load_dotenv()

if not os.path.exists(".cache"):
    os.mkdir("./cache")

if not os.path.exists(".cache/files"):
    os.mkdir("./cache/files")

if not os.path.exists(".embeddings"):
    os.mkdir("./embeddings")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "chain" not in st.session_state:
    st.session_state["chain"] = None


def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(chat_message.content)

def add_message(role, message):
    st.session_state["message"].append(ChatMessage(role=role, content=message))

@st.cache_resource(show_spinner= "업로드한 파일을 처리해 주세요")
def embed_file(file):
    file_content = file.read()
    file_path = f".cache/files/file"