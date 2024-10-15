import streamlit as st
from langchain_core.messages.chat import ChatMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_teddynote.prompts import load_prompt
from langchain import hub
from dotenv import load_dotenv

load_dotenv()

st.title("Pratice streamlit ChatGPT💬")


# 처음 1번만 실행하기 위한 코드, st.session_state: 변수를 저장하기 위함.
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# 사이드바
with st.sidebar:
    # 대화 초기화 버튼
    clear_btn = st.button("대화 초기화")

    # 셀렉트박스
    selected_prompt = st.selectbox(
        "프롬프트를 선택해주세요", ("기본모드", "SNS 게시글", "요약"), index=0
    )


# 이전 대화를 출력
def print_messages():
    for chat_message in st.session_state["messages"]:
        st.chat_message(chat_message.role).write(
            chat_message.content
        )  # 롤을 작성하고 내용을 작성.


# 새로운 메세지를 추가
# from langchain_core.messages.chat import ChatMessage
def add_message(role, message):
    st.session_state["messages"].append(ChatMessage(role=role, content=message))


def create_chain(prompt_type):

    # 프롬프트-기본모드
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 친절한 AI 어시스턴트 입니다. 다음의 질문에 간결하게 답변해 주세요.",
            ),
            (
                "user",
                """
                "#Context: \n {context}"
                "#Question: \n {question}"
                """,
            ),
        ]
    )

    if prompt_type == "SNS 게시글":
        prompt = load_prompt("prompts/sns.yaml", encoding="utf-8")

    elif prompt_type == "요약":
        prompt = hub.pull("teddynote/chain-of-density-korean:946ed62d")

    # Load data
    docs = PyPDFDirectoryLoader("data/RAG")
    docs = docs.load()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    documents = text_splitter.split_documents(docs)

    # Embedding
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Vector index
    vector_store = FAISS.from_documents(documents, embeddings)
    retriever = vector_store.as_retriever()

    # Model
    llm = ChatOpenAI(model_name="gpt-4o-mini")

    # 출력파서
    output_parser = StrOutputParser()
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )
    return chain


# 초기화 버튼이 눌리면
if clear_btn:
    st.session_state["messages"] = []

# 이전 대화기록 출력
print_messages()

# 사용자의 입력
user_input = st.chat_input("궁금한 내용을 물어보세요!")

# 만약, 사용자의 입력이 들어오면
if user_input:
    # 사용자의 입력
    st.chat_message("user").write(user_input)
    # chain을 생성
    chain = create_chain(selected_prompt)

    # 스트리밍 호출
    # response = chain.stream({"question": user_input})
    response = chain.stream(user_input)
    with st.chat_message("assistant"):

        # 빈 공간(컨테이너를) 만들어서, 여기에 토큰을 생성한다.
        container = st.empty()

        ai_answer = ""
        for token in response:
            ai_answer += token
            container.markdown(ai_answer)

    # 대화 기록을 저장한다.
    add_message("user", user_input)
    add_message("assistant", ai_answer)


# https://github.com/teddylee777/langchain-kr/blob/main/19-Streamlit/01-MyProject/main.py
# streamlit run 19-Streamlit\MyProject\main.py
