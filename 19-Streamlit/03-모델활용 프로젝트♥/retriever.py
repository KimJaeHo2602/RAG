from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# Indexing
def create_retriever(file_path):
    # 1단계: 문서 로드
    loader = PDFPlumberLoader(file_path)
    docs = loader.load()

    # 2단계: 문서 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    documents = text_splitter.split_documents(docs)

    # 3단계: 임베딩 생성
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 4단계: 벡터 스토어 생성
    vector_store = FAISS.from_documents(documents, embeddings)

    # 5단계: 검색기(Retriever) 생성
    retriever = vector_store.as_retriever()
    return retriever
