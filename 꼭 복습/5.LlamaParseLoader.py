import os
from typing import Iterator
from typing import Any, Dict, List, Optional
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader
import nest_asyncio

nest_asyncio.apply()


class LlamaParseLoader(BaseLoader):
    """파일을 한 줄씩 읽어오는 문서 로더의 예시입니다."""

    def __init__(self, file_paths: List[str], parsing_instructions="") -> None:
        """로더를 파일 경로와 함께 초기화합니다.
        Args:
            file_paths: 로드할 파일의 경로입니다.
        """
        # LlamaParse 설정
        parser = LlamaParse(
            # api_key="llx-...",  # API 키 (환경 변수 LLAMA_CLOUD_API_KEY에 저장 가능)
            result_type="markdown",  # 결과 타입: "markdown" 또는 "text"
            num_workers=4,  # 여러 파일 처리 시 API 호출 분할 수
            verbose=True,
            language="ko",  # 언어 설정 (기본값: 'en')
            invalidate_cache=True,
            skip_diagonal_text=True,
            use_vendor_multimodal_model=True,
            vendor_multimodal_model_name="openai-gpt4o",
            vendor_multimodal_api_key=os.environ.get("OPENAI_API_KEY"),
            parsing_instruction=parsing_instructions,
        )

        file_extractor = {".pdf": parser}

        self.document_reader = SimpleDirectoryReader(
            input_files=file_paths,
            file_extractor=file_extractor,
        )

    def lazy_load(self) -> Iterator[Document]:  # <-- 인자를 받지 않습니다
        """파일을 한 줄씩 읽어오는 지연 로더입니다.

        지연 로드 메소드를 구현할 때는, 문서를 하나씩 생성하여 반환하는 제너레이터를 사용해야 합니다.
        """
        documents = self.document_reader.load_data()
        langchain_documents = [doc.to_langchain_format() for doc in documents]
        return langchain_documents
