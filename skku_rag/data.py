"""논문 PDF 불러오기와 조각(chunk) 나누기."""
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from skku_rag.config import PAPER_DIR


def load_papers(paper_dir: Path = PAPER_DIR) -> list[Document]:
    """paper_data 폴더의 모든 PDF를 페이지 단위 Document 목록으로 읽어옵니다."""
    pdf_paths = sorted(Path(paper_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"{paper_dir} 폴더에 PDF 파일이 없습니다.")
    return [page for path in pdf_paths for page in PyMuPDFLoader(str(path)).load()]


def split_documents(
    docs: list[Document], chunk_size: int = 1000, chunk_overlap: int = 100
) -> list[Document]:
    """긴 문서를 검색하기 좋은 크기의 조각으로 나눕니다."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)
