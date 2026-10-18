"""벡터DB(Chroma) 만들기·불러오기 도우미 (06_Vector_RAG 에서 자세히 설명).

흐름: 논문 로드 → 머리글/바닥글 청소(clean) → 조각 나누기 → 임베딩 → Chroma 저장.
한 번 만들어 두면 db/chroma 폴더에 저장되어, 다음부터는 바로 불러옵니다.
"""
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from skku_rag.config import DB_DIR, get_embeddings
from skku_rag.data import load_papers, split_documents

CHROMA_DIR = DB_DIR / "chroma"
COLLECTION_NAME = "papers"

# 모든 페이지에 반복해서 찍히는 머리글/바닥글(저널 이름, 쪽 번호, 저작권 문구) 패턴.
# 내용과 무관한 문구가 조각마다 섞이면 검색 품질이 떨어지므로 지웁니다.
_BOILERPLATE_PATTERNS = [
    r"This journal is © The Royal Society of Chemistry \d{4}",
    r"(\d+ \| )?J\. Mater\. Chem\. C, \d{4}, \d+, [\d–-]+( \| \d+)?",
    r"Journal of Materials Chemistry C",
    r"Review",
    r"G\. Kim et al\. / Scripta Materialia \d+ \(\d{4}\) [\d–-]+",
]
_BOILERPLATE_RE = re.compile(r"^\s*(?:" + "|".join(_BOILERPLATE_PATTERNS) + r")\s*$")


def clean_text(text: str) -> str:
    """페이지 텍스트에서 반복되는 머리글/바닥글 줄만 골라 지운 새 문자열을 돌려줍니다."""
    lines = [line for line in text.splitlines() if not _BOILERPLATE_RE.match(line)]
    return "\n".join(lines)


def clean_documents(docs: list[Document]) -> list[Document]:
    """각 Document 의 본문을 clean_text 로 청소한 새 Document 목록을 돌려줍니다(원본은 그대로)."""
    return [
        Document(page_content=clean_text(d.page_content), metadata=dict(d.metadata))
        for d in docs
    ]


def load_chunks(chunk_size: int = 1000, chunk_overlap: int = 100) -> list[Document]:
    """논문 로드 → 청소 → 조각 나누기까지 한 번에 수행합니다."""
    return split_documents(clean_documents(load_papers()), chunk_size, chunk_overlap)


def format_docs(docs: list[Document]) -> str:
    """검색된 조각들을 '[출처: 파일명, p.쪽]' 머리표와 함께 하나의 문자열로 합칩니다.

    page 메타데이터는 0부터 세므로, 사람이 읽는 쪽 번호로 +1 해서 표시합니다.
    """
    blocks = []
    for d in docs:
        name = Path(d.metadata.get("source", "?")).name
        page = d.metadata.get("page", -1) + 1
        blocks.append(f"[출처: {name}, p.{page}]\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def get_vectorstore(rebuild: bool = False) -> Chroma:
    """논문 조각이 저장된 Chroma 벡터DB를 돌려줍니다.

    db/chroma 에 이미 만들어져 있으면 그대로 불러오고(빠름),
    없거나 rebuild=True 이면 처음부터 임베딩해서 새로 만듭니다(1~3분).
    """
    store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
    )
    if rebuild:
        store.reset_collection()  # 기존 내용을 비우고 같은 이름으로 다시 만듦
    if store._collection.count() == 0:
        chunks = load_chunks()
        ids = [f"chunk-{i:04d}" for i in range(len(chunks))]
        store.add_documents(chunks, ids=ids)
    return store
