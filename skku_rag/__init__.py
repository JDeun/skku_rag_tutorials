"""SKKU RAG 튜토리얼 공용 도우미 모듈.

노트북에서 반복되는 설정(모델 이름, 경로, 논문 로드)을 한 곳에 모아 둔 것입니다.
핵심 개념은 모두 노트북 안에서 직접 구현하며, 이 모듈은 '반복되는 준비 작업'만 맡습니다.
"""
from skku_rag.config import (
    DB_DIR,
    EMBED_MODEL,
    LLM_MODEL,
    OLLAMA_BASE_URL,
    PAPER_DIR,
    PROJECT_ROOT,
    get_embeddings,
    get_llm,
)
from skku_rag.data import load_papers, split_documents

__all__ = [
    "DB_DIR",
    "EMBED_MODEL",
    "LLM_MODEL",
    "OLLAMA_BASE_URL",
    "PAPER_DIR",
    "PROJECT_ROOT",
    "get_embeddings",
    "get_llm",
    "load_papers",
    "split_documents",
]
