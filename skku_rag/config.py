"""모델 이름과 폴더 경로 설정.

다른 모델을 쓰고 싶다면 이 파일의 LLM_MODEL / EMBED_MODEL 만 바꾸거나,
환경변수로 덮어쓰면 됩니다. 예) 터미널에서 `export SKKU_LLM_MODEL=qwen3.5:9b`
모델을 바꾼 뒤에는 `ollama pull <모델이름>` 으로 먼저 내려받아야 합니다.
"""
import os
from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings

# 대화(답변 생성)에 쓰는 언어모델. 사양이 좋은 PC라면 더 큰 모델(예: qwen3.5:9b)로 바꿔 보세요.
LLM_MODEL = os.getenv("SKKU_LLM_MODEL", "qwen3.5:4b")
# 문장을 숫자 벡터로 바꾸는 임베딩 모델. 바꾸면 벡터DB를 처음부터 다시 만들어야 합니다.
EMBED_MODEL = os.getenv("SKKU_EMBED_MODEL", "embeddinggemma")
# Ollama 서버 주소 (보통 바꿀 필요 없음)
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PAPER_DIR = PROJECT_ROOT / "paper_data"
DB_DIR = PROJECT_ROOT / "db"
DB_DIR.mkdir(exist_ok=True)


def get_llm(temperature: float = 0, **kwargs) -> ChatOllama:
    """설정된 언어모델을 돌려줍니다.

    기본값 reasoning=False 로 '생각 과정(thinking)' 출력을 끄면 답이 훨씬 빨리 나옵니다.
    생각 과정을 켜 보고 싶다면 get_llm(reasoning=True) 처럼 부르세요.
    """
    kwargs.setdefault("reasoning", False)
    return ChatOllama(
        model=LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        temperature=temperature,
        **kwargs,
    )


def get_embeddings() -> OllamaEmbeddings:
    """설정된 임베딩 모델을 돌려줍니다."""
    return OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
