#!/usr/bin/env bash
# Google Colab 전용 설치 스크립트: 파이썬 패키지 + Ollama + 모델 다운로드
# 여러 번 실행해도 안전합니다(이미 설치된 것은 건너뜀).
set -e
cd "$(dirname "$0")/.."

echo "[1/4] 파이썬 패키지 설치"
pip install -q -r requirements.txt

echo "[2/4] Ollama 설치"
if ! command -v ollama >/dev/null 2>&1; then
  apt-get update -qq >/dev/null && apt-get install -y -qq zstd pciutils >/dev/null
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo "[3/4] Ollama 서버 실행"
if ! curl -s http://localhost:11434 >/dev/null; then
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  # 서버가 응답할 때까지 최대 60초 기다립니다 (고정 5초 대기는 첫 실행 때 부족할 수 있음)
  for _ in $(seq 1 60); do
    curl -s http://localhost:11434 >/dev/null && break
    sleep 1
  done
fi

echo "[4/4] 모델 다운로드 (처음 한 번만 몇 분 걸립니다)"
LLM=$(python -c "from skku_rag.config import LLM_MODEL; print(LLM_MODEL)")
EMB=$(python -c "from skku_rag.config import EMBED_MODEL; print(EMBED_MODEL)")
ollama pull "$LLM"
ollama pull "$EMB"
echo "완료! LLM=$LLM, EMBED=$EMB"
