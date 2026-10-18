# SKKU RAG 튜토리얼 — 소재 연구자를 위한 RAG · 데이터베이스 · 에이전트 입문

프로그래밍이나 컴퓨터과학을 따로 배우지 않은 **소재(재료) 연구자**를 위한 실습 자료입니다.
논문 PDF 2편을 데이터로 삼아 다음 개념을 하나씩 직접 만들어 보며 익힙니다.

- **RAG**(검색 증강 생성)
- 세 가지 데이터베이스: **관계형 DB**, **벡터 DB**, **그래프 DB**
- **에이전트**

마지막에는 **Vector RAG**, **Graph RAG**, **Agentic RAG**를 처음부터 구현합니다.

> 모든 LLM은 [Ollama](https://ollama.com)로 **내 컴퓨터(또는 Colab)에서 무료로** 실행합니다. API 키나 비용이 필요 없습니다.

## 학습 순서

| # | 노트북 | 배우는 것 | Colab |
|---|---|---|---|
| 00 | [00_Setup.ipynb](00_Setup.ipynb) | 실습 환경 준비 (Colab / 내 컴퓨터) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JDeun/skku_rag_tutorials/blob/main/00_Setup.ipynb) |
| 01 | [01_What_is_RAG.ipynb](01_What_is_RAG.ipynb) | LLM의 한계와 RAG의 아이디어 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JDeun/skku_rag_tutorials/blob/main/01_What_is_RAG.ipynb) |
| 02 | [02_RDB_SQLite.ipynb](02_RDB_SQLite.ipynb) | 관계형 DB와 SQL, Text2SQL | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JDeun/skku_rag_tutorials/blob/main/02_RDB_SQLite.ipynb) |
| 03 | [03_VectorDB.ipynb](03_VectorDB.ipynb) | 임베딩과 벡터 DB, 의미 검색 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JDeun/skku_rag_tutorials/blob/main/03_VectorDB.ipynb) |
| 04 | [04_GraphDB_Kuzu.ipynb](04_GraphDB_Kuzu.ipynb) | 그래프 DB와 Cypher | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JDeun/skku_rag_tutorials/blob/main/04_GraphDB_Kuzu.ipynb) |
| 05 | [05_Agent.ipynb](05_Agent.ipynb) | 도구를 쓰는 LLM, 에이전트 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JDeun/skku_rag_tutorials/blob/main/05_Agent.ipynb) |
| 06 | [06_Vector_RAG.ipynb](06_Vector_RAG.ipynb) | **Vector RAG** 구현 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JDeun/skku_rag_tutorials/blob/main/06_Vector_RAG.ipynb) |
| 07 | [07_Graph_RAG.ipynb](07_Graph_RAG.ipynb) | **Graph RAG** 구현 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JDeun/skku_rag_tutorials/blob/main/07_Graph_RAG.ipynb) |
| 08 | [08_Agentic_RAG.ipynb](08_Agentic_RAG.ipynb) | **Agentic RAG** 구현 | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JDeun/skku_rag_tutorials/blob/main/08_Agentic_RAG.ipynb) |

노트북 01~05는 **개념편**, 06~08은 **구현편**입니다.
**데이터베이스(SQLite, Chroma, Kùzu)는 미리 만들어 두지 않았습니다.** 노트북을 순서대로 실행하면서 여러분이 직접 만들게 됩니다. 특히 08은 02·06·07에서 만든 DB를 모두 사용합니다. 노트북에서 막힐 때 함께 볼 읽기 자료는 [`docs/`](docs/README.md)에 있습니다.

- [자료구조 기초](docs/01_자료구조_기초.md): 리스트, 딕셔너리, 벡터, 그래프
- [데이터베이스 기초](docs/02_데이터베이스_기초.md): 관계형 DB, NoSQL, 벡터 DB, 그래프 DB 비교

```
            ┌──────────── 개념편 ────────────┐      ┌────────── 구현편 ──────────┐
 00 환경 → 01 RAG → 02 RDB → 03 VectorDB → 04 GraphDB → 05 Agent → 06 Vector RAG → 07 Graph RAG → 08 Agentic RAG
                      │          │              │          │          ▲              ▲              ▲
                      │          └──────────────┼──────────┼──────────┘              │              │
                      │                         └──────────┼─────────────────────────┘              │
                      └────────────────────────────────────┴────────────────────────────────────────┘
```

## 시작하기

### 방법 A. Google Colab (설치 없음, 권장)

1. 위 표의 **Open in Colab** 배지를 누릅니다.
2. 메뉴에서 `런타임 → 런타임 유형 변경 → T4 GPU`를 선택합니다.
3. 첫 번째 코드 셀을 실행하면 저장소, 패키지, Ollama, 모델이 자동으로 설치됩니다. 처음 한 번은 5~10분 정도 걸립니다.

### 방법 B. 내 컴퓨터 (로컬 Jupyter)

1. [Ollama 설치](https://ollama.com/download) 후 모델 내려받기
   ```bash
   ollama pull qwen3.5:4b       # 답변을 만드는 언어모델 (약 4GB)
   ollama pull embeddinggemma   # 문장을 벡터로 바꾸는 임베딩 모델 (약 0.6GB)
   ```
2. **Python 3.10~3.13** 준비 (권장 3.12). Kùzu 0.11.3은 macOS/Windows용 Python 3.14를 지원하지 않습니다.
3. 저장소 받기와 패키지 설치
   ```bash
   git clone https://github.com/JDeun/skku_rag_tutorials.git
   cd skku_rag_tutorials
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt jupyterlab
   jupyter lab
   ```

자세한 설명과 문제 해결 방법은 [00_Setup.ipynb](00_Setup.ipynb)에 있습니다.

### 모델 바꾸기

모델 이름은 [`skku_rag/config.py`](skku_rag/config.py) 한 곳에서만 관리합니다.
환경변수로도 바꿀 수 있습니다.

```bash
export SKKU_LLM_MODEL=qwen3.5:9b        # 사양이 좋은 PC라면 더 큰 모델로
export SKKU_EMBED_MODEL=embeddinggemma  # 임베딩 모델을 바꾸면 벡터 DB를 다시 만들어야 합니다
```

## 사용 기술

| 역할 | 사용 도구 |
|---|---|
| 언어모델 / 임베딩 | Ollama (`qwen3.5:4b`, `embeddinggemma`) |
| 관계형 DB | SQLite (파이썬 내장) |
| 벡터 DB | Chroma |
| 그래프 DB | Kùzu 0.11.3 (설치가 간단한 내장형 그래프 DB, Cypher 사용) |
| RAG / 에이전트 프레임워크 | LangChain, LangGraph |

> **Kùzu 안내**: Kùzu 프로젝트는 2025년 10월 개발이 중단되었고, 0.11.3이 마지막 버전입니다.
> 학습용으로는 안정적이며, 여기서 배우는 Cypher 문법은 Neo4j 등 다른 그래프 DB에서도 거의 그대로 쓸 수 있습니다.

## 폴더 구조

```
skku_rag_tutorials/
├── 00_Setup.ipynb ~ 08_Agentic_RAG.ipynb   # 실습 노트북
├── docs/                 # 읽기 자료 (자료구조, 데이터베이스 기초)
├── paper_data/           # 실습 데이터: 논문 PDF 2편 + 논문 표를 옮겨 적은 CSV
├── skku_rag/             # 노트북에서 반복되는 준비 작업을 모아 둔 도우미 코드
├── scripts/colab_setup.sh
├── db/                   # 여러분이 노트북을 실행하며 직접 만드는 데이터베이스 (처음엔 비어 있음)
└── requirements.txt
```

## 실습 데이터

| 논문 | 내용 |
|---|---|
| G. Kim, X. Chai, L. Yu, X. Cheng, D. S. Gianola, *Interplay between grain boundary segregation and electrical resistivity in dilute nanocrystalline Cu alloys*, **Scripta Materialia** (2016) | 나노결정 Cu-Nb, Cu-Fe 합금의 결정립계 편석과 비저항 |
| J. Y. Sung et al., *Recent progress in alternative metals for advanced interconnects*, **J. Mater. Chem. C** 14, 6558 (2026) | Cu 배선을 대체할 금속(Ru, Co, Mo, 금속간화합물, 위상 준금속) 리뷰 |

논문의 저작권은 각 출판사와 저자에게 있습니다. 이 저장소의 PDF는 수업 실습 목적으로만 사용하세요.

## 참고 자료

- 테디노트, [〈랭체인LangChain 노트〉 - LangChain 한국어 튜토리얼](https://wikidocs.net/book/14314) / [GitHub langchain-kr](https://github.com/teddylee777/langchain-kr)
- 공원나연, [Neo4j GraphRAG 파이썬 패키지 가이드북](https://wikidocs.net/book/16760)
- [Ollama](https://ollama.com) · [LangChain](https://python.langchain.com) · [LangGraph](https://langchain-ai.github.io/langgraph/) · [Chroma](https://docs.trychroma.com) · [Kùzu 문서](https://kuzudb.github.io/docs)
