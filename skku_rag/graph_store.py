"""논문 지식그래프(Kùzu) 만들기·불러오기·검색 도우미 (07_Graph_RAG 담당).

- 추출 결과: `DB_DIR/"kg_triples.json"` — LLM 이 논문 조각마다 뽑은 개체·관계.
  07 노트북(또는 extract_knowledge_graph())으로 학생이 직접 만듭니다. (git 에 올리지 않음)
- 그래프 DB 파일: `DB_DIR/"paper_graph.kuzu"` — get_graph() 가 위 JSON 에서 만듭니다.

그래프 구조(스키마)
    (Chunk)-[:PART_OF]->(Paper)            조각은 어느 논문의 일부인가
    (Chunk)-[:MENTIONS]->(Entity)          조각이 언급하는 개체(재료·물성·현상 …)
    (Entity)-[:RELATED_TO {relation}]->(Entity)   개체 사이의 관계 (예: Ru -ALTERNATIVE_TO-> Cu)

다른 노트북(08 Agentic RAG 등)에서 쓰는 함수
    extract_knowledge_graph(max_chunks=None) -> Path  (LLM 추출, 중단 후 이어하기 가능, 약 1시간)
    get_graph(rebuild=False)                 -> kuzu.Connection
    find_entities(conn, keywords, limit=20)  -> pandas.DataFrame[name, type, degree]
    get_neighbors(conn, names, hops=1, relations=None) -> pandas.DataFrame[source, relation, target, chunk_id, hop]
    chunks_for_entities(conn, names, limit=5)-> pandas.DataFrame[id, paper_id, page, text, hits]
    run_cypher(conn, query, params=None)     -> pandas.DataFrame (읽기 전용 검사 포함)
"""
from __future__ import annotations

import json
import re
import shutil
import time
from collections import Counter
from pathlib import Path

import kuzu
import pandas as pd

from skku_rag.config import DB_DIR, LLM_MODEL, get_llm
from skku_rag.data import load_papers, split_documents

GRAPH_DB_PATH = DB_DIR / "paper_graph.kuzu"
TRIPLES_PATH = DB_DIR / "kg_triples.json"   # LLM 추출 결과 (학생이 07 에서 직접 생성, git 에 올리지 않음)
CHUNK_SIZE, CHUNK_OVERLAP = 1500, 150         # 추출 호출 수를 줄이려고 03/06(1000자)보다 크게

# 개체 종류(type)와 관계 종류(relation) — 추출 프롬프트와 Text2Cypher 가 함께 씁니다.
ENTITY_TYPES = ["Material", "Property", "Phenomenon", "Process", "Application", "Method"]
RELATION_TYPES = [
    "ALTERNATIVE_TO",  # A 는 B 를 대체할 후보다        (Ru -> Cu)
    "HAS_PROPERTY",    # 재료 A 는 물성 B 를 가진다      (Ru -> low EMFP)
    "INCREASES",       # A 는 B 를 증가시킨다            (grain boundary scattering -> resistivity)
    "DECREASES",       # A 는 B 를 감소시킨다
    "CAUSES",          # A 는 B 를 일으킨다              (size effect -> resistivity increase)
    "CONTAINS",        # 합금/화합물 A 는 원소 B 를 포함  (NiAl -> Ni)
    "USED_IN",         # A 는 B 에 쓰인다                (Cu -> interconnect)
    "PROCESSED_BY",    # A 는 공정 B 로 만들어진다/가공된다 (Co -> dry etching)
    "RELATED_TO",      # 위에 딱 맞지 않는 기타 관계
]

GRAPH_SCHEMA = """\
[노드(Node) 테이블]
- Paper(id STRING 기본키, title STRING, year INT64)      : 논문. id 예) 'kim2016', 'sung2026'
- Chunk(id STRING 기본키, paper_id STRING, page INT64, text STRING) : 논문 조각(영어 원문). page 는 1부터
- Entity(name STRING 기본키, type STRING)                : 개체. type 은 Material, Property, Phenomenon, Process, Application, Method 중 하나
  (관계에만 등장해 종류를 모르는 개체는 'Concept')

[관계(Relationship) 테이블]
- (Chunk)-[:PART_OF]->(Paper)
- (Chunk)-[:MENTIONS]->(Entity)
- (Entity)-[:RELATED_TO {relation STRING, chunk_id STRING}]->(Entity)
  relation 값: ALTERNATIVE_TO, HAS_PROPERTY, INCREASES, DECREASES, CAUSES, CONTAINS, USED_IN, PROCESSED_BY, RELATED_TO

[이름 규칙]
- 원소·화합물은 화학식 그대로: 'Cu', 'Ru', 'Co', 'Mo', 'W', 'NiAl', 'CuAl2', 'Cu-Nb'
- 그 밖의 개체 이름은 영어 소문자 단수형: 'resistivity', 'grain boundary scattering', 'electromigration', 'interconnect'
- 관계 종류는 r.relation 속성으로 구분합니다. 예) MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity) WHERE r.relation = 'ALTERNATIVE_TO'
"""

# 논문 파일 이름 → (짧은 id, 제목, 연도)
PAPER_INFO = {
    "1-s2.0-S1359646216302512-main.pdf": (
        "kim2016",
        "Interplay between grain boundary segregation and electrical resistivity "
        "in dilute nanocrystalline Cu alloys",
        2016,
    ),
    "Recent progress in alternative metals for advanced interconnects.pdf": (
        "sung2026",
        "Recent progress in alternative metals for advanced interconnects",
        2026,
    ),
}

# 같은 대상을 가리키는 다른 이름(별칭) → 대표 이름
ALIASES = {
    "copper": "Cu", "ruthenium": "Ru", "cobalt": "Co", "molybdenum": "Mo",
    "tungsten": "W", "nickel": "Ni", "aluminum": "Al", "aluminium": "Al",
    "niobium": "Nb", "iron": "Fe", "tantalum": "Ta", "titanium": "Ti",
    "rhodium": "Rh", "iridium": "Ir", "platinum": "Pt", "silver": "Ag",
    "gold": "Au", "chromium": "Cr", "osmium": "Os", "graphene": "graphene",
    "tantalum nitride": "TaN", "titanium nitride": "TiN",
    "electron mean free path": "EMFP", "mean free path": "EMFP",
    "electron mean free path (emfp)": "EMFP", "emfp": "EMFP",
    "electrical resistivity": "resistivity", "electric resistivity": "resistivity",
    "interconnects": "interconnect", "interconnection": "interconnect",
    "gb": "grain boundary", "gb scattering": "grain boundary scattering",
    "grain-boundary scattering": "grain boundary scattering",
    "grain boundary segregation": "grain boundary segregation",
    "size effect": "size effect", "resistivity size effect": "size effect",
    "em": "electromigration", "tsv": "through-silicon via", "tsvs": "through-silicon via",
}

# 'Cu', 'NiAl', 'CuAl2', 'Cu-Nb', 'Ti2SiC', 'MoS2' 처럼 화학식으로 보이는 이름
_FORMULA = re.compile(r"^(?:[A-Z][a-z]?\d*(?:\.\d+)?)+(?:[-/](?:[A-Z][a-z]?\d*(?:\.\d+)?)+)*$")
_NO_SINGULAR = ("ss", "us", "is", "ics", "sis")  # 'process', 'analysis', 'electronics' 등은 그대로
# s 로 끝나지만 단수형(또는 단·복수 같은 꼴)인 단어 — 's' 를 떼면 뜻이 망가짐 ('series' → 'sery')
_NO_SINGULAR_WORDS = {
    "series", "species", "bias", "gas", "lens", "class", "status", "apparatus", "physics",
    "kinetics", "mechanics", "dynamics", "thermodynamics", "optics", "electronics",
    "means", "news", "alias", "atlas", "chaos",
}


def normalize_name(name: str) -> str:
    """개체 이름을 대표 이름으로 맞춥니다. (예: 'Copper'→'Cu', 'Grain Boundaries'→'grain boundary')"""
    raw = re.sub(r"\s+", " ", str(name)).strip().strip(".,;:'\"")
    if not raw:
        return ""
    raw = re.sub(r"^(the|a|an) ", "", raw, flags=re.I)  # 앞의 관사 제거
    if raw.lower() in ALIASES:  # 별칭 사전 우선 ('copper' → 'Cu', 'EM' → 'electromigration')
        return ALIASES[raw.lower()]
    if _FORMULA.match(raw):  # 화학식은 대소문자를 그대로 둡니다
        return raw
    # 단어별로: 화학식('Cu', 'NiAl')은 그대로, 나머지는 소문자
    words = [w if _FORMULA.match(w) and len(w) > 1 else w.lower() for w in raw.split(" ")]
    last = words[-1]
    if last.lower() in _NO_SINGULAR_WORDS:  # 'series', 'bias' 등은 그대로
        pass
    elif last.endswith("ies") and len(last) > 4:  # boundaries -> boundary
        last = last[:-3] + "y"
    elif last.endswith(("sses", "ches", "shes", "xes")):  # losses -> loss, fluxes -> flux
        last = last[:-2]
    elif last.endswith("s") and len(last) > 3 and not last.endswith(_NO_SINGULAR) and not _FORMULA.match(last):
        last = last[:-1]  # metals -> metal
    name = " ".join(words[:-1] + [last])
    return ALIASES.get(name.lower(), name)


def normalize_relation(relation: str) -> str:
    """관계 이름을 대문자_밑줄 형태로 맞추고, 목록에 없으면 RELATED_TO 로 둡니다."""
    rel = re.sub(r"[^A-Z]+", "_", str(relation).upper()).strip("_")
    return rel if rel in RELATION_TYPES else "RELATED_TO"


def paper_meta(source: str) -> tuple[str, str, int]:
    """PDF 경로 → (논문 id, 제목, 연도). 목록에 없는 PDF 는 파일 이름으로 id 를 만듭니다."""
    fname = Path(source).name
    if fname in PAPER_INFO:
        return PAPER_INFO[fname]
    slug = re.sub(r"[^a-z0-9]+", "_", Path(fname).stem.lower())[:30]
    return slug, Path(fname).stem, 0


def make_chunk_records(chunks) -> list[dict]:
    """split_documents() 결과(LangChain Document 목록) → 그래프용 조각 기록(id/paper_id/page/text).

    조각 id 는 '논문id_c순번' (예: 'sung2026_c012') 입니다. 같은 분할 설정이면 항상 같은 id 가 나옵니다.
    """
    counter: Counter = Counter()
    records = []
    for doc in chunks:
        paper_id, _, _ = paper_meta(doc.metadata.get("source", "unknown"))
        idx = counter[paper_id]
        counter[paper_id] += 1
        records.append({
            "id": f"{paper_id}_c{idx:03d}",
            "paper_id": paper_id,
            "page": int(doc.metadata.get("page", 0)) + 1,  # 사람이 읽는 쪽 번호(1부터)
            "text": doc.page_content,
        })
    return records


# ---------------------------------------------------------------- LLM 개체·관계 추출
# 07 노트북에서 한 단계씩 설명한 코드를 한곳에 모은 것입니다.
EXTRACTION_PROMPT = """You are a materials-science expert building a knowledge graph from a research paper.
Extract entities and relations from the TEXT below.

Entity types: {entity_types}
- Material: elements, alloys, compounds (e.g. Cu, Ru, NiAl, Cu-Nb, graphene)
- Property: measurable properties (e.g. resistivity, electron mean free path, melting point)
- Phenomenon: physical effects (e.g. size effect, grain boundary scattering, electromigration)
- Process: fabrication or treatment steps (e.g. sputtering, annealing, dry etching)
- Application: uses and devices (e.g. interconnect, barrier layer, transistor)
- Method: measurement or simulation techniques (e.g. TEM, DFT, four-point probe)

Relation types: {relation_types}

Rules:
- Use short English names. Chemical elements/compounds as formulas (Cu, Ru, CuAl2). Other names in lowercase singular.
- Only extract facts stated in the TEXT. Do not invent. At most 8 entities and 8 relations (the most important ones).
- Use ONLY the relation types listed above.
- source and target of every relation must also appear in entities.
- Return ONLY JSON in this format:
{{"entities": [{{"name": "Ru", "type": "Material"}}, {{"name": "Cu", "type": "Material"}}],
 "relations": [{{"source": "Ru", "relation": "ALTERNATIVE_TO", "target": "Cu"}}]}}

TEXT:
{text}
"""


def parse_json(raw: str) -> dict:
    """LLM 출력에서 {...} 부분만 꺼내 파싱합니다. 실패하면 빈 결과를 돌려줍니다."""
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    if not isinstance(data, dict):
        data = {}
    return {"entities": list(data.get("entities") or []), "relations": list(data.get("relations") or [])}


def extract_graph(text: str, llm=None, retries: int = 1) -> dict:
    """조각 하나에서 개체·관계를 추출합니다(원시 결과). 결과가 비면 retries 번 더 시도합니다."""
    llm = llm or get_llm()
    prompt = EXTRACTION_PROMPT.format(
        entity_types=", ".join(ENTITY_TYPES), relation_types=", ".join(RELATION_TYPES), text=text)
    result = {"entities": [], "relations": []}
    for _ in range(retries + 1):
        result = parse_json(llm.invoke(prompt).content)
        if result["entities"]:
            break
    return result


def clean_extraction(ex: dict) -> dict:
    """추출 결과 정리: 이름 정규화, 목록에 없는 type/관계 보정, 중복·자기 자신 관계 제거."""
    entities = {}
    for e in ex.get("entities", []):
        name = normalize_name(e.get("name", "")) if isinstance(e, dict) else ""
        if name:
            entities.setdefault(name, e.get("type") if e.get("type") in ENTITY_TYPES else "Concept")
    relations = []
    for r in ex.get("relations", []):
        if not isinstance(r, dict):
            continue
        src, dst = normalize_name(r.get("source", "")), normalize_name(r.get("target", ""))
        item = {"source": src, "relation": normalize_relation(r.get("relation", "")), "target": dst}
        if src and dst and src != dst and item not in relations:
            relations.append(item)
    return {"entities": [{"name": n, "type": t} for n, t in entities.items()], "relations": relations}


def reference_chunk_ids(records: list[dict]) -> set[str]:
    """논문마다 'References' 가 처음 나오는 조각 이후(참고문헌 목록)는 추출에서 뺍니다."""
    skip, seen_ref = set(), set()
    for r in records:
        if r["paper_id"] in seen_ref:
            skip.add(r["id"])
        elif re.search(r"\bReferences\b|Notes and references", r["text"]):
            seen_ref.add(r["paper_id"])
    return skip


def load_chunk_records() -> tuple[list[dict], list[dict]]:
    """논문을 그래프용 조각(CHUNK_SIZE)으로 나눠 (논문 목록, 조각 기록)을 돌려줍니다."""
    docs = load_papers()
    chunks = split_documents(docs, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    papers = sorted({paper_meta(d.metadata["source"]) for d in docs})
    return [{"id": p, "title": t, "year": y} for p, t, y in papers], make_chunk_records(chunks)


def extract_knowledge_graph(max_chunks: int | None = None, verbose: bool = True, llm=None) -> Path:
    """모든 본문 조각에서 개체·관계를 추출해 db/kg_triples.json 에 저장합니다.

    - 조각 하나 끝날 때마다 파일에 저장하므로, 중간에 멈춰도 다시 실행하면 이어서 합니다.
    - max_chunks: 이번 실행에서 새로 추출할 최대 조각 수 (느린 PC 는 30 정도로 먼저 시험)
    """
    papers, records = load_chunk_records()
    skip = reference_chunk_ids(records)
    targets = [r for r in records if r["id"] not in skip]
    data = json.loads(TRIPLES_PATH.read_text(encoding="utf-8")) if TRIPLES_PATH.exists() else {}
    done = {ex["chunk_id"]: ex for ex in data.get("extractions", [])}
    meta = {"llm": LLM_MODEL, "chunk_size": CHUNK_SIZE, "chunk_overlap": CHUNK_OVERLAP,
            "skipped_reference_chunks": len(skip), "target_chunks": len(targets),
            "elapsed_sec": data.get("meta", {}).get("elapsed_sec", 0.0)}
    todo = [r for r in targets if r["id"] not in done][: max_chunks]
    if verbose:
        print(f"추출 대상 {len(targets)}개 중 완료 {len(done)}개 → 이번에 {len(todo)}개 추출")
    llm = llm or get_llm()
    for r in todo:
        t0 = time.time()
        ex = clean_extraction(extract_graph(r["text"], llm=llm))
        done[r["id"]] = {"chunk_id": r["id"], **ex}
        meta["elapsed_sec"] = round(meta["elapsed_sec"] + time.time() - t0, 1)
        _save_triples(meta, papers, records, [done[t["id"]] for t in targets if t["id"] in done])
        if verbose:
            print(f"[{len(done)}/{len(targets)}] {r['id']} 개체 {len(ex['entities'])}개, "
                  f"관계 {len(ex['relations'])}개 ({time.time() - t0:.0f}초)")
    if not todo:  # 이미 모두 끝났어도 파일은 최신 형식으로 저장
        _save_triples(meta, papers, records, [done[t["id"]] for t in targets if t["id"] in done])
    return TRIPLES_PATH


def _save_triples(meta: dict, papers: list, records: list, extractions: list) -> None:
    data = {"meta": {**meta, "extracted_chunks": len(extractions)},
            "papers": papers, "chunks": records, "extractions": extractions}
    TRIPLES_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- 그래프 만들기
def create_schema(conn: kuzu.Connection) -> None:
    """빈 DB 에 노드/관계 테이블을 만듭니다."""
    for stmt in [
        "CREATE NODE TABLE Paper(id STRING, title STRING, year INT64, PRIMARY KEY(id))",
        "CREATE NODE TABLE Chunk(id STRING, paper_id STRING, page INT64, text STRING, PRIMARY KEY(id))",
        "CREATE NODE TABLE Entity(name STRING, type STRING, PRIMARY KEY(name))",
        "CREATE REL TABLE PART_OF(FROM Chunk TO Paper)",
        "CREATE REL TABLE MENTIONS(FROM Chunk TO Entity)",
        "CREATE REL TABLE RELATED_TO(FROM Entity TO Entity, relation STRING, chunk_id STRING)",
    ]:
        conn.execute(stmt)


def vote_entity_types(extractions: list[dict]) -> dict[str, str]:
    """같은 개체가 조각마다 다른 type 으로 뽑혔으면 가장 많이 나온 type 으로 정합니다."""
    votes: dict[str, Counter] = {}
    for ex in extractions:
        for ent in ex.get("entities", []):
            name = normalize_name(ent.get("name", ""))
            if name:
                votes.setdefault(name, Counter())[ent.get("type") or "Material"] += 1
        for rel in ex.get("relations", []):  # 관계에만 등장한 개체도 노드로 만듭니다
            for key in ("source", "target"):
                name = normalize_name(rel.get(key, ""))
                if name:
                    votes.setdefault(name, Counter())
    return {n: (c.most_common(1)[0][0] if c else "Concept") for n, c in votes.items()}


def load_triples_into(conn: kuzu.Connection, data: dict) -> None:
    """kg_triples.json 형식의 데이터를 그래프에 적재합니다(MERGE 사용 → 중복 없이)."""
    for p in data["papers"]:
        conn.execute("MERGE (p:Paper {id: $id}) SET p.title = $title, p.year = $year", p)
    for c in data["chunks"]:
        conn.execute(
            "MERGE (c:Chunk {id: $id}) SET c.paper_id = $paper_id, c.page = $page, c.text = $text",
            {k: c[k] for k in ("id", "paper_id", "page", "text")},
        )
        conn.execute(
            "MATCH (c:Chunk {id: $cid}), (p:Paper {id: $pid}) MERGE (c)-[:PART_OF]->(p)",
            {"cid": c["id"], "pid": c["paper_id"]},
        )
    types = vote_entity_types(data["extractions"])
    for name, etype in types.items():
        conn.execute("MERGE (e:Entity {name: $name}) SET e.type = $type", {"name": name, "type": etype})
    for ex in data["extractions"]:
        _load_one_extraction(conn, ex)


def _load_one_extraction(conn: kuzu.Connection, ex: dict) -> None:
    """조각 하나의 추출 결과 → MENTIONS / RELATED_TO 관계."""
    cid = ex["chunk_id"]
    names = {normalize_name(e.get("name", "")) for e in ex.get("entities", [])}
    for rel in ex.get("relations", []):
        src, dst = normalize_name(rel.get("source", "")), normalize_name(rel.get("target", ""))
        if not src or not dst or src == dst:
            continue
        names.update([src, dst])
        conn.execute(
            "MATCH (a:Entity {name: $s}), (b:Entity {name: $t}) "
            "MERGE (a)-[:RELATED_TO {relation: $r, chunk_id: $c}]->(b)",
            {"s": src, "t": dst, "r": normalize_relation(rel.get("relation", "")), "c": cid},
        )
    for name in names - {""}:
        conn.execute(
            "MATCH (c:Chunk {id: $c}), (e:Entity {name: $n}) MERGE (c)-[:MENTIONS]->(e)",
            {"c": cid, "n": name},
        )


def _remove_db_files(path: Path) -> None:
    """Kùzu DB 파일(또는 예전 버전의 폴더)과 .wal 파일을 지웁니다."""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)
    Path(str(path) + ".wal").unlink(missing_ok=True)


def _build_graph(db_path: Path) -> None:
    """임시 경로에 그래프를 다 만든 뒤, 성공했을 때만 db_path 로 바꿔 놓습니다(중간 실패 시 반쪽 DB 방지)."""
    tmp_path = db_path.with_name(db_path.name + ".tmp")
    _remove_db_files(tmp_path)
    db = kuzu.Database(str(tmp_path))
    conn = kuzu.Connection(db)
    try:
        create_schema(conn)
        load_triples_into(conn, json.loads(TRIPLES_PATH.read_text(encoding="utf-8")))
    except Exception:
        conn.close()
        db.close()
        _remove_db_files(tmp_path)  # 만들다 만 임시 DB 정리
        raise
    conn.close()
    db.close()  # 닫아야 파일 잠금이 풀리고 .wal 내용이 본 파일에 반영됨
    _remove_db_files(db_path)
    tmp_path.replace(db_path)
    tmp_wal = Path(str(tmp_path) + ".wal")
    if tmp_wal.exists():
        tmp_wal.replace(Path(str(db_path) + ".wal"))


def get_graph(rebuild: bool = False, db_path: Path = GRAPH_DB_PATH) -> kuzu.Connection:
    """논문 지식그래프에 연결합니다.

    DB 가 없거나, rebuild=True 이거나, kg_triples.json 이 DB 보다 새 파일이면 JSON 에서 다시 만듭니다.
    """
    db_path = Path(db_path)
    stale = (db_path.exists() and TRIPLES_PATH.exists()
             and TRIPLES_PATH.stat().st_mtime > db_path.stat().st_mtime)  # 추출 결과가 더 최신
    if rebuild or stale or not db_path.exists():
        if not TRIPLES_PATH.exists():
            raise FileNotFoundError(
                "db/kg_triples.json(LLM 추출 결과)이 없습니다. 먼저 07_Graph_RAG 노트북의 '전체 추출' 셀을 "
                "실행하거나, from skku_rag.graph_store import extract_knowledge_graph; "
                "extract_knowledge_graph() 를 실행하세요. (약 1시간, 중단해도 이어서 실행 가능)")
        _build_graph(db_path)
    return kuzu.Connection(kuzu.Database(str(db_path)))


# ---------------------------------------------------------------- 검색 도우미
def find_entities(conn: kuzu.Connection, keywords: list[str], limit: int = 20) -> pd.DataFrame:
    """키워드(영어)와 이름이 같거나 이름에 포함되는 개체를 찾습니다. 연결이 많은 개체부터.

    예) find_entities(conn, ["Ru", "grain boundary"]) → Ru, grain boundary, grain boundary scattering …
    """
    keys = [normalize_name(k) for k in keywords if str(k).strip()]
    if not keys:
        return pd.DataFrame(columns=["name", "type", "degree"])
    query = """
        MATCH (e:Entity)
        WHERE e.name IN $keys
           OR any(k IN $long_keys WHERE lower(e.name) CONTAINS lower(k))
        OPTIONAL MATCH (e)-[r:RELATED_TO]-(:Entity)
        RETURN e.name AS name, e.type AS type, count(r) AS degree,
               CASE WHEN e.name IN $keys THEN 1 ELSE 0 END AS exact
        ORDER BY exact DESC, degree DESC LIMIT $limit
    """
    # 'Co', 'W' 처럼 짧은 이름은 부분일치하면 엉뚱한 개체('cost', 'low')까지 잡히므로 정확일치만 허용
    # Kùzu 는 빈 리스트 매개변수의 자료형을 정하지 못해 오류가 나므로, 없을 때는 아무것도 안 걸리는 값을 넣음
    long_keys = [k for k in keys if len(k) >= 4] or ["\x00"]
    df = conn.execute(query, {"keys": keys, "long_keys": long_keys, "limit": limit}).get_as_df()
    return df.drop(columns="exact")


def get_neighbors(
    conn: kuzu.Connection, names: list[str], hops: int = 1, limit: int = 60,
    relations: list[str] | None = None,
) -> pd.DataFrame:
    """개체들에서 출발해 hops 단계 안에 있는 관계(source, relation, target, chunk_id)를 모읍니다.

    hops=1: 바로 연결된 관계, hops=2: 이웃의 이웃까지(결과가 빠르게 늘어나니 limit 로 자릅니다).
    relations 를 주면(예: ["ALTERNATIVE_TO"]) 그 관계를 먼저 보여 줍니다.
    정렬: ① 원하는 관계 ② 출발 개체에 가까운(hop 이 작은) 관계
    """
    query = """
        MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
        WHERE a.name IN $frontier OR b.name IN $frontier
        RETURN a.name AS source, r.relation AS relation, b.name AS target, r.chunk_id AS chunk_id
    """
    if not names:
        return pd.DataFrame(columns=["source", "relation", "target", "chunk_id", "hop"])
    frontier, seen, parts = list(names), set(names), []
    for hop in range(1, max(1, min(int(hops), 3)) + 1):  # 한 단계씩 바깥으로 넓혀 가기(BFS)
        edges = conn.execute(query, {"frontier": frontier}).get_as_df().assign(hop=hop)
        parts.append(edges)
        new_nodes = (set(edges.source) | set(edges.target)) - seen
        seen |= new_nodes
        # 다음 단계는 새로 만난 개체 중 자주 등장한 15개만 (결과 폭발 방지)
        counts = pd.concat([edges.source, edges.target]).value_counts()
        frontier = [n for n in counts.index if n in new_nodes][:15]
        if not frontier:
            break
    result = pd.concat(parts).drop_duplicates(["source", "relation", "target"])
    wanted = set(relations or [])
    return (
        result.assign(_want=~result.relation.isin(wanted))
        .sort_values(["_want", "hop"], kind="stable")
        .drop(columns="_want").head(limit).reset_index(drop=True)
    )


def chunks_for_entities(conn: kuzu.Connection, names: list[str], limit: int = 5) -> pd.DataFrame:
    """개체들을 많이 언급하는 조각을 찾습니다(hits = 조각이 언급한 대상 개체 수)."""
    if not names:
        return pd.DataFrame(columns=["id", "paper_id", "page", "text", "hits", "entities"])
    query = """
        MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
        WHERE e.name IN $names
        RETURN c.id AS id, c.paper_id AS paper_id, c.page AS page, c.text AS text,
               count(DISTINCT e) AS hits, collect(DISTINCT e.name) AS entities
        ORDER BY hits DESC, id LIMIT $limit
    """
    return conn.execute(query, {"names": list(names), "limit": limit}).get_as_df()


_READ_START = re.compile(r"^(MATCH|OPTIONAL\s+MATCH|WITH|UNWIND|RETURN)\b", re.I)  # 허용하는 시작 단어
_WRITE_WORDS = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|ALTER|COPY|INSTALL|LOAD|ATTACH|CALL"
    r"|EXPORT|IMPORT|CHECKPOINT|USE|BEGIN|COMMIT|ROLLBACK)\b", re.I
)


def _strip_query(query: str) -> str:
    """앞뒤 공백과 맨 끝의 세미콜론 하나를 뗍니다."""
    query = str(query).strip()
    return query[:-1].rstrip() if query.endswith(";") else query


def is_read_only(query: str) -> bool:
    """Cypher 가 데이터를 바꾸지 않는(읽기 전용) 쿼리인지 간단히 검사합니다(허용 목록 방식).

    MATCH / OPTIONAL MATCH / WITH / UNWIND / RETURN 으로 시작하고, 중간에 ';'(여러 문장 이어 붙이기)가 없으며,
    쓰기·관리 명령(CREATE, DELETE, SET, CALL, EXPORT …)이 없어야 통과합니다.
    """
    q = _strip_query(query)
    return bool(_READ_START.match(q)) and ";" not in q and not _WRITE_WORDS.search(q)


def run_cypher(conn: kuzu.Connection, query: str, params: dict | None = None) -> pd.DataFrame:
    """읽기 전용 Cypher 만 실행해 표(DataFrame)로 돌려줍니다. (LLM 이 만든 쿼리를 실행할 때 안전장치)"""
    if not is_read_only(query):
        raise ValueError("읽기 전용 쿼리(MATCH … RETURN)만 실행할 수 있습니다.")
    return conn.execute(_strip_query(query), params or {}).get_as_df()
