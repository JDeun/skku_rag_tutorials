"""논문 표(Table)에서 옮겨 적은 물성 데이터를 SQLite 데이터베이스로 만드는 도우미.

원본 데이터는 저장소에 함께 들어 있는 CSV 파일입니다.
  - paper_data/papers.csv           : 논문 정보 (논문 1편 = 1행)
  - paper_data/metal_properties.csv : Sung et al. 2026 리뷰의 Table 1, Table 2
  - paper_data/alloy_films.csv      : Kim et al. 2016 의 Table 1 (Cu-Nb, Cu-Fe 박막)

모든 수치는 논문 표에서 그대로 옮겼고, 표에 없는 값은 비워 두었습니다(NULL).
"""
import csv
import sqlite3
from pathlib import Path

from skku_rag.config import DB_DIR, PAPER_DIR

# 테이블 정의: (테이블 이름, 원본 CSV 파일 이름, CREATE TABLE 문)
_TABLES = [
    (
        "papers",
        "papers.csv",
        """CREATE TABLE papers (
            paper_id TEXT PRIMARY KEY,
            title    TEXT NOT NULL,
            authors  TEXT,
            journal  TEXT,
            year     INTEGER,
            volume   TEXT,
            pages    TEXT,
            doi      TEXT,
            pdf_file TEXT
        )""",
    ),
    (
        "metals",
        "metal_properties.csv",
        """CREATE TABLE metals (
            metal_id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            material                    TEXT NOT NULL,
            category                    TEXT,
            crystal_structure           TEXT,
            value_type                  TEXT,
            resistivity_uohm_cm         REAL,
            emfp_nm                     REAL,
            emfp_nm_alt                 REAL,
            rho_lambda_1e16_ohm_m2      REAL,
            rho_lambda_alt_1e16_ohm_m2  REAL,
            fermi_velocity_1e6_m_s      REAL,
            r_film                      REAL,
            r_wire                      REAL,
            cohesive_energy_ev_atom     REAL,
            paper_id                    TEXT REFERENCES papers(paper_id),
            source                      TEXT,
            note                        TEXT,
            UNIQUE (material, source)
        )""",
    ),
    (
        "alloy_films",
        "alloy_films.csv",
        """CREATE TABLE alloy_films (
            film_id               INTEGER PRIMARY KEY AUTOINCREMENT,
            solute                TEXT,
            cu_target_power_w     REAL,
            alloy_target_power_w  REAL,
            thickness_nm          REAL,
            thickness_err_nm      REAL,
            solute_at_pct         REAL,
            solute_at_pct_err     REAL,
            grain_size_nm         REAL,
            grain_size_err_nm     REAL,
            paper_id              TEXT REFERENCES papers(paper_id),
            source                TEXT
        )""",
    ),
]

SQL_SCHEMA_DESCRIPTION = """\
SQLite 데이터베이스 (SQLite database) — 테이블 3개 / 3 tables.
빈 값은 NULL 입니다 (missing values are NULL). 모든 수치는 논문 표에서 옮긴 값입니다.

1) papers — 논문 정보 (one row per paper)
   - paper_id TEXT PRIMARY KEY  : 'kim2016' 또는 'sung2026'
   - title TEXT, authors TEXT ('; ' 로 구분), journal TEXT, year INTEGER, volume TEXT, pages TEXT, doi TEXT, pdf_file TEXT

2) metals — 배선(interconnect) 후보 금속/화합물의 물성 (Sung et al. 2026 리뷰, Table 1 & Table 2)
   - metal_id INTEGER PRIMARY KEY
   - material TEXT : 화학식 (e.g. 'Cu', 'Ru', 'Co', 'NiIr3', 'PtCoO2'). 'Cu' appears twice (Table 1 and Table 2).
   - category TEXT : 'element'(단일 원소 금속), 'compound'(화합물), 'oxide'(산화물)
   - crystal_structure TEXT : 'fcc', 'hcp', 'bcc' (Table 1 only; NULL for Table 2)
   - value_type TEXT : 'representative'(Table 1, 대표 실온값) or 'calculated'(Table 2, 계산값)
   - resistivity_uohm_cm REAL : 벌크 비저항 rho0, 단위 μΩ·cm (bulk resistivity). Cu = 1.678
   - emfp_nm REAL : 전자 평균 자유 행로 lambda (electron mean free path, EMFP), 단위 nm
   - emfp_nm_alt REAL : hcp 금속(Co, Ru, Os)의 두 번째 EMFP 값 (표에 'a/b' 로 두 값이 있음)
   - rho_lambda_1e16_ohm_m2 REAL : rho0×lambda (성능지수 FOM), 단위 1e-16 Ω·m². 작을수록 미세배선에 유리
   - rho_lambda_alt_1e16_ohm_m2 REAL : hcp 금속의 두 번째 rho0×lambda 값
   - fermi_velocity_1e6_m_s REAL : 페르미 속도 vF, 단위 1e6 m/s (Table 2 only)
   - r_film REAL, r_wire REAL : 박막/선 형상별 비저항 스케일링 계수 (Table 2 only, 작을수록 유리)
   - cohesive_energy_ev_atom REAL : 응집 에너지 (cohesive energy), eV/atom (Table 2 only)
   - paper_id TEXT FOREIGN KEY -> papers.paper_id
   - source TEXT : 'sung2026 Table 1' or 'sung2026 Table 2'
   - note TEXT

3) alloy_films — 스퍼터링한 나노결정 Cu 합금 박막 (Kim et al. 2016, Table 1)
   - film_id INTEGER PRIMARY KEY
   - solute TEXT : 첨가 원소 'Nb', 'Fe', 또는 순수 Cu는 'none'
   - cu_target_power_w REAL, alloy_target_power_w REAL : 스퍼터링 파워 (W)
   - thickness_nm REAL, thickness_err_nm REAL : 막 두께 및 오차 (nm)
   - solute_at_pct REAL, solute_at_pct_err REAL : 용질 농도 및 오차 (at.%)
   - grain_size_nm REAL, grain_size_err_nm REAL : 평균 결정립 크기 및 오차 (nm)
   - paper_id TEXT FOREIGN KEY -> papers.paper_id
   - source TEXT : 'kim2016 Table 1'
"""


def _to_value(text: str):
    """CSV 문자열 하나를 SQLite 값으로 바꿉니다: 빈칸 → None(NULL).

    숫자 변환은 SQLite 가 열의 타입(REAL/INTEGER)을 보고 자동으로 해 줍니다.
    """
    text = text.strip()
    return None if text == "" else text


def _load_csv(conn: sqlite3.Connection, table: str, csv_path: Path) -> int:
    """CSV 파일의 모든 행을 table 에 INSERT 하고, 넣은 행 수를 돌려줍니다."""
    with open(csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    columns = list(rows[0].keys())
    placeholders = ", ".join("?" for _ in columns)
    sql = f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    conn.executemany(sql, [tuple(_to_value(r[c]) for c in columns) for r in rows])
    return len(rows)


def build_sqlite(db_path: Path = DB_DIR / "papers.sqlite", overwrite: bool = False) -> Path:
    """CSV 파일들로 SQLite DB 파일을 만들고 그 경로를 돌려줍니다.

    이미 DB 파일이 있으면(overwrite=False) 그대로 두고 경로만 돌려줍니다.
    """
    db_path = Path(db_path)
    if db_path.exists() and not overwrite:
        return db_path
    if db_path.exists():
        db_path.unlink()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(".tmp")
    tmp_path.unlink(missing_ok=True)
    try:
        with sqlite3.connect(tmp_path) as conn:
            for table, csv_name, create_sql in _TABLES:
                conn.execute(create_sql)
                _load_csv(conn, table, PAPER_DIR / csv_name)
        conn.close()
        tmp_path.replace(db_path)
    except Exception as error:
        tmp_path.unlink(missing_ok=True)
        raise RuntimeError(f"SQLite DB 생성 실패: {error}") from error
    return db_path
