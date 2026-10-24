"""pgvector 저장소. 근사 인덱스 없이 코사인 거리로 전부 비교한다(정확 검색).

임베딩 모델마다 벡터 차원이 달라서 차원을 정하지 않은 vector 열에 모델 이름과 함께 저장한다.
"""

import json

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .config import DATABASE_URL
from .corpus import Chunk, Span

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    model text NOT NULL,
    chunk_id text NOT NULL,
    strategy text NOT NULL,
    doc_id text NOT NULL,
    law_name text NOT NULL,
    text text NOT NULL,
    span_start int NOT NULL,
    span_end int NOT NULL,
    articles jsonb NOT NULL,
    embedding_input text NOT NULL,
    embedding vector NOT NULL,
    PRIMARY KEY (model, chunk_id)
);
"""
COLUMNS = "chunk_id, doc_id, law_name, text, span_start, span_end, articles"


def connect() -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)


def replace_strategy(conn: psycopg.Connection, model: str, strategy: str,
                     rows: list[tuple[Chunk, str, list[float]]]) -> None:
    with conn.transaction():
        conn.execute("DELETE FROM chunks WHERE model = %s AND strategy = %s", (model, strategy))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                [(model, c.chunk_id, c.strategy, c.doc_id, c.law_name, c.text, c.span.start, c.span.end,
                  json.dumps(list(c.articles)), text, np.array(vec)) for c, text, vec in rows],
            )


def count(conn: psycopg.Connection, model: str, strategy: str) -> int:
    return conn.execute("SELECT count(*) FROM chunks WHERE model = %s AND strategy = %s",
                        (model, strategy)).fetchone()[0]


def _row(r) -> dict:
    return {"chunk_id": r[0], "doc_id": r[1], "law_name": r[2], "text": r[3],
            "span": Span(r[1], r[4], r[5]), "articles": tuple(r[6])}


def all_chunks(conn: psycopg.Connection, model: str, strategy: str) -> list[dict]:
    rows = conn.execute(f"SELECT {COLUMNS} FROM chunks WHERE model = %s AND strategy = %s ORDER BY chunk_id",
                        (model, strategy)).fetchall()
    return [_row(r) for r in rows]


def search(conn: psycopg.Connection, model: str, strategy: str, query_vec: list[float], k: int) -> list[dict]:
    rows = conn.execute(
        f"SELECT {COLUMNS}, embedding <=> %s AS distance FROM chunks WHERE model = %s AND strategy = %s "
        "ORDER BY distance, chunk_id LIMIT %s",
        (np.array(query_vec), model, strategy, k),
    ).fetchall()
    return [_row(r) | {"distance": float(r[7])} for r in rows]
