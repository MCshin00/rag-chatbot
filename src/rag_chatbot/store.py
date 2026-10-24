"""pgvector 저장소. 근사 인덱스 없이 코사인 거리로 전부 비교한다(정확 검색)."""

import json

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from .config import DATABASE_URL, EMBEDDING_DIM
from .corpus import Chunk, Span

SCHEMA = f"""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id text PRIMARY KEY,
    strategy text NOT NULL,
    doc_id text NOT NULL,
    law_name text NOT NULL,
    text text NOT NULL,
    span_start int NOT NULL,
    span_end int NOT NULL,
    articles jsonb NOT NULL,
    embedding_input text NOT NULL,
    embedding vector({EMBEDDING_DIM}) NOT NULL
);
"""


def connect() -> psycopg.Connection:
    conn = psycopg.connect(DATABASE_URL, autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init(conn: psycopg.Connection) -> None:
    conn.execute(SCHEMA)


def replace_strategy(conn: psycopg.Connection, strategy: str, rows: list[tuple[Chunk, str, list[float]]]) -> None:
    with conn.transaction():
        conn.execute("DELETE FROM chunks WHERE strategy = %s", (strategy,))
        with conn.cursor() as cur:
            cur.executemany(
                "INSERT INTO chunks VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                [(c.chunk_id, c.strategy, c.doc_id, c.law_name, c.text, c.span.start, c.span.end,
                  json.dumps(list(c.articles)), text, np.array(vec)) for c, text, vec in rows],
            )


def count(conn: psycopg.Connection, strategy: str) -> int:
    return conn.execute("SELECT count(*) FROM chunks WHERE strategy = %s", (strategy,)).fetchone()[0]


def search(conn: psycopg.Connection, strategy: str, query_vec: list[float], k: int) -> list[dict]:
    rows = conn.execute(
        "SELECT chunk_id, doc_id, law_name, text, span_start, span_end, articles, embedding <=> %s AS distance "
        "FROM chunks WHERE strategy = %s ORDER BY distance, chunk_id LIMIT %s",
        (np.array(query_vec), strategy, k),
    ).fetchall()
    return [
        {"chunk_id": r[0], "doc_id": r[1], "law_name": r[2], "text": r[3],
         "span": Span(r[1], r[4], r[5]), "articles": tuple(r[6]), "distance": float(r[7])}
        for r in rows
    ]
