"""실행 설정. 평가 비교에서 두 방식에 똑같이 쓰는 값은 여기 한 곳에 둔다."""

import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://rag:rag@127.0.0.1:5433/rag")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_PRICES = {"text-embedding-3-small": 0.02, "text-embedding-3-large": 0.13}  # $/1M, 2026-10-24 공식 모델 페이지
RETRIEVAL = "vector"  # "vector" | "hybrid"(벡터 + BM25를 RRF로 결합)
HYBRID_POOL = 20  # 결합 전에 각 검색에서 가져오는 수
RRF_K = 60
CHAT_MODEL = "gpt-5.4-mini-2026-03-17"
REASONING_EFFORT = "none"
MAX_OUTPUT_TOKENS = 800
TOP_K = 5
PROMPT_VERSION = "p1"
EMBED_WITH_LAW_NAME = True
