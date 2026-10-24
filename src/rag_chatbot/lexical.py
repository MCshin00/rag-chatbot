"""키워드 검색(BM25)과 순위 결합(RRF).

한국어는 조사가 붙어("지급한도는") 띄어쓰기 단위로는 같은 말이 다른 단어가 된다.
그래서 낱말마다 두 글자씩 겹쳐 자른 바이그램을 단어로 쓴다(한 글자 낱말은 그대로).
"""

import math
import re
from collections import Counter

WORD = re.compile(r"[0-9A-Za-z가-힣]+")


def bigrams(text: str) -> list[str]:
    out = []
    for w in WORD.findall(text):
        out += [w] if len(w) == 1 else [w[i:i + 2] for i in range(len(w) - 1)]
    return out


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75):
        self.tfs = [Counter(bigrams(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter(t for tf in self.tfs for t in tf)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.k1, self.b = k1, b

    def scores(self, query: str) -> list[float]:
        terms = [t for t in bigrams(query) if t in self.idf]
        out = []
        for tf, n in zip(self.tfs, self.lens):
            norm = self.k1 * (1 - self.b + self.b * n / self.avg)
            out.append(sum(self.idf[t] * tf[t] * (self.k1 + 1) / (tf[t] + norm) for t in terms if t in tf))
        return out


def rrf(rankings: list[list[str]], k: int) -> list[str]:
    """여러 순위 목록을 1/(k + 순위)의 합으로 합친다. 동점은 ID 순."""
    score = Counter()
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            score[item] += 1 / (k + rank)
    return sorted(score, key=lambda item: (-score[item], item))
