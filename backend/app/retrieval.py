"""
retrieval.py
------------
Section 4.3 of the project doc.

Hybrid retrieval = BM25 (sparse, exact-term) + a dense/semantic component,
combined and then reranked, rather than relying on either signal alone.

Why hybrid, concretely:
- BM25 wins when the regulation uses an exact legal term ("material
  adverse change", "foreclosure charge") that also appears verbatim in
  the internal policy -- dense embeddings can under-rank an exact match
  if the surrounding sentence structure differs.
- Dense/semantic search wins when the regulation and the internal clause
  express the same obligation in different words (e.g. "erase personal
  data" vs. "does not provide a mechanism for deletion") -- BM25 alone
  would miss this because there's no shared vocabulary.
- Relying on dense embeddings alone is flagged explicitly in the project
  doc (Section 9) as a "tutorial-level" tell, because generic compliance
  boilerplate makes many unrelated clauses look semantically similar,
  producing false positives that only exact-term/BM25 signal can filter.

IMPLEMENTATION NOTE (read this before treating this as "the" dense
model): to keep this project runnable with zero external services, no
API key, and completely deployable within serverless constraints (e.g. Vercel),
the retrieval and ranking algorithms are implemented in pure Python.
TF-IDF cosine similarity stands in for a trained embedding model here.
It captures shared-vocabulary semantic similarity but NOT true paraphrase understanding.
For a production / resume-defensible version, swap `_dense_scores()` below for one of:
  - sentence-transformers (e.g. `all-MiniLM-L6-v2`) run locally, or
  - a hosted embedding API (OpenAI text-embedding-3, Voyage, Cohere).
The rest of the pipeline (BM25 fusion, reranking, hop logic) is
unaffected by that swap -- that's the point of separating the layers.

Similarly, `_rerank()` is a heuristic stand-in for a real cross-encoder
(e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2` via sentence-transformers).
A cross-encoder scores the (query, candidate) pair jointly instead of
comparing independent vectors, which is what actually filters
false-positive candidates before they reach the LLM reasoning step.
"""

from dataclasses import dataclass
from collections import Counter
import math
import re

# ---------------------------------------------------------------------------
# Stop words (Standard English Stop Words)
# ---------------------------------------------------------------------------
ENGLISH_STOP_WORDS = frozenset([
    'a', 'about', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost',
    'alone', 'along', 'already', 'also', 'although', 'always', 'am', 'among', 'amongst', 'amoungst',
    'amount', 'an', 'and', 'another', 'any', 'anyhow', 'anyone', 'anything', 'anyway', 'anywhere',
    'are', 'around', 'as', 'at', 'back', 'be', 'became', 'because', 'become', 'becomes', 'becoming',
    'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 'between',
    'beyond', 'bill', 'both', 'bottom', 'but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con',
    'could', 'couldnt', 'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during',
    'each', 'eg', 'eight', 'either', 'eleven', 'else', 'elsewhere', 'empty', 'enough', 'etc', 'even',
    'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 'fifty',
    'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four',
    'from', 'front', 'full', 'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he',
    'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 'hers', 'herself', 'him',
    'himself', 'his', 'how', 'however', 'hundred', 'i', 'ie', 'if', 'in', 'inc', 'indeed', 'interest',
    'into', 'is', 'it', 'its', 'itself', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd',
    'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 'mine', 'more', 'moreover', 'most',
    'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never',
    'nevertheless', 'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now',
    'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 'only', 'onto', 'or', 'other', 'others',
    'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'part', 'per', 'perhaps', 'please',
    'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several',
    'she', 'should', 'show', 'side', 'since', 'sincere', 'six', 'sixty', 'so', 'some', 'somehow',
    'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 'take',
    'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there',
    'thereafter', 'thereby', 'therefore', 'therein', 'thereupon', 'these', 'they', 'thick', 'thin',
    'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to',
    'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until',
    'up', 'upon', 'us', 'very', 'via', 'was', 'we', 'well', 'were', 'what', 'whatever', 'when',
    'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon',
    'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose',
    'why', 'will', 'with', 'within', 'without', 'would', 'yet', 'you', 'your', 'yours', 'yourself',
    'yourselves'
])

# ---------------------------------------------------------------------------
# Relevance gate
# ---------------------------------------------------------------------------
RELEVANCE_THRESHOLD: float = 0.38
GENERIC_RELEVANCE_TERMS = {
    "account", "business", "customer", "customers", "data", "days", "entity",
    "entities", "information", "policy", "provide", "provided", "record",
    "records", "request", "requests", "required", "shall", "within",
}
RELEVANCE_TOPIC_GROUPS = {
    "erase": "data_lifecycle",
    "erasure": "data_lifecycle",
    "delete": "data_lifecycle",
    "deletion": "data_lifecycle",
    "retained": "data_lifecycle",
    "retention": "data_lifecycle",
    "breach": "incident_response",
    "incident": "incident_response",
    "incidents": "incident_response",
    "notify": "incident_response",
    "notification": "incident_response",
    "reported": "incident_response",
    "reporting": "incident_response",
    "cybersecurity": "security_controls",
    "encryption": "security_controls",
    "security": "security_controls",
    "vendor": "vendor_security",
    "processor": "vendor_security",
    "subprocessor": "vendor_security",
}

_WORD_PATTERN = re.compile(r"[a-zA-Z]+")
_TOKEN_PATTERN = re.compile(r"(?u)\b\w\w+\b")


def _tokenize(text: str) -> list[str]:
    return _WORD_PATTERN.findall(text.lower())


def _extract_ngrams(text: str) -> list[str]:
    words = _TOKEN_PATTERN.findall(text.lower())
    unigrams = [w for w in words if w not in ENGLISH_STOP_WORDS]
    bigrams = [
        f"{words[i]} {words[i+1]}"
        for i in range(len(words) - 1)
        if words[i] not in ENGLISH_STOP_WORDS and words[i+1] not in ENGLISH_STOP_WORDS
    ]
    return unigrams + bigrams


class LightweightTfidf:
    """Lightweight, zero-dependency TF-IDF vectorizer matching scikit-learn smooth_idf."""

    def __init__(self):
        self.idf = {}
        self.doc_vectors = []

    def fit_transform(self, docs: list[str]):
        n_docs = len(docs)
        if n_docs == 0:
            return self
        doc_ngrams = [_extract_ngrams(d) for d in docs]
        df = Counter()
        for ngrams in doc_ngrams:
            df.update(set(ngrams))

        # Sklearn formula: log((1 + n_samples) / (1 + df)) + 1
        self.idf = {
            term: math.log((1 + n_docs) / (1 + count)) + 1.0
            for term, count in df.items()
        }

        self.doc_vectors = []
        for ngrams in doc_ngrams:
            counts = Counter(ngrams)
            vec = {t: counts[t] * self.idf[t] for t in counts if t in self.idf}
            norm = math.sqrt(sum(v * v for v in vec.values()))
            if norm > 0:
                vec = {t: v / norm for t, v in vec.items()}
            self.doc_vectors.append(vec)
        return self

    def score(self, query: str) -> list[float]:
        q_ngrams = _extract_ngrams(query)
        counts = Counter(q_ngrams)
        q_vec = {t: counts[t] * self.idf[t] for t in counts if t in self.idf}
        norm = math.sqrt(sum(v * v for v in q_vec.values()))
        if norm > 0:
            q_vec = {t: v / norm for t, v in q_vec.items()}
        else:
            return [0.0] * len(self.doc_vectors)

        scores = []
        for d_vec in self.doc_vectors:
            s = sum(q_vec[t] * d_val for t, d_val in d_vec.items() if t in q_vec)
            scores.append(s)
        return scores


class PureBM25Okapi:
    """Pure-Python BM25Okapi implementation without external dependencies."""

    def __init__(self, corpus_tokens: list[list[str]], k1=1.5, b=0.75, epsilon=0.25):
        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.corpus_size = len(corpus_tokens)
        self.doc_len = [len(doc) for doc in corpus_tokens]
        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0
        self.doc_freqs = [Counter(doc) for doc in corpus_tokens]
        self.idf = {}
        self._calc_idf()

    def _calc_idf(self):
        df = Counter()
        for doc in self.doc_freqs:
            df.update(doc.keys())
        idf_sum = 0
        negative_idfs = []
        for word, freq in df.items():
            idf = math.log(self.corpus_size - freq + 0.5) - math.log(freq + 0.5)
            self.idf[word] = idf
            idf_sum += idf
            if idf < 0:
                negative_idfs.append(word)
        avg_idf = idf_sum / len(self.idf) if self.idf else 0
        eps = self.epsilon * avg_idf
        for word in negative_idfs:
            self.idf[word] = eps

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = [0.0] * self.corpus_size
        for i in range(self.corpus_size):
            d_len = self.doc_len[i]
            d_freqs = self.doc_freqs[i]
            denom_const = self.k1 * (1 - self.b + self.b * d_len / self.avgdl) if self.avgdl > 0 else 1.0
            doc_score = 0.0
            for q in query_tokens:
                q_freq = d_freqs.get(q, 0)
                if q_freq > 0:
                    idf_val = self.idf.get(q, 0.0)
                    doc_score += idf_val * (q_freq * (self.k1 + 1)) / (q_freq + denom_const)
            scores[i] = doc_score
        return scores


@dataclass
class RetrievalCandidate:
    clause_id: str
    doc_title: str
    section: str
    text: str
    bm25_score: float
    dense_score: float
    hybrid_score: float
    rerank_score: float
    below_threshold: bool = False  # True → skip classification, log as irrelevant


class HybridRetriever:
    """
    Built fresh over the current internal-clause corpus. Pure Python implementation
    optimized for fast execution and zero external dependencies.
    """

    def __init__(self, corpus_clauses: list[dict]):
        self.corpus = corpus_clauses
        self.texts = [c["text"] for c in corpus_clauses]
        self._tokenized = [_tokenize(t) for t in self.texts]
        self.bm25 = PureBM25Okapi(self._tokenized) if self.texts else None
        self.vectorizer = LightweightTfidf().fit_transform(self.texts) if self.texts else None

    def _bm25_scores(self, query: str) -> list[float]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(_tokenize(query))
        return self._minmax(scores)

    def _dense_scores(self, query: str) -> list[float]:
        if not self.vectorizer:
            return []
        return self.vectorizer.score(query)

    def _minmax(self, arr: list[float]) -> list[float]:
        if not arr:
            return []
        lo, hi = min(arr), max(arr)
        if hi - lo < 1e-9:
            return [0.0] * len(arr)
        return [(x - lo) / (hi - lo) for x in arr]

    def _rerank(self, query: str, candidate_text: str, dense_score: float) -> float:
        """
        Absolute relevance proxy used by the pre-classification gate.
        """
        query_terms = {
            t for t in _tokenize(query)
            if len(t) > 2
            and t not in ENGLISH_STOP_WORDS
            and t not in GENERIC_RELEVANCE_TERMS
        }
        candidate_terms = {
            t for t in _tokenize(candidate_text)
            if t not in GENERIC_RELEVANCE_TERMS
        }
        raw_shared_terms = query_terms & candidate_terms
        query_topics = {
            RELEVANCE_TOPIC_GROUPS[t] for t in query_terms if t in RELEVANCE_TOPIC_GROUPS
        }
        candidate_topics = {
            RELEVANCE_TOPIC_GROUPS[t] for t in candidate_terms if t in RELEVANCE_TOPIC_GROUPS
        }
        shared_topics = query_topics & candidate_topics
        if not shared_topics and len(raw_shared_terms) < 2:
            return 0.0
        shared_terms = shared_topics or raw_shared_terms
        coverage = len(shared_terms) / max(len(query_terms), 1)
        topic_bonus = 0.20 if shared_topics else 0.0
        return min(1.0, topic_bonus + 0.65 * dense_score + 0.35 * coverage)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.5,
        threshold: float = RELEVANCE_THRESHOLD,
    ) -> list[RetrievalCandidate]:
        if not self.corpus:
            return []

        bm25_scores = self._bm25_scores(query)
        dense_scores = self._dense_scores(query)
        hybrid = [
            bm25_weight * b + (1 - bm25_weight) * d
            for b, d in zip(bm25_scores, dense_scores)
        ]

        ranked_idx = sorted(range(len(hybrid)), key=lambda i: hybrid[i], reverse=True)[:top_k]
        results = []
        for i in ranked_idx:
            c = self.corpus[i]
            fused = float(hybrid[i])
            rerank_score = self._rerank(query, c["text"], float(dense_scores[i]))
            results.append(
                RetrievalCandidate(
                    clause_id=c["id"],
                    doc_title=c["doc_title"],
                    section=c["section"],
                    text=c["text"],
                    bm25_score=float(bm25_scores[i]) if bm25_scores else 0.0,
                    dense_score=float(dense_scores[i]) if dense_scores else 0.0,
                    hybrid_score=fused,
                    rerank_score=rerank_score,
                    below_threshold=rerank_score < threshold,
                )
            )
        results.sort(key=lambda r: -r.rerank_score)
        return results
