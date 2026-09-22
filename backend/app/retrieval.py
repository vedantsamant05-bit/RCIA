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
model): to keep this project runnable with zero external services and
no API key, the "dense" component here is TF-IDF cosine similarity, not
a trained embedding model. It captures shared-vocabulary semantic
similarity but NOT true paraphrase understanding. For a production /
resume-defensible version, swap `_dense_scores()` below for one of:
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
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re

# ---------------------------------------------------------------------------
# Relevance gate
# ---------------------------------------------------------------------------
# Candidates whose rerank_score falls below this threshold are returned but
# flagged as irrelevant BEFORE they reach the agent classifier. This prevents
# a weak keyword overlap from being promoted to contradicts/tightens/loosens
# purely because numbers happen to co-occur.
#
# The gate uses an absolute score, not a per-query rank score. This prevents
# an unrelated candidate from becoming score=1 simply because it was the
# least-unrelated result in a weak candidate set.
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


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())


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
    Built fresh over the current internal-clause corpus. For a corpus this
    small (tens to low-hundreds of clauses) rebuilding the index per query
    is cheap; at real scale you'd persist the BM25/vector index instead of
    rebuilding it in __init__ every time (e.g. Elasticsearch + FAISS/
    Weaviate/Pinecone, per the tech stack in Section 6).
    """

    def __init__(self, corpus_clauses: list[dict]):
        # corpus_clauses: [{id, doc_title, section, text}, ...]
        self.corpus = corpus_clauses
        self.texts = [c["text"] for c in corpus_clauses]
        self._tokenized = [_tokenize(t) for t in self.texts]
        self.bm25 = BM25Okapi(self._tokenized) if self.texts else None

        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self._tfidf_matrix = (
            self.vectorizer.fit_transform(self.texts) if self.texts else None
        )

    def _bm25_scores(self, query: str) -> np.ndarray:
        if not self.bm25:
            return np.array([])
        scores = np.array(self.bm25.get_scores(_tokenize(query)))
        return _minmax(scores)

    def _dense_scores(self, query: str) -> np.ndarray:
        # See module docstring: TF-IDF cosine similarity stands in for a
        # trained embedding model here.
        if self._tfidf_matrix is None:
            return np.array([])
        q_vec = self.vectorizer.transform([query])
        return cosine_similarity(q_vec, self._tfidf_matrix)[0]

    def _rerank(self, query: str, candidate_text: str, dense_score: float) -> float:
        """
        Absolute relevance proxy used by the pre-classification gate.

        Dense similarity captures broader topical matches while meaningful
        query-term coverage makes the score resistant to generic compliance
        boilerplate. Numeric overlap is deliberately not a relevance boost:
        dates and limits are useful for classification only after a candidate
        has established that it discusses the same obligation.
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
        """Return the top-k candidates from the corpus, each annotated with
        `below_threshold=True` when the rerank score is too weak to justify
        running the full agent pipeline. Callers decide what to do with those;
        typically they still log the candidate (for auditability) but skip
        classification so no false-positive impacts are generated."""
        if not self.corpus:
            return []

        bm25_scores = self._bm25_scores(query)
        dense_scores = self._dense_scores(query)
        hybrid = bm25_weight * bm25_scores + (1 - bm25_weight) * dense_scores

        ranked_idx = np.argsort(-hybrid)[:top_k]
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
                    bm25_score=float(bm25_scores[i]) if len(bm25_scores) else 0.0,
                    dense_score=float(dense_scores[i]) if len(dense_scores) else 0.0,
                    hybrid_score=fused,
                    rerank_score=rerank_score,
                    below_threshold=rerank_score < threshold,
                )
            )
        # Final ordering is by rerank_score, matching a real
        # retrieve-then-rerank pipeline.
        results.sort(key=lambda r: -r.rerank_score)
        return results


def _minmax(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-9:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)
