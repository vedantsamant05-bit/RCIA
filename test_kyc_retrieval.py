import re
from backend.app.corpus_data import INTERNAL_POLICIES
from backend.app.database import get_conn
from backend.app.nlp_preprocessing import segment_into_clauses
from test_eval_retrieval import rerank, TOPICS, STOP, GENERIC, LightweightTfidf, PureBM25Okapi, _tokenize

with get_conn() as conn:
    row = conn.execute('SELECT raw_text FROM regulations WHERE id = ?', ('reg-c9a9a6d0',)).fetchone()
    raw_text = row['raw_text']

clauses = segment_into_clauses(raw_text)

corpus = []
for doc, doc_clauses in INTERNAL_POLICIES.items():
    for sec, txt in doc_clauses:
        corpus.append({'id': f"{doc}::{sec}", 'doc_title': doc, 'section': sec, 'text': txt})

texts = [c['text'] for c in corpus]
vectorizer = LightweightTfidf().fit_transform(texts)
tokenized = [_tokenize(t) for t in texts]
bm25 = PureBM25Okapi(tokenized)

print(f"Total KYC test clauses: {len(clauses)}\n")
for i, cl in enumerate(clauses):
    q = cl.text
    d_scores = vectorizer.score(q)
    b_scores = bm25.get_scores(_tokenize(q))
    lo, hi = min(b_scores), max(b_scores)
    b_norm = [(x - lo)/(hi - lo) if hi - lo > 1e-9 else 0.0 for x in b_scores]
    hybrid = [0.5 * b + 0.5 * d for b, d in zip(b_norm, d_scores)]
    ranked_idx = sorted(range(len(hybrid)), key=lambda idx: hybrid[idx], reverse=True)[:3]
    print(f"=== Clause {i+1}: {q[:75]}... ===")
    for idx in ranked_idx:
        c = corpus[idx]
        score = rerank(q, c['text'], d_scores[idx])
        print(f"  [{score:.3f}] {c['id']}")
