import re
from backend.app.corpus_data import INTERNAL_POLICIES
from backend.app.database import get_conn
from backend.app.nlp_preprocessing import segment_into_clauses
from test_eval_retrieval import rerank, TOPICS, STOP, GENERIC, LightweightTfidf, PureBM25Okapi, _tokenize
from test_eval_classifier import classify_test

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

print(f"=== KYC TEST BREAKDOWN ({len(clauses)} clauses) ===\n")
for i, cl in enumerate(clauses):
    q = cl.text
    d_scores = vectorizer.score(q)
    b_scores = bm25.get_scores(_tokenize(q))
    lo, hi = min(b_scores), max(b_scores)
    b_norm = [(x - lo)/(hi - lo) if hi - lo > 1e-9 else 0.0 for x in b_scores]
    hybrid = [0.5 * b + 0.5 * d for b, d in zip(b_norm, d_scores)]
    ranked_idx = sorted(range(len(hybrid)), key=lambda idx: hybrid[idx], reverse=True)[:3]
    print(f"----------------------------------------------------------------------")
    print(f"REGULATORY CLAUSE {i+1}:")
    print(f"\"{q}\"")
    for idx in ranked_idx:
        c = corpus[idx]
        score = rerank(q, c['text'], d_scores[idx])
        pred_type, reason = classify_test(q, c['text'])
        # If below threshold, it's auto-dismissed
        below_thresh = score < 0.28
        final_impact = "no-op" if below_thresh else pred_type
        print(f"\n  Matched Internal Clause: {c['id']}")
        print(f"  Similarity/Relevance Score: {score:.3f} (below_threshold: {below_thresh})")
        print(f"  Classification: {final_impact}")
        print(f"  Reason: {reason}")
