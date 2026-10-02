import re
from backend.app.corpus_data import INTERNAL_POLICIES, EVAL_LABELS
from backend.app.retrieval import _tokenize, ENGLISH_STOP_WORDS, LightweightTfidf, PureBM25Okapi

STOP = set(ENGLISH_STOP_WORDS) - {'system', 'interest'}
GENERIC = {'entity', 'entities', 'provide', 'provided', 'shall', 'must', 'within', 'also', 'including', 'relevant', 'appropriate'}

TOPICS = {
    'erase': 'data_lifecycle', 'erasure': 'data_lifecycle', 'delete': 'data_lifecycle', 'deletion': 'data_lifecycle',
    'retained': 'data_lifecycle', 'retention': 'data_lifecycle', 'purge': 'data_lifecycle', 'purged': 'data_lifecycle',
    'breach': 'incident_security', 'incident': 'incident_security', 'incidents': 'incident_security',
    'notify': 'incident_security', 'notification': 'incident_security', 'reported': 'incident_security', 'reporting': 'incident_security',
    'encryption': 'incident_security', 'security': 'incident_security', 'access': 'incident_security', 'privilege': 'incident_security',
    'vendor': 'vendor_transfer', 'processor': 'vendor_transfer', 'subprocessor': 'vendor_transfer',
    'transfer': 'vendor_transfer', 'transfers': 'vendor_transfer', 'cross-border': 'vendor_transfer', 'jurisdiction': 'vendor_transfer',
    'kyc': 'kyc_onboarding', 're-kyc': 'kyc_onboarding', 'rekyc': 'kyc_onboarding', 'onboarding': 'kyc_onboarding',
    'verification': 'kyc_onboarding', 'verify': 'kyc_onboarding', 'verified': 'kyc_onboarding', 'identification': 'kyc_onboarding',
    'pep': 'kyc_onboarding', 'peps': 'kyc_onboarding', 'v-cip': 'kyc_onboarding', 'vcip': 'kyc_onboarding',
    'beneficial': 'kyc_onboarding', 'ownership': 'kyc_onboarding', 'updation': 'kyc_onboarding',
    'risk': 'risk_monitoring', 'profile': 'risk_monitoring', 'profiling': 'risk_monitoring', 'classification': 'risk_monitoring',
    'monitoring': 'risk_monitoring', 'monitor': 'risk_monitoring', 'monitored': 'risk_monitoring',
    'alert': 'risk_monitoring', 'alerts': 'risk_monitoring', 'overdue': 'risk_monitoring', 'deadline': 'risk_monitoring',
    'escalate': 'risk_monitoring', 'escalated': 'risk_monitoring', 'escalation': 'risk_monitoring',
    'loan': 'lending_credit', 'loans': 'lending_credit', 'lending': 'lending_credit', 'borrower': 'lending_credit', 'borrowers': 'lending_credit',
    'lender': 'lending_credit', 'foreclosure': 'lending_credit', 'prepayment': 'lending_credit', 'interest': 'lending_credit',
    'rate': 'lending_credit', 'floating': 'lending_credit', 'disbursal': 'lending_credit',
    'grievance': 'grievance', 'grievances': 'grievance', 'redressal': 'grievance', 'ombudsman': 'grievance', 'complaint': 'grievance',
    'marketing': 'marketing_consent', 'consent': 'marketing_consent', 'opt-out': 'marketing_consent', 'ndnc': 'marketing_consent',
    'audit': 'audit_controls', 'trail': 'audit_controls', 'remediation': 'audit_controls', 'controls': 'audit_controls',
    'procedure': 'audit_controls', 'procedures': 'audit_controls', 'workflow': 'audit_controls', 'crm': 'audit_controls'
}

def rerank(query, cand, dense):
    q_terms = {t for t in _tokenize(query) if len(t) > 2 and t not in STOP and t not in GENERIC}
    c_terms = {t for t in _tokenize(cand) if len(t) > 2 and t not in STOP and t not in GENERIC}
    raw_shared = q_terms & c_terms
    q_topics = {TOPICS[t] for t in q_terms if t in TOPICS}
    c_topics = {TOPICS[t] for t in c_terms if t in TOPICS}
    shared_topics = q_topics & c_topics

    if not shared_topics and len(raw_shared) == 0 and dense < 0.12:
        return 0.0

    topic_bonus = 0.22 if shared_topics else 0.0
    coverage = min(1.0, len(raw_shared) / max(len(q_terms), 1))
    if shared_topics:
        score = topic_bonus + 0.50 * dense + 0.28 * coverage
    else:
        score = 0.60 * dense + 0.40 * coverage
    return min(1.0, max(0.0, score))

corpus = []
for doc, clauses in INTERNAL_POLICIES.items():
    for sec, txt in clauses:
        corpus.append({'id': f"{doc}::{sec}", 'doc_title': doc, 'section': sec, 'text': txt})

texts = [c['text'] for c in corpus]
vectorizer = LightweightTfidf().fit_transform(texts)
tokenized = [_tokenize(t) for t in texts]
bm25 = PureBM25Okapi(tokenized)

print('--- EVAL SET RETRIEVAL WITH NEW SCORING ---')
for lbl in EVAL_LABELS:
    q = lbl['regulation_clause_text']
    d_scores = vectorizer.score(q)
    b_scores = bm25.get_scores(_tokenize(q))
    lo, hi = min(b_scores), max(b_scores)
    b_norm = [(x - lo)/(hi - lo) if hi - lo > 1e-9 else 0.0 for x in b_scores]
    hybrid = [0.5 * b + 0.5 * d for b, d in zip(b_norm, d_scores)]
    ranked_idx = sorted(range(len(hybrid)), key=lambda i: hybrid[i], reverse=True)[:5]
    cands = [corpus[i] for i in ranked_idx]
    target = f"{lbl['internal_doc']}::{lbl['internal_section']}"
    cand_scores = [rerank(q, c['text'], d_scores[corpus.index(c)]) for c in cands]
    top_cand = cands[0]['id']
    target_in_top5 = any(c['id'] == target for c in cands)
    target_score = [s for c, s in zip(cands, cand_scores) if c['id'] == target]
    t_score = target_score[0] if target_score else 0.0
    print(f"{target:45} | in_top5={target_in_top5} | score={t_score:.3f} | top={top_cand}")
