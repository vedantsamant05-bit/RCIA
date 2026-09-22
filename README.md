# RCIA — Regulatory Change-Impact Agent

A working reference implementation of the RCIA design doc: an agentic
pipeline that ingests a new regulatory circular, retrieves the internal
policy clauses it affects, classifies the type of impact, drafts a
redline suggestion with citations, self-critiques the draft against the
source text, and routes the finding to a human review queue — with a
full audit trail.

```
rcia-project/
├── backend/
│   ├── app/
│   │   ├── main.py                # FastAPI app + all endpoints
│   │   ├── database.py            # SQLite schema + connection helper
│   │   ├── nlp_preprocessing.py   # clause segmentation, obligation extraction
│   │   ├── retrieval.py           # hybrid BM25 + TF-IDF retrieval + rerank
│   │   ├── agent.py               # 5-step agent state machine
│   │   └── corpus_data.py         # synthetic internal corpus + sample regs + eval labels
│   └── requirements.txt
├── frontend/
│   ├── index.html                 # dashboard shell
│   ├── style.css                  # design tokens + components
│   └── app.js                     # API calls + rendering (vanilla JS, no build step)
└── README.md
```

---

## 1. Quick start

### Backend

```bash
cd backend
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

The first request to the API auto-creates `backend/rcia.db` (SQLite) and
seeds it with:
- the synthetic internal policy corpus (8 documents, ~30 clauses)
- a hand-labeled evaluation set (10 pairs) for the `/api/eval` endpoint

Visit `http://localhost:8000/docs` for the interactive OpenAPI docs.

### Frontend

The frontend is plain HTML/CSS/JS — no build step. Just serve the folder:

```bash
cd frontend
python3 -m http.server 5500
```

Open `http://localhost:5500` in a browser. It talks to the backend at
`http://localhost:8000` (CORS is open for `*` in `main.py` — tighten this
before deploying anywhere real).

### Optional: enable real LLM reasoning

Without an API key, `classify_impact`, `draft_redline`, and `self_critique`
in `agent.py` all fall back to transparent rule-based heuristics — the
whole system runs for free, deterministically, with no external calls.
This is deliberate: it makes the demo reproducible and makes failure
cases easy to reason about without LLM non-determinism in the way.

To use Claude for the reasoning steps instead:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
uvicorn app.main:app --reload --port 8000
```

`agent.py` detects the environment variable at import time and routes
`classify_impact` / `draft_redline` / `self_critique` through the
Anthropic API, with a fallback to the heuristic path if the call fails.

---

## 2. Walking through the pipeline yourself

1. Open the **Ingest regulation** tab, click **Load a sample regulation**
   (there are 3 seeded samples — a data-erasure amendment, a digital
   lending rate-reset circular, and a KYC re-verification update), then
   **Run pipeline**.
2. Behind the scenes:
   - `nlp_preprocessing.segment_into_clauses()` splits the pasted text on
     numbered clause headers (`Clause 4(a):`, `Para 3.1:`, etc.) and
     extracts the obligated actor, action, and any deadline.
   - For each clause, `retrieval.HybridRetriever` scores every internal
     clause with BM25 + TF-IDF cosine similarity, fuses the two, then
     reranks with a heuristic that boosts shared numeric/legal tokens
     (the stand-in for a real cross-encoder — see the docstring in
     `retrieval.py` for exactly what to swap in for production).
   - The top candidates go through `agent.run_agent_pipeline()`: classify
     → second-hop cross-reference check → draft redline → self-critique
     citation check → confidence/risk scoring → routing decision.
   - Every step's output is written to `review_items` with a full JSON
     reasoning trace.
3. Open **Review queue** to see what got flagged, drafted, and routed.
   Expand an item to see the side-by-side clause comparison, the drafted
   redline, and the numbered reasoning trace. Approve or reject — this
   updates the audit log but never touches the regulation or corpus text
   itself (no auto-apply, per the guardrails in Section 4.5 of the
   design doc).
4. Open **Audit log** for the full read-only trail across every
   regulation ever ingested.
5. Open **Evaluation** and click **Run evaluation** to score the pipeline
   against the 10 hand-labeled (regulation clause, internal clause) pairs
   in `corpus_data.EVAL_LABELS`, reporting retrieval recall@5,
   classification precision/recall/F1, impact-type accuracy, and citation
   hallucination rate — this is what Section 5 of the design doc calls
   "what separates this from 80% of resume projects."

---

## 3. Design decisions worth defending in an interview

**Why clause-level segmentation instead of fixed-token chunking?**
`nlp_preprocessing.py` splits on numbered clause headers first, falling
back to paragraph/sentence boundaries. A fixed-token window over legal
text routinely cuts a single obligation in half or merges two unrelated
obligations into one chunk, which corrupts both retrieval and
downstream classification.

**Why hybrid retrieval?** Pure dense embeddings over-match on generic
compliance boilerplate (see `retrieval.py` docstring): many unrelated
policy clauses "sound" similar semantically even when legally unrelated.
BM25 catches the exact legal-term matches dense search sometimes
under-ranks; dense catches paraphrases BM25 misses entirely. Fusing both,
then reranking, is what keeps false positives out of the expensive
reasoning step.

**Why an explicit state machine instead of one big prompt?** Each step
(classify → hop-decision → draft → self-critique → score) has a
different failure mode and a different cost profile. Classification runs
on every retrieved candidate and should be cheap; drafting only runs on
clauses that already passed classification as impactful, so it's the
right place to spend a stronger/more expensive model's budget on. A
single mega-prompt collapses this cost/latency tradeoff and makes it much
harder to debug which step produced a wrong answer.

**Guardrails, concretely, in this codebase:**
- `POST /api/regulations/ingest` never writes to `internal_clauses` —
  only to `review_items`. There is no code path that applies a drafted
  redline automatically.
- `self_critique()` checks whether the draft's quoted citation actually
  appears in the source regulatory clause text; if it doesn't, the item
  is marked `citation_verified = False` and `score_confidence_risk()`
  forces its routing to `escalate` regardless of confidence.
- `score_confidence_risk()` implements confidence-threshold routing:
  below 0.72 confidence, or a low-confidence pass on `contradicts`/
  `tightens`/`loosens`, routes to `escalate` rather than a silent
  low-confidence pass.

**A real failure case found and fixed during development of this repo**
(this is the kind of story Section 8 of the design doc asks you to have
ready): the first version of `classify_impact()`'s negation check used
the raw substring `"shall not"`, which matched inside `"shall notify"` —
a pure notice-timing clause was wrongly classified as `contradicts`
because "notify" contains "not". Fixed by adding a `\b` word boundary
after "not" in the regex. This is exactly the class of bug you'd expect
from a keyword/regex-based classifier and is the concrete argument for
why the harder classification cases benefit from an LLM (or at minimum a
trained NLI/entailment model) rather than pattern matching — see the next
point.

**An honest limitation to disclose unprompted:** the heuristic classifier
still misses *double-negation* cases — e.g. a regulation clause reading
"shall not waive periodic re-KYC" against an internal clause reading
"re-KYC may be waived" both contain negation-adjacent tokens ("not",
"waive"/"waived"), so the current keyword heuristic cancels them out and
misclassifies this as `no-op` when it should be `contradicts`. This is
in `corpus_data.EVAL_LABELS` as one of the 10 labeled pairs specifically
so `/api/eval` surfaces it as a classification miss rather than hiding
it. Fixing this properly needs real semantic entailment (an LLM call or
a trained NLI model), which is exactly why `agent.py` is wired to prefer
Claude when `ANTHROPIC_API_KEY` is set — the heuristic path is a
zero-cost fallback for demos, not the intended production classifier.

---

## 4. What to build next (mapped to the original design doc's Phase 6+)

- Swap `retrieval._dense_scores()` for real embeddings (sentence-transformers
  locally, or a hosted embedding API) and `retrieval._rerank()` for an
  actual cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
- Move `database.py` from SQLite to PostgreSQL for concurrent writers,
  per the original tech-stack section.
- Expand `corpus_data.EVAL_LABELS` well past 10 pairs — the design doc
  targets 30–50 — and track precision/recall over time as the retrieval
  and classification logic changes.
- Add authentication and role-based access to the review queue
  (compliance reviewer vs. auditor-read-only) before this touches any
  real internal policy data.
