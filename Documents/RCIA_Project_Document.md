# Regulatory Change-Impact Agent (RCIA)
### Agentic AI + RAG + NLP for Automated Regulatory Compliance Monitoring

---

## 1. Problem Statement

Regulated organizations (banks, fintechs, insurers, healthcare providers) maintain large internal corpora of policies, SOPs, and contracts. Regulators (RBI, SEBI, GDPR, DPDP Act, SEC, etc.) issue amendments and circulars continuously. Today, compliance teams manually read every new circular and manually cross-reference it against internal documents to determine which policies are now outdated or non-compliant.

This process is:
- **Slow** — manual review can take days per circular
- **Error-prone** — humans miss indirect/multi-clause impacts
- **Expensive** — requires dedicated compliance headcount
- **Unauditable at scale** — no systematic trail of what was checked, when, against what

**Goal:** Build a system that ingests new regulatory text, retrieves the internal documents/clauses it affects, reasons about the *type* of impact, drafts a suggested revision with citations, and routes it to a human for approval — with full audit logging.

---

## 2. Why This Project (Not Generic RAG)

| Generic "Chat with PDF" RAG | RCIA |
|---|---|
| Single document, single-hop Q&A | Multi-document, multi-hop comparative reasoning |
| Keyword or basic dense retrieval | Hybrid retrieval (BM25 + dense) with legal clause-level chunking |
| No evaluation beyond "it answers questions" | Precision/recall on impact detection, citation hallucination rate, human-agreement rate |
| Agent = wrapper around one LLM call | Agent = decompose → retrieve → classify → decide on deeper retrieval → draft → self-critique → escalate |
| No guardrails | Confidence thresholds, mandatory citations, audit trail, no auto-apply of changes |

This maps to a real, recognizable industry problem — every compliance/legal/fintech team faces this. It is not a toy use case.

---

## 3. System Architecture

```
┌─────────────────────┐
│ Regulatory Source    │  (RBI/SEBI circulars, GDPR amendments — public sources)
│ Ingestion            │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐
│ NLP Preprocessing     │  clause segmentation, entity/obligation extraction,
│                       │  effective-date extraction
└──────────┬───────────┘
           │
           ▼
┌─────────────────────┐        ┌──────────────────────┐
│ Hybrid Retrieval      │◄──────┤ Internal Policy Corpus │
│ (BM25 + dense + rerank)│      │ (self-built/synthetic) │
└──────────┬───────────┘        └──────────────────────┘
           │  candidate clauses
           ▼
┌─────────────────────────────────────────────┐
│ Agent Orchestrator (state machine)            │
│  1. Classify impact type                       │
│    (contradicts / tightens / loosens / no-op)  │
│  2. Decide if second retrieval hop needed        │
│     (cross-clause dependency check)             │
│  3. Draft redline suggestion                    │
│  4. Self-critique against source regulation       │
│  5. Assign confidence + risk score               │
└──────────┬────────────────────────────────────┘
           │
           ▼
┌─────────────────────┐
│ Human Review Queue    │  full audit trail: retrieved clauses, agent
│ + Audit Log           │  reasoning steps, confidence, citations
└──────────────────────┘
```

---

## 4. Core Components

### 4.1 Data
- **Regulatory source data:** Public circulars from RBI/SEBI, or GDPR/DPDP Act amendment texts (freely available, real).
- **Internal policy corpus:** Since real internal bank policies aren't accessible, build a **synthetic but realistic** corpus — 30–50 policy documents generated to mimic real SOP/contract structure, seeded with intentional overlaps with real regulatory clauses so impact detection is testable.

### 4.2 NLP Layer
- Clause-level segmentation (not fixed-token chunking — legal text breaks on obligations, not paragraphs)
- Obligation/entity extraction (who must do what, by when)
- Effective-date and jurisdiction extraction

### 4.3 Retrieval Layer
- Hybrid search: BM25 (exact regulatory term matches like "material adverse change") + dense embeddings (semantic similarity)
- Re-ranking stage (cross-encoder) to filter false-positive candidate clauses
- Justify chunk size/strategy choice explicitly — this is a key interview talking point

### 4.4 Agentic Reasoning Layer
- Step-wise state machine (not a single mega-prompt):
  1. Classify each candidate clause's relationship to the new regulation
  2. Trigger a second retrieval hop if cross-referenced clauses exist elsewhere in the corpus
  3. Draft a redline suggestion with inline citation to the exact regulatory source clause
  4. Self-critique step: does the citation actually say what the draft claims? (mitigates hallucination)
  5. Confidence + risk scoring to decide auto-flag vs. auto-dismiss vs. escalate

### 4.5 Guardrails (non-negotiable, and a strong interview topic)
- No auto-application of any change — always human-in-the-loop
- Every claim must carry a source citation; unverified claims are discarded, not shown
- Full audit log per decision: inputs, retrieved evidence, reasoning trace, confidence, reviewer action
- Confidence-threshold-based routing (low confidence → mandatory review, not silent pass)

---

## 5. Evaluation Plan (this is what separates this from 80% of resume projects)

Build a small **labeled test set** yourself:
- 30–50 (regulation, internal clause) pairs manually labeled as impacted / not impacted, with impact type
- Metrics to report:
  - **Retrieval precision/recall** — did the system find the right candidate clauses?
  - **Classification accuracy** — did it correctly identify impact type?
  - **Citation hallucination rate** — % of citations that don't actually support the claim (check manually)
  - **Human-agreement rate** — how often does a human reviewer agree with the agent's assessment?

Document at least one real failure case you found and fixed (e.g., "the agent flagged unrelated clauses because dense retrieval alone over-matched on generic compliance language; fixed by adding BM25 hybrid weighting").

---

## 6. Tech Stack (suggested, not mandatory)

- **LLM:** Claude or GPT-4-class model for reasoning/drafting; a cheaper/smaller model for classification steps (cost/latency tradeoff — talk about this explicitly)
- **Retrieval:** FAISS/Weaviate/Pinecone for dense; Elasticsearch/BM25 for sparse; a cross-encoder re-ranker
- **Orchestration:** LangGraph, or a hand-rolled state machine (be ready to defend why you chose framework vs. custom)
- **Backend:** FastAPI
- **Audit/state store:** PostgreSQL for structured audit trail
- **Frontend (optional, for demo):** simple review dashboard showing flagged documents, citations, confidence

---

## 7. Build Order (suggested phased approach)

1. **Phase 1 — Data:** Collect real regulatory texts; build synthetic internal policy corpus with deliberate overlaps.
2. **Phase 2 — Retrieval:** Implement clause-level chunking, hybrid retrieval, re-ranking. Evaluate retrieval precision/recall before touching the agent.
3. **Phase 3 — Agent:** Build the state-machine reasoning pipeline (classify → hop decision → draft → self-critique → score).
4. **Phase 4 — Guardrails + Audit:** Confidence thresholds, citation verification, audit logging.
5. **Phase 5 — Evaluation:** Build labeled test set, measure metrics, document failure cases and fixes.
6. **Phase 6 — Demo layer:** Minimal review dashboard for presentation purposes.

---

## 8. Talking Points for Interviews / Resume Defense (30–45 min)

- Why naive single-hop RAG fails on regulatory/legal text, and what you did instead
- Chunking strategy decision and why (clause-level vs. fixed-token)
- Why hybrid retrieval was necessary (exact legal term matching + semantic similarity)
- Why this needed to be agentic — specific failure mode of a single LLM call you observed
- Cost/latency tradeoffs — which steps used cheaper vs. more capable models, and why
- Evaluation methodology and actual numbers (even if the dataset is small — a real number beats "it worked well")
- A specific failure case you diagnosed and how you fixed it
- Guardrail design — why no auto-apply, why citation-mandatory, why audit logging matters for a regulated-industry buyer
- Honest limitation you'd disclose unprompted (e.g., confidence scoring still noisy on ambiguous regulatory language, synthetic corpus doesn't fully capture real legal complexity)

---

## 9. What Will Get You Flagged as "Tutorial-Level" — Avoid This

- Wiring LangChain + GPT-4 + a vector DB with no evaluation and calling it done
- No labeled test set, no metrics beyond "it works when I tried it"
- No hybrid retrieval — relying only on dense embeddings
- No guardrails — implying the system auto-applies changes
- Inability to explain a specific failure case you personally diagnosed and fixed
