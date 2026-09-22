"""
main.py
-------
FastAPI backend for RCIA (Regulatory Change-Impact Agent).

Run with:
    uvicorn app.main:app --reload --port 8000

Endpoints map directly onto the architecture diagram in the project doc:
  POST /api/regulations/ingest   -> ingestion + NLP + retrieval + agent pipeline
  GET  /api/regulations          -> list ingested regulations
  GET  /api/regulations/{id}/results -> review-queue items for one regulation
  GET  /api/corpus               -> browse internal policy corpus
  POST /api/corpus               -> add a new internal policy document
  GET  /api/review               -> full review queue (all regulations)
  POST /api/review/{item_id}     -> reviewer approves/rejects/escalates
  GET  /api/audit                -> full audit log (read-only trace)
  GET  /api/eval                 -> compute Section 5 metrics against the
                                     hand-labeled test set
"""

import uuid
from typing import Optional, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from . import database as db
from .corpus_data import INTERNAL_POLICIES, SAMPLE_REGULATIONS, EVAL_LABELS
from .nlp_preprocessing import segment_into_clauses
from .retrieval import HybridRetriever, RELEVANCE_THRESHOLD
from .agent import run_agent_pipeline

app = FastAPI(title="RCIA - Regulatory Change-Impact Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # demo only; restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Startup: init DB + seed synthetic corpus if empty
# ---------------------------------------------------------------------------
@app.on_event("startup")
def on_startup():
    db.init_db()
    with db.get_conn() as conn:
        count = conn.execute("SELECT COUNT(*) c FROM internal_clauses").fetchone()["c"]
        if count == 0:
            _seed_corpus(conn)
            _seed_eval_labels(conn)


def _seed_corpus(conn):
    for doc_title, clauses in INTERNAL_POLICIES.items():
        for section, text in clauses:
            conn.execute(
                "INSERT INTO internal_clauses (id, doc_title, section, clause_text, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (f"ic-{uuid.uuid4().hex[:8]}", doc_title, section, text, db.now_iso()),
            )


def _seed_eval_labels(conn):
    for row in EVAL_LABELS:
        conn.execute(
            "INSERT INTO eval_labels (id, regulation_clause_text, internal_clause_id, "
            "is_impacted, true_impact_type, notes) VALUES (?, ?, ?, ?, ?, ?)",
            (
                f"ev-{uuid.uuid4().hex[:8]}",
                row["regulation_clause_text"],
                f"{row['internal_doc']}::{row['internal_section']}",
                row["is_impacted"],
                row.get("true_impact_type"),
                None,
            ),
        )


def _load_corpus(conn) -> list[dict]:
    rows = conn.execute("SELECT * FROM internal_clauses").fetchall()
    return [
        {"id": r["id"], "doc_title": r["doc_title"], "section": r["section"], "text": r["clause_text"]}
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class RegulationIn(BaseModel):
    title: str
    source: Optional[str] = None
    text: str


class CorpusDocIn(BaseModel):
    doc_title: str
    # list of {section, text}
    clauses: list[dict]


class ReviewActionIn(BaseModel):
    action: Literal["approve", "reject", "escalate_confirmed"]
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Ingestion + pipeline
# ---------------------------------------------------------------------------
@app.post("/api/regulations/ingest")
def ingest_regulation(reg: RegulationIn, top_k: int = 4):
    with db.get_conn() as conn:
        reg_id = f"reg-{uuid.uuid4().hex[:8]}"
        conn.execute(
            "INSERT INTO regulations (id, title, source, raw_text, ingested_at) VALUES (?, ?, ?, ?, ?)",
            (reg_id, reg.title, reg.source, reg.text, db.now_iso()),
        )

        # NLP preprocessing: clause segmentation + obligation extraction
        clauses = segment_into_clauses(reg.text, id_prefix="regc")

        corpus = _load_corpus(conn)
        retriever = HybridRetriever(corpus)

        created_items = []
        for clause in clauses:
            conn.execute(
                "INSERT INTO regulation_clauses (id, regulation_id, clause_text, obligation_actor, "
                "obligation_action, effective_date) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    clause.id,
                    reg_id,
                    clause.text,
                    clause.obligation_actor,
                    clause.obligation_action,
                    clause.effective_date,
                ),
            )

            candidates = retriever.retrieve(clause.text, top_k=top_k)

            for cand in candidates:
                internal_clause = {
                    "id": cand.clause_id,
                    "text": cand.text,
                    "doc_title": next((c["doc_title"] for c in corpus if c["id"] == cand.clause_id), "Internal Policy"),
                    "section": next((c["section"] for c in corpus if c["id"] == cand.clause_id), "Section"),
                }

                # ── Relevance gate ──────────────────────────────────────────
                # Candidates below the retrieval threshold are logged for
                # auditability but skip the agent pipeline entirely. This
                # prevents weak keyword matches from being mis-classified as
                # contradicts/tightens/loosens based on incidental number
                # co-occurrence.
                if cand.below_threshold:
                    item_id = f"rv-{uuid.uuid4().hex[:8]}"
                    conn.execute(
                        """INSERT INTO review_items
                        (id, regulation_id, regulation_clause_id, internal_policy_id, internal_clause_id, impact_type,
                         draft_redline, business_action, below_threshold, citation_verified,
                         confidence, risk, routing, second_hop_clauses, reasoning_trace,
                         retrieval_score, status, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            item_id, reg_id, clause.id, cand.doc_title, cand.clause_id,
                            "no-op",
                            f"CURRENT:\n{cand.text}\n\nPROPOSED:\nNo redline — candidate dismissed by relevance filter (score {cand.rerank_score:.2f} < threshold).",
                            "No business action required — candidate dismissed by relevance filter.",
                            1,  # below_threshold = True
                            1,  # citation_verified
                            cand.rerank_score,
                            "low",
                            "auto_dismiss",
                            db.dumps([]),
                            db.dumps([{"step": "relevance_gate",
                                       "detail": f"Candidate dismissed: rerank_score={cand.rerank_score:.3f} below threshold={RELEVANCE_THRESHOLD:.2f}. Skipped agent pipeline."}]),
                            cand.rerank_score,
                            "auto_dismissed_irrelevant",
                            db.now_iso(),
                        ),
                    )
                    created_items.append(item_id)
                    continue
                # ── End relevance gate ──────────────────────────────────────

                result = run_agent_pipeline(
                    regulation_clause_text=clause.text,
                    internal_clause=internal_clause,
                    retrieval_rerank_score=cand.rerank_score,
                    corpus=corpus,
                    regulation_id=reg_id,
                    regulation_clause_id=clause.id,
                    internal_policy_id=cand.doc_title,
                    internal_clause_id=cand.clause_id,
                )

                item_id = f"rv-{uuid.uuid4().hex[:8]}"
                conn.execute(
                    """INSERT INTO review_items
                    (id, regulation_id, regulation_clause_id, internal_policy_id, internal_clause_id, impact_type,
                     draft_redline, business_action, below_threshold, citation_verified, confidence,
                     risk, routing, second_hop_clauses, reasoning_trace, retrieval_score, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        item_id,
                        reg_id,
                        clause.id,
                        cand.doc_title,
                        cand.clause_id,
                        result.impact_type,
                        result.draft_redline,
                        result.business_action,
                        0,  # below_threshold = False
                        int(result.citation_verified),
                        result.confidence,
                        result.risk,
                        result.routing,
                        db.dumps(result.second_hop_refs),
                        db.dumps([{"step": s.step, "detail": s.detail} for s in result.trace]),
                        cand.rerank_score,
                        "auto_dismissed" if result.routing == "auto_dismiss" else "pending",
                        db.now_iso(),
                    ),
                )
                created_items.append(item_id)

        # Build a meaningful summary so the ingest result panel and the
        # frontend summary bar both have accurate, intentional numbers.
        all_items_in_reg = conn.execute(
            "SELECT impact_type, below_threshold, status FROM review_items WHERE regulation_id = ?",
            (reg_id,)
        ).fetchall()
        material_impacts = sum(
            1 for r in all_items_in_reg
            if r["impact_type"] != "no-op" and not r["below_threshold"]
        )
        irrelevant_dismissed = sum(1 for r in all_items_in_reg if r["below_threshold"])
        noops_dismissed = sum(
            1 for r in all_items_in_reg
            if r["impact_type"] == "no-op" and not r["below_threshold"]
        )
        human_review_findings = sum(
            1 for r in all_items_in_reg
            if r["status"] == "pending" and r["impact_type"] != "no-op"
        )

        return {
            "regulation_id": reg_id,
            "clauses_extracted": len(clauses),
            "candidates_retrieved": len(created_items),
            "material_impacts": material_impacts,
            "irrelevant_dismissed": irrelevant_dismissed,
            "noops_dismissed": noops_dismissed,
            "human_review_findings": human_review_findings,
            "review_items_created": len(created_items),
        }


@app.get("/api/regulations")
def list_regulations():
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, source, ingested_at FROM regulations ORDER BY ingested_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


@app.get("/api/regulations/{regulation_id}/results")
def get_regulation_results(regulation_id: str):
    with db.get_conn() as conn:
        reg = conn.execute("SELECT * FROM regulations WHERE id = ?", (regulation_id,)).fetchone()
        if not reg:
            raise HTTPException(404, "Regulation not found")

        items = conn.execute(
            """SELECT rv.*, rc.clause_text as regulation_clause_text, rc.effective_date,
                      rc.obligation_actor, ic.doc_title, ic.section, ic.clause_text as internal_clause_text
               FROM review_items rv
               JOIN regulation_clauses rc ON rv.regulation_clause_id = rc.id
               JOIN internal_clauses ic ON rv.internal_clause_id = ic.id
               WHERE rv.regulation_id = ?
               ORDER BY rv.confidence DESC""",
            (regulation_id,),
        ).fetchall()

        return {
            "regulation": dict(reg),
            "items": [_serialize_review_item(r) for r in items],
        }


def _serialize_review_item(row) -> dict:
    d = dict(row)
    d["reasoning_trace"] = db.loads(d["reasoning_trace"])
    d["second_hop_clauses"] = db.loads(d["second_hop_clauses"])
    d["citation_verified"] = bool(d["citation_verified"])
    return d


# ---------------------------------------------------------------------------
# Corpus browsing / management
# ---------------------------------------------------------------------------
@app.get("/api/corpus")
def get_corpus():
    with db.get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM internal_clauses ORDER BY doc_title, section"
        ).fetchall()
        docs: dict[str, list[dict]] = {}
        for r in rows:
            docs.setdefault(r["doc_title"], []).append(
                {"id": r["id"], "section": r["section"], "text": r["clause_text"]}
            )
        return docs


@app.post("/api/corpus")
def add_corpus_doc(doc: CorpusDocIn):
    with db.get_conn() as conn:
        ids = []
        for c in doc.clauses:
            cid = f"ic-{uuid.uuid4().hex[:8]}"
            conn.execute(
                "INSERT INTO internal_clauses (id, doc_title, section, clause_text, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (cid, doc.doc_title, c.get("section", ""), c["text"], db.now_iso()),
            )
            ids.append(cid)
        return {"doc_title": doc.doc_title, "clause_ids": ids}


# ---------------------------------------------------------------------------
# Human review queue (Section 4.5 guardrail: nothing auto-applies)
# ---------------------------------------------------------------------------
@app.get("/api/review")
def get_review_queue(status: Optional[str] = None):
    with db.get_conn() as conn:
        query = """SELECT rv.*, r.title as regulation_title, rc.clause_text as regulation_clause_text,
                          ic.doc_title, ic.section, ic.clause_text as internal_clause_text
                   FROM review_items rv
                   JOIN regulations r ON rv.regulation_id = r.id
                   JOIN regulation_clauses rc ON rv.regulation_clause_id = rc.id
                   JOIN internal_clauses ic ON rv.internal_clause_id = ic.id"""
        params = ()
        if status == "pending_material":
            query += " WHERE rv.status = 'pending' AND rv.impact_type != 'no-op'"
        elif status:
            query += " WHERE rv.status = ?"
            params = (status,)
        query += " ORDER BY rv.confidence DESC"
        rows = conn.execute(query, params).fetchall()
        return [_serialize_review_item(r) for r in rows]



@app.post("/api/review/{item_id}")
def act_on_review_item(item_id: str, action_in: ReviewActionIn):
    with db.get_conn() as conn:
        row = conn.execute("SELECT * FROM review_items WHERE id = ?", (item_id,)).fetchone()
        if not row:
            raise HTTPException(404, "Review item not found")

        new_status = {
            "approve": "approved",
            "reject": "rejected",
            "escalate_confirmed": "escalated_confirmed",
        }[action_in.action]

        conn.execute(
            "UPDATE review_items SET status = ?, reviewer_action = ?, reviewer_notes = ?, "
            "reviewed_at = ? WHERE id = ?",
            (new_status, action_in.action, action_in.notes, db.now_iso(), item_id),
        )
        return {"id": item_id, "status": new_status}


# ---------------------------------------------------------------------------
# Audit log (Section 4.5: full trail of inputs, evidence, reasoning, confidence,
# reviewer action)
# ---------------------------------------------------------------------------
@app.get("/api/audit")
def get_audit_log():
    with db.get_conn() as conn:
        rows = conn.execute(
            """SELECT rv.id, rv.created_at, rv.impact_type, rv.confidence, rv.risk, rv.routing,
                      rv.status, rv.reviewer_action, rv.reviewer_notes, rv.reviewed_at,
                      rv.citation_verified, rv.reasoning_trace,
                      r.title as regulation_title, ic.doc_title, ic.section
               FROM review_items rv
               JOIN regulations r ON rv.regulation_id = r.id
               JOIN internal_clauses ic ON rv.internal_clause_id = ic.id
               ORDER BY rv.created_at DESC"""
        ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["reasoning_trace"] = db.loads(d["reasoning_trace"])
            d["citation_verified"] = bool(d["citation_verified"])
            out.append(d)
        return out


# ---------------------------------------------------------------------------
# Evaluation (Section 5): score the pipeline against the hand-labeled set
# ---------------------------------------------------------------------------
@app.get("/api/eval")
def run_eval():
    with db.get_conn() as conn:
        labels = conn.execute("SELECT * FROM eval_labels").fetchall()
        corpus = _load_corpus(conn)

    if not labels:
        raise HTTPException(400, "No eval labels seeded")

    retriever = HybridRetriever(corpus)

    tp = fp = fn = tn = 0
    correct_impact_type = 0
    total_impactful = 0
    hallucination_flags = 0
    total_drafts = 0
    retrieval_hits = 0

    detail = []

    for lbl in labels:
        target_key = lbl["internal_clause_id"]  # "DocTitle::Section"
        doc_title, section = target_key.split("::")
        target_row = next(
            (c for c in corpus if c["doc_title"] == doc_title and c["section"] == section), None
        )

        candidates = retriever.retrieve(lbl["regulation_clause_text"], top_k=5)
        retrieved_ids = {c.clause_id for c in candidates}
        target_retrieved = bool(target_row) and target_row["id"] in retrieved_ids
        if target_retrieved:
            retrieval_hits += 1

        predicted_impacted = False
        predicted_type = "no-op"
        citation_verified = True

        target_promotable = False
        if target_row and target_retrieved:
            cand = next(c for c in candidates if c.clause_id == target_row["id"])
            target_promotable = not cand.below_threshold
        if target_row and target_promotable:
            result = run_agent_pipeline(
                lbl["regulation_clause_text"], target_row, cand.rerank_score, corpus,
                regulation_id=f"eval-{lbl['id']}",
                regulation_clause_id=f"eval-clause-{lbl['id']}",
                internal_policy_id=target_row["doc_title"],
                internal_clause_id=target_row["id"],
            )
            predicted_impacted = result.impact_type != "no-op"
            predicted_type = result.impact_type
            citation_verified = result.citation_verified
            if predicted_impacted:
                total_drafts += 1
                if not citation_verified:
                    hallucination_flags += 1

        actual_impacted = bool(lbl["is_impacted"])

        if actual_impacted and predicted_impacted:
            tp += 1
        elif actual_impacted and not predicted_impacted:
            fn += 1
        elif not actual_impacted and predicted_impacted:
            fp += 1
        else:
            tn += 1

        if actual_impacted:
            total_impactful += 1
            if predicted_type == lbl["true_impact_type"]:
                correct_impact_type += 1

        detail.append(
            {
                "regulation_clause_text": lbl["regulation_clause_text"][:100] + "...",
                "target": target_key,
                "target_retrieved": target_retrieved,
                "actual_impacted": actual_impacted,
                "predicted_impacted": predicted_impacted,
                "true_impact_type": lbl["true_impact_type"],
                "predicted_impact_type": predicted_type if target_retrieved else None,
            }
        )

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "n_labels": len(labels),
        "retrieval_recall_at_5": round(retrieval_hits / len(labels), 3),
        "classification_precision": round(precision, 3),
        "classification_recall": round(recall, 3),
        "classification_f1": round(f1, 3),
        "impact_type_accuracy": round(
            correct_impact_type / total_impactful, 3
        ) if total_impactful else None,
        "citation_hallucination_rate": round(
            hallucination_flags / total_drafts, 3
        ) if total_drafts else 0.0,
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Convenience: load sample regulations for demo purposes
# ---------------------------------------------------------------------------
@app.get("/api/samples")
def get_sample_regulations():
    return SAMPLE_REGULATIONS


@app.get("/api/health")
def health():
    return {"status": "ok", "llm_enabled": bool(__import__("os").environ.get("ANTHROPIC_API_KEY"))}
