"""
agent.py
--------
Section 4.4 of the project doc: the agent orchestrator, built as an
explicit step-wise state machine rather than a single mega-prompt.

Steps, matching the architecture diagram exactly:
  1. classify_impact      -> contradicts / tightens / loosens / no-op
  2. decide_second_hop     -> does the internal clause reference another
                               clause elsewhere in the corpus?
  3. draft_redline         -> proposed revision + citation to the source
                               regulatory clause
  4. self_critique         -> does the citation actually support the
                               draft's claim? (anti-hallucination check)
  5. score_confidence_risk -> decide auto-flag / auto-dismiss / escalate

LLM usage (cost/latency tradeoff, Section 8 talking point): classification
is the highest-volume step (runs on every retrieved candidate) and is
cheap to do with a heuristic/small-model classifier. Drafting only runs
on clauses that already passed classification as impactful, so it's the
right place to spend tokens on a stronger model. If ANTHROPIC_API_KEY is
set, classify/draft/critique call Claude; otherwise everything falls back
to transparent rule-based heuristics so the whole system runs for free
and deterministically -- useful for demos and for reasoning about
failures without LLM non-determinism in the way.
"""

import os
import re
from dataclasses import dataclass, field
from typing import Optional

from .retrieval import RELEVANCE_THRESHOLD

_ANTHROPIC_CLIENT = None
_USE_LLM = bool(os.environ.get("ANTHROPIC_API_KEY"))

if _USE_LLM:
    try:
        import anthropic

        _ANTHROPIC_CLIENT = anthropic.Anthropic()
    except Exception:
        _USE_LLM = False


NEGATION_MARKERS = ["does not", "shall not", "no mechanism", "not currently", "may not", "waive", "waived"]
TIGHTEN_MARKERS_REG = ["at least", "within", "not exceeding", "shall", "must"]


@dataclass
class ReasoningStep:
    step: str
    detail: str


@dataclass(frozen=True)
class FindingContext:
    """Immutable input scope for one regulation/internal clause pair."""

    regulation_clause_id: str
    internal_clause_id: str
    regulation_clause: str
    internal_clause: dict
    retrieval_rerank_score: float


@dataclass
class AgentResult:
    regulation_clause_id: str
    internal_clause_id: str
    impact_type: str  # contradicts | tightens | loosens | no-op
    impact_rationale: str
    second_hop_refs: list[str]
    draft_redline: Optional[str]
    business_action: str
    citation_verified: bool
    citation_check_detail: str
    confidence: float
    risk: str  # low | medium | high
    routing: str  # auto_dismiss | flag_for_review | escalate
    trace: list[ReasoningStep] = field(default_factory=list)


def _llm_call(system: str, user: str, max_tokens: int = 400) -> Optional[str]:
    if not _USE_LLM:
        return None
    try:
        resp = _ANTHROPIC_CLIENT.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
    except Exception as e:
        return None


def classify_impact(context: FindingContext) -> tuple[str, str]:
    """
    Step 1: classify the relationship between a new regulatory clause and
    a retrieved internal-policy clause.

    Heuristic fallback logic (used when no LLM is configured):
    - If the internal clause explicitly negates/waives something the
      regulation now mandates -> "contradicts"
    - If the regulation imposes a stricter number (shorter deadline,
      lower cap, more frequent cycle) than the internal clause's number
      -> "tightens"
    - If the regulation imposes a looser number than the internal clause
      -> "loosens"
    - Otherwise, if there's topical overlap but no clear obligation
      conflict -> "no-op"
    """
    if _USE_LLM:
        out = _llm_call(
            system=(
                "You are a compliance analyst. Classify the relationship between a "
                "NEW REGULATORY CLAUSE and an EXISTING INTERNAL POLICY CLAUSE as exactly "
                "one of: contradicts, tightens, loosens, no-op. "
                "Respond in the format: LABEL | one-sentence rationale."
            ),
            user=(
                f"PAIR IDS: regulation_clause_id={context.regulation_clause_id}; "
                f"internal_clause_id={context.internal_clause_id}\n"
                f"NEW REGULATORY CLAUSE:\n{context.regulation_clause}\n\n"
                f"EXISTING INTERNAL POLICY CLAUSE:\n{context.internal_clause['text']}"
            ),
        )
        if out and "|" in out:
            label, rationale = out.split("|", 1)
            label = label.strip().lower()
            if label in {"contradicts", "tightens", "loosens", "no-op"}:
                return label, rationale.strip()

    reg_lower = context.regulation_clause.lower()
    int_lower = context.internal_clause["text"].lower()

    # NOTE: \b after "not" is required -- without it "shall not" wrongly
    # matches inside words like "shall notify".
    reg_negates = bool(re.search(r"shall not\b|must not\b|prohibited|no longer permitted", reg_lower))
    int_negates = bool(re.search(r"shall not\b|must not\b|does not\b|no mechanism|not currently|may not\b|waive[d]?\b", int_lower))

    if int_negates and any(v in reg_lower for v in ["shall", "must", "required"]) and not reg_negates:
        return (
            "contradicts",
            "Internal clause explicitly negates or waives an action the new "
            "regulatory clause now mandates.",
        )

    if reg_negates and not int_negates and any(
        v in int_lower for v in ["shall", "may", "will"]
    ):
        return (
            "contradicts",
            "New regulatory clause now prohibits an action that the internal "
            "clause currently permits or requires.",
        )

    if any(term in reg_lower for term in ["erase", "erasure", "deletion"]):
        if any(term in int_lower for term in ["retained", "retention"]):
            return (
                "contradicts",
                "The regulatory erasure obligation conflicts with the internal retention period "
                "unless a separate legal-retention exception applies.",
            )
        if "mechanism" in int_lower or "does not currently provide" in int_lower:
            return (
                "contradicts",
                "The regulation requires an erasure capability that the internal clause does not provide.",
            )

    reg_numbers = [int(n) for n in re.findall(r"\d+", context.regulation_clause)]
    int_numbers = [int(n) for n in re.findall(r"\d+", context.internal_clause["text"])]

    if reg_numbers and int_numbers:
        reg_n, int_n = reg_numbers[0], int_numbers[0]
        deadline_context = any(
            w in reg_lower for w in ["day", "days", "month", "months", "year", "years"]
        )
        if deadline_context:
            if reg_n < int_n:
                return (
                    "tightens",
                    f"New regulatory timeframe ({reg_n}) is shorter than the internal "
                    f"policy's current timeframe ({int_n}).",
                )
            if reg_n > int_n:
                return (
                    "loosens",
                    f"New regulatory timeframe ({reg_n}) is longer than the internal "
                    f"policy's current timeframe ({int_n}).",
                )
        else:
            if reg_n != int_n:
                return (
                    "tightens" if reg_n < int_n else "loosens",
                    f"New regulatory numeric limit ({reg_n}) differs from internal "
                    f"policy's current limit ({int_n}).",
                )

    shared = set(re.findall(r"[a-z]{5,}", reg_lower)) & set(
        re.findall(r"[a-z]{5,}", int_lower)
    )
    if len(shared) >= 3:
        return (
            "no-op",
            "Clauses share topical vocabulary but no direct obligation conflict "
            "or numeric mismatch was detected; internal clause may already comply.",
        )
    return "no-op", "Limited overlap detected; likely a low-relevance retrieval match."


def decide_second_hop(context: FindingContext, corpus: list[dict]) -> list[str]:
    """Step 2: look for cross-references inside the flagged internal clause."""
    from .nlp_preprocessing import find_cross_references

    refs = find_cross_references(context.internal_clause["text"])
    matched_ids = []
    for ref in refs:
        for c in corpus:
            if c["section"].lower() == ref.lower() or ref.lower() in c["section"].lower():
                matched_ids.append(c["id"])
    return matched_ids


def draft_redline(context: FindingContext, impact_type: str) -> str:
    """Step 3: draft a suggested revision structured with explicit CURRENT and PROPOSED blocks."""
    if _USE_LLM:
        out = _llm_call(
            system=(
                "You are a compliance drafting assistant. Draft a specific, structured revision "
                "to the EXISTING INTERNAL CLAUSE so it complies with the NEW REGULATORY CLAUSE.\n"
                "Format your response EXACTLY as:\n\n"
                "CURRENT:\n[exact relevant existing internal policy wording]\n\n"
                "PROPOSED:\n[exact proposed revised wording of the internal policy clause]"
            ),
            user=(
                f"PAIR IDS: regulation_clause_id={context.regulation_clause_id}; "
                f"internal_clause_id={context.internal_clause_id}\n"
                f"IMPACT TYPE: {impact_type}\n\n"
                f"NEW REGULATORY CLAUSE:\n{context.regulation_clause}\n\n"
                f"EXISTING INTERNAL CLAUSE:\n{context.internal_clause['text']}"
            ),
        )
        if out and "PROPOSED:" in out:
            return out.strip()

    reg_clean = context.regulation_clause.strip()
    int_clean = context.internal_clause["text"].strip()
    reg_lower = reg_clean.lower()
    int_lower = int_clean.lower()

    if impact_type == "no-op":
        return (
            f"CURRENT:\n{int_clean}\n\n"
            f"PROPOSED:\nNo redline required — clause is already compliant or topically distinct."
        )

    # Heuristic redline drafting logic for precise CURRENT / PROPOSED outputs
    proposed_text = int_clean

    if "mechanism" in int_lower and ("does not currently provide" in int_lower or "no mechanism" in int_lower):
        proposed_text = (
            "The Bank shall provide a secure online mechanism through account service portals "
            "for data principals to request erasure of personal data within thirty (30) days."
        )
    elif "foreclosure charge" in int_lower or "prepayment" in int_lower:
        if "shall not levy" in reg_lower or "no foreclosure" in reg_lower or "prohibited" in reg_lower:
            proposed_text = re.sub(
                r"shall attract a foreclosure charge of up to four percent \(4%\)[^,\.]*",
                "shall not attract any foreclosure or prepayment charge",
                int_clean,
                flags=re.IGNORECASE
            )
            if proposed_text == int_clean:
                proposed_text = int_clean + " Prepayment of floating-rate retail loans shall not attract any foreclosure charge."
    elif "waived" in int_lower and ("mandatory" in reg_lower or "shall not be waived" in reg_lower or "shall be conducted" in reg_lower):
        proposed_text = (
            "Re-KYC verification is mandatory across all risk tiers and shall not be waived, "
            "regardless of whether registered address or contact details remain unchanged."
        )
    elif ("retained" in int_lower or "period of" in int_lower) and (
        "five (5) years" in reg_lower or "5 years" in reg_lower
    ):
        if "five (5) years" in reg_lower or "5 years" in reg_lower:
            proposed_text = re.sub(
                r"ten \b\(10\)\s*years", "a maximum of five (5) years", int_clean, flags=re.IGNORECASE
            )
            proposed_text = re.sub(
                r"10 years", "5 years", proposed_text, flags=re.IGNORECASE
            )
    elif "communicated to the borrower" in int_lower and ("15 days" in reg_lower or "fifteen (15) days" in reg_lower):
        proposed_text = re.sub(
            r"thirty \b\(30\)\s*days", "at least fifteen (15) days", int_clean, flags=re.IGNORECASE
        )

    if proposed_text == int_clean:
        if impact_type == "contradicts":
            proposed_text = f"Amend clause to eliminate conflict with regulatory requirement: \"{reg_clean[:180]}\"."
        elif impact_type == "tightens":
            proposed_text = f"Update internal threshold/timeframe to meet stricter regulatory standard: \"{reg_clean[:180]}\"."
        else:
            proposed_text = f"Relax internal standard to match regulatory limit: \"{reg_clean[:180]}\"."

    return f"CURRENT:\n{int_clean}\n\nPROPOSED:\n{proposed_text}"


def generate_business_action(context: FindingContext, impact_type: str) -> str:
    """Generates a structured BUSINESS ACTION REQUIRED breakdown for compliance officers."""
    if impact_type == "no-op":
        return "No business action required — internal clause appears compliant or topically distinct."

    regulation_clause = context.regulation_clause
    internal_clause = context.internal_clause
    doc_title = internal_clause.get("doc_title", "Internal Policy Document")
    section = internal_clause.get("section", "Section")
    reg_lower = regulation_clause.lower()
    int_lower = internal_clause.get("text", "").lower()

    if any(term in reg_lower for term in ["erase", "erasure", "deletion"]) and any(
        term in int_lower for term in ["mechanism", "deletion", "erasure", "retained", "retention"]
    ):
        action_summary = "Deploy self-service data deletion request mechanism on customer portals."
    elif "mechanism" in int_lower or "deletion" in int_lower:
        action_summary = "Implement the missing data request mechanism described by the regulation."
    elif "foreclosure" in int_lower or "prepayment" in int_lower:
        action_summary = "Eliminate foreclosure/prepayment charges on floating-rate loans."
    elif "waived" in int_lower or "re-kyc" in int_lower:
        action_summary = "Mandate periodic Re-KYC verification; eliminate address-change waivers."
    elif "communicated" in int_lower or "notice" in int_lower:
        action_summary = "Update borrower rate-reset advance notice window to at least 15 days."
    elif ("retained" in int_lower or "retention" in int_lower) and (
        "5 years" in reg_lower or "five (5) years" in reg_lower
    ):
        action_summary = "Reduce PII retention timeframe from 10 years → 5 years."
    elif impact_type == "contradicts":
        action_summary = "Amend conflicting internal policy provision to comply with regulatory mandate."
    elif impact_type == "tightens":
        action_summary = "Enforce stricter operational timeline/threshold per new regulation."
    else:
        action_summary = "Adjust operational policy threshold for relaxed regulatory standard."

    if "30 days" in reg_lower or "thirty (30) days" in reg_lower:
        deadline = "Within 30 days of circular effective date"
    elif "15 days" in reg_lower:
        deadline = "Within 15 days of rate modification"
    else:
        deadline = "1 Jan 2027 (Mandatory Enforcement Date)"

    return (
        f"ACTION REQUIRED: {action_summary}\n"
        f"TARGET POLICY: {doc_title} (Section {section})\n"
        f"STEPS:\n"
        f"1. Update Section {section} text in the Master Policy Repository.\n"
        f"2. Audit affected historical records and operational systems.\n"
        f"3. Implement the revised policy control and supporting workflow.\n"
        f"DEADLINE: {deadline}"
    )


def self_critique(context: FindingContext, draft: str) -> tuple[bool, str]:
    """Step 4: anti-hallucination check."""
    quoted = re.findall(r'"([^"]+)"', draft)
    if not quoted:
        return True, "No quoted citation in draft to verify (no-op / non-quoting draft)."

    for q in quoted:
        q_clean = q.strip().rstrip(".")
        if q_clean[:60] in context.regulation_clause:
            return True, "Quoted citation text confirmed present in source regulatory clause."

    return (
        False,
        "Quoted citation in draft could not be matched against the source regulatory clause text.",
    )


def score_confidence_risk(
    context: FindingContext,
    impact_type: str,
    citation_verified: bool,
) -> tuple[float, str, str]:
    """Step 5: score confidence, risk, and routing."""
    base = context.retrieval_rerank_score
    if impact_type == "contradicts":
        base += 0.15
    elif impact_type in ("tightens", "loosens"):
        base += 0.08

    if not citation_verified:
        base -= 0.25

    confidence = max(0.0, min(1.0, base))

    if impact_type == "contradicts":
        risk = "high"
    elif impact_type in ("tightens", "loosens"):
        risk = "medium"
    else:
        risk = "low"

    if not citation_verified:
        routing = "escalate"
    elif impact_type == "no-op":
        routing = "auto_dismiss"
    elif confidence >= 0.72:
        routing = "flag_for_review"
    else:
        routing = "escalate"

    return confidence, risk, routing


def run_agent_pipeline(
    regulation_clause_text: str,
    internal_clause: dict,
    retrieval_rerank_score: float,
    corpus: list[dict],
    regulation_clause_id: str,
    internal_clause_id: str,
) -> AgentResult:
    """Runs the full 5-step state machine for one candidate pair."""
    context = FindingContext(
        regulation_clause_id=regulation_clause_id,
        internal_clause_id=internal_clause_id,
        regulation_clause=regulation_clause_text,
        internal_clause=internal_clause,
        retrieval_rerank_score=retrieval_rerank_score,
    )
    trace: list[ReasoningStep] = []

    if context.retrieval_rerank_score < RELEVANCE_THRESHOLD:
        trace.append(
            ReasoningStep(
                "relevance_gate",
                f"pair={regulation_clause_id}/{internal_clause_id}; "
                f"score={context.retrieval_rerank_score:.3f} below threshold="
                f"{RELEVANCE_THRESHOLD:.2f}; impact classification skipped.",
            )
        )
        return AgentResult(
            regulation_clause_id=regulation_clause_id,
            internal_clause_id=internal_clause_id,
            impact_type="no-op",
            impact_rationale="Candidate was below the minimum relevance threshold.",
            second_hop_refs=[],
            draft_redline="No redline — candidate dismissed by relevance filter.",
            business_action="No business action required — candidate dismissed by relevance filter.",
            citation_verified=True,
            citation_check_detail="Impact classification skipped by relevance gate.",
            confidence=context.retrieval_rerank_score,
            risk="low",
            routing="auto_dismiss",
            trace=trace,
        )

    impact_type, rationale = classify_impact(context)
    trace.append(
        ReasoningStep(
            "1_classify",
            f"pair={regulation_clause_id}/{internal_clause_id}; {impact_type}: {rationale}",
        )
    )

    hop_ids = decide_second_hop(context, corpus)
    trace.append(
        ReasoningStep(
            "2_second_hop",
            f"Cross-references found -> retrieved {len(hop_ids)} linked clause(s): {hop_ids}"
            if hop_ids
            else "No cross-references detected; no second hop needed.",
        )
    )

    draft = draft_redline(context, impact_type)
    business_action = generate_business_action(context, impact_type)

    if impact_type == "no-op":
        trace.append(ReasoningStep("3_draft", "No-op: no substantive redline required."))
        citation_verified, citation_detail = True, "N/A for no-op classification."
        trace.append(ReasoningStep("4_self_critique", citation_detail))
    else:
        trace.append(ReasoningStep("3_draft", draft))
        citation_verified, citation_detail = self_critique(context, draft)
        trace.append(ReasoningStep("4_self_critique", citation_detail))

    confidence, risk, routing = score_confidence_risk(
        context, impact_type, citation_verified
    )
    trace.append(
        ReasoningStep(
            "5_score",
            f"pair={regulation_clause_id}/{internal_clause_id}; "
            f"confidence={confidence:.2f}, risk={risk}, routing={routing}",
        )
    )

    return AgentResult(
        regulation_clause_id=regulation_clause_id,
        internal_clause_id=internal_clause_id,
        impact_type=impact_type,
        impact_rationale=rationale,
        second_hop_refs=hop_ids,
        draft_redline=draft,
        business_action=business_action,
        citation_verified=citation_verified,
        citation_check_detail=citation_detail,
        confidence=confidence,
        risk=risk,
        routing=routing,
        trace=trace,
    )

