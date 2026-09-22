"""
agent.py
--------
RCIA agent orchestrator.

The pipeline is implemented as an explicit state machine:
  1. classify_impact       -> contradicts / tightens / loosens / no-op
  2. decide_second_hop     -> follow internal cross-references
  3. draft_redline         -> proposed revision grounded in the source clause
  4. self_critique         -> citation + claim grounding checks
  5. score_confidence_risk -> route to review / escalation

LLM behavior:
- If ANTHROPIC_API_KEY is available, classification / drafting / critique may
  use Claude.
- Otherwise deterministic heuristics are used.

Safety/grounding guardrails:
- No default regulatory deadline is invented.
- No fixed "30 days" is added to an erasure-mechanism clause.
- Proposed numeric/date claims must be supported by the source clause.
- The foreclosure redline cannot invert a floating-rate prohibition.
- Unsupported claims force escalation.
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


# ---------------------------------------------------------------------------
# Heuristic markers
# ---------------------------------------------------------------------------
NEGATION_MARKERS = [
    "does not",
    "shall not",
    "must not",
    "no mechanism",
    "not currently",
    "may not",
    "waive",
    "waived",
]

TIGHTEN_MARKERS_REG = [
    "at least",
    "within",
    "not exceeding",
    "shall",
    "must",
]

NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
    "seventy-two": 72,
}

MATERIAL_TOPIC_GROUPS = {
    "retention": "retention",
    "retained": "retention",
    "deletion": "erasure",
    "delete": "erasure",
    "erasure": "erasure",
    "erase": "erasure",
    "mechanism": "mechanism",
    "kyc": "kyc",
    "re-kyc": "kyc",
    "verification": "kyc",
    "foreclosure": "foreclosure",
    "prepayment": "foreclosure",
    "interest": "interest",
    "rate": "rate",
    "borrower": "borrower",
    "grievance": "grievance",
    "vendor": "vendor",
    "processor": "vendor",
    "subprocessor": "vendor",
    "transfer": "transfer",
    "encryption": "encryption",
    "breach": "breach",
    "incident": "incident",
    "notification": "notification",
    "notify": "notification",
    "reporting": "reporting",
    "report": "reporting",
    "consent": "consent",
    "log": "logging",
    "logs": "logging",
    "records": "records",
    "record": "records",
    "channel": "channel",
    "access": "access",
    "approval": "approval",
    "authorized": "authorization",
    "authorization": "authorization",
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class ReasoningStep:
    step: str
    detail: str


@dataclass(frozen=True)
class FindingContext:
    """Immutable input scope for one regulation/internal clause pair."""

    regulation_id: str
    regulation_clause_id: str
    internal_policy_id: str
    internal_clause_id: str
    regulation_clause: str
    internal_clause: dict
    retrieval_rerank_score: float


@dataclass
class AgentResult:
    regulation_id: str
    regulation_clause_id: str
    internal_policy_id: str
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


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------
def _extract_numbers(text: str) -> list[int]:
    """Return unique numeric values, preserving first-seen order."""
    values: list[int] = []
    seen: set[int] = set()

    for raw in re.findall(r"\d+", text):
        value = int(raw)
        if value not in seen:
            values.append(value)
            seen.add(value)

    lower = text.lower()
    for word, value in NUMBER_WORDS.items():
        if re.search(rf"\b{re.escape(word)}\b", lower) and value not in seen:
            values.append(value)
            seen.add(value)

    return values


def _extract_source_deadline(text: str) -> Optional[str]:
    """Extract an explicit source timeframe/date; never invent a fallback."""
    patterns = [
        r"within\s+(?:\w+\s+)?\(?\d+\)?\s*(?:day|days|business day|business days|month|months|year|years)",
        r"at least\s+(?:\w+\s+)?\(?\d+\)?\s*(?:day|days|business day|business days|month|months|year|years)\s*(?:in advance|prior|before)?",
        r"not exceeding\s+(?:\w+\s+)?\(?\d+\)?\s*(?:day|days|month|months|year|years)",
        r"(?:by|before|no later than)\s+\d{1,2}[/-]\d{1,2}[/-]\d{2,4}",
        r"(?:by|before|no later than)\s+\d{1,2}\s+[A-Za-z]+\s+\d{4}",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip().rstrip(".")

    return None


def _extract_years_and_dates(text: str) -> set[str]:
    """Return explicit year/date tokens for conservative grounding checks."""
    values = set(re.findall(r"\b(?:19|20)\d{2}\b", text))
    values.update(re.findall(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b", text))
    values.update(
        m.group(0).lower()
        for m in re.finditer(
            r"\b\d{1,2}\s+[A-Za-z]{3,12}\s+\d{4}\b",
            text,
        )
    )
    return values


def _material_topic_groups(text: str) -> set[str]:
    """Map substantive terms in text to normalized topic groups."""
    lower = text.lower()
    groups: set[str] = set()

    for term, group in MATERIAL_TOPIC_GROUPS.items():
        if re.search(rf"\b{re.escape(term)}\b", lower):
            groups.add(group)

    return groups


def _extract_proposed_section(draft: str) -> str:
    """Extract only PROPOSED text so CURRENT policy values are ignored."""
    if not draft:
        return ""

    if "PROPOSED:" in draft:
        return draft.split("PROPOSED:", 1)[1].strip()

    return draft.strip()


def _extract_quoted_text(draft: str) -> list[str]:
    return [
        q.strip().rstrip(".")
        for q in re.findall(r'"([^"]+)"', draft)
    ]


# ---------------------------------------------------------------------------
# Optional LLM helper
# ---------------------------------------------------------------------------
def _llm_call(
    system: str,
    user: str,
    max_tokens: int = 400,
) -> Optional[str]:
    if not _USE_LLM or _ANTHROPIC_CLIENT is None:
        return None

    try:
        resp = _ANTHROPIC_CLIENT.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )

        return "".join(
            block.text
            for block in resp.content
            if block.type == "text"
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Step 1: classify impact
# ---------------------------------------------------------------------------
def classify_impact(context: FindingContext) -> tuple[str, str]:
    """Classify the relationship between a regulation and internal clause."""

    if _USE_LLM:
        out = _llm_call(
            system=(
                "You are a compliance analyst. Classify the relationship between a "
                "NEW REGULATORY CLAUSE and an EXISTING INTERNAL POLICY CLAUSE as exactly "
                "one of: contradicts, tightens, loosens, no-op. Do not infer facts that "
                "are not stated. Respond: LABEL | one-sentence rationale."
            ),
            user=(
                f"PAIR IDS: regulation_clause_id={context.regulation_clause_id}; "
                f"internal_clause_id={context.internal_clause_id}\n"
                f"NEW REGULATORY CLAUSE:\n{context.regulation_clause}\n\n"
                f"EXISTING INTERNAL POLICY CLAUSE:\n"
                f"{context.internal_clause['text']}"
            ),
        )

        if out and "|" in out:
            label, rationale = out.split("|", 1)
            label = label.strip().lower()
            if label in {"contradicts", "tightens", "loosens", "no-op"}:
                return label, rationale.strip()

    reg_lower = context.regulation_clause.lower()
    int_lower = context.internal_clause["text"].lower()

    meaningful_reg_terms = {
        t
        for t in re.findall(r"[a-z]{4,}", reg_lower)
        if t not in {
            "shall",
            "must",
            "required",
            "within",
            "days",
            "policy",
            "data",
        }
    }

    meaningful_int_terms = set(re.findall(r"[a-z]{4,}", int_lower))
    shared_terms = meaningful_reg_terms & meaningful_int_terms

    shared_topics = {
        "incident",
        "breach",
        "reported",
        "reporting",
        "notification",
        "notify",
        "encryption",
        "security",
        "vendor",
        "processor",
        "subprocessor",
        "retention",
        "retained",
        "deletion",
        "erasure",
        "erase",
        "foreclosure",
        "prepayment",
        "kyc",
        "grievance",
        "interest",
        "borrower",
    } & shared_terms

    if len(shared_terms) < 2 and not shared_topics:
        return "no-op", "No meaningful obligation evidence links this candidate pair."

    # Word boundaries avoid substring bugs such as matching `not` inside `notify`.
    reg_negates = bool(
        re.search(
            r"shall not\b|must not\b|prohibited|no longer permitted",
            reg_lower,
        )
    )

    int_negates = bool(
        re.search(
            r"shall not\b|must not\b|does not\b|no mechanism|"
            r"not currently|may not\b|waive[d]?\b",
            int_lower,
        )
    )

    if (
        int_negates
        and any(v in reg_lower for v in ["shall", "must", "required"])
        and not reg_negates
        and any(
            term in reg_lower
            for term in [
                "mechanism",
                "deletion",
                "erasure",
                "erase",
                "waive",
            ]
        )
    ):
        return (
            "contradicts",
            "Internal clause explicitly negates or omits an action the new "
            "regulatory clause mandates.",
        )

    if (
        reg_negates
        and not int_negates
        and any(v in int_lower for v in ["shall", "may", "will"])
    ):
        return (
            "contradicts",
            "New regulatory clause prohibits an action that the internal "
            "clause permits or requires.",
        )

    if any(term in reg_lower for term in ["erase", "erasure", "deletion"]):
        if any(term in int_lower for term in ["retained", "retention"]):
            return (
                "contradicts",
                "The regulatory erasure obligation may conflict with the "
                "internal retention provision unless another legal-retention "
                "exception applies.",
            )

        if "mechanism" in int_lower or "does not currently provide" in int_lower:
            return (
                "contradicts",
                "The regulation requires an erasure capability that the "
                "internal clause does not provide.",
            )

    reg_numbers = _extract_numbers(context.regulation_clause)
    int_numbers = _extract_numbers(context.internal_clause["text"])

    if reg_numbers and int_numbers:
        transition = re.search(
            r"from\s+[^.]{0,80}?to\s+([^,.]+)",
            reg_lower,
        )
        transition_numbers = (
            _extract_numbers(transition.group(1)) if transition else []
        )

        reg_n = transition_numbers[0] if transition_numbers else reg_numbers[-1]
        int_n = int_numbers[0]

        deadline_context = any(
            word in reg_lower
            for word in [
                "day",
                "days",
                "month",
                "months",
                "year",
                "years",
            ]
        )

        if deadline_context:
            if reg_n < int_n:
                return (
                    "tightens",
                    f"New regulatory timeframe ({reg_n}) is shorter than the "
                    f"internal policy's current timeframe ({int_n}).",
                )

            if reg_n > int_n:
                return (
                    "loosens",
                    f"New regulatory timeframe ({reg_n}) is longer than the "
                    f"internal policy's current timeframe ({int_n}).",
                )
        elif reg_n != int_n:
            return (
                "tightens" if reg_n < int_n else "loosens",
                f"New regulatory numeric limit ({reg_n}) differs from the "
                f"internal policy's current limit ({int_n}).",
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


# ---------------------------------------------------------------------------
# Step 2: second-hop cross-reference lookup
# ---------------------------------------------------------------------------
def decide_second_hop(
    context: FindingContext,
    corpus: list[dict],
) -> list[str]:
    """Look for cross-references inside the flagged internal clause."""

    from .nlp_preprocessing import find_cross_references

    refs = find_cross_references(context.internal_clause["text"])
    matched_ids: list[str] = []

    for ref in refs:
        for c in corpus:
            if (
                c["section"].lower() == ref.lower()
                or ref.lower() in c["section"].lower()
            ):
                matched_ids.append(c["id"])

    return matched_ids


# ---------------------------------------------------------------------------
# Step 3: draft redline
# ---------------------------------------------------------------------------
def draft_redline(
    context: FindingContext,
    impact_type: str,
) -> str:
    """Draft a specific revision with CURRENT / PROPOSED blocks."""

    if _USE_LLM:
        out = _llm_call(
            system=(
                "You are a compliance drafting assistant. Draft a revision to the "
                "EXISTING INTERNAL CLAUSE so it complies with the NEW REGULATORY CLAUSE. "
                "Use ONLY requirements stated in the new regulatory clause. Do not "
                "invent dates, deadlines, retention periods, percentages, thresholds, "
                "exceptions, or implementation facts. If the source does not state a "
                "value, do not add one. Preserve conditions such as 'floating-rate' "
                "exactly; never invert a prohibition by adding an exception.\n\n"
                "Format EXACTLY as:\n\n"
                "CURRENT:\n[exact relevant existing internal policy wording]\n\n"
                "PROPOSED:\n[proposed revised wording]"
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
            "PROPOSED:\n"
            "No redline required — clause is already compliant or topically distinct."
        )

    proposed_text = int_clean

    # ---------------------------------------------------------------
    # Erasure mechanism
    # ---------------------------------------------------------------
    # IMPORTANT: Do not hard-code "30 days" here. The mechanism clause
    # can be separate from a processing-deadline clause.
    if "mechanism" in int_lower and (
        "does not currently provide" in int_lower
        or "no mechanism" in int_lower
    ):
        proposed_text = (
            "The Bank shall provide a secure mechanism through account service "
            "channels for data principals to request erasure of personal data."
        )

    # ---------------------------------------------------------------
    # Foreclosure / prepayment charge
    # ---------------------------------------------------------------
    elif "foreclosure charge" in int_lower or "prepayment" in int_lower:
        if (
            "shall not levy" in reg_lower
            or "shall not attract" in reg_lower
            or "no foreclosure" in reg_lower
            or "prohibited" in reg_lower
        ):
            # Replace the ENTIRE conflicting sentence. This intentionally
            # removes the old "except where the loan carries a floating rate"
            # condition so the proposed wording cannot reverse the regulation.
            proposed_text = (
                "Prepayment of the loan, in part or in full, shall not attract "
                "any foreclosure or prepayment charge where the loan is a "
                "floating-rate retail loan availed by an individual borrower."
            )

    # ---------------------------------------------------------------
    # Re-KYC waiver
    # ---------------------------------------------------------------
    elif "waived" in int_lower and (
        "mandatory" in reg_lower
        or "shall not be waived" in reg_lower
        or "shall be conducted" in reg_lower
    ):
        proposed_text = (
            "Re-KYC verification is mandatory and shall not be waived solely "
            "on the basis that registered address or contact details remain "
            "unchanged."
        )

    # ---------------------------------------------------------------
    # Retention timeframe
    # ---------------------------------------------------------------
    elif (
        ("retained" in int_lower or "retention" in int_lower)
        and any(term in reg_lower for term in ["retention", "retained"])
    ):
        required_match = re.search(
            r"\b(\d+)\s*\(?\d*\)?\s*years?\b",
            reg_lower,
        )

        if required_match:
            required_period = required_match.group(1)
            proposed_text = re.sub(
                r"\b(?:ten|10)\s*\(?10\)?\s*years?\b",
                f"{required_period} years",
                int_clean,
                flags=re.IGNORECASE,
            )

    # ---------------------------------------------------------------
    # Rate-reset notice
    # ---------------------------------------------------------------
    elif "communicated to the borrower" in int_lower and (
        "15 days" in reg_lower
        or "fifteen (15) days" in reg_lower
    ):
        proposed_text = re.sub(
            r"thirty \(30\)\s*days",
            "at least fifteen (15) days",
            int_clean,
            flags=re.IGNORECASE,
        )

    # ---------------------------------------------------------------
    # Generic grounded fallback
    # ---------------------------------------------------------------
    if proposed_text == int_clean:
        if impact_type == "contradicts":
            proposed_text = (
                "Amend the clause to eliminate the conflict with the regulatory "
                f"requirement: \"{reg_clean[:220]}\"."
            )
        elif impact_type == "tightens":
            proposed_text = (
                "Update the internal threshold/timeframe to meet the stricter "
                f"regulatory standard: \"{reg_clean[:220]}\"."
            )
        else:
            proposed_text = (
                "Relax the internal standard to match the regulatory limit: "
                f"\"{reg_clean[:220]}\"."
            )

    # Keep the output clean for the frontend's CURRENT / PROPOSED split.
    return f"CURRENT:\n{int_clean}\n\nPROPOSED:\n{proposed_text}"


# ---------------------------------------------------------------------------
# Business action generation
# ---------------------------------------------------------------------------
def generate_business_action(
    context: FindingContext,
    impact_type: str,
) -> str:
    """Generate an auditable business action without inventing deadlines."""

    if impact_type == "no-op":
        return (
            "No business action required — internal clause appears compliant "
            "or topically distinct."
        )

    regulation_clause = context.regulation_clause
    internal_clause = context.internal_clause

    doc_title = internal_clause.get("doc_title", "Internal Policy Document")
    section = internal_clause.get("section", "Section")
    reg_lower = regulation_clause.lower()
    int_lower = internal_clause.get("text", "").lower()

    if (
        any(term in reg_lower for term in ["erase", "erasure", "deletion"])
        and any(
            term in int_lower
            for term in [
                "mechanism",
                "deletion",
                "erasure",
                "retained",
                "retention",
            ]
        )
    ):
        action_summary = (
            "Deploy a secure mechanism for customer data-erasure requests "
            "through the applicable account-service channel."
        )

    elif (
        any(
            term in reg_lower
            for term in ["mechanism", "deletion", "erasure", "erase"]
        )
        and ("mechanism" in int_lower or "deletion" in int_lower)
    ):
        action_summary = (
            "Implement the missing data-erasure request mechanism described "
            "by the regulation."
        )

    elif "foreclosure" in int_lower or "prepayment" in int_lower:
        action_summary = (
            "Eliminate foreclosure/prepayment charges where the new regulation "
            "prohibits them."
        )

    elif "waived" in int_lower or "re-kyc" in int_lower:
        action_summary = (
            "Update the periodic Re-KYC control to remove the prohibited "
            "waiver condition."
        )

    elif "communicated to the borrower" in int_lower or "notice" in int_lower:
        if "15 days" in reg_lower or "fifteen (15) days" in reg_lower:
            action_summary = (
                "Update the borrower rate-reset notice control to meet the "
                "source requirement."
            )
        else:
            action_summary = (
                "Update the borrower notification control to match the "
                "source requirement."
            )

    elif (
        ("retained" in int_lower or "retention" in int_lower)
        and any(term in reg_lower for term in ["retention", "retained"])
    ):
        regulation_periods = _extract_numbers(reg_lower)

        if regulation_periods:
            required_period = regulation_periods[-1]
            action_summary = (
                "Update the internal retention requirement to the "
                f"{required_period}-year period stated in the regulation."
            )
        else:
            action_summary = (
                "Update the internal retention requirement to match the "
                "regulatory standard."
            )

    elif impact_type == "contradicts":
        action_summary = (
            "Amend the conflicting internal policy provision to comply with "
            "the regulatory requirement."
        )

    elif impact_type == "tightens":
        action_summary = (
            "Update the operational control to meet the stricter regulatory "
            "requirement."
        )

    else:
        action_summary = (
            "Adjust the operational control to align with the regulatory "
            "requirement."
        )

    # Crucial: a missing deadline stays missing. We never invent a date.
    source_deadline = _extract_source_deadline(regulation_clause)
    deadline = source_deadline if source_deadline else "Not specified in source clause"

    return (
        f"ACTION REQUIRED: {action_summary}\n"
        f"TARGET POLICY: {doc_title} (Section {section})\n"
        "STEPS:\n"
        f"1. Update Section {section} text in the Master Policy Repository.\n"
        "2. Review affected operational records and systems.\n"
        "3. Implement the revised policy control and supporting workflow.\n"
        f"DEADLINE: {deadline}"
    )


# ---------------------------------------------------------------------------
# Step 4: self-critique / grounding
# ---------------------------------------------------------------------------
def self_critique(
    context: FindingContext,
    draft: str,
    business_action: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Verify that newly generated claims are grounded in the source clause.

    Checks:
      1. Explicit quoted citations exist in the source.
      2. New numeric claims in PROPOSED are supported by the source.
      3. New dates/years in PROPOSED are supported by the source.
      4. Material business-action topics are supported by the source.
      5. Known semantic inversion patterns are rejected.
      6. Optional LLM grounding review can veto the result.
    """

    source = context.regulation_clause.strip()
    source_lower = source.lower()
    proposed = _extract_proposed_section(draft)
    proposed_lower = proposed.lower()

    # ------------------------------------------------------------------
    # 1. Validate explicit quotes/citations.
    # ------------------------------------------------------------------
    quoted = _extract_quoted_text(draft)

    for q in quoted:
        if not q:
            continue

        q_lower = q.lower()
        if q_lower not in source_lower:
            prefix = q_lower[:60]
            if prefix and prefix not in source_lower:
                return (
                    False,
                    "Quoted citation in draft could not be matched against "
                    "the source regulatory clause.",
                )

    # ------------------------------------------------------------------
    # 2. Validate numeric claims.
    # ------------------------------------------------------------------
    source_numbers = set(_extract_numbers(source))
    proposed_numbers = set(_extract_numbers(proposed))
    unsupported_numbers = proposed_numbers - source_numbers

    if unsupported_numbers:
        return (
            False,
            "Source-grounding failed: proposed wording introduces numeric "
            "value(s) not present in the regulatory clause: "
            + ", ".join(str(v) for v in sorted(unsupported_numbers))
            + ".",
        )

    # ------------------------------------------------------------------
    # 3. Validate dates / years.
    # ------------------------------------------------------------------
    source_dates = _extract_years_and_dates(source)
    proposed_dates = _extract_years_and_dates(proposed)
    unsupported_dates = proposed_dates - source_dates

    if unsupported_dates:
        return (
            False,
            "Source-grounding failed: proposed wording introduces date/year "
            "value(s) not present in the regulatory clause: "
            + ", ".join(sorted(unsupported_dates))
            + ".",
        )

    # ------------------------------------------------------------------
    # 4. Known semantic inversion guardrail: floating-rate prohibition.
    # ------------------------------------------------------------------
    source_prohibits_float_charge = (
        (
            "shall not levy" in source_lower
            or "shall not attract" in source_lower
            or "no foreclosure" in source_lower
        )
        and (
            "floating-rate" in source_lower
            or "floating rate" in source_lower
        )
        and (
            "foreclosure" in source_lower
            or "prepayment" in source_lower
        )
    )

    proposed_exempts_float = (
        "except where the loan carries a floating rate" in proposed_lower
        or "except where the loan is a floating-rate" in proposed_lower
        or "except for floating-rate" in proposed_lower
        or "except for floating rate" in proposed_lower
    )

    if source_prohibits_float_charge and proposed_exempts_float:
        return (
            False,
            "Source-grounding failed: proposed redline reverses the regulatory "
            "condition by exempting floating-rate loans from a prohibition "
            "that specifically applies to floating-rate loans.",
        )

    # ------------------------------------------------------------------
    # 5. Conservative business-action topic grounding.
    # ------------------------------------------------------------------
    if business_action:
        action_lines = [
            line.split(":", 1)[1].strip()
            for line in business_action.splitlines()
            if line.startswith("ACTION REQUIRED:")
        ]

        action_text = " ".join(action_lines).strip()

        if action_text:
            action_groups = _material_topic_groups(action_text)
            source_groups = _material_topic_groups(source)
            unsupported_groups = action_groups - source_groups

            # Generic implementation vocabulary isn't treated as an
            # unsupported regulatory requirement by itself.
            unsupported_groups -= {
                "records",
                "access",
                "channel",
            }

            if unsupported_groups:
                return (
                    False,
                    "Source-grounding failed: business action introduces "
                    "material topic(s) not supported by the regulatory clause: "
                    + ", ".join(sorted(unsupported_groups))
                    + ".",
                )

    # ------------------------------------------------------------------
    # 6. Optional LLM grounding check.
    # ------------------------------------------------------------------
    if _USE_LLM:
        critique = _llm_call(
            system=(
                "You are a strict compliance evidence checker. Determine whether "
                "the proposed revision and business action are fully supported by "
                "the NEW REGULATORY CLAUSE. Any invented deadline, number, retention "
                "period, threshold, date, exception, or material requirement means "
                "FAIL. Preserve conditional scope such as 'floating-rate'. "
                "Respond exactly: PASS | reason or FAIL | reason."
            ),
            user=(
                f"NEW REGULATORY CLAUSE:\n{source}\n\n"
                f"PROPOSED REDLINE:\n{draft}\n\n"
                f"BUSINESS ACTION:\n{business_action or 'None'}"
            ),
            max_tokens=220,
        )

        if critique and "|" in critique:
            verdict, reason = critique.split("|", 1)
            verdict = verdict.strip().upper()

            if verdict == "FAIL":
                return (
                    False,
                    "LLM source-grounding check failed: " + reason.strip(),
                )

            if verdict != "PASS":
                return (
                    False,
                    "LLM source-grounding check returned an invalid verdict; "
                    "escalation required.",
                )

    return (
        True,
        "Source-grounding checks passed: citations, numeric/date claims, "
        "semantic conditions, and material business-action topics are supported "
        "by the regulatory clause.",
    )


# ---------------------------------------------------------------------------
# Step 5: confidence / risk / routing
# ---------------------------------------------------------------------------
def score_confidence_risk(
    context: FindingContext,
    impact_type: str,
    citation_verified: bool,
) -> tuple[float, str, str]:
    """Score confidence, risk, and routing."""

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


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_agent_pipeline(
    regulation_clause_text: str,
    internal_clause: dict,
    retrieval_rerank_score: float,
    corpus: list[dict],
    regulation_id: str,
    regulation_clause_id: str,
    internal_policy_id: str,
    internal_clause_id: str,
) -> AgentResult:
    """Run the complete RCIA state machine for one candidate pair."""

    context = FindingContext(
        regulation_id=regulation_id,
        regulation_clause_id=regulation_clause_id,
        internal_policy_id=internal_policy_id,
        internal_clause_id=internal_clause_id,
        regulation_clause=regulation_clause_text,
        internal_clause=internal_clause,
        retrieval_rerank_score=retrieval_rerank_score,
    )

    trace: list[ReasoningStep] = []

    # ------------------------------------------------------------------
    # Relevance gate
    # ------------------------------------------------------------------
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
            regulation_id=regulation_id,
            regulation_clause_id=regulation_clause_id,
            internal_policy_id=internal_policy_id,
            internal_clause_id=internal_clause_id,
            impact_type="no-op",
            impact_rationale="Candidate was below the minimum relevance threshold.",
            second_hop_refs=[],
            draft_redline="No redline — candidate dismissed by relevance filter.",
            business_action=(
                "No business action required — candidate dismissed by relevance filter."
            ),
            citation_verified=True,
            citation_check_detail="Impact classification skipped by relevance gate.",
            confidence=context.retrieval_rerank_score,
            risk="low",
            routing="auto_dismiss",
            trace=trace,
        )

    # ------------------------------------------------------------------
    # Step 1: classification
    # ------------------------------------------------------------------
    impact_type, rationale = classify_impact(context)

    trace.append(
        ReasoningStep(
            "1_classify",
            f"pair={regulation_clause_id}/{internal_clause_id}; "
            f"{impact_type}: {rationale}",
        )
    )

    # ------------------------------------------------------------------
    # Step 2: second-hop cross-reference
    # ------------------------------------------------------------------
    hop_ids = decide_second_hop(context, corpus)

    trace.append(
        ReasoningStep(
            "2_second_hop",
            (
                f"Cross-references found -> retrieved {len(hop_ids)} linked "
                f"clause(s): {hop_ids}"
                if hop_ids
                else "No cross-references detected; no second hop needed."
            ),
        )
    )

    # ------------------------------------------------------------------
    # Step 3: draft + business action
    # ------------------------------------------------------------------
    draft = draft_redline(context, impact_type)
    business_action = generate_business_action(context, impact_type)

    if impact_type == "no-op":
        trace.append(
            ReasoningStep(
                "3_draft",
                "No-op: no substantive redline required.",
            )
        )
        citation_verified = True
        citation_detail = "N/A for no-op classification."
        trace.append(
            ReasoningStep(
                "4_self_critique",
                citation_detail,
            )
        )
    else:
        trace.append(
            ReasoningStep(
                "3_draft",
                draft,
            )
        )

        citation_verified, citation_detail = self_critique(
            context,
            draft,
            business_action,
        )

        trace.append(
            ReasoningStep(
                "4_self_critique",
                citation_detail,
            )
        )

    # ------------------------------------------------------------------
    # Step 5: confidence / risk / routing
    # ------------------------------------------------------------------
    confidence, risk, routing = score_confidence_risk(
        context,
        impact_type,
        citation_verified,
    )

    trace.append(
        ReasoningStep(
            "5_score",
            f"pair={regulation_clause_id}/{internal_clause_id}; "
            f"confidence={confidence:.2f}, risk={risk}, routing={routing}",
        )
    )

    return AgentResult(
        regulation_id=regulation_id,
        regulation_clause_id=regulation_clause_id,
        internal_policy_id=internal_policy_id,
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
