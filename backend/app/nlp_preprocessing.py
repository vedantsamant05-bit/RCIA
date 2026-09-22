"""
nlp_preprocessing.py
---------------------
Section 4.2 of the project doc.

Deliberate design choice: clause-level segmentation, NOT fixed-token
chunking. Regulatory/legal text carries meaning at the level of an
obligation ("who must do what, by when"), and a fixed-token window will
routinely cut a clause in half or merge two unrelated obligations into
one chunk. Splitting on clause boundaries (numbered paragraphs, "shall"
statements, semicolon-joined provisos) keeps each unit semantically
whole, which materially improves downstream retrieval precision.

This module is intentionally regex/heuristic-based rather than a trained
model -- it's transparent, fast, needs no GPU, and is easy to defend
clause-by-clause in an interview. A production version would swap the
segmenter for a fine-tuned sentence/clause boundary model, and the
extractor for an NER model trained on regulatory obligation spans.
"""

import re
import uuid
from dataclasses import dataclass, field
from typing import Optional

# Matches numbered clause headers like "4(a):", "Para 3.1:", "Clause 2.2:",
# "Section 5.1 -" etc. at the start of a line/paragraph.
CLAUSE_HEADER_RE = re.compile(
    r"(?m)^\s*(?:Clause|Para|Section|Article|Regulation)?\s*"
    r"(\d+(?:\.\d+)*(?:\([a-zA-Z]\))?)\s*[:.\-]\s*"
)

OBLIGATION_VERBS = [
    "shall", "must", "is required to", "are required to", "shall not",
    "must not", "will be required to", "may not",
]

ACTOR_PATTERNS = [
    r"[Ee]very (?:data fiduciary|regulated entity|vendor|customer|borrower)",
    r"Regulated Entit(?:y|ies)",
    r"[Tt]he Bank",
    r"[Tt]he Borrower",
    r"[Tt]he Lender",
    r"[Vv]endor",
    r"[Cc]ustomers?",
    r"[Ee]mployees?",
    r"[Dd]ata fiduciar(?:y|ies)",
]

DATE_PATTERNS = [
    r"within\s+(?:\w+\s+)?\(?\d+\)?\s*(?:day|days|business day|business days|month|months|year|years)",
    r"at least\s+(?:\w+\s+)?\(?\d+\)?\s*(?:day|days|business day|business days|month|months|year|years)\s*(?:in advance|prior|before)?",
    r"not exceeding\s+(?:\w+\s+)?\(?\d+\)?\s*(?:day|days|month|months|year|years)",
]


@dataclass
class Clause:
    id: str
    text: str
    section_label: Optional[str] = None
    obligation_actor: Optional[str] = None
    obligation_action: Optional[str] = None
    effective_date: Optional[str] = None


def _extract_actor(text: str) -> Optional[str]:
    for pat in ACTOR_PATTERNS:
        m = re.search(pat, text)
        if m:
            return m.group(0)
    return None


def _extract_obligation_verb_phrase(text: str) -> Optional[str]:
    for verb in OBLIGATION_VERBS:
        idx = text.find(verb)
        if idx != -1:
            # Return the verb phrase + up to ~12 following words as the
            # "action" summary -- good enough for a demo audit trail.
            snippet = text[idx: idx + 140]
            return snippet.strip().rstrip(".") + ("..." if len(snippet) == 140 else "")
    return None


def _extract_date_clause(text: str) -> Optional[str]:
    for pat in DATE_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return m.group(0)
    return None


def segment_into_clauses(raw_text: str, id_prefix: str = "reg") -> list[Clause]:
    """
    Split raw regulatory text into clause-level units.

    Strategy:
    1. If numbered clause headers are present (Clause 4(a):, Para 3.1:, etc.),
       split on those -- this is the strongest, most reliable signal in
       real regulatory text.
    2. Otherwise, fall back to splitting on blank lines, then on sentence
       boundaries containing an obligation verb ("shall", "must").
    """
    raw_text = raw_text.strip()
    headers = list(CLAUSE_HEADER_RE.finditer(raw_text))

    segments = []
    if headers:
        for i, h in enumerate(headers):
            start = h.start()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(raw_text)
            label = h.group(1)
            body = raw_text[h.end():end].strip()
            if body:
                segments.append((label, body))
    else:
        # Fallback: split on blank lines / paragraphs
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", raw_text) if p.strip()]
        for i, p in enumerate(paragraphs):
            segments.append((f"p{i+1}", p))

    clauses = []
    for label, body in segments:
        clause = Clause(
            id=f"{id_prefix}-{uuid.uuid4().hex[:8]}",
            text=body,
            section_label=label,
            obligation_actor=_extract_actor(body),
            obligation_action=_extract_obligation_verb_phrase(body),
            effective_date=_extract_date_clause(body),
        )
        clauses.append(clause)
    return clauses


# Cross-reference detector used by the agent's "second retrieval hop"
# decision (Section 4.4, step 2). Looks for phrases like "as defined in
# Section 4.1" or "pursuant to clause 2.2" or "Annexure A" that suggest
# the clause's real meaning depends on another clause elsewhere.
CROSS_REF_RE = re.compile(
    r"(?:as defined in|pursuant to|as set out in|as specified in|per)\s+"
    r"(?:Section|Clause|Para|Article|Annexure)\s+([A-Za-z0-9.]+)",
    flags=re.IGNORECASE,
)


def find_cross_references(text: str) -> list[str]:
    return CROSS_REF_RE.findall(text)
