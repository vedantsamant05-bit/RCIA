import re
from backend.app.corpus_data import INTERNAL_POLICIES, EVAL_LABELS
from test_eval_retrieval import rerank, TOPICS, STOP, GENERIC, LightweightTfidf, PureBM25Okapi, _tokenize

# Test classification logic
def classify_test(reg_clause, int_clause_text):
    reg_lower = reg_clause.lower()
    int_lower = int_clause_text.lower()

    meaningful_reg_terms = {
        t for t in re.findall(r"[a-z]{4,}", reg_lower)
        if t not in {"shall", "must", "required", "within", "entity", "entities", "also"}
    }
    meaningful_int_terms = {
        t for t in re.findall(r"[a-z]{4,}", int_lower)
        if t not in {"shall", "must", "required", "within", "entity", "entities", "also"}
    }
    shared_terms = meaningful_reg_terms & meaningful_int_terms

    reg_topics = {TOPICS[t] for t in meaningful_reg_terms if t in TOPICS}
    int_topics = {TOPICS[t] for t in meaningful_int_terms if t in TOPICS}
    shared_topics = reg_topics & int_topics

    if not shared_topics and len(shared_terms) < 2:
        return "no-op", "No substantive obligation or domain link connects this candidate pair."

    # 1. Contradiction: Explicit Prohibition vs Permission / Waiver
    reg_prohibits = bool(re.search(r"\b(?:shall not|must not|prohibited|no longer permitted|cannot|may not)\b", reg_lower))
    int_permits_or_waives = bool(re.search(r"\b(?:may be waived|waived?|permitted|allowed|may|attract[s]? a charge)\b", int_lower))
    if reg_prohibits and int_permits_or_waives:
        for action_stem in ["waive", "foreclosure", "prepayment", "charge", "transfer", "disclose", "market"]:
            if action_stem in reg_lower and action_stem in int_lower:
                return "contradicts", f"New regulation prohibits or restricts an action ('{action_stem}') that internal policy permits or waives."

    # 2. Contradiction: Mandate vs Stated Lack of Capability / Mechanism
    int_lacks_capability = bool(re.search(r"\b(?:does not currently provide|no mechanism|does not support|cannot provide|does not permit)\b", int_lower))
    reg_mandates = any(v in reg_lower for v in ["shall", "must", "required to", "shall implement", "shall provide"])
    if int_lacks_capability and reg_mandates and not reg_prohibits:
        for cap_stem in ["mechanism", "deletion", "erasure", "erase", "portal", "channel", "request"]:
            if cap_stem in reg_lower or cap_stem in int_lower:
                return "contradicts", "Internal clause explicitly lacks a mechanism or capability that the new regulation mandates."

    if any(t in reg_lower for t in ["erase", "erasure", "deletion"]):
        if any(t in int_lower for t in ["retained for a period", "shall be retained"]):
            if "does not" in int_lower or "deletion" in int_lower or "prior to the standard retention" in int_lower:
                return "contradicts", "Regulatory erasure mandate directly conflicts with internal retention policy."

    # 3. Numeric & Timeframe / Frequency Comparison
    from backend.app.agent import _extract_numbers
    reg_numbers = _extract_numbers(reg_clause)
    int_numbers = _extract_numbers(int_clause_text)

    if reg_numbers and int_numbers:
        reg_num_set = set(reg_numbers)
        int_num_set = set(int_numbers)
        new_intervals = reg_num_set - int_num_set
        deadline_context = any(w in reg_lower for w in ["year", "years", "month", "months", "day", "days"])
        if new_intervals and deadline_context:
            if "medium" in reg_lower and not ("medium" in int_lower):
                return "tightens", f"Regulation introduces a mandatory timeframe ({min(new_intervals)} years) for customer tiers not covered in internal policy."
            if min(reg_num_set) < min(int_num_set):
                return "tightens", f"New regulatory minimum timeframe ({min(reg_num_set)}) is stricter than internal policy's minimum ({min(int_num_set)})."

        if deadline_context:
            transition = re.search(r"from\s+[^.]{0,80}?to\s+([^,.]+)", reg_lower)
            transition_numbers = _extract_numbers(transition.group(1)) if transition else []
            reg_n = transition_numbers[0] if transition_numbers else reg_numbers[-1]
            int_n = int_numbers[0]
            if reg_n < int_n:
                return "tightens", f"New regulatory timeframe ({reg_n}) is shorter than internal policy timeframe ({int_n})."
            elif reg_n > int_n and not new_intervals:
                return "loosens", f"New regulatory timeframe ({reg_n}) is longer than internal policy timeframe ({int_n})."

    # 4. Event-Driven / Trigger-Based Reviews
    has_event_trigger = bool(re.search(r"\b(?:material change|trigger(?:ed)?|event-driven|beneficial ownership|ownership structure|transaction behaviour|business activity)\b", reg_lower))
    if has_event_trigger and any(w in reg_lower for w in ["review", "updation", "kyc", "due diligence"]):
        if any(w in int_lower for w in ["periodic", "every", "years", "verification cycle"]):
            return "tightens", "Regulation introduces mandatory trigger-based reviews upon material customer or risk changes, expanding beyond periodic schedules."

    # 5. Automated Alerts & Overdue Identification
    has_alert_requirement = bool(re.search(r"\b(?:generate alert[s]?|automated alert[s]?|identify overdue|approaching.*deadline|overdue.*monitored)\b", reg_lower))
    if has_alert_requirement:
        if not re.search(r"\b(?:alert[s]?|overdue|automated)\b", int_lower):
            return "tightens", "Regulation introduces mandatory system alerts for approaching deadlines and overdue case tracking."

    # 6. Customer Advance Notification & Overdue Escalation
    has_notification_escalation = bool(re.search(r"\b(?:notified before.*due|reminder communication[s]?|escalated to.*compliance|escalate.*overdue)\b", reg_lower))
    if has_notification_escalation:
        if not re.search(r"\b(?:reminder|escalat(?:ed|ion))\b", int_lower):
            return "tightens", "Regulation mandates advance customer notifications and formal compliance escalation procedures."

    # 7. Change-Level Audit Trail Requirements
    has_audit_trail_mandate = bool(re.search(r"\b(?:audit trail|date of change|action performed|relevant customer record)\b", reg_lower))
    if has_audit_trail_mandate:
        if not re.search(r"\b(?:action performed)\b", int_lower):
            return "tightens", "Regulation mandates change-level audit trails capturing date of change, record ID, and action performed."

    # 8. Mandatory Compliance Tracking Metadata Fields
    has_field_tracking_mandate = bool(re.search(r"\b(?:next.*due date|review due date|outstanding.*requirements|current.*review status|last.*update)\b", reg_lower))
    if has_field_tracking_mandate:
        return "tightens", "Regulation requires maintaining structured tracking metadata (due dates, review status, outstanding requirements)."

    # 9. Multi-Factor Risk Classification Criteria
    has_multi_factor_risk = bool(re.search(r"\b(?:classified according to.*risk profile|geographical exposure|transaction behaviour|beneficial ownership|risk indicators)\b", reg_lower))
    if has_multi_factor_risk and any(w in int_lower for w in ["pep", "politically exposed", "risk", "onboarding"]):
        return "tightens", "Regulation mandates comprehensive multi-factor risk profiling criteria."

    # 10. Expanded Channel / Interface Scope
    has_expanded_scope = bool(re.search(r"\b(?:every page|all pages|same channel|all channels|not solely)\b", reg_lower))
    if has_expanded_scope:
        if any(narrow in int_lower for narrow in ["home screen", "branch manager", "written request"]):
            return "tightens", "Regulation expands control scope across all application pages or digital service channels."

    # 11. Policy Review, System Configuration & Pre-Implementation Testing
    has_remediation_testing = bool(re.search(r"\b(?:review existing.*policies|identify gaps|remediation shall be implemented|updated and tested before implementation|assess the impact.*system)\b", reg_lower))
    if has_remediation_testing and any(w in int_lower for w in ["sop", "procedure", "policy", "onboarding", "kyc", "system"]):
        return "tightens", "Regulation imposes mandatory policy and system gap assessment, remediation, and pre-implementation testing controls."

    # 12. Already compliant / satisfied
    return "no-op", "Clauses share domain vocabulary but internal policy clause already appears compliant or no operational gap was identified."

corpus = []
for doc, clauses in INTERNAL_POLICIES.items():
    for sec, txt in clauses:
        corpus.append({'id': f"{doc}::{sec}", 'doc_title': doc, 'section': sec, 'text': txt})

print('=== TESTING EVAL SET WITH NEW CLASSIFIER ===')
tp = fp = fn = tn = 0
correct_type = 0
total_impactful = 0
for lbl in EVAL_LABELS:
    q = lbl['regulation_clause_text']
    target_key = f"{lbl['internal_doc']}::{lbl['internal_section']}"
    target_clause = next(c for c in corpus if c['id'] == target_key)
    pred_type, reason = classify_test(q, target_clause['text'])
    actual_impacted = bool(lbl['is_impacted'])
    pred_impacted = (pred_type != 'no-op')
    true_type = lbl['true_impact_type']

    if actual_impacted and pred_impacted: tp += 1
    elif actual_impacted and not pred_impacted: fn += 1
    elif not actual_impacted and pred_impacted: fp += 1
    else: tn += 1

    if actual_impacted:
        total_impactful += 1
        if pred_type == true_type:
            correct_type += 1

    print(f"[{'PASS' if (pred_impacted == actual_impacted and (not actual_impacted or pred_type == true_type)) else 'FAIL'}] target={target_key[:35]:35} | actual={true_type:10} | pred={pred_type:10} | {reason[:60]}")

prec = tp / (tp + fp) if (tp + fp) else 0.0
rec = tp / (tp + fn) if (tp + fn) else 0.0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
acc = correct_type / total_impactful if total_impactful else 0.0
print(f"\nEval Results: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}, Impact Type Acc={acc:.3f} (tp={tp}, fp={fp}, fn={fn}, tn={tn})")
