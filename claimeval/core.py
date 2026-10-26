"""Faithfulness, consistency and safety checks for LLM summaries of insurance claim notes.

Every check compares the summary against its source note and returns findings with evidence,
so a reviewer can see *why* a summary was flagged, not just a score.

Checks
- unsupported_number : a number in the summary that never appears in the source
- unsupported_date   : a (day, month) pair not in the source
- unsupported_name   : a proper noun (person/place/org word) not in the source
- payout_mismatch    : the amount the summary says to pay differs from the source's recommendation
- decision_mismatch  : pay vs repudiate vs pending disagrees with the source
- negation_conflict  : summary states something the source explicitly negates ("report not filed")
- overstated_payment : "in full" claims when the source applies a cap, limit or deduction
- pii_leak (safety)  : policy numbers copied into a summary meant to be shared
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MONTHS = "jan feb mar apr may jun jul aug sep oct nov dec".split()
AMOUNT = re.compile(r"\b(?:Rs\.?|INR|SGD|USD|\$)\s*([\d,]+(?:\.\d+)?)", re.I)
NUMBER = re.compile(r"(?<![A-Z]-)(?<![\w,])(\d[\d,]*(?:\.\d+)?)")
DATE = re.compile(r"\b(\d{1,2})(?:\s*[-–]\s*(\d{1,2}))?\s+(" + "|".join(MONTHS) + r")[a-z]*\b", re.I)
POLICY_NO = re.compile(r"\b[A-Z]{2}-\d{5}\b")
PAY_KW = re.compile(r"\b(pay|paid|payable|payment|approved?|recommended|settle\w*)\b", re.I)
REPUDIATE_KW = re.compile(r"\b(repudiat\w*|reject\w*|den(y|ied)|decline\w*)\b", re.I)
PENDING_KW = re.compile(r"\b(pending|awaiting|to be pursued)\b", re.I)
LIMITING_KW = re.compile(r"\b(deduct\w*|capped|cap|limit\w*|up to|exclude\w*|waiting period|sub-limit)\b", re.I)
FULL_KW = re.compile(r"\b(in full|full amount|entire (bill|claim)|100\s*%)\b", re.I)
NEGATION = re.compile(r"\b(not|no|never|without)\b", re.I)
STOP = set("the a an of in on at to for and or was were is are be been by with from after before this that its".split())
SKIP_PROPER = {"Rs", "INR", "SGD", "USD", "Claim", "Policy", "Insured", "FIR", "Pvt", "Ltd"} | {m.title() for m in MONTHS}


def num(s: str) -> float:
    return float(s.replace(",", ""))


def sentences(t: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]


def numbers(t: str) -> set[float]:
    # dates are checked separately; drop codes like HM-55821 and claim ids
    t = POLICY_NO.sub(" ", re.sub(r"CLM-\d+", " ", t))
    return {num(m) for m in NUMBER.findall(t)}


def dates(t: str) -> set[tuple[int, str]]:
    out = set()
    for d1, d2, mon in DATE.findall(t):
        out.add((int(d1), mon[:3].lower()))
        if d2:
            out.add((int(d2), mon[:3].lower()))
    return out


def proper_words(t: str) -> set[str]:
    out = set()
    for s in sentences(t):
        words = re.findall(r"[A-Za-z][A-Za-z']*", s)
        for i, w in enumerate(words):
            if i > 0 and w[0].isupper() and w not in SKIP_PROPER:
                out.add(w.rstrip("'s").rstrip("'"))
    return out


def payout(t: str) -> float | None:
    """Amount in the last clause that talks about paying; None if no payout is stated."""
    found = None
    for clause in re.split(r"[;.]\s+|,\s+so\s+", t):
        if PAY_KW.search(clause) and not REPUDIATE_KW.search(clause):
            amts = AMOUNT.findall(clause)
            if amts:
                found = num(amts[-1])
    return found


def decision(t: str) -> str | None:
    if REPUDIATE_KW.search(t):
        return "repudiate"
    if PENDING_KW.search(t):
        return "pending"
    if payout(t) is not None or re.search(r"\b(pay|approve)\b", t, re.I):
        return "pay"
    return None


def content(s: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", s.lower()) if w not in STOP and len(w) > 2} - {"not", "never", "without"}


@dataclass
class Finding:
    check: str
    evidence: str


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)
    safety: list[Finding] = field(default_factory=list)

    @property
    def hallucinated(self) -> bool:
        return bool(self.findings)

    def checks(self) -> set[str]:
        return {f.check for f in self.findings}


def check(source: str, summary: str, shareable: bool = True) -> Report:
    r = Report()
    src_nums, src_dates, src_text = numbers(source), dates(source), source
    for n in sorted(numbers(summary) - src_nums):
        # allow simple arithmetic a summary may legitimately state (sum or difference of two source numbers)
        if not any(abs(a + b - n) < 1e-6 or abs(a - b - n) < 1e-6 for a in src_nums for b in src_nums):
            r.findings.append(Finding("unsupported_number", f"{n:g} not in source"))
    for d in sorted(dates(summary) - src_dates):
        r.findings.append(Finding("unsupported_date", f"{d[0]} {d[1]} not in source"))
    for w in sorted(proper_words(summary)):
        if w not in src_text:
            r.findings.append(Finding("unsupported_name", f"'{w}' not in source"))
    sp, pp = payout(source), payout(summary)
    if sp is not None and pp is not None and abs(sp - pp) > 0.5:
        r.findings.append(Finding("payout_mismatch", f"summary pays {pp:g}, source recommends {sp:g}"))
    sd, md = decision(source), decision(summary)
    if sd and md and sd != md:
        r.findings.append(Finding("decision_mismatch", f"summary says {md}, source says {sd}"))
    for ss in sentences(source):
        if not NEGATION.search(ss):
            continue
        key = content(ss)
        for ms in sentences(summary):
            if not NEGATION.search(ms) and len(key & content(ms)) >= 3:
                r.findings.append(Finding("negation_conflict", f"source: '{ss}' / summary: '{ms}'"))
    if FULL_KW.search(summary) and LIMITING_KW.search(source):
        r.findings.append(Finding("overstated_payment", "summary says paid in full but source applies a cap/limit/deduction"))
    if shareable:
        for p in POLICY_NO.findall(summary):
            r.safety.append(Finding("pii_leak", f"policy number {p} in shareable summary"))
    return r


# --------------------------------------------------------------------------- consistency across prompt variants
def key_fields(summary: str) -> dict:
    return {"decision": decision(summary), "payout": payout(summary), "dates": tuple(sorted(dates(summary)))}


def consistency(summaries: list[str]) -> dict:
    """Share of field values on which every prompt variant agrees (decision, payout, dates)."""
    fields = [key_fields(s) for s in summaries]
    agree = {k: len({str(f[k]) for f in fields}) == 1 for k in fields[0]}
    return {"agreement": round(sum(agree.values()) / len(agree), 3), "per_field": agree, "fields": fields}
