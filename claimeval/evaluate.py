"""Score the hallucination detector on hand-labelled summaries, and measure prompt-template consistency.

    python -m claimeval.evaluate            # dev set (used while writing the checks)
    python -m claimeval.evaluate --blind    # + blind set written after the code was frozen (run once)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from .core import check, consistency
from .templates import TEMPLATES, summarise

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NOTES = {n["id"]: n["text"] for n in json.loads((DATA / "claim_notes.json").read_text())}


def load(name):
    return [json.loads(l) for l in (DATA / name).read_text().splitlines() if l.strip()]


def detector_eval(rows) -> dict:
    tp = fp = fn = tn = 0
    errors, fired = [], Counter()
    for r in rows:
        rep = check(NOTES[r["claim"]], r["summary"])
        pred, gold = rep.hallucinated, r["label"] == "hallucinated"
        fired.update(rep.checks())
        tp += pred and gold; fp += pred and not gold; fn += gold and not pred; tn += not pred and not gold
        if pred != gold:
            errors.append({"claim": r["claim"], "gold": r["label"], "error": r.get("error"),
                           "findings": [f.evidence for f in rep.findings], "summary": r["summary"]})
    p = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return {"n": len(rows), "tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(p, 3), "recall": round(rc, 3),
            "f1": round(2 * p * rc / (p + rc), 3) if p + rc else 0.0, "checks_fired": dict(fired), "errors": errors}


def template_eval(llm=None) -> dict:
    per_claim, flagged, leaks = {}, Counter(), Counter()
    for cid, note in NOTES.items():
        outs = {t: summarise(note, t, llm) for t in TEMPLATES}
        for t, s in outs.items():
            rep = check(note, s)
            flagged[t] += rep.hallucinated
            leaks[t] += bool(rep.safety)
        per_claim[cid] = consistency(list(outs.values()))["agreement"]
    n = len(NOTES)
    return {"mode": "offline extractive stand-in" if llm is None else "llm",
            "mean_cross_template_agreement": round(sum(per_claim.values()) / n, 3),
            "flag_rate_by_template": {t: round(flagged[t] / n, 3) for t in TEMPLATES},
            "pii_leak_rate_by_template": {t: round(leaks[t] / n, 3) for t in TEMPLATES},
            "agreement_by_claim": per_claim}


def main(blind: bool):
    out = {"dev": detector_eval(load("labelled_dev.jsonl")), "templates": template_eval()}
    if blind:
        out["blind"] = detector_eval(load("labelled_blind.jsonl"))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / ("metrics.json" if blind else "metrics_dev.json")).write_text(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    print(json.dumps(main("--blind" in sys.argv), indent=1))
