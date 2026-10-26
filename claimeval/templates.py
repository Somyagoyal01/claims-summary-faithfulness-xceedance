"""Prompt templates (zero-shot, few-shot, chain-of-thought, ReAct-style) and an offline summariser stand-in.

`summarise(note, template, llm=None)` sends the filled prompt to `llm` (any callable str->str,
e.g. a Hugging Face pipeline or a hosted API). Without one, an extractive stand-in is used so the
evaluation pipeline runs offline; it picks different sentences per template so consistency can be measured.
"""
from __future__ import annotations

from typing import Callable, Optional

from .core import sentences

RULES = ("Use only facts in the note. Copy amounts and dates exactly. State the decision "
         "(pay amount / repudiate / pending). Do not include policy numbers.")

TEMPLATES = {
    "zero_shot": "{rules}\n\nClaim note:\n{note}\n\nSummary (2 sentences):",
    "few_shot": "{rules}\n\nExample note: Insured: A. Policy XX-00000. Loss 01 Jan 2026. Repair Rs 10,000, deductible Rs 1,000. Pay Rs 9,000.\n"
                "Example summary: A's loss on 01 Jan 2026 cost Rs 10,000; after the Rs 1,000 deductible, pay Rs 9,000.\n\n"
                "Claim note:\n{note}\n\nSummary:",
    "cot": "{rules}\n\nClaim note:\n{note}\n\nFirst list the loss, the amounts, any caps or deductions and the decision. "
           "Then write a 2-sentence summary after the line 'SUMMARY:'.",
    "react": "{rules}\nYou may call lookup(field) to re-read a field of the note before writing.\n\nClaim note:\n{note}\n\n"
             "Thought / Action / Observation ... then 'SUMMARY:'.",
}


def build_prompt(note: str, template: str) -> str:
    return TEMPLATES[template].format(rules=RULES, note=note)


def _offline(note: str, template: str) -> str:
    s = [x for x in sentences(note) if not x.startswith(("Claim CLM", "Policy"))]
    s = [x.replace("Insured: ", "") for x in s]
    picks = {"zero_shot": [0, -1], "few_shot": [1, -1], "cot": [0, 1, -1], "react": [1, 2, -1]}[template]
    return " ".join(dict.fromkeys(s[i] for i in picks if -len(s) <= i < len(s)))


def summarise(note: str, template: str, llm: Optional[Callable[[str], str]] = None) -> str:
    if llm is None:
        return _offline(note, template)
    out = llm(build_prompt(note, template))
    return out.split("SUMMARY:")[-1].strip()
