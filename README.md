# Claim-summary faithfulness checks

This is a personal project by Somya Goyal, built while applying to Xceedance (Junior GenAI Developer, Delhi). The claim notes and summaries are fictional and written by hand. No company data is used.

## Why this matters
When an LLM summarises a claim note, the costly errors are small and confident: a payout that ignores the deductible, a surveyor's name from another file, "approved in full" when a sub-limit applies. This toolkit checks each summary against its source note. When it flags a summary, it shows the evidence, so a reviewer can see why. It also measures how much prompt templates agree with each other.

## What it checks (`claimeval/core.py`)

| Check | Catches |
|---|---|
| unsupported_number / _date / _name | figures, dates or proper nouns not in the source note |
| payout_mismatch | stated payout differs from the source recommendation |
| decision_mismatch | pay vs repudiate vs pending reversed |
| negation_conflict | summary asserts what the source negates ("police report not filed") |
| overstated_payment | "in full" when the source applies a cap, limit or deduction |
| pii_leak (safety) | policy numbers in a summary meant to be shared |

`claimeval/templates.py` holds zero-shot, few-shot, chain-of-thought and ReAct-style prompts. `summarise(note, template, llm=...)` takes any model callable, such as a Hugging Face pipeline or a hosted API. `consistency()` scores agreement on the decision, payout and dates across template outputs.

## Results (as measured)

| Labelled set | n | Precision | Recall | F1 |
|---|---|---|---|---|
| Dev (written while building the checks) | 16 | 1.00 | 1.00 | 1.00 |
| **Blind** (written after the code was frozen, run once) | 15 | **0.56** | **0.56** | **0.56** |

The dev score confirms itself, because those labels were written alongside the checks. **The blind score is the honest number.** The code hashes at freeze time are in `results/code_freeze_sha256.txt`, and the full output is in `results/metrics.json`.

What the blind set showed:
- **Caught:**
  - a surveyor name taken from a different claim
  - a reversed decision
  - an invented room-rent deduction
  - a wrong city
  - a "closed" status on a pending claim
- **False alarms (4):**
  - Indian "lakh" notation and spelled-out month names were treated as unsupported.
  - The payout parser picked the wrong amount from a sentence that mentioned several amounts.
- **Missed (4).** These are relational errors where every word appears somewhere in the source:
  - a real date attached to the wrong event
  - "FIR a week later"
  - a "fire" cause the source never mentions
  - "declared jewellery" when the source says it was not declared

**Takeaway:** surface-level entity checks catch copied-wrong facts. Errors about relations and events need a claim-level check, such as an NLI model or an LLM judge given the source, and that check should be scored against the same blind set.

Template consistency (offline extractive stand-in, so this tests the metric, not a model): mean cross-template agreement is 0.667 across 8 claims. With a real model, run `template_eval(llm=...)`.

## Run
```bash
pip install -r requirements.txt
python -m claimeval.evaluate            # dev
python -m claimeval.evaluate --blind    # + blind set
pytest -q
```

## Limitations
- The checks are rules, not a model. They are tuned to the style of these notes, which uses rupee amounts, "DD Mon YYYY" dates and "Recommended:" lines.
- The labelled sets are small (16 + 15), so treat the numbers as directional only.
- Lakh notation is a known gap and is kept as an `xfail` test.
