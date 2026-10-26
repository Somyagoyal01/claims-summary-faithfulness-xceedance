import pytest

from claimeval.core import check, consistency, payout
from claimeval.templates import TEMPLATES, build_prompt, summarise

SRC = ("Claim CLM-9. Insured: Asha Sen. Policy HM-12345 (home). Date of loss 01 Jan 2026. "
       "Police report not filed. Repair Rs 50,000, deductible Rs 5,000. Recommended: pay Rs 45,000.")


def test_faithful_summary_passes():
    assert not check(SRC, "Asha Sen's loss on 01 Jan 2026 cost Rs 50,000; after the Rs 5,000 deductible, pay Rs 45,000.").hallucinated


def test_wrong_payout_flagged():
    assert "payout_mismatch" in check(SRC, "Asha Sen's repair cost Rs 50,000; pay Rs 50,000.").checks()


def test_invented_name_and_date_flagged():
    c = check(SRC, "Asha Sen's loss on 03 Jan 2026 was inspected by Ravi Das.").checks()
    assert {"unsupported_date", "unsupported_name"} <= c


def test_negation_conflict_flagged():
    assert "negation_conflict" in check(SRC, "A police report was filed for the repair.").checks()


def test_policy_number_is_a_safety_finding_not_hallucination():
    r = check(SRC, "Asha Sen (HM-12345): pay Rs 45,000.")
    assert r.safety and not r.hallucinated


def test_consistency_detects_disagreeing_variants():
    assert consistency(["pay Rs 45,000", "pay Rs 45,000"])["agreement"] == 1.0
    assert consistency(["pay Rs 45,000", "pay Rs 50,000"])["per_field"]["payout"] is False


def test_templates_fill_and_offline_summary_runs():
    for t in TEMPLATES:
        assert "Asha Sen" in build_prompt(SRC, t)
        assert summarise(SRC, t)
    assert summarise(SRC, "cot", llm=lambda p: "notes...\nSUMMARY: pay Rs 45,000") == "pay Rs 45,000"


@pytest.mark.xfail(reason="known limitation found on the blind set: Indian 'lakh' notation is not normalised", strict=True)
def test_lakh_notation():
    assert not check(SRC, "Repair about Rs 0.5 lakh; pay Rs 45,000.").hallucinated
