"""Run with `python -m pytest tests` or plain `python tests/test_pipeline.py`."""
import os, sys
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from refunds.clean import load_tickets, clean_tickets
from refunds.classify import detect_double_payout

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "tickets.csv")
POS = [
    "Issued refund + replacement both, TL aware.",
    "Refunded Rs 5199 and also raised replacement, cx was very upset.",
    "Cx got the amount back, also sending a new set to keep him happy.",
    "Money back to source and a fresh pair shipped.",
]
NEG = [
    "Replacement was offered earlier by chat team, cx opted for refund.",
    "Refunded. Replacement request rejected as per policy.",
    "refund issued; rplc not applicable (out of stock).",
    "cx did not want replacement.",
]


def _hit(note):
    return bool(detect_double_payout(pd.DataFrame({"agent_notes": [note], "replacement_issued": ["N"]}))[1].iloc[0])


def test_double_payout_regex():
    assert all(_hit(n) for n in POS)
    assert not any(_hit(n) for n in NEG)


def test_cleaning_invariants():
    if not os.path.exists(DATA):
        return
    raw = load_tickets(DATA)
    c, log = clean_tickets(raw)
    assert c.ticket_id.is_unique
    assert len(c) == raw.ticket_id.nunique()
    assert c.amt.max() < 30000, "legacy paise not rescaled"
    assert dict(log)["re-import pairs whose rupee refund differs after /100 (should be 0)"] == 0


if __name__ == "__main__":
    test_double_payout_regex(); test_cleaning_invariants(); print("all tests passed")
