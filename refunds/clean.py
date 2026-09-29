"""Load tickets, remove migration duplicates, fix legacy money units.

Decisions (see DECISIONS.md):
  * A ticket_id present in both helpdesk and legacy_fd is a re-import. Keep the helpdesk copy.
  * legacy_fd money is in paise (proved: on all duplicated refund pairs legacy == 100 x helpdesk).
  * Month = month of created_at (the export has no refund-date field).
"""
import numpy as np
import pandas as pd

PAISE = 100


def load_tickets(path):
    t = pd.read_csv(path)
    t["created_dt"] = pd.to_datetime(t["created_at"])
    return t


def clean_tickets(raw):
    """Return (clean_df, log) where log is a list of (check, value) for the reconciliation sheet."""
    t = raw.copy()
    log = [("raw rows in tickets.csv", len(t))]

    t["refund_as_exported"] = t["refund_amount_inr"]
    is_legacy = t["source_system"] == "legacy_fd"
    t["amt"] = np.where(is_legacy, t["refund_amount_inr"] / PAISE, t["refund_amount_inr"])

    both = t.groupby("ticket_id")["source_system"].nunique()
    dup_ids = set(both[both > 1].index)
    h = t[(t.source_system == "helpdesk") & t.ticket_id.isin(dup_ids)].set_index("ticket_id").sort_index()
    l = t[(t.source_system == "legacy_fd") & t.ticket_id.isin(dup_ids)].set_index("ticket_id").sort_index()
    money_mismatch = int(((h["amt"].fillna(0) - l["amt"].fillna(0)).abs() > 0.5).sum())
    log.append(("ticket_ids present under both source systems (re-imports)", len(dup_ids)))
    log.append(("re-import pairs whose rupee refund differs after /100 (should be 0)", money_mismatch))
    log.append(("repeat of same ticket_id inside ONE system (should be 0)",
                int(t.duplicated(subset=["ticket_id", "source_system"]).sum())))

    drop = is_legacy & t.ticket_id.isin(dup_ids)
    log.append(("re-import rows dropped (legacy copy)", int(drop.sum())))
    c = t[~drop].copy().reset_index(drop=True)
    assert c.ticket_id.is_unique, "ticket_id not unique after de-dup"
    log.append(("clean tickets", len(c)))

    c["month"] = c["created_dt"].dt.to_period("M").astype(str)
    c["quarter"] = c["created_dt"].dt.to_period("Q").astype(str)
    return c, log
