"""Build the monthly refund pack: by reason, by agent, reconciliation, leakage lists."""
import numpy as np
import pandas as pd
from .classify import GOODWILL_CAP

REPL_SHIP = 340  # policy s5: unit cost + Rs 340 reverse pickup and forward shipping


def build_frames(clean, refunds_cls, products, agents, metrics, log):
    r = refunds_cls.copy()
    prod = products.set_index("sku")
    flag, txt = r["_flag"], r["_txt"]
    r["double_payout"] = flag | txt
    r["double_payout_basis"] = np.where(flag & txt, "flag + note", np.where(flag, "flag only", np.where(txt, "note only", "")))
    r["replacement_cost"] = np.where(r.double_payout, r.product_sku.map(prod.unit_cost_inr) + REPL_SHIP, 0)
    r["gw_over_cap"] = r.true_reason.isin(["GW-OTHER (goodwill stated)"]) & (r.amt > GOODWILL_CAP)
    r["gw_code_over_cap"] = (r.refund_reason_code == "GW-OTHER") & (r.amt > GOODWILL_CAP)
    ag = agents.set_index("agent_id")
    r["agent_name"] = r.agent_id.map(ag.name)
    r["agent_team"] = r.agent_id.map(ag.team)
    r["agent_tier"] = r.agent_id.map(ag.tier)

    monthly_reason = r.pivot_table(index="true_reason", columns="month", values="amt", aggfunc="sum", fill_value=0)
    monthly_reason["Total"] = monthly_reason.sum(axis=1)
    monthly_reason.loc["TOTAL"] = monthly_reason.sum()
    monthly_reason_as_coded = r.pivot_table(index="refund_reason_code", columns="month", values="amt", aggfunc="sum", fill_value=0)
    monthly_reason_as_coded["Total"] = monthly_reason_as_coded.sum(axis=1)
    monthly_reason_as_coded.loc["TOTAL"] = monthly_reason_as_coded.sum()

    idx = r.agent_id.astype(str) + " " + r.agent_name.fillna("") + " (" + r.agent_team.fillna("") + ")"
    monthly_agent = r.assign(agent=idx).pivot_table(index="agent", columns="month", values="amt", aggfunc="sum", fill_value=0)
    monthly_agent["Total"] = monthly_agent.sum(axis=1)
    monthly_agent = monthly_agent.sort_values("Total", ascending=False)
    monthly_agent.loc["TOTAL"] = monthly_agent.sum()

    handled = clean.groupby("agent_id").size().rename("tickets_handled")
    sc = r.groupby("agent_id").agg(refunds=("amt", "size"), refund_inr=("amt", "sum"),
                                   gw_coded_pct=("refund_reason_code", lambda s: (s == "GW-OTHER").mean()),
                                   unresolved_reason=("true_reason", lambda s: (s == "GW-OTHER (unresolved)").sum()),
                                   goodwill_stated=("true_reason", lambda s: (s == "GW-OTHER (goodwill stated)").sum()),
                                   double_payouts=("double_payout", "sum"),
                                   replacement_cost_inr=("replacement_cost", "sum")).join(handled)
    sc["refunds_per_100_tickets"] = sc.refunds / sc.tickets_handled * 100
    sc["double_payout_rate_pct"] = sc.double_payouts / sc.refunds * 100
    sc = sc.join(ag[["name", "team", "site", "tier"]])
    sc["team_median_refunds_per_100"] = sc.groupby("team").refunds_per_100_tickets.transform("median")
    sc["note"] = np.where(sc.tier == 2, "Tier 2: multi-touch, do not rank against Tier 1 on volume (policy s6)", "")
    sc = sc.reset_index().sort_values("double_payouts", ascending=False)

    team = r.groupby("agent_team").agg(refunds=("amt", "size"), refund_inr=("amt", "sum"), double_payouts=("double_payout", "sum"))
    team["share_of_refund_inr_pct"] = team.refund_inr / team.refund_inr.sum() * 100
    team = team.sort_values("refund_inr", ascending=False)

    quarterly = r.groupby("quarter").agg(refund_inr=("amt", "sum"), refunds=("amt", "size"),
                                         double_payouts=("double_payout", "sum"),
                                         replacement_cost_inr=("replacement_cost", "sum"))
    quarterly["pulse2_refund_inr"] = r[r.product_sku == "VA-EB-PL2"].groupby("quarter").amt.sum()
    quarterly = quarterly.fillna(0)
    quarterly["ex_pulse2_refund_inr"] = quarterly.refund_inr - quarterly.pulse2_refund_inr

    # --- reconciliation
    raw_sum = clean.refund_as_exported.sum()
    rec = pd.DataFrame(log, columns=["check", "value"])
    total = r.amt.sum()
    extra = [
        ("Sum of refund column exactly as exported (incl. duplicates, paise as rupees) - what a naive sum gives", raw_sum),
        ("Clean total refunds Jan25-Jun26 (Rs)", total),
        ("  = sum of monthly-by-reason table", monthly_reason.loc["TOTAL", "Total"]),
        ("  = sum of monthly-by-agent table", monthly_agent.loc["TOTAL", "Total"]),
        ("Refund tickets", len(r)),
        ("Refund tickets still open/pending (money raised, ticket not closed)", int(r.status.isin(["open", "pending"]).sum())),
        ("  ...their refund value (Rs)", r[r.status.isin(["open", "pending"])].amt.sum()),
    ]
    rec = pd.concat([rec, pd.DataFrame(extra, columns=["check", "value"])], ignore_index=True)
    assert abs(monthly_reason.loc["TOTAL", "Total"] - total) < 0.5
    assert abs(monthly_agent.loc["TOTAL", "Total"] - total) < 0.5

    dl = r[r.double_payout].sort_values("created_dt")[["ticket_id", "created_at", "month", "agent_id", "agent_name", "agent_team", "product_sku",
                                                       "amt", "replacement_cost", "double_payout_basis", "refund_reason_code", "agent_notes"]]
    reclass = r[r.refund_reason_code == "GW-OTHER"][["ticket_id", "month", "agent_id", "amt", "true_reason", "reason_conf", "reason_source", "agent_notes"]]
    return dict(monthly_by_reason=monthly_reason, monthly_by_reason_as_coded=monthly_reason_as_coded, monthly_by_agent=monthly_agent,
                agent_scorecard=sc, team_summary=team, quarterly=quarterly, reconciliation=rec, double_payouts=dl, gw_other_reclass=reclass,
                reason_model_per_class=metrics["per_class"]), r
