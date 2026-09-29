# Vireo refund summary tool

Monthly refunds by reason code and by agent, reconciled to the ticket export, plus a list of
refund-and-replacement double payouts.

## Run (clean machine, Python 3.10+)
```
pip install -r requirements.txt
# put the pack's CSVs (tickets, products, agents, ...) in ./data
python run.py --data data --out out
python tests/test_pipeline.py        # or: python -m pytest tests
```
Takes a few seconds. No API keys, no network, no paid calls.

## Output: `out/vireo_refund_summary.xlsx`
| Sheet | What it is |
|---|---|
| reconciliation | raw export -> clean total, duplicates removed, checks that must be 0, tie-outs |
| monthly_by_reason | Rs by *recovered* reason x month (GW-OTHER split back into real reasons) |
| monthly_by_reason_as_coded | same, using the agent's dropdown code as exported |
| monthly_by_agent | Rs by agent x month, total row reconciles to the same number |
| agent_scorecard | per agent: refunds per 100 tickets handled, double payouts, unresolved reasons. Tier 2 flagged |
| team_summary, quarterly | team shares; quarterly totals with and without Pulse 2 |
| double_payouts | every refund+replacement ticket with the evidence text |
| gw_other_reclass | every GW-OTHER ticket with the reason we assigned, confidence and source |
| reason_model_per_class | out-of-fold precision/recall of the reason model |

`out/refund_tickets_enriched.csv` = one row per refund ticket with all derived fields.

## How it works
1. `refunds/clean.py` - drops the 638 legacy re-import duplicates (keeps the helpdesk copy), divides legacy_fd money by 100 (paise).
2. `refunds/classify.py` - (a) reason model: TF-IDF + logistic regression trained on tickets where the agent
   picked a non-default code, applied to GW-OTHER tickets; below 0.60 confidence or with an empty note it stays
   `GW-OTHER (unresolved)`; (b) double-payout detector: `replacement_issued = Y` OR note says both were given, minus offered/declined wording.
3. `refunds/report.py` - pivots, scorecards, reconciliation asserts (monthly tables must equal the clean total).

Read `DECISIONS.md` for every judgment call and `SUBMISSION_ANSWERS.md` for known weaknesses.
