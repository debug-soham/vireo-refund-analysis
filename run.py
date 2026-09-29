import argparse, os
import pandas as pd
from refunds.clean import load_tickets, clean_tickets
from refunds.classify import classify_reasons, detect_double_payout
from refunds.report import build_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="out")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    raw = load_tickets(os.path.join(a.data, "tickets.csv"))
    clean, log = clean_tickets(raw)
    refunds = clean[clean.amt.notna()].copy()
    flag, txt = detect_double_payout(refunds)
    refunds["_flag"], refunds["_txt"] = flag, txt
    refunds, metrics = classify_reasons(refunds)
    frames, full = build_frames(clean, refunds, pd.read_csv(os.path.join(a.data, "products.csv")),
                                pd.read_csv(os.path.join(a.data, "agents.csv")), metrics, log)
    path = os.path.join(a.out, "vireo_refund_summary.xlsx")
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        for name, df in frames.items():
            df.to_excel(xw, sheet_name=name[:31], index=name in ("monthly_by_reason", "monthly_by_reason_as_coded", "monthly_by_agent", "team_summary", "quarterly"))
            ws = xw.sheets[name[:31]]
            ws.freeze_panes = "B2"
            for col in ws.columns:
                ws.column_dimensions[col[0].column_letter].width = min(60, max(10, max(len(str(c.value or "")) for c in col[:50]) + 2))
    full.drop(columns=["_flag", "_txt"]).to_csv(os.path.join(a.out, "refund_tickets_enriched.csv"), index=False)
    print("clean total Rs", round(frames["reconciliation"].set_index("check").loc["Clean total refunds Jan25-Jun26 (Rs)", "value"]))
    print("wrote", path)
    print({k: round(v, 3) for k, v in metrics.items() if k != "per_class"})


if __name__ == "__main__":
    main()
