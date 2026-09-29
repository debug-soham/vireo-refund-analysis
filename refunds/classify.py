"""Recover the real refund reason where the agent left the dropdown default (GW-OTHER),
and detect refund+replacement double payouts hidden in free-text notes.

Reason model: TF-IDF + logistic regression trained on tickets where the agent chose a
NON-default code (assumed mostly right), applied to GW-OTHER tickets. Low-confidence
or uninformative notes stay 'GW-OTHER (unresolved)'. No paid API calls.
"""
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

CONF_MIN = 0.60
GOODWILL_CAP = 500  # policy s5
UNRESOLVED = "GW-OTHER (unresolved)"
GOODWILL_STATED = "GW-OTHER (goodwill stated)"

_sig = re.compile(r"(//|~|--|-)\s?[a-z]{2,8}\s*$")


def _norm(s):
    s = str(s).lower().strip()
    s = re.sub(r"\(sop [\d\.]+\)", "", s)
    s = _sig.sub("", s.strip())
    return re.sub(r"rs\.? ?[\d,]+|\d+", "#", s)


def _text(df):
    return df.agent_notes.fillna("").map(_norm) + " || " + df.customer_message.fillna("").map(lambda s: _norm(s)[:300])


def _uninformative(notes):
    n = notes.fillna("").map(_norm).str.replace(r"[^a-z ]", "", regex=True).str.strip()
    return n.str.len() < 12


GOODWILL_RE = re.compile(r"goodwill|one-time gesture|as a gesture", re.I)

BOTH_POS = re.compile(
    r"(\bboth\b|\brfnd\b\s*\+\s*rplc|refund\s*\+\s*(rplc|replacement)|refund (and|&) (rplc|replacement)|"
    r"rfnd (and|&) rplc|also raised (a )?(replacement|rplc)|and also raised|"
    r"replacement unit also|new unit also|unit also shipped|and also shipped|"
    r"new unit (sent )?(and|&) released|released the money as well|money back to source (and|&) a fresh|"
    r"credited full amount, new unit|(refund|rfnd) done (and|&) new unit|"
    r"(refund|rfnd) issued (and|&) (rplc|replacement)|"
    r"amount returned to card; fresh unit|"
    r"amount back, also sending a new|processed fr (and|&) raised rma|fresh unit shipped|reversed the payment (and|&) dispatched a new one|new piece dispatched)", re.I)
BOTH_NEG = re.compile(
    r"(offered|declined|opted|preferred|rejected|not applicable|did not want|out of stock|explained policy|rfnd only|refund only|"
    r"asked for rplc \+ rfnd|asked for rplc and rfnd)", re.I)


def detect_double_payout(df):
    """Returns (flag_says_Y, note_says_both) boolean Series."""
    note = df.agent_notes.fillna("")
    text_hit = note.map(lambda s: bool(BOTH_POS.search(s)) and not bool(BOTH_NEG.search(s)))
    return df.replacement_issued.eq("Y"), text_hit


def classify_reasons(refunds):
    r = refunds.copy()
    r["txt"] = _text(r)
    lab = r[r.refund_reason_code != "GW-OTHER"]
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(lab.txt)
    y = lab.refund_reason_code.values
    clf = LogisticRegression(max_iter=2000, C=5)

    cv = StratifiedKFold(5, shuffle=True, random_state=0)
    oof_p = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")
    classes = np.array(sorted(set(y)))
    oof = classes[oof_p.argmax(1)]
    covered = oof_p.max(1) >= CONF_MIN
    metrics = {
        "n_labelled": int(len(y)),
        "oof_accuracy_all": float((oof == y).mean()),
        "oof_coverage_at_threshold": float(covered.mean()),
        "oof_accuracy_when_confident": float((oof[covered] == y[covered]).mean()),
        "per_class": pd.DataFrame({
            "code": classes,
            "precision": [((oof == c) & (y == c)).sum() / max((oof == c).sum(), 1) for c in classes],
            "recall": [((oof == c) & (y == c)).sum() / max((y == c).sum(), 1) for c in classes],
            "support": [int((y == c).sum()) for c in classes]}),
    }

    clf.fit(X, y)
    r["true_reason"] = r.refund_reason_code
    r["reason_conf"] = np.nan
    r["reason_source"] = "agent code"
    gw = r.refund_reason_code == "GW-OTHER"
    p = clf.predict_proba(vec.transform(r.loc[gw, "txt"]))
    pred, pc = clf.classes_[p.argmax(1)], p.max(1)
    unin = _uninformative(r.loc[gw, "agent_notes"]).values
    good = r.loc[gw, "agent_notes"].fillna("").str.contains(GOODWILL_RE).values
    ok = (pc >= CONF_MIN) & ~unin
    out = np.where(ok, pred, UNRESOLVED)
    src = np.where(ok, "reclassified from notes", "unresolved")
    out = np.where(good, GOODWILL_STATED, out)
    src = np.where(good, "note says goodwill", src)
    r.loc[gw, "true_reason"] = out
    r.loc[gw, "reason_conf"] = pc
    r.loc[gw, "reason_source"] = src
    return r.drop(columns=["txt"]), metrics
