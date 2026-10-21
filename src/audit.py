"""Audit a credit approval model for fairness, try two mitigations, and write a model card.

The data is synthetic. Two applicant groups (A and B) have the same underlying ability
to repay, but group B was historically pushed toward certain postal regions and lower
incomes. The model never sees the group directly, yet it can still treat groups
differently through *proxy* features such as region. That is the core lesson here.
"""
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
RNG = np.random.default_rng(11)
THRESHOLD = 0.5  # approve if predicted probability of repaying is at least this


def make_data(n=10000):
    group = np.where(RNG.random(n) < 0.35, "B", "A")
    is_b = group == "B"
    region_r = (RNG.random(n) < np.where(is_b, 0.7, 0.2)).astype(int)  # proxy for group
    income = RNG.lognormal(np.where(is_b, 10.9, 11.1), 0.35)            # historical gap
    history_years = RNG.integers(0, 25, n)
    debt_ratio = RNG.beta(2, 5, n)
    prior_defaults = RNG.poisson(0.3, n)
    # True ability to repay does NOT depend on group or region.
    z = (1.2 * (np.log(income) - 11) + 0.08 * history_years - 3.0 * debt_ratio
         - 0.9 * prior_defaults + 1.0)
    repaid = (RNG.random(n) < 1 / (1 + np.exp(-z))).astype(int)
    # Historical labels carry bias: some group-B repayers were recorded as defaults.
    biased = repaid.copy()
    flip = is_b & (repaid == 1) & (RNG.random(n) < 0.12)
    biased[flip] = 0
    return pd.DataFrame({"group": group, "region_r": region_r, "income": income,
                         "history_years": history_years, "debt_ratio": debt_ratio,
                         "prior_defaults": prior_defaults, "repaid_true": repaid,
                         "repaid_recorded": biased})


def fit(train, features):
    scaler = StandardScaler().fit(train[features])
    model = LogisticRegression(max_iter=1000).fit(scaler.transform(train[features]),
                                                  train["repaid_recorded"])
    return lambda df: model.predict_proba(scaler.transform(df[features]))[:, 1]


def group_metrics(df, approved):
    rows = []
    for g in ["A", "B"]:
        m = df["group"] == g
        good = m & (df["repaid_true"] == 1)
        bad = m & (df["repaid_true"] == 0)
        rows.append({"group": g,
                     "approval_rate": approved[m].mean(),
                     "true_positive_rate": approved[good].mean(),   # good payers approved
                     "false_positive_rate": approved[bad].mean()})  # defaulters approved
    t = pd.DataFrame(rows).set_index("group")
    return t, {
        "demographic_parity_diff": t.loc["A", "approval_rate"] - t.loc["B", "approval_rate"],
        "disparate_impact_ratio": t.loc["B", "approval_rate"] / t.loc["A", "approval_rate"],
        "equal_opportunity_diff": t.loc["A", "true_positive_rate"] - t.loc["B", "true_positive_rate"],
        "accuracy": (approved == df["repaid_true"]).mean(),
    }


def equalize_tpr_thresholds(df, scores, target_tpr):
    """Post-processing mitigation: pick a threshold per group so good payers are approved
    at the same rate in both groups. Uses true outcomes on a validation set."""
    thresholds = {}
    for g in ["A", "B"]:
        good = (df["group"] == g) & (df["repaid_true"] == 1)
        thresholds[g] = np.quantile(scores[good], 1 - target_tpr)
    return thresholds


def main():
    data = make_data()
    train, rest = train_test_split(data, test_size=0.4, random_state=0, stratify=data["group"])
    val, test = train_test_split(rest, test_size=0.5, random_state=0, stratify=rest["group"])
    all_feats = ["region_r", "income", "history_years", "debt_ratio", "prior_defaults"]
    no_proxy = [f for f in all_feats if f != "region_r"]

    results, tables = {}, {}
    s_base = fit(train, all_feats)(test)
    tables["1. Baseline (uses region proxy)"], results["1. Baseline (uses region proxy)"] = \
        group_metrics(test, (s_base >= THRESHOLD).astype(int))

    model_np = fit(train, no_proxy)
    s_np = model_np(test)
    tables["2. Proxy removed"], results["2. Proxy removed"] = \
        group_metrics(test, (s_np >= THRESHOLD).astype(int))

    base_tpr = tables["2. Proxy removed"]["true_positive_rate"]["A"]
    th = equalize_tpr_thresholds(val, model_np(val), base_tpr)  # tune on validation, not test
    approved = np.where(test["group"] == "A", s_np >= th["A"], s_np >= th["B"]).astype(int)
    tables["3. Proxy removed + equal-opportunity thresholds"], \
        results["3. Proxy removed + equal-opportunity thresholds"] = group_metrics(test, approved)

    summary = pd.DataFrame(results).T.round(3)
    print(summary.to_string())

    lines = ["# Fairness audit results\n", "## Summary\n", summary.to_markdown(), "\n",
             "Rules of thumb: demographic parity and equal opportunity differences near 0 are fairer; "
             "a disparate impact ratio below 0.8 is a common red flag (the \"four-fifths rule\").\n"]
    for name, t in tables.items():
        lines += [f"## {name}\n", t.round(3).to_markdown(), "\n"]
    (ROOT / "results.md").write_text("\n".join(lines))

    card = (ROOT / "model_card_template.md").read_text()
    card = card.replace("{{DATE}}", str(date.today())).replace("{{SUMMARY_TABLE}}", summary.to_markdown())
    (ROOT / "model_card.md").write_text(card)
    print("\nWrote results.md and model_card.md")


if __name__ == "__main__":
    main()
