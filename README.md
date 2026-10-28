# Auditing a credit model for fairness

## Why I built it

A lot of AI governance happens at review time. Someone hands you a model, some metrics and a model card, and you decide whether it's ready to ship. I wanted to do the work underneath that myself, so I know what good looks like and which questions to ask. It's also how I learn best: build something small, break it on purpose, see what happens.

## What it does

[src/audit.py](src/audit.py) generates 10,000 synthetic loan applicants in two groups with the same real ability to repay, then adds two kinds of bias common in lending data. Group B is concentrated in one postal region, so region becomes a proxy for group. And about 12% of group B applicants who repaid are recorded as defaults.

Because I generated the data, I know the true outcomes. So I can measure what the model actually does to each group, not just how well it matches biased history. That's the part I'm happiest with.

It trains a logistic regression three ways:

| Approach | Parity diff | Impact ratio | Equal opp. diff | Accuracy |
|---|---|---|---|---|
| Baseline (uses region) | 0.121 | 0.854 | 0.081 | 0.714 |
| Region removed | 0.063 | 0.923 | 0.030 | 0.720 |
| Region removed + per-group thresholds | -0.002 | 1.003 | -0.010 | 0.719 |

Parity diff is the gap in approval rates. Impact ratio is B's approval rate over A's (under 0.8 fails the four-fifths rule). Equal opportunity diff is the gap among people who actually repaid.

Dropping region halved the gap and accuracy didn't drop. Per-group thresholds closed the rest, but they use group membership at decision time, which is a legal and policy call, not just a technical one.

## Tech stack

Python, numpy, pandas, scikit-learn and tabulate. I chose logistic regression because it's easy to inspect and close to what credit teams use, so any unfairness comes from the data, not a black box. I wrote the fairness metrics by hand instead of using a library, so every number traces back to a few lines of pandas.

## What's next

Tests. I want unit tests for the metric functions, plus one that checks a full run still reproduces the table above.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/audit.py
```

This prints the summary table and writes `results.md` (per-group breakdowns) and `model_card.md` (filled in from [model_card_template.md](model_card_template.md)) to the repo root.
