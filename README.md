# Responsible AI Audit of a Credit Approval Model

A credit model can treat groups unfairly even when it never sees a protected attribute.
This project builds a simple loan approval model, audits it for fairness, tests two
mitigations, and generates a **model card**: the kind of documentation AI governance
teams review before a model goes live.

## The scenario

10,000 synthetic loan applicants in two groups, A and B, with the **same underlying
ability to repay**. Two realistic sources of bias are built in:

- **A proxy feature.** Group B applicants were historically concentrated in certain postal
  regions, so "region" quietly stands in for group membership.
- **Biased labels.** Some group B applicants who repaid were recorded as defaults in the
  historical data the model learns from.

The model never receives the group as an input, yet it still disadvantages group B.

## What it measures

| Metric | Question it answers | Fairer when |
|---|---|---|
| Demographic parity difference | Are groups approved at similar rates? | Close to 0 |
| Disparate impact ratio | Group B's approval rate ÷ group A's | Above 0.8 (four-fifths rule) |
| Equal opportunity difference | Are *good payers* in each group approved equally? | Close to 0 |

## Results (from one run)

| Approach | Parity diff | Impact ratio | Equal opp. diff | Accuracy |
|---|---|---|---|---|
| 1. Baseline (uses region proxy) | 0.121 | 0.854 | 0.081 | 0.714 |
| 2. Proxy removed | 0.063 | 0.923 | 0.030 | 0.720 |
| 3. Proxy removed + equal-opportunity thresholds | -0.002 | 1.003 | -0.010 | 0.719 |

**Lessons learned**

- **Removing the proxy halved the unfairness** with no loss in accuracy.
- **Group-specific thresholds closed the remaining gap**, but they require using group
  membership at decision time. That can raise legal and policy questions, which is why
  mitigation is a governance decision, not just a technical one.
- **Fairness and accuracy aren't always in conflict.** Here, the fairest model was as
  accurate as the baseline.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/audit.py        # writes results.md and model_card.md
```

## Stretch challenges

1. **Use a fairness library.** Recompute the metrics with Fairlearn or IBM AIF360 and check
   they match.
2. **Try in-processing mitigation.** Train with Fairlearn's `ExponentiatedGradient` and
   compare it with the threshold approach.
3. **Explain decisions.** Add SHAP values and write an "adverse action" reason for each
   declined applicant.
4. **Monitor drift.** Simulate a shift in the applicant population and show how fairness
   metrics change after deployment.
5. **Audit a generative AI system.** Design a review checklist for an LLM chatbot covering
   accuracy, bias, privacy, and harmful outputs.

## Skills demonstrated

AI governance · fairness metrics · proxy and label-bias detection · bias mitigation ·
model documentation (model cards) · risk controls · Python, pandas, scikit-learn
