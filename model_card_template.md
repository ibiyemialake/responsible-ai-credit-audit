# Model Card: Synthetic Credit Approval Model

*Generated {{DATE}}. All data is synthetic and for learning only.*

## Intended use
Ranks loan applicants by predicted likelihood of repayment to support, not replace, a human
credit decision. Not intended for fully automated decisions.

## Data
10,000 synthetic applicants. Features: income, credit history length, debt ratio, prior
defaults, and a postal region indicator. The applicant's group is **not** a model input.
Historical labels contain known bias: some group B applicants who repaid were recorded as
defaults.

## Fairness evaluation
{{SUMMARY_TABLE}}

## Known risks and limitations
- **Proxy discrimination:** region is correlated with group membership, so including it
  lets the model treat groups differently without ever seeing group directly.
- **Label bias:** the model learns from biased historical outcomes, which can understate
  group B's true creditworthiness.
- **Mitigation trade-offs:** group-specific thresholds improve equal opportunity but require
  using group membership at decision time, which may raise legal and policy questions.

## Recommended controls
- Remove or justify every feature that acts as a proxy for a protected attribute.
- Monitor approval rates and true positive rates by group after deployment.
- Re-audit after every retraining and whenever the applicant population shifts.
- Keep a human reviewer in the loop for declined applications near the threshold.

## Owner and review
Owner: *your name* · Reviewers: *risk, compliance, legal* · Next review: *date*
