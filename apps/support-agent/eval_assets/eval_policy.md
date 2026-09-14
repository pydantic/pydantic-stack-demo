# Eval policy

One page for the three decisions that keep an eval loop honest.

## Annotation (human labels)

Reviewers label runs in Logfire (Evals -> Human review) with: pass / neutral / fail,
the failure mode, the expected output, a comment, and whether it is a regression
candidate. Human labels are the calibration set for the LLM judge: when judge and
humans disagree on more than a handful of cases, fix the rubric, not the humans.

## Sampling (online)

- Deterministic checks (`NoSensitiveDataLeaked`, `ResolutionShape`) run on 100% of traffic.
  They are free.
- The `support_resolution_feasible` judge samples 10% of runs with at most three
  concurrent judge calls (`OnlineEvaluator(sample_rate=0.1, max_concurrency=3)`).
- Raise the rate for high-risk categories; `support-agent-traffic --judge-all` forces
  100% for a demo without changing the deployed default.
- Failed results go to human review first.

## Regression cases (offline)

A production trace becomes a case in `dataset.jsonl` when:

1. a human reviewer confirmed the behaviour is wrong,
2. the failure mode is named (`failure_mode`),
3. the expected behaviour is written down (`expected`, `expects_escalation`),
4. it reproduces offline from the `input` alone.

Keep the case names stable: Logfire groups experiment history by case name.
