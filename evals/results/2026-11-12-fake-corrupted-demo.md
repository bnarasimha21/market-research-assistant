# Eval results: 2026-11-12, --llm fake

Fake mode: every stage is answered by a scripted FakeLLM built directly from each trio's golden.yaml, so this checks the scoring and plumbing are correct, not a real model's quality. A perfect run scores at or near 1.0 on every metric.

| trio | category_coverage | grounding_rate | fact_recall | no_evidence_precision | duration_s | tokens |
|---|---|---|---|---|---|---|
| trio_docs | 1.00 | 1.00 | 1.00 | 1.00 | 0.055 | 10048 |
| trio_ide | 1.00 | 1.00 | 1.00 | 1.00 | 0.047 | 10173 |
| trio_nl | 1.00 | 1.00 | 1.00 | 1.00 | 0.047 | 10268 |
| trio_pricing (corrupted) | 1.00 | 0.67 | 0.67 | 1.00 | 0.048 | 9374 |
| trio_scale | 1.00 | 1.00 | 1.00 | 1.00 | 0.048 | 9953 |
| trio_support | 1.00 | 1.00 | 1.00 | 1.00 | 0.048 | 9986 |

## Corruption demonstration

`--corrupt-trio trio_pricing` mangled that trio's first competitor's first golden quote before running it. Its grounding_rate above (0.67) is lower than a normal run of the same trio, proving the deterministic grounding check (quote_is_grounded) really detects a broken fixture instead of the metric always reading 1.0 by construction.

Average: category_coverage=1.00, grounding_rate=0.94, fact_recall=0.94, no_evidence_precision=1.00
