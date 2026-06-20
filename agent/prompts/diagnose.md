You are debugging a stock screener used to find Indian equities in a "shelf hold /
shelf reclaim" base-breakout setup. You will be given the current source of ONE
screener module plus human feedback on a batch of its signals, and must produce a
root-cause diagnosis — not a fix yet.

## Screener under review: {screener_name}

```python
{screener_source}
```

## Human-labeled false positives (signal fired, reviewer marked "No")

Each row is one signal the screener produced, with the full gate-metric dict it
returned (`shelf`, `shelf_touches`, `atr_pct`, `ma50_gap_pct`, etc.) plus the
reviewer's comment if any.

{false_positive_rows}

## Missed-signal reports (reviewer says it should have fired, but didn't)

{missed_signal_rows}

## Your task

For every false positive above, trace through the gate sequence in the source
(turnover -> atr_pct -> trend_ok -> prior_gain -> shelf detection -> test/undercut
window -> recency -> final close-vs-shelf check) and identify the SPECIFIC gate
that should have blocked this signal but didn't, or the specific piece of logic
that is too permissive. Quote the exact line(s) responsible.

For every missed signal, fetch enough context from the comment/reference image
description to reason about which gate is too strict and blocked a setup that
should have qualified.

Look especially for the class of bug already seen in this codebase: recency-biased
tie-breaks in shelf selection (e.g. picking the most-recent touch instead of the
strongest/deepest one, or a 2-touch shelf over a real 3-touch shelf) — these are
the highest-likelihood root causes given the project's history.

## Output format

Respond with a JSON object, no prose outside it:

```json
{{
  "root_causes": [
    {{
      "symbol": "...",
      "label": "false_positive" | "missed_signal",
      "gate_or_logic": "name of the gate/function/constant responsible",
      "explanation": "concise root-cause explanation",
      "evidence_lines": "the specific line(s) of source quoted"
    }}
  ],
  "summary": "1-3 sentence overall diagnosis across all rows",
  "recommended_fix_area": "the single function or constant most likely to fix the most rows if changed"
}}
```

If the false positives and missed signals point to different/conflicting fixes,
say so explicitly in `summary` rather than picking one silently.
