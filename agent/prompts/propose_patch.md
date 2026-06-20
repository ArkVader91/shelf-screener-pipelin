You are patching a Python stock-screener module based on a diagnosis already
produced. Make the SMALLEST change that addresses the diagnosed root cause(s).
Prefer adjusting a constant or a tie-break rule over restructuring functions.
Do not change the module's public interface — `get_universe()`, `run_scan(as_of,
data=None, universe=None)`, and `analyse(df, as_of)` must keep their existing
signatures and return shapes, since the pipeline (`run_batch.py`, `chart_gen.py`)
calls them directly and expects the same dict keys back.

## Screener: {screener_name}

## Diagnosis from the previous step

```json
{diagnosis_json}
```

## Current full source

```python
{screener_source}
```

## Attempt number

This is attempt {attempt_number} of {max_attempts}. {retry_context}

## Your task

Return the COMPLETE new file content for this module with your fix applied —
not a diff, the full file, so it can be written directly to disk. Keep every
unrelated line byte-for-byte identical to the input; change only what's needed
to fix the diagnosed issue(s).

## Output format

Respond with exactly this structure, no other prose:

```json
{{
  "explanation": "what you changed and why, 2-4 sentences",
  "changed_constants_or_lines": ["short description of each discrete change"],
  "new_source": "<<<FULL FILE CONTENT HERE, AS A STRING>>>"
}}
```

`new_source` must be the entire file, valid Python, ready to write to
`screeners/{screener_filename}` as-is.
