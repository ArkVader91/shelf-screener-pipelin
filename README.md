# Shelf-Screener Feedback-Loop Pipeline

Automates the three shelf-hold / shelf-reclaim screeners end to end: scheduled
scan -> chart generation -> human review -> precision gate -> LangGraph debug
agent that patches the screener code and opens a PR, until precision per
batch reaches 80%. Full design rationale is in `screener_pipeline_design.md`
in the parent `Trading` folder; this README covers what's actually here and
how to run it.

This is additive to, and does not touch, the existing local automation
(`run_all_screeners.py`, `bot_listener.py`, Task Scheduler, Telegram) that
already runs daily off Kite/yfinance. Everything in this repo runs in GitHub
Actions, on yfinance only -- no Kite session, no 2FA, nothing to refresh.

## Layout

```
screeners/            copies of the 3 screener modules (HOLD/FLEX/STRICT)
pipeline/
  run_batch.py         runs all 3 screeners off one shared yfinance download
  chart_gen.py         candlestick + 20/50/200 DMA + shelf annotations
  compute_accuracy.py  precision gate from a reviewed labels.csv
review_batches/<date>/
  results.json         full gate-metric dict per signal
  labels.csv           human fills in human_label (Yes/No), reviewer, comment
  charts/<SYM>_<SCREENER>.png
  accuracy.json         written by compute_accuracy.py
missed_signals/<date>/
  notes.csv             false negatives a human noticed (see TEMPLATE_notes.csv)
agent/
  debug_agent.py        LangGraph graph: diagnose -> patch -> validate -> PR
  tools.py              feedback loading, regression check, git/PR plumbing
  prompts/               diagnose.md, propose_patch.md
.github/workflows/
  daily_scan.yml         cron (weekday 18:30 IST) -> run_batch + chart_gen
  debug_on_feedback.yml  on push to labels.csv/notes.csv -> accuracy + agent
```

**Note on `.github/workflows/`:** the two workflow YAML files are NOT in this
commit -- the GitHub App used to push this repo doesn't have the separate
`workflows` permission GitHub requires for writing to that path. They need to
be added manually once (see the setup note that comes with them).

## One-time setup

1. Push this repo to GitHub (already done if you're reading this from there).
2. Add the two workflow files under `.github/workflows/` (`daily_scan.yml`,
   `debug_on_feedback.yml`) via the GitHub web UI -- Add file -> Create new
   file -- since they couldn't be pushed automatically (see note above).
3. Repo secret **`ANTHROPIC_API_KEY`** -- only needed for `debug_on_feedback.yml`
   (the daily scan itself makes zero LLM calls). Settings -> Secrets and
   variables -> Actions -> New repository secret.
4. Actions need to be enabled for the repo (Settings -> Actions -> General ->
   Allow all actions). `daily_scan.yml`'s commit-back step and
   `debug_on_feedback.yml`'s PR-opening step both need the default
   `GITHUB_TOKEN` to have **write** permission: Settings -> Actions -> General
   -> Workflow permissions -> "Read and write permissions".
5. Optional: run `daily_scan.yml` once manually (Actions tab -> Daily
   shelf-screener scan -> Run workflow) with a small `max_symbols` (e.g. 50)
   before relying on the cron schedule, to sanity-check run time on a
   GitHub-hosted runner against the full ~650-symbol universe (this hasn't
   been benchmarked -- see "Known limitations" below).

## Day-to-day flow

1. **Scan runs automatically** weekday evenings (or trigger manually). It
   commits `review_batches/<date>/results.json`, `labels.csv`, and `charts/`.
2. **You review**: open `review_batches/<date>/charts/`, look at each PNG
   alongside `labels.csv`, fill in `human_label` (`Yes`/`No`), your name in
   `reviewer`, and any `comment`. Commit and push `labels.csv` (directly to
   `main`, or via a quick PR -- either way, push triggers the next step).
3. **Missed a setup?** Copy `missed_signals/TEMPLATE_notes.csv` to
   `missed_signals/<date>/notes.csv`, fill in a row per stock you think should
   have fired, push it. This is tracked separately from `labels.csv` since
   it's a false-negative signal, not a false-positive judgment on what fired.
4. **Pushing either file** triggers `debug_on_feedback.yml`: it computes
   precision via `compute_accuracy.py`, and if any screener is below 80% (or
   has a missed-signal report, which triggers the agent regardless of
   precision), runs the debug agent for that screener.
5. **Debug agent opens a PR** with its diagnosis, the patch, and a regression
   check against previously-confirmed signals. **It never merges its own
   PRs** -- review the diff and merge by hand.

## Known limitations / deliberate tradeoffs

Flagging these explicitly rather than letting them surface as surprises:

- **Patch strategy is full-file replacement, not a diff.** The design doc
  suggested preferring small diffs; in practice, asking the LLM to return a
  complete corrected file (while instructing it to change as little as
  possible) is far more reliable to apply mechanically than parsing and
  applying an LLM-generated unified diff without a test harness for the
  diffing code itself. The tradeoff is a noisier PR diff if the model touches
  unrelated whitespace -- worth re-checking PRs for unintended changes.
- **`debug_agent.py` is unverified end-to-end.** `langchain-anthropic` and
  `langgraph` aren't installable in the sandbox this was built in (no PyPI
  access), so it was validated with `py_compile`, prompt-template formatting,
  and reading the LangGraph/LangChain APIs carefully -- not by actually
  running the graph. Treat the first real run in CI as the real test; check
  the Action's logs and the resulting PR closely.
- **Missed-signal auto-triggering only looks at the most recent
  `missed_signals/<date>/` batch** when deciding which screeners to debug
  (`debug_agent.py --screener` lets you target one manually if you need to
  reach further back).
- **Universe size vs. CI run time hasn't been benchmarked** (open question
  from the design doc) -- do a `max_symbols`-limited manual run first.
- **No auto-merge.** Every agent PR needs a human to read the diagnosis and
  merge. This is intentional, not a gap -- see the design doc's risk section
  on runaway auto-edit loops.

## Local testing

`pipeline/run_batch.py`, `chart_gen.py`, and `compute_accuracy.py` were each
smoke-tested locally against synthetic OHLCV data and synthetic `labels.csv`
batches before being committed (monkeypatched `yfinance.download`, so no
network calls). `agent/debug_agent.py` and `agent/tools.py` were checked with
`python -m py_compile` and manual prompt-template rendering only, for the
reasons above.

To re-run a scan locally against a small slice of the universe (needs real
network access to yfinance and niftyindices.com, which this build sandbox
didn't have):

```
pip install -r requirements.txt
python pipeline/run_batch.py --max-symbols 20
python pipeline/chart_gen.py
python pipeline/compute_accuracy.py
```
