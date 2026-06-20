"""
debug_agent.py — stage 5 of the shelf-screener feedback loop (design doc section 3.5).

Triggered by .github/workflows/debug_on_feedback.yml when a batch's precision
drops below the 80% gate (compute_accuracy.py) or on a push to labels.csv /
missed_signals/**. Builds a small LangGraph graph per failing screener:

  load_feedback_and_code -> diagnose -> propose_patch -> apply_patch -> validate
        ^                                                                  |
        |-------------------- (regression found, attempt < max) -----------|
                                                                            |
                                                          (no regression) open_pr
                                                          (attempts exhausted) END

The agent only ever opens a PR (agent/tools.py: create_branch/commit_and_push/
open_pr) — merging stays a manual step, per the design doc's "no unsupervised
compounding changes" requirement.

This module needs `langchain-anthropic` and `langgraph` (declared in
requirements.txt) and an `ANTHROPIC_API_KEY` repo secret. It is not runnable in
a network-isolated sandbox; it was validated with `python -m py_compile` only —
real execution happens in GitHub Actions.

Usage:
  python agent/debug_agent.py --date 2026-06-19
  python agent/debug_agent.py --date 2026-06-19 --screener FLEX --max-retries 2
"""

import argparse
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent import tools  # noqa: E402

PROMPTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts")
GATE_THRESHOLD = 0.80
DEFAULT_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-6")


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

from typing import Any, Dict, List, Optional, TypedDict  # noqa: E402


class DebugState(TypedDict):
    screener_name: str
    repo_root: str
    false_positives: List[Dict[str, Any]]
    missed_signals: List[Dict[str, Any]]
    screener_source: str
    diagnosis: Optional[Dict[str, Any]]
    patch: Optional[Dict[str, Any]]
    validation: Optional[Dict[str, Any]]
    attempt: int
    max_attempts: int
    branch_name: str
    pr_url: Optional[str]
    status: str
    error: Optional[str]


# ---------------------------------------------------------------------------
# Prompt + LLM helpers
# ---------------------------------------------------------------------------

def _load_prompt(name):
    with open(os.path.join(PROMPTS_DIR, name)) as f:
        return f.read()


def _extract_json(text):
    """LLMs reliably wrap JSON in ```json fences despite instructions not to.
    Strip fences first, then fall back to grabbing the largest {...} block."""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    candidate = fence.group(1).strip() if fence else text
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end == -1:
            raise
        return json.loads(candidate[start:end + 1])


def _get_llm():
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(model=DEFAULT_MODEL, temperature=0, max_tokens=8000)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def load_feedback_and_code(state: DebugState) -> dict:
    name, repo_root = state["screener_name"], state["repo_root"]
    return {
        "false_positives": tools.load_labeled_feedback(name, repo_root),
        "missed_signals": tools.load_missed_signals(name, repo_root),
        "screener_source": tools.load_screener_source(name, repo_root),
    }


def diagnose(state: DebugState) -> dict:
    prompt = _load_prompt("diagnose.md").format(
        screener_name=state["screener_name"],
        screener_source=state["screener_source"],
        false_positive_rows=tools.render_rows_markdown(state["false_positives"]),
        missed_signal_rows=tools.render_rows_markdown(state["missed_signals"]),
    )
    response = _get_llm().invoke(prompt)
    diagnosis = _extract_json(response.content)
    return {"diagnosis": diagnosis}


def propose_patch(state: DebugState) -> dict:
    attempt = state["attempt"]
    retry_context = ""
    if state.get("validation") and not state["validation"]["ok"]:
        retry_context = (
            "The previous patch caused a regression on previously-confirmed "
            f"signals: {json.dumps(state['validation']['regressions'])}. "
            "Address this without reintroducing the original false positives."
        )
    prompt = _load_prompt("propose_patch.md").format(
        screener_name=state["screener_name"],
        diagnosis_json=json.dumps(state["diagnosis"], indent=2),
        screener_source=state["screener_source"],
        attempt_number=attempt,
        max_attempts=state["max_attempts"],
        retry_context=retry_context,
        screener_filename=tools.SCREENER_FILES[state["screener_name"]],
    )
    response = _get_llm().invoke(prompt)
    patch = _extract_json(response.content)
    if "new_source" not in patch or not patch["new_source"].strip():
        raise ValueError("propose_patch: LLM response missing non-empty 'new_source'")
    return {"patch": patch}


def apply_patch(state: DebugState) -> dict:
    """Writes the candidate patch into the working tree on the debug branch
    (created once, on the first attempt) so validate() can import it from disk
    via the real module path -- but does NOT commit yet. Commit only happens
    in open_pr(), once validation has actually passed.
    """
    branch_name = state.get("branch_name") or (
        f"debug-agent/{state['screener_name'].lower()}-"
        f"{os.environ.get('GITHUB_RUN_ID', 'local')}"
    )
    if state["attempt"] == 1:
        try:
            tools.create_branch(branch_name, state["repo_root"])
        except Exception as e:
            return {"branch_name": branch_name, "error": f"create_branch failed: {e}"}
    tools.write_screener_source(state["screener_name"], state["patch"]["new_source"],
                                 state["repo_root"])
    return {"branch_name": branch_name}


def validate(state: DebugState) -> dict:
    result = tools.validate_no_regression(
        state["screener_name"], state["patch"]["new_source"], state["repo_root"]
    )
    return {"validation": result}


def open_pr_node(state: DebugState) -> dict:
    name = state["screener_name"]
    filename = tools.SCREENER_FILES[name]
    rel_path = os.path.join("screeners", filename)
    message = f"debug-agent: patch {name} screener after batch review feedback"
    try:
        tools.commit_and_push(state["branch_name"], message, [rel_path], state["repo_root"])
        body = (
            f"Automated patch proposed by the LangGraph debug agent.\n\n"
            f"**Diagnosis**\n```json\n{json.dumps(state['diagnosis'], indent=2)}\n```\n\n"
            f"**Patch ({state['attempt']} attempt(s))**\n"
            f"{state['patch'].get('explanation', '')}\n\n"
            f"Changed: {state['patch'].get('changed_constants_or_lines', [])}\n\n"
            f"**Regression check** against {state['validation']['checked']} "
            f"historically-confirmed signal(s): no regressions found.\n\n"
            f"_Manual review and merge required — this agent never merges its own PRs._"
        )
        pr_url = tools.open_pr(
            state["branch_name"],
            title=f"[debug-agent] patch {name} screener",
            body=body,
            repo_root=state["repo_root"],
        )
        return {"pr_url": pr_url, "status": "opened_pr"}
    except Exception as e:
        return {"status": "failed", "error": f"open_pr failed: {e}"}


def route_after_validate(state: DebugState) -> str:
    if state.get("error"):
        return "end_failed"
    if state["validation"]["ok"]:
        return "open_pr"
    if state["attempt"] >= state["max_attempts"]:
        return "end_failed"
    return "retry"


def bump_attempt(state: DebugState) -> dict:
    return {"attempt": state["attempt"] + 1}


def mark_failed(state: DebugState) -> dict:
    if state.get("status") == "failed":
        return {}
    reason = state.get("error") or "max retries exhausted without a non-regressing patch"
    return {"status": "failed", "error": reason}


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    from langgraph.graph import StateGraph, END

    g = StateGraph(DebugState)
    g.add_node("load_feedback_and_code", load_feedback_and_code)
    g.add_node("diagnose", diagnose)
    g.add_node("propose_patch", propose_patch)
    g.add_node("apply_patch", apply_patch)
    g.add_node("validate", validate)
    g.add_node("bump_attempt", bump_attempt)
    g.add_node("open_pr", open_pr_node)
    g.add_node("mark_failed", mark_failed)

    g.set_entry_point("load_feedback_and_code")
    g.add_edge("load_feedback_and_code", "diagnose")
    g.add_edge("diagnose", "propose_patch")
    g.add_edge("propose_patch", "apply_patch")
    g.add_edge("apply_patch", "validate")
    g.add_conditional_edges("validate", route_after_validate, {
        "open_pr": "open_pr",
        "retry": "bump_attempt",
        "end_failed": "mark_failed",
    })
    g.add_edge("bump_attempt", "diagnose")
    g.add_edge("open_pr", END)
    g.add_edge("mark_failed", END)
    return g.compile()


# ---------------------------------------------------------------------------
# Driving it from a batch's accuracy.json
# ---------------------------------------------------------------------------

def failing_screeners(batch_dir):
    """Screeners in this batch whose precision is below GATE_THRESHOLD, using
    the same per_screener breakdown compute_accuracy.py already wrote."""
    acc_path = os.path.join(batch_dir, "accuracy.json")
    with open(acc_path) as f:
        acc = json.load(f)
    failing = []
    for name, counts in acc.get("per_screener", {}).items():
        denom = counts["yes"] + counts["no"]
        if denom == 0:
            continue
        precision = counts["yes"] / denom
        if precision < GATE_THRESHOLD:
            failing.append(name)
    return failing


def run_for_screener(screener_name, repo_root, max_attempts=3):
    graph = build_graph()
    initial: DebugState = {
        "screener_name": screener_name,
        "repo_root": repo_root,
        "false_positives": [],
        "missed_signals": [],
        "screener_source": "",
        "diagnosis": None,
        "patch": None,
        "validation": None,
        "attempt": 1,
        "max_attempts": max_attempts,
        "branch_name": "",
        "pr_url": None,
        "status": "running",
        "error": None,
    }
    return graph.invoke(initial, config={"recursion_limit": max_attempts * 6 + 10})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default=None, help="Batch date to read accuracy.json from")
    ap.add_argument("--batch-root", default=os.path.join(REPO_ROOT, "review_batches"))
    ap.add_argument("--screener", default=None, choices=["HOLD", "FLEX", "STRICT"],
                     help="Restrict to one screener instead of auto-detecting failing ones")
    ap.add_argument("--max-retries", type=int, default=3)
    args = ap.parse_args()

    if args.date:
        batch_dir = os.path.join(args.batch_root, args.date)
    else:
        dated = sorted(d for d in os.listdir(args.batch_root)
                        if os.path.isdir(os.path.join(args.batch_root, d)))
        if not dated:
            sys.exit(f"No batches found under {args.batch_root}")
        batch_dir = os.path.join(args.batch_root, dated[-1])

    if args.screener:
        targets = [args.screener]
    else:
        targets = set()
        if os.path.exists(os.path.join(batch_dir, "accuracy.json")):
            targets.update(failing_screeners(batch_dir))
        else:
            print(f"(no accuracy.json in {batch_dir} yet -- skipping the precision-gate check)")
        # Missed-signal reports are their own trigger (design doc 3.5), independent
        # of whether the precision gate on existing signals passed -- a screener
        # can be 100% precise on what it fires and still be too strict.
        for name in tools.SCREENER_FILES:
            if tools.load_missed_signals(name, REPO_ROOT, max_batches=1):
                targets.add(name)
        targets = sorted(targets)

    if not targets:
        print("No screeners below the precision gate and no missed-signal reports -- "
              "nothing to debug.")
        return

    exit_code = 0
    for name in targets:
        print(f"\n=== Debug agent: {name} ===")
        final_state = run_for_screener(name, REPO_ROOT, max_attempts=args.max_retries)
        status = final_state.get("status")
        if status == "opened_pr":
            print(f"  PR opened: {final_state.get('pr_url')}")
        else:
            print(f"  FAILED after {final_state.get('attempt')} attempt(s): "
                  f"{final_state.get('error')}")
            exit_code = 1

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
