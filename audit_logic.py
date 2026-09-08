"""Local, descriptive audit helpers. Hashes detect changes, not authenticity."""
from copy import deepcopy
from datetime import datetime, timezone
from statistics import mean

from evaluation_logic import input_signature, is_reviewable


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def freeze_plan(inputs, scenarios, criteria, critical_ids, previous=None, evidence_exists=False):
    payload = {"inputs": deepcopy(inputs), "scenarios": deepcopy(scenarios),
               "criteria": deepcopy(criteria), "critical_scenario_ids": list(critical_ids)}
    return {"version": (previous or {}).get("version", 0) + 1,
            "frozen_at_utc": utc_now(), "signature": input_signature(payload),
            "post_results": bool(evidence_exists), "payload": payload}


def plan_matches(plan, inputs):
    return bool(plan) and plan["payload"]["inputs"] == inputs


def validate_plan(plan, inputs=None):
    if not isinstance(plan, dict) or not isinstance(plan.get("payload"), dict):
        raise ValueError("Invalid plan")
    payload = plan["payload"]
    if plan.get("signature") != input_signature(payload) or type(plan.get("version")) is not int or plan["version"] < 1:
        raise ValueError("Invalid plan signature or version")
    if not isinstance(payload.get("scenarios"), list) or not isinstance(payload.get("criteria"), dict) or not isinstance(payload.get("critical_scenario_ids"), list):
        raise ValueError("Invalid plan fields")
    if inputs is not None and not plan_matches(plan, inputs):
        raise ValueError("Plan does not match run inputs")


def note_reveal(result):
    if not result.get("identity_revealed"):
        result["identity_revealed"] = True
        result["identity_revealed_at_utc"] = utc_now()


def record_review(result, review, reviewer, revealed):
    """Append only changes; never erase earlier revisions or a prior reveal."""
    revealed = bool(revealed or result.get("identity_revealed") or result.get("review_mode") == "identified")
    if revealed:
        note_reveal(result)
    snapshot = {"review": deepcopy(review), "reviewer": reviewer.strip() or None,
                "identity_revealed": revealed}
    history = result.setdefault("review_history", [])
    if not history or history[-1]["snapshot"] != snapshot:
        history.append({"revision": len(history) + 1, "timestamp_utc": utc_now(), "snapshot": snapshot})
    result.update(review)
    result["identity_revealed"] = revealed
    result["reviewer_id"] = snapshot["reviewer"]
    result["review_updated_at_utc"] = history[-1]["timestamp_utc"]


def paired_analysis(runs, critical_ids=()):
    """Pair within run AND trial, never across different experiments."""
    rows = []
    for run in runs:
        comparison = input_signature({"inputs": {k: v for k, v in run["evaluation_inputs"].items() if k != "requested_trial_count"}})[:12]
        groups = {}
        for result in run["individual_trial_results"]:
            groups.setdefault(result["trial_number"], {})[result["configuration_key"]] = result
        for trial, pair in sorted(groups.items()):
            a, b = pair.get("A"), pair.get("B")
            usable = bool(a and b and is_reviewable(a) and is_reviewable(b))
            reviewed = usable and all(x.get("review_status") == "complete" for x in (a, b))
            score = b["total_human_score"] - a["total_human_score"] if reviewed else None
            latency = b["latency_seconds"] - a["latency_seconds"] if usable else None
            tokens = b["total_tokens"] - a["total_tokens"] if usable and all(x.get("total_tokens") is not None for x in (a, b)) else None
            reasons = []
            if score is not None and score < 0:
                reasons.append("lower quality score")
            if latency is not None and latency > 0:
                reasons.append("higher latency")
            if tokens is not None and tokens > 0:
                reasons.append("higher token usage")
            if a and b:
                if is_reviewable(a) and not is_reviewable(b):
                    reasons.append("B output unusable")
                if reviewed and a.get("expected_decision_match") == "Yes" and b.get("expected_decision_match") != "Yes":
                    reasons.append("decision-match regression")
                if b.get("unacceptable_behavior") and not a.get("unacceptable_behavior"):
                    reasons.append("new unacceptable behavior")
            rows.append({"Run": run["run_id"], "Scenario": run["scenario_id"], "Comparison group": comparison, "Trial": trial,
                         "Score B-A": score, "Latency B-A": latency, "Tokens B-A": tokens,
                         "Pair usable": usable, "Regression": "; ".join(reasons),
                         "Critical regression": run["scenario_id"] in critical_ids and bool(reasons)})
    summaries = []
    for sid, group in sorted({(r["Scenario"], r["Comparison group"]) for r in rows}):
        selected = [r for r in rows if r["Scenario"] == sid and r["Comparison group"] == group]
        summary = {"Scenario": sid, "Comparison group": group, "Pairs": len(selected)}
        for metric in ("Score B-A", "Latency B-A", "Tokens B-A"):
            values = [r[metric] for r in selected if r[metric] is not None]
            summary["Mean " + metric] = mean(values) if values else None
            summary[metric + " pair count"] = len(values)
        summaries.append(summary)
    return {"pairs": rows, "scenarios": summaries}
