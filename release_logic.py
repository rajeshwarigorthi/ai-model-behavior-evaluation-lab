"""Offline evidence validation and scenario-level release gates."""
from copy import deepcopy
from math import isfinite
from evaluation_logic import aggregate_review_summary, aggregate_trial_results, input_signature, build_review, SCORE_DIMENSIONS


def prompt_identity(prompt):
    return "sha256:" + input_signature({"prompt": prompt})


def output_status(response):
    if getattr(response, "status", None) != "completed":
        return "incomplete"
    return "success" if (getattr(response, "output_text", "") or "").strip() else "empty"


def import_evidence(document):
    """Validate before importing; recompute summaries rather than trusting exports."""
    if not isinstance(document, dict):
        raise ValueError("Expected a JSON object.")
    if document.get("release_decision") is not None and not isinstance(document["release_decision"], dict):
        raise ValueError("Invalid decision record.")
    runs = document.get("runs", [document])
    if not isinstance(runs, list) or not runs:
        raise ValueError("Expected evaluation JSON or a workspace bundle.")
    validated = {}
    for source in runs:
        run = deepcopy(source)
        for field in ("run_id", "scenario_id", "scenario_title", "scenario_category", "scenario_risk_level", "scenario", "model", "synthetic_data_statement"):
            if not isinstance(run.get(field), str) or not run[field]:
                raise ValueError("Missing run metadata.")
        inputs = run["evaluation_inputs"]
        if type(run["requested_trial_count"]) is not int or run["requested_trial_count"] not in range(1, 6):
            raise ValueError("Unsupported trial count.")
        if not isinstance(run["run_id"], str) or not run["run_id"] or run["run_id"] in validated:
            raise ValueError("Invalid or duplicate run ID.")
        if run["input_signature"] != input_signature(inputs):
            raise ValueError("Input signature does not match saved inputs.")
        for field in ("scenario_id", "model", "requested_trial_count", "reasoning_effort", "max_output_tokens"):
            if run[field] != inputs[field]:
                raise ValueError("Run metadata does not match inputs.")
        if run["scenario"] != inputs["scenario_text"]:
            raise ValueError("Scenario does not match inputs.")
        required = run["required_considerations"]
        if not isinstance(required, list) or any(not isinstance(x, str) for x in required):
            raise ValueError("Invalid reference considerations.")
        if run["expected_decision"] not in ("GO", "NO_GO", "CONDITIONAL_GO"):
            raise ValueError("Invalid reference decision.")
        for field in ("expected_decision", "required_considerations"):
            if field in inputs and inputs[field] != run[field]:
                raise ValueError("Reference labels do not match signed inputs.")
        records = run["individual_trial_results"]
        if run.get("total_expected_api_request_count") != 2 * run["requested_trial_count"]:
            raise ValueError("Incorrect expected request count.")
        if len(records) != 2 * run["requested_trial_count"]:
            raise ValueError("Incomplete trial record set.")
        seen = set()
        for result in records:
            for field in ("configuration_name", "model", "prompt_version", "prompt"):
                if not isinstance(result.get(field), str):
                    raise ValueError("Missing request metadata.")
            if result.get("status") == "failure":
                result["safe_error_message"] = "The recorded request failed."
            if "unacceptable_behavior" in result and type(result["unacceptable_behavior"]) is not bool:
                raise ValueError("Invalid behavior flag.")
            key = result["configuration_key"]
            trial = result["trial_number"]
            if key not in ("A", "B") or type(trial) is not int or not 1 <= trial <= run["requested_trial_count"] or (trial, key) in seen:
                raise ValueError("Invalid or duplicate trial identity.")
            seen.add((trial, key))
            if result["prompt"] != inputs[f"configuration_{key.lower()}_prompt"]:
                raise ValueError("Prompt does not match inputs.")
            if result["model"] != inputs["model"]:
                raise ValueError("Request model does not match inputs.")
            if result["status"] not in ("success", "failure", "empty", "incomplete"):
                raise ValueError("Invalid outcome.")
            if result["execution_position"] not in ("first", "second"):
                raise ValueError("Invalid execution position.")
            if not isinstance(result.get("response"), (str, type(None))):
                raise ValueError("Invalid response.")
            if result["status"] == "success" and not (result.get("response") or "").strip():
                raise ValueError("Successful responses must be nonempty.")
            if not isinstance(result["latency_seconds"], (int, float)) or not isfinite(result["latency_seconds"]) or result["latency_seconds"] < 0:
                raise ValueError("Invalid latency.")
            for field in ("input_tokens", "output_tokens", "total_tokens"):
                value = result.get(field)
                if value is not None and (type(value) is not int or value < 0):
                    raise ValueError("Invalid usage.")
            scores = result.get("human_scores") or dict.fromkeys(SCORE_DIMENSIONS)
            if set(scores) != set(SCORE_DIMENSIONS) or any(v is not None and (type(v) is not int or not 1 <= v <= 5) for v in scores.values()):
                raise ValueError("Invalid scores.")
            result.update(build_review(scores, result.get("expected_decision_match"), result.get("addressed_considerations") or [], required, result.get("addressed_considerations") is not None, result.get("reviewer_notes") or "", result.get("review_status") == "complete"))
        for trial in range(1, run["requested_trial_count"] + 1):
            if {r["execution_position"] for r in records if r["trial_number"] == trial} != {"first", "second"}:
                raise ValueError("Conflicting execution positions.")
        run["aggregate_metrics"] = aggregate_trial_results(records)
        validated[run["run_id"]] = run
    return validated


def release_report(runs, scenario_ids, criteria):
    """Require consistent prompts/settings and full evidence before applying gates."""
    rows, missing = [], []
    signatures = set()
    for run in runs:
        inp = run["evaluation_inputs"]
        signatures.add(input_signature({k: inp[k] for k in ("configuration_a_prompt", "configuration_b_prompt", "model", "reasoning_effort", "max_output_tokens")}))
    if len(signatures) > 1:
        return {"recommendation": "Insufficient evidence", "reasons": ["Selected runs mix prompts or model settings."], "rows": []}
    for sid in scenario_ids:
        selected = [r for r in runs if r["scenario_id"] == sid]
        if not selected:
            missing.append(sid)
            continue
        reference_versions = {input_signature({"text": r["scenario"], "decision": r["expected_decision"], "considerations": r["required_considerations"]}) for r in selected}
        if len(reference_versions) > 1:
            missing.append(sid + " (inconsistent scenario references)")
            continue
        records = [x for r in selected for x in r["individual_trial_results"]]
        quality, performance = aggregate_review_summary(records), aggregate_trial_results(records)
        for key in ("A", "B"):
            q, p = quality[key], performance[key]
            count = sum(x["configuration_key"] == key for x in records)
            complete = q["completed_review_count"]
            ready = count >= criteria["minimum_trials"] and q["incomplete_review_count"] == 0 and complete > 0
            passed = ready and p["failed_trial_count"] / count <= criteria["maximum_failure_rate"] and q["expected_decision_matches"] / complete >= criteria["minimum_match_rate"] and q["mean_human_score"] >= criteria["minimum_score"] and p["mean_latency_seconds"] <= criteria["maximum_latency"] and (p["mean_total_tokens"] is not None and p["mean_total_tokens"] <= criteria["maximum_tokens"]) and not any(x.get("unacceptable_behavior") for x in records if x["configuration_key"] == key)
            rows.append({"Scenario": sid, "Configuration": key, "Requests": count, "Reviewed": complete, "Mean score": q["mean_human_score"], "Mean latency": p["mean_latency_seconds"], "Mean tokens": p["mean_total_tokens"], "Ready": ready, "Pass": bool(passed)})
    if missing or not rows or any(not x["Ready"] for x in rows):
        recommendation = "Insufficient evidence"
    else:
        a = all(x["Pass"] for x in rows if x["Configuration"] == "A")
        b = all(x["Pass"] for x in rows if x["Configuration"] == "B")
        recommendation = "No clear winner" if a and b else "Release B" if b else "Keep A" if a else "Improve and retest"
    return {"recommendation": recommendation, "reasons": ["Missing or inconsistent evidence: " + ", ".join(missing)] if missing else ["Every selected scenario must satisfy the configured gates; averages cannot offset a failed scenario."], "rows": rows}
