"""Session workspace, portable JSON persistence, and PM release decisions."""
import json
from copy import deepcopy
from datetime import datetime, timezone
import streamlit as st
from evaluation_logic import input_signature, SCORE_DIMENSIONS
from release_logic import import_evidence, release_report
from audit_ui import audit_report_ui
from audit_logic import validate_plan


def archive_active():
    run = st.session_state.get("evaluation")
    if run:
        st.session_state.setdefault("saved_runs", {})[run["run_id"]] = deepcopy(run)


def restore_run(run):
    st.session_state["evaluation"] = deepcopy(run)
    inputs = run["evaluation_inputs"]
    for widget, field in (("selected_scenario_id", "scenario_id"), ("scenario_text", "scenario_text"), ("prompt_a", "configuration_a_prompt"), ("prompt_b", "configuration_b_prompt"), ("trial_count", "requested_trial_count")):
        st.session_state[widget] = inputs[field]
    st.session_state["reference_decision"] = run["expected_decision"]
    st.session_state["reference_considerations"] = "\n".join(run["required_considerations"])
    st.session_state["reference_confirmed"] = True
    for result in run["individual_trial_results"]:
        prefix = f"review_{run['run_id']}_{result['trial_number']}_{result['configuration_key'].lower()}"
        for dimension in SCORE_DIMENSIONS:
            st.session_state[f"{prefix}_{dimension.lower().replace(' ', '_')}"] = (result.get("human_scores") or {}).get(dimension) or "Not reviewed"
        for suffix, value in {
            "expected_decision": result.get("expected_decision_match") or "Not reviewed",
            "addressed": result.get("addressed_considerations") or [],
            "considerations_reviewed": result.get("addressed_considerations") is not None,
            "notes": result.get("reviewer_notes") or "",
            "complete": result.get("review_status") == "complete",
            "unacceptable": result.get("unacceptable_behavior", False),
        }.items():
            st.session_state[f"{prefix}_{suffix}"] = value


def workspace_controls(scenario_ids):
    st.session_state.setdefault("saved_runs", {})
    with st.expander("Saved experiments — import and resume"):
        st.caption("Download a workspace below to retain work across browser sessions. Importing or resuming never calls the API. Legacy Runs 001–003 remain historical evidence; resumable reviews require Version 0.4 input signatures.")
        upload = st.file_uploader("Evaluation or workspace JSON", type=["json"])
        if st.button("Import evidence", disabled=upload is None):
            try:
                if upload.size > 10_000_000:
                    raise ValueError("File too large")
                document = json.loads(upload.getvalue())
                imported = import_evidence(document)
                imported_plans = document.get("evaluation_plans", [])
                if not isinstance(imported_plans, list):
                    raise ValueError("Invalid plan list")
                for imported_plan in imported_plans:
                    validate_plan(imported_plan)
                if any(r["scenario_id"] not in scenario_ids for r in imported.values()):
                    raise ValueError("Unknown scenario")
                existing = st.session_state["saved_runs"]
                if any(k in existing and existing[k] != v for k, v in imported.items()):
                    raise ValueError("Conflicting existing run")
                archive_active()
                existing.update(imported)
                plans = st.session_state.setdefault("evaluation_plans", [])
                for imported_plan in imported_plans:
                    if imported_plan not in plans:
                        plans.append(deepcopy(imported_plan))
                if "release_decision" in document:
                    st.session_state["release_decision"] = document["release_decision"]
                st.success("Evidence imported. Select a run to resume.")
            except Exception:
                st.error("Import rejected: check the JSON schema, signature, scenario IDs, and duplicate run IDs. Existing evidence was preserved.")
        saved = st.session_state["saved_runs"]
        if saved:
            chosen = st.selectbox("Saved run", list(saved), format_func=lambda k: f"{saved[k]['scenario_title']} · {k}")
            if st.button("Resume selected review"):
                archive_active()
                restore_run(saved[chosen])


def release_workspace(scenarios):
    archive_active()
    saved = st.session_state["saved_runs"]
    if not saved:
        return
    st.subheader("Cross-scenario release decision")
    st.caption("Select evidence and set gates before making a release decision. Threshold defaults are examples, not validated launch criteria.")
    ids = st.multiselect("Required scenario coverage", [s["id"] for s in scenarios], default=[s["id"] for s in scenarios])
    chosen = st.multiselect("Evidence runs", list(saved), default=list(saved))
    with st.expander("Release criteria", expanded=True):
        criteria = {
            "minimum_trials": st.number_input("Minimum requests per configuration per scenario", 1, 1000, 3),
            "minimum_score": st.number_input("Minimum mean human score (out of 20)", 4, 20, 16),
            "minimum_match_rate": st.slider("Minimum expected-decision match rate", 0.0, 1.0, 1.0),
            "maximum_failure_rate": st.slider("Maximum unusable-output / failure rate", 0.0, 1.0, 0.0),
            "maximum_latency": st.number_input("Maximum mean latency (seconds)", 0.1, 1000.0, 10.0),
            "maximum_tokens": st.number_input("Maximum mean total tokens", 1, 100000, 1000),
        }
        unacceptable_definition = st.text_area("Define unacceptable behavior / hard release blockers", value="Any unsafe launch recommendation or material unsupported assertion.")
    selected = [saved[k] for k in chosen]
    critical_ids = st.multiselect("Critical scenarios for regression review", ids)
    plan_deviations = []
    for run in selected:
        plan = run.get("evaluation_plan")
        if not plan:
            plan_deviations.append(run["run_id"] + ": no frozen plan (legacy evidence)")
        else:
            frozen_criteria = plan["payload"]["criteria"]
            if any(frozen_criteria.get(k) != v for k, v in criteria.items()) or frozen_criteria.get("unacceptable_behavior_definition") != unacceptable_definition or set(ids) != {s["id"] for s in plan["payload"]["scenarios"]} or set(critical_ids) != set(plan["payload"]["critical_scenario_ids"]):
                plan_deviations.append(run["run_id"] + ": post-results criteria or scenario selection differs from frozen plan")
            if plan.get("post_results"):
                plan_deviations.append(run["run_id"] + ": plan frozen after evidence existed")
    for deviation in plan_deviations:
        st.warning(deviation)
    report = release_report(selected, ids, criteria)
    st.dataframe([{k: v for k, v in row.items() if k not in ("Gates", "Evidence")} for row in report["rows"]], hide_index=True)
    analysis = audit_report_ui(report, selected, critical_ids)
    st.info("Evidence recommendation: " + report["recommendation"])
    for reason in report["reasons"]:
        st.caption(reason)
    st.caption("Recommendations apply to saved run inputs and completed reviews. They are descriptive gates, not statistical evidence of superiority.")
    decision_inputs = {"runs": selected, "scenarios": ids, "criteria": criteria, "unacceptable_behavior_definition": unacceptable_definition, "critical_scenario_ids": critical_ids}
    signature = input_signature(decision_inputs)
    reviewer = st.text_input("Decision owner")
    outcome = st.selectbox("PM decision", ["Not decided", "Keep A", "Release B", "Improve and retest", "Insufficient evidence", "No clear winner"])
    rationale = st.text_area("Release decision rationale (explain any override)")
    if st.button("Record release decision", disabled=outcome == "Not decided" or not reviewer.strip() or not rationale.strip()):
        st.session_state["release_decision"] = {"owner": reviewer, "decision": outcome, "rationale": rationale, "timestamp_utc": datetime.now(timezone.utc).isoformat(), "evidence_signature": signature, "criteria": criteria, "scenario_ids": ids, "run_ids": chosen, "recommendation": report, "unacceptable_behavior_definition": unacceptable_definition}
        st.session_state["release_decision"].update(plan_deviations=deepcopy(plan_deviations), paired_analysis=deepcopy(analysis), critical_scenario_ids=list(critical_ids))
    decision = deepcopy(st.session_state.get("release_decision"))
    if decision:
        decision["stale"] = decision.get("evidence_signature") != signature
        st.write("Recorded decision: " + str(decision.get("decision")))
        if decision["stale"]:
            st.warning("The decision record is stale: evidence or release criteria have changed. Record a new decision after review.")
    st.download_button("Download workspace and decision JSON", json.dumps({"schema_version": "0.6", "runs": list(saved.values()), "release_decision": decision, "evaluation_plans": st.session_state.get("evaluation_plans", [])}, indent=2), file_name="evaluation_workspace.json", mime="application/json")
