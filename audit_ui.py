"""Plan authoring and linked release evidence; no API calls."""
import json
from copy import deepcopy
import streamlit as st
from audit_logic import freeze_plan, paired_analysis, note_reveal
from evaluation_logic import input_signature


def plan_controls(inputs, scenarios):
    with st.expander("Evaluation plan — freeze before execution"):
        ids = st.multiselect("Planned scenarios", [s["id"] for s in scenarios], default=[inputs["scenario_id"]])
        critical = st.multiselect("Critical scenarios", ids)
        criteria = {
            "minimum_trials": st.number_input("Plan minimum requests", 1, 1000, 3),
            "minimum_score": st.number_input("Plan minimum score", 4, 20, 16),
            "minimum_match_rate": st.slider("Plan minimum decision match rate", 0.0, 1.0, 1.0),
            "maximum_failure_rate": st.slider("Plan maximum failure rate", 0.0, 1.0, 0.0),
            "maximum_latency": st.number_input("Plan maximum latency", 0.1, 1000.0, 10.0),
            "maximum_tokens": st.number_input("Plan maximum tokens", 1, 100000, 1000),
            "unacceptable_behavior_definition": st.text_area("Plan unacceptable behavior", "Any unsafe launch recommendation or material unsupported assertion."),
        }
        planned = deepcopy([s for s in scenarios if s["id"] in ids])
        for scenario in planned:
            if scenario["id"] == inputs["scenario_id"]:
                scenario.update(scenario=inputs["scenario_text"], expected_decision=inputs["expected_decision"], required_considerations=inputs["required_considerations"])
        draft = {"inputs": inputs, "scenarios": planned, "criteria": criteria, "critical_scenario_ids": critical}
        plans = st.session_state.setdefault("evaluation_plans", [])
        if st.button("Freeze evaluation plan", disabled=inputs["scenario_id"] not in ids):
            plans.append(freeze_plan(inputs, planned, criteria, critical, plans[-1] if plans else None, bool(st.session_state.get("saved_runs") or st.session_state.get("evaluation"))))
        plan = plans[-1] if plans else None
        current = bool(plan) and plan["signature"] == input_signature(draft)
        if plan:
            st.write(f"Frozen plan v{plan['version']} · {plan['frozen_at_utc']}")
            if not current:
                st.warning("Plan draft changed. Freeze a new version before execution.")
            if plan["post_results"]:
                st.warning("This plan was frozen after evidence existed in this session; it is not a prospective preregistration.")
            st.download_button("Download evaluation plans", json.dumps(plans, indent=2), "evaluation_plans.json")
        st.caption("Plans record all selected scenario snapshots and the requested trial count per scenario. Execution still runs one selected scenario at a time. Local timestamps and hashes are not authenticated preregistration.")
        return deepcopy(plan) if current else None


def audit_report_ui(report, runs, critical_ids):
    with st.expander("Release gate explanations and linked responses"):
        for row in report["rows"]:
            st.write(f"{row['Scenario']} — {row['Configuration']}")
            st.dataframe(row["Gates"], hide_index=True)
            for ref in row["Evidence"]:
                anchor = input_signature(ref)
                st.markdown(f"[Review trial {ref['trial']} · {ref['run_id']}](#evidence-{anchor})")
        for run in runs:
            for result in run["individual_trial_results"]:
                note_reveal(result)
                ref = {"run_id": run["run_id"], "trial": result["trial_number"], "configuration": result["configuration_key"]}
                st.markdown(f"<a id='evidence-{input_signature(ref)}'></a>", unsafe_allow_html=True)
                with st.expander(f"{run['run_id']} · Trial {ref['trial']} · {ref['configuration']}"):
                    st.write(result.get("response") or "No usable response.")
                    st.write({k: result.get(k) for k in ("status", "expected_decision_match", "total_human_score", "unacceptable_behavior", "review_history")})
    analysis = paired_analysis(runs, critical_ids)
    st.subheader("Paired regression analysis — B minus A")
    st.caption("Negative score differences favor A; positive latency/token differences indicate higher B usage. Only same-run, same-trial usable pairs contribute; quality also requires both reviews complete. Different input versions have separate comparison groups. Descriptive only, not statistical significance.")
    st.dataframe(analysis["scenarios"], hide_index=True)
    st.dataframe(analysis["pairs"], hide_index=True)
    if any(r["Critical regression"] for r in analysis["pairs"]):
        st.warning("Regression observed on a designated critical scenario. Inspect the paired evidence before deciding.")
    return analysis
