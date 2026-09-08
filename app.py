import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4
from random import SystemRandom

import streamlit as st
from openai import APIConnectionError, APIError, AuthenticationError, OpenAI, RateLimitError

from evaluation_logic import (
    ALLOWED_TRIAL_COUNTS,
    SCORE_DIMENSIONS,
    aggregate_trial_results,
    aggregate_review_summary,
    build_review,
    empty_review,
    execution_order,
    expected_request_count,
    format_consideration_coverage,
    format_mean_consideration_coverage,
    input_signature,
    is_reviewable,
    results_are_stale,
)
from release_logic import prompt_identity, output_status
from workspace_ui import workspace_controls, release_workspace, archive_active


MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 700
PROMPT_VERSION_A = "v1-general"
PROMPT_VERSION_B = "v1-operational"
CONFIGURATION_A = "Configuration A — General Analysis"
CONFIGURATION_B = "Configuration B — Operational Readiness Analysis"
EXECUTION_ORDER_STRATEGY = (
    "Sequential alternating order with randomized starting configuration. Even trial counts balance first positions exactly."
)
SCENARIOS_PATH = Path(__file__).parent / "data" / "scenarios.json"


def load_scenarios():
    with SCENARIOS_PATH.open(encoding="utf-8") as scenarios_file:
        return json.load(scenarios_file)


def token_usage(response):
    usage = getattr(response, "usage", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def safe_error_details(error):
    """Map exceptions to safe categories and fixed messages."""
    if isinstance(error, AuthenticationError):
        return "authentication", "Authentication failed. Check the OPENAI_API_KEY environment variable."
    if isinstance(error, RateLimitError):
        return "rate_or_spend_limit", "The API rate or spend limit was reached. Check account limits and try again later."
    if isinstance(error, APIConnectionError):
        return "connection", "The OpenAI API could not be reached. Check the network connection and try again."
    if isinstance(error, APIError):
        return "openai_api", "The OpenAI API returned an unexpected error. Try again later."
    return "unexpected", "An unexpected error occurred while running the evaluation."


def run_configuration(client, trial_number, execution_position, configuration):
    started = time.perf_counter()
    common = {
        "trial_number": trial_number,
        "execution_position": execution_position,
        "configuration_key": configuration["key"],
        "configuration_name": configuration["name"],
        "prompt_version": configuration["prompt_version"],
        "prompt": configuration["prompt"],
        "model": MODEL,
    }
    try:
        response = client.responses.create(
            model=MODEL,
            instructions=configuration["prompt"],
            input=configuration["scenario"],
            reasoning={"effort": REASONING_EFFORT},
            max_output_tokens=MAX_OUTPUT_TOKENS,
        )
    except Exception as error:
        category, message = safe_error_details(error)
        return {
            **common,
            "response": None,
            "latency_seconds": time.perf_counter() - started,
            "input_tokens": None,
            "output_tokens": None,
            "total_tokens": None,
            "status": "failure",
            "error_category": category,
            "safe_error_message": message,
        }

    usage = token_usage(response)
    return {
        **common,
        "model": getattr(response, "model", MODEL),
        "response": response.output_text,
        "latency_seconds": time.perf_counter() - started,
        **usage,
        "status": output_status(response),
        "error_category": None if output_status(response) == "success" else output_status(response),
        "safe_error_message": None,
    }


def clear_human_scores():
    for key in tuple(st.session_state):
        if key.startswith("score_"):
            del st.session_state[key]


def clear_evaluation():
    archive_active()
    st.session_state.pop("evaluation", None)
    clear_human_scores()


def display_number(value, decimals=2):
    return "Unavailable" if value is None else f"{value:.{decimals}f}"


st.set_page_config(page_title="AI Model Behavior Evaluation Lab", page_icon="🔬", layout="wide")
st.title("AI Model Behavior Evaluation Lab")
st.write("Compare two prompt configurations across repeated trials using synthetic production-readiness scenarios.")

scenarios = load_scenarios()
scenarios_by_id = {item["id"]: item for item in scenarios}
workspace_controls(list(scenarios_by_id))


def change_scenario():
    selected = scenarios_by_id[st.session_state["selected_scenario_id"]]
    st.session_state["scenario_text"] = selected["scenario"]
    st.session_state["reference_decision"] = selected["expected_decision"]
    st.session_state["reference_considerations"] = "\n".join(selected["required_considerations"])
    st.session_state["reference_confirmed"] = False


st.subheader("Synthetic scenario library")
selected_scenario_id = st.selectbox(
    "Select a scenario",
    options=list(scenarios_by_id),
    format_func=lambda scenario_id: scenarios_by_id[scenario_id]["title"],
    key="selected_scenario_id",
    on_change=change_scenario,
)
selected_scenario = scenarios_by_id[selected_scenario_id]
metadata_columns = st.columns(3)
metadata_columns[0].write(f"**Title:** {selected_scenario['title']}")
metadata_columns[1].write(f"**Category:** {selected_scenario['category']}")
metadata_columns[2].write(f"**Risk level:** {selected_scenario['risk_level']}")

if "scenario_text" not in st.session_state:
    st.session_state["scenario_text"] = selected_scenario["scenario"]

def invalidate_reference_confirmation():
    st.session_state["reference_confirmed"] = False

scenario = st.text_area(
    "Scenario to evaluate",
    height=120,
    key="scenario_text",
    on_change=invalidate_reference_confirmation,
)
st.caption(selected_scenario["synthetic_data_statement"])
st.session_state.setdefault("reference_decision", selected_scenario["expected_decision"])
st.session_state.setdefault("reference_considerations", "\n".join(selected_scenario["required_considerations"]))
with st.expander("Scenario reference labels"):
    expected_decision = st.selectbox("Expected decision for the edited scenario", ("GO", "CONDITIONAL_GO", "NO_GO"), key="reference_decision")
    considerations_text = st.text_area("Required considerations (one per line)", key="reference_considerations")
    required_reference = list(dict.fromkeys(x.strip() for x in considerations_text.splitlines() if x.strip()))
    reference_confirmed = st.checkbox("I confirm these reference labels apply to the edited scenario", key="reference_confirmed")
    st.caption("If the facts change, update the expected decision and considerations before evaluating.")

st.subheader("Trial settings")
trial_count = st.radio(
    "Number of trials",
    options=ALLOWED_TRIAL_COUNTS,
    index=0,
    horizontal=True,
    key="trial_count",
)
request_count = expected_request_count(trial_count)
st.info(
    f"Each trial makes two OpenAI API requests. This evaluation will make "
    f"{request_count} requests only after you click Run evaluation."
)
st.caption(EXECUTION_ORDER_STRATEGY)
if trial_count % 2:
    st.caption("An odd trial count leaves one extra first position. Use 2 or 4 trials for exact balance; the starting configuration is randomized each run.")
blind_review = st.checkbox("Blind response review (hide configuration, order, and performance)", value=True)
with st.expander("Scoring guidance — apply the same anchors to both responses"):
    st.markdown("1: Incorrect or unsafe, major omissions.\n\n2: Major weaknesses requiring substantial correction.\n\n3: Partly adequate with meaningful gaps.\n\n4: Sound with minor gaps.\n\n5: Fully meets the scenario criteria with clear support.")
    st.caption("Correctness: valid conclusion and claims. Risk awareness: material hazards and safeguards. Actionability: concrete steps, owners, and exit criteria. Evidence quality: supported reasoning and explicit missing evidence. Score substance rather than length. Blinding hides labels, but writing style can still reveal identity.")

st.subheader("Prompt configurations")
configuration_a, configuration_b = st.columns(2, gap="large")
with configuration_a:
    st.markdown(f"### {CONFIGURATION_A}")
    st.session_state.setdefault("prompt_a", "You are a technical program manager. Analyze the situation and recommend whether the application should be approved for production migration.")
    prompt_a = st.text_area(
        "Prompt",
        height=180,
        key="prompt_a",
    )
with configuration_b:
    st.markdown(f"### {CONFIGURATION_B}")
    st.session_state.setdefault("prompt_b", "You are responsible for production-readiness evaluation. Assess combined-workload performance, dependencies, rollback readiness, operational risk, missing evidence, and required mitigations before recommending a launch decision.")
    prompt_b = st.text_area(
        "Prompt",
        height=180,
        key="prompt_b",
    )

current_inputs = {
    "scenario_id": selected_scenario["id"],
    "scenario_text": scenario,
    "configuration_a_prompt": prompt_a,
    "configuration_b_prompt": prompt_b,
    "requested_trial_count": trial_count,
    "model": MODEL,
    "reasoning_effort": REASONING_EFFORT,
    "max_output_tokens": MAX_OUTPUT_TOKENS,
    "expected_decision": expected_decision,
    "required_considerations": required_reference,
}

if st.button("Run evaluation", type="primary"):
    if not scenario.strip():
        st.error("Enter a scenario before running the evaluation.")
    elif scenario != selected_scenario["scenario"] and not reference_confirmed:
        st.error("Confirm or update the reference labels for the edited scenario before running.")
    elif not os.environ.__contains__("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is missing. Set it in the Windows environment and restart the app.")
    else:
        clear_evaluation()
        try:
            client = OpenAI(max_retries=0, timeout=60.0)
        except Exception as error:
            _, message = safe_error_details(error)
            st.error(message)
        else:
            configurations = {
                "A": {
                    "key": "A",
                    "name": CONFIGURATION_A,
                    "prompt_version": prompt_identity(prompt_a),
                    "prompt": prompt_a,
                    "scenario": scenario,
                },
                "B": {
                    "key": "B",
                    "name": CONFIGURATION_B,
                    "prompt_version": prompt_identity(prompt_b),
                    "prompt": prompt_b,
                    "scenario": scenario,
                },
            }
            trial_results = []
            first_configuration = SystemRandom().choice(("A", "B"))
            blind_labels = dict(zip(SystemRandom().sample(["A", "B"], 2), ["Response X", "Response Y"]))
            for trial_number in range(1, trial_count + 1):
                for position_index, configuration_key in enumerate(
                    execution_order(trial_number, first_configuration), start=1
                ):
                    trial_results.append(
                        run_configuration(
                            client,
                            trial_number,
                            "first" if position_index == 1 else "second",
                            configurations[configuration_key],
                        )
                    )

            timestamp = datetime.now(timezone.utc)
            st.session_state["evaluation"] = {
                "run_id": f"run-{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}",
                "scenario_id": selected_scenario["id"],
                "scenario_title": selected_scenario["title"],
                "scenario_category": selected_scenario["category"],
                "scenario_risk_level": selected_scenario["risk_level"],
                "expected_decision": expected_decision,
                "required_considerations": required_reference,
                "scenario": scenario,
                "evaluation_inputs": current_inputs,
                "input_signature": input_signature(current_inputs),
                "requested_trial_count": trial_count,
                "total_expected_api_request_count": request_count,
                "execution_order_strategy": EXECUTION_ORDER_STRATEGY,
                "first_configuration": first_configuration,
                "blind_labels": blind_labels,
                "individual_trial_results": trial_results,
                "aggregate_metrics": aggregate_trial_results(trial_results),
                "prompt_version_a": prompt_identity(prompt_a),
                "prompt_version_b": prompt_identity(prompt_b),
                "model": MODEL,
                "reasoning_effort": REASONING_EFFORT,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "timestamp_utc": timestamp.isoformat(),
                "synthetic_data_statement": selected_scenario["synthetic_data_statement"],
            }

evaluation = st.session_state.get("evaluation")
if evaluation:
    evaluation.setdefault("blind_labels", {"A": "Response X", "B": "Response Y"})
    def review_label(key):
        return evaluation["blind_labels"][key] if blind_review else (CONFIGURATION_A if key == "A" else CONFIGURATION_B)
    signature_inputs = current_inputs
    if "expected_decision" not in evaluation.get("evaluation_inputs", {}):
        signature_inputs = {k: v for k, v in current_inputs.items() if k not in ("expected_decision", "required_considerations")}
    stale_results = results_are_stale(evaluation.get("input_signature"), signature_inputs) or expected_decision != evaluation["expected_decision"] or required_reference != evaluation["required_considerations"]
    st.divider()
    if stale_results:
        st.warning(
            "These results are stale because the evaluation inputs changed after this run. "
            "Run the evaluation again before scoring or comparing the updated inputs."
        )

    st.subheader("Saved experiment results")
    st.caption("Descriptive metrics only; these results do not establish statistical significance.")
    aggregate_rows = []
    for key, name in (("A", CONFIGURATION_A), ("B", CONFIGURATION_B)):
        metrics = evaluation["aggregate_metrics"][key]
        aggregate_rows.append(
            {
                "Configuration": review_label(key),
                "Successful trials": metrics["successful_trial_count"],
                "Failed trials": metrics["failed_trial_count"],
                "Result basis": metrics["result_basis"],
                "Mean latency (s)": display_number(metrics["mean_latency_seconds"]),
                "Median latency (s)": display_number(metrics["median_latency_seconds"]),
                "Min latency (s)": display_number(metrics["minimum_latency_seconds"]),
                "Max latency (s)": display_number(metrics["maximum_latency_seconds"]),
                "Mean input tokens": display_number(metrics["mean_input_tokens"], 1),
                "Mean output tokens": display_number(metrics["mean_output_tokens"], 1),
                "Mean total tokens": display_number(metrics["mean_total_tokens"], 1),
            }
        )
        if metrics["successful_trial_count"] == 0 and not blind_review:
            st.warning(f"{name} has no successful results; aggregate performance metrics are unavailable.")
        elif metrics["successful_trial_count"] == 1 and not blind_review:
            st.info(f"{name} metrics are single-run results, not repeated-trial estimates.")
    if not blind_review:
        st.dataframe(aggregate_rows, use_container_width=True, hide_index=True)
    else:
        st.caption("Performance and execution order are hidden during blind review. Uncheck blind review to reveal them after scoring.")

    st.subheader("Individual trial responses")
    required_considerations = evaluation["required_considerations"]
    for trial_number in range(1, evaluation["requested_trial_count"] + 1):
        trial_records = [
            result
            for result in evaluation["individual_trial_results"]
            if result["trial_number"] == trial_number
        ]
        first_name = next(
            result["configuration_name"]
            for result in trial_records
            if result["execution_position"] == "first"
        )
        st.markdown(f"#### Trial {trial_number}" if blind_review else f"#### Trial {trial_number} — first: {first_name}")
        if blind_review:
            trial_records = sorted(trial_records, key=lambda r: review_label(r["configuration_key"]))
        for result in trial_records:
            label = (
                f"{review_label(result['configuration_key'])} · "
                f"{result['status']}"
            )
            with st.expander(label):
                if not blind_review:
                    st.write(f"Prompt version: {result['prompt_version']}")
                    st.write(f"Model: {result['model']}")
                    st.write(f"Latency: {result['latency_seconds']:.2f} seconds")
                if is_reviewable(result):
                    st.markdown(result["response"])
                    if not blind_review:
                        st.write(f"Input tokens: {result['input_tokens'] if result['input_tokens'] is not None else 'Unavailable'}")
                        st.write(f"Output tokens: {result['output_tokens'] if result['output_tokens'] is not None else 'Unavailable'}")
                        st.write(f"Total tokens: {result['total_tokens'] if result['total_tokens'] is not None else 'Unavailable'}")
                    st.markdown("##### Human review")
                    st.write(f"Expected decision: **{evaluation['expected_decision']}**")
                    key_prefix = (
                        f"review_{evaluation['run_id']}_{trial_number}_"
                        f"{result['configuration_key'].lower()}"
                    )
                    selected_scores = {}
                    for dimension in SCORE_DIMENSIONS:
                        selected = st.selectbox(
                            f"{dimension} score",
                            options=("Not reviewed", 1, 2, 3, 4, 5),
                            key=f"{key_prefix}_{dimension.lower().replace(' ', '_')}",
                            disabled=stale_results,
                        )
                        selected_scores[dimension] = (
                            None if selected == "Not reviewed" else selected
                        )

                    expected_selection = st.selectbox(
                        "Did this response reach the expected decision?",
                        options=("Not reviewed", "Yes", "No", "Unclear"),
                        key=f"{key_prefix}_expected_decision",
                        disabled=stale_results,
                    )
                    expected_assessment = (
                        None if expected_selection == "Not reviewed" else expected_selection
                    )
                    st.markdown("**Required considerations**")
                    st.markdown(
                        "\n".join(f"- {item}" for item in required_considerations)
                        or "- No required considerations are defined."
                    )
                    addressed = st.multiselect(
                        "Considerations addressed by this response",
                        options=required_considerations,
                        key=f"{key_prefix}_addressed",
                        disabled=stale_results,
                    )
                    considerations_reviewed = not required_considerations or st.checkbox(
                        "Required-considerations review performed",
                        key=f"{key_prefix}_considerations_reviewed",
                        disabled=stale_results,
                    )
                    notes = st.text_area(
                        "Optional reviewer notes",
                        key=f"{key_prefix}_notes",
                        disabled=stale_results,
                    )
                    unacceptable = st.checkbox("Unacceptable behavior / release blocker observed", key=f"{key_prefix}_unacceptable", disabled=stale_results)
                    prerequisites_met = (
                        all(value is not None for value in selected_scores.values())
                        and expected_assessment is not None
                        and considerations_reviewed
                    )
                    mark_complete = st.checkbox(
                        "Review complete",
                        key=f"{key_prefix}_complete",
                        disabled=stale_results or not prerequisites_met,
                    )
                    review = build_review(
                        selected_scores,
                        expected_assessment,
                        addressed,
                        required_considerations,
                        considerations_reviewed,
                        notes,
                        mark_complete,
                    )
                    if not stale_results:
                        result.update(review)
                        result["unacceptable_behavior"] = unacceptable
                        result["review_mode"] = "blind" if blind_review and result.get("review_mode") != "identified" else "identified"
                    else:
                        review = {field: result.get(field, default) for field, default in empty_review().items()}
                    status_label = {
                        "not_reviewed": "Not reviewed",
                        "review_incomplete": "Review incomplete",
                        "complete": "Review complete",
                    }[review["review_status"]]
                    st.write(f"Review status: **{status_label}**")
                    if review["review_status"] == "complete":
                        covered = len(review["addressed_considerations"])
                        required = len(required_considerations)
                        st.write(f"Total human score: **{review['total_human_score']}/20**")
                        coverage_display = format_consideration_coverage(
                            review["consideration_coverage"], covered, required
                        )
                        st.write(f"Consideration coverage: **{coverage_display}**")
                else:
                    result.update(empty_review())
                    if result["status"] == "failure":
                        st.error(result["safe_error_message"])
                        st.write(f"Safe error category: {result['error_category']}")
                    elif result.get("response"):
                        st.warning("Incomplete output retained for diagnosis; excluded from quality review and successful-output metrics.")
                        st.write(result["response"])
                    st.info(
                        "This response cannot be reviewed because no usable model output was produced."
                    )

    review_summary = aggregate_review_summary(evaluation["individual_trial_results"])
    completed_reviews = sum(
        item["completed_review_count"] for item in review_summary.values()
    )
    usable_responses = sum(
        item["usable_response_count"] for item in review_summary.values()
    )
    incomplete_reviews = usable_responses - completed_reviews

    st.subheader("Human review summary")
    st.write(f"**{completed_reviews} of {usable_responses} responses reviewed.**")
    review_rows = []
    for key, name in (("A", CONFIGURATION_A), ("B", CONFIGURATION_B)):
        summary = review_summary[key]
        reviewed = summary["completed_review_count"]
        coverage = summary["mean_consideration_coverage"]
        review_rows.append(
            {
                "Configuration": review_label(key),
                "Usable responses": summary["usable_response_count"],
                "Completed reviews": reviewed,
                "Expected-decision matches": f"{summary['expected_decision_matches']} of {reviewed} reviewed",
                "Mismatches": f"{summary['expected_decision_mismatches']} of {reviewed} reviewed",
                "Unclear": f"{summary['expected_decision_unclear']} of {reviewed} reviewed",
                "Mean human score": (
                    "Unavailable"
                    if summary["mean_human_score"] is None
                    else f"{summary['mean_human_score']:.1f}/20"
                ),
                "Mean consideration coverage": format_mean_consideration_coverage(
                    coverage, reviewed
                ),
            }
        )
    st.dataframe(review_rows, use_container_width=True, hide_index=True)

    all_reviews_complete = all(
        review_summary[key]["usable_response_count"] > 0
        and review_summary[key]["incomplete_review_count"] == 0
        for key in ("A", "B")
    )
    if stale_results:
        st.warning("No winner is presented while results are stale.")
    elif not all_reviews_complete:
        st.info("No clear winner. Complete every usable response review before comparing human scores.")
    else:
        mean_a = review_summary["A"]["mean_human_score"]
        mean_b = review_summary["B"]["mean_human_score"]
        if mean_a > mean_b:
            st.success(review_label("A") + " has the higher mean completed-review score. Apply release gates before choosing a prompt.")
        elif mean_b > mean_a:
            st.success(review_label("B") + " has the higher mean completed-review score. Apply release gates before choosing a prompt.")
        else:
            st.info("No clear winner. The mean completed-review scores are tied.")

    export = {
        **evaluation,
        "completed_review_count": completed_reviews,
        "incomplete_review_count": incomplete_reviews,
        "configuration_review_summary": review_summary,
        "stale_results": stale_results,
    }
    st.download_button(
        "Download evaluation JSON",
        data=json.dumps(export, indent=2),
        file_name="model_behavior_evaluation.json",
        mime="application/json",
    )

if not blind_review:
    release_workspace(scenarios)
else:
    archive_active()
    st.caption("Reveal configurations to compare saved experiments and record a release decision. Download individual evaluation JSON above to save review progress.")
