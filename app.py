import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

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


MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 700
PROMPT_VERSION_A = "v1-general"
PROMPT_VERSION_B = "v1-operational"
CONFIGURATION_A = "Configuration A — General Analysis"
CONFIGURATION_B = "Configuration B — Operational Readiness Analysis"
EXECUTION_ORDER_STRATEGY = (
    "Sequential alternating order: odd trials A then B; even trials B then A."
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
        "status": "success",
        "error_category": None,
        "safe_error_message": None,
    }


def clear_human_scores():
    for key in tuple(st.session_state):
        if key.startswith("score_"):
            del st.session_state[key]


def clear_evaluation():
    st.session_state.pop("evaluation", None)
    clear_human_scores()


def display_number(value, decimals=2):
    return "Unavailable" if value is None else f"{value:.{decimals}f}"


st.set_page_config(page_title="AI Model Behavior Evaluation Lab", page_icon="🔬", layout="wide")
st.title("AI Model Behavior Evaluation Lab")
st.write("Compare two prompt configurations across repeated trials using synthetic production-readiness scenarios.")

scenarios = load_scenarios()
scenarios_by_id = {item["id"]: item for item in scenarios}


def change_scenario():
    selected = scenarios_by_id[st.session_state["selected_scenario_id"]]
    st.session_state["scenario_text"] = selected["scenario"]


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

scenario = st.text_area(
    "Scenario to evaluate",
    height=120,
    key="scenario_text",
)
st.caption(selected_scenario["synthetic_data_statement"])

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

st.subheader("Prompt configurations")
configuration_a, configuration_b = st.columns(2, gap="large")
with configuration_a:
    st.markdown(f"### {CONFIGURATION_A}")
    prompt_a = st.text_area(
        "Prompt",
        value="You are a technical program manager. Analyze the situation and recommend whether the application should be approved for production migration.",
        height=180,
        key="prompt_a",
    )
with configuration_b:
    st.markdown(f"### {CONFIGURATION_B}")
    prompt_b = st.text_area(
        "Prompt",
        value="You are responsible for production-readiness evaluation. Assess combined-workload performance, dependencies, rollback readiness, operational risk, missing evidence, and required mitigations before recommending a launch decision.",
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
}

if st.button("Run evaluation", type="primary"):
    if not scenario.strip():
        st.error("Enter a scenario before running the evaluation.")
    elif not os.environ.__contains__("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY is missing. Set it in the Windows environment and restart the app.")
    else:
        clear_evaluation()
        try:
            client = OpenAI()
        except Exception as error:
            _, message = safe_error_details(error)
            st.error(message)
        else:
            configurations = {
                "A": {
                    "key": "A",
                    "name": CONFIGURATION_A,
                    "prompt_version": PROMPT_VERSION_A,
                    "prompt": prompt_a,
                    "scenario": scenario,
                },
                "B": {
                    "key": "B",
                    "name": CONFIGURATION_B,
                    "prompt_version": PROMPT_VERSION_B,
                    "prompt": prompt_b,
                    "scenario": scenario,
                },
            }
            trial_results = []
            for trial_number in range(1, trial_count + 1):
                for position_index, configuration_key in enumerate(
                    execution_order(trial_number), start=1
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
                "expected_decision": selected_scenario["expected_decision"],
                "required_considerations": selected_scenario["required_considerations"],
                "scenario": scenario,
                "evaluation_inputs": current_inputs,
                "input_signature": input_signature(current_inputs),
                "requested_trial_count": trial_count,
                "total_expected_api_request_count": request_count,
                "execution_order_strategy": EXECUTION_ORDER_STRATEGY,
                "individual_trial_results": trial_results,
                "aggregate_metrics": aggregate_trial_results(trial_results),
                "prompt_version_a": PROMPT_VERSION_A,
                "prompt_version_b": PROMPT_VERSION_B,
                "model": MODEL,
                "reasoning_effort": REASONING_EFFORT,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "timestamp_utc": timestamp.isoformat(),
                "synthetic_data_statement": selected_scenario["synthetic_data_statement"],
            }

evaluation = st.session_state.get("evaluation")
if evaluation:
    stale_results = results_are_stale(evaluation.get("input_signature"), current_inputs)
    st.divider()
    if stale_results:
        st.warning(
            "These results are stale because the evaluation inputs changed after this run. "
            "Run the evaluation again before scoring or comparing the updated inputs."
        )

    st.subheader("Aggregate comparison")
    st.caption("Descriptive metrics only; these results do not establish statistical significance.")
    aggregate_rows = []
    for key, name in (("A", CONFIGURATION_A), ("B", CONFIGURATION_B)):
        metrics = evaluation["aggregate_metrics"][key]
        aggregate_rows.append(
            {
                "Configuration": name,
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
        if metrics["successful_trial_count"] == 0:
            st.warning(f"{name} has no successful results; aggregate performance metrics are unavailable.")
        elif metrics["successful_trial_count"] == 1:
            st.info(f"{name} metrics are single-run results, not repeated-trial estimates.")
    st.dataframe(aggregate_rows, use_container_width=True, hide_index=True)

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
        st.markdown(f"#### Trial {trial_number} — first: {first_name}")
        for result in trial_records:
            label = (
                f"{result['configuration_name']} · {result['execution_position']} · "
                f"{result['status']}"
            )
            with st.expander(label):
                st.write(f"Prompt version: {result['prompt_version']}")
                st.write(f"Model: {result['model']}")
                st.write(f"Latency: {result['latency_seconds']:.2f} seconds")
                if is_reviewable(result):
                    st.markdown(result["response"])
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
                    considerations_reviewed = st.checkbox(
                        "Required-considerations review performed",
                        key=f"{key_prefix}_considerations_reviewed",
                        disabled=stale_results,
                    )
                    notes = st.text_area(
                        "Optional reviewer notes",
                        key=f"{key_prefix}_notes",
                        disabled=stale_results,
                    )
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
                    result.update(review)
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
                "Configuration": name,
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
            st.success("Configuration A has the higher mean completed-review score.")
        elif mean_b > mean_a:
            st.success("Configuration B has the higher mean completed-review score.")
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
