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
    aggregate_trial_results,
    execution_order,
    expected_request_count,
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
SCORE_DIMENSIONS = ("Correctness", "Risk awareness", "Actionability", "Evidence quality")
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
    clear_evaluation()


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
    on_change=clear_evaluation,
)
st.caption(selected_scenario["synthetic_data_statement"])

st.subheader("Trial settings")
trial_count = st.radio(
    "Number of trials",
    options=ALLOWED_TRIAL_COUNTS,
    index=0,
    horizontal=True,
    key="trial_count",
    on_change=clear_evaluation,
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
                    "scenario": scenario.strip(),
                },
                "B": {
                    "key": "B",
                    "name": CONFIGURATION_B,
                    "prompt_version": PROMPT_VERSION_B,
                    "prompt": prompt_b,
                    "scenario": scenario.strip(),
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
                "scenario": scenario.strip(),
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
    st.divider()
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
                if result["status"] == "success":
                    st.markdown(result["response"])
                    st.write(f"Input tokens: {result['input_tokens'] if result['input_tokens'] is not None else 'Unavailable'}")
                    st.write(f"Output tokens: {result['output_tokens'] if result['output_tokens'] is not None else 'Unavailable'}")
                    st.write(f"Total tokens: {result['total_tokens'] if result['total_tokens'] is not None else 'Unavailable'}")
                else:
                    st.error(result["safe_error_message"])
                    st.write(f"Safe error category: {result['error_category']}")

    st.subheader("Human evaluation scores")
    st.caption("These evaluator-assigned scores summarize each configuration; they are not automated model scores.")
    score_columns = st.columns(2, gap="large")
    human_scores = {}
    for column, key, name in zip(
        score_columns,
        ("A", "B"),
        (CONFIGURATION_A, CONFIGURATION_B),
    ):
        with column:
            st.markdown(f"#### {name}")
            human_scores[key] = {
                dimension: st.slider(
                    dimension,
                    1,
                    5,
                    3,
                    key=f"score_{key.lower()}_{dimension.lower().replace(' ', '_')}",
                )
                for dimension in SCORE_DIMENSIONS
            }

    totals = {key: sum(scores.values()) for key, scores in human_scores.items()}
    difference = totals["A"] - totals["B"]
    summary_columns = st.columns(3)
    summary_columns[0].metric("Configuration A total", totals["A"])
    summary_columns[1].metric("Configuration B total", totals["B"])
    summary_columns[2].metric("Score difference (A − B)", difference)
    if difference > 0:
        st.success("Configuration A currently has the higher human score.")
    elif difference < 0:
        st.success("Configuration B currently has the higher human score.")
    else:
        st.info("The configurations are tied.")

    export = {
        **evaluation,
        "human_scores": human_scores,
        "total_human_scores": totals,
    }
    st.download_button(
        "Download evaluation JSON",
        data=json.dumps(export, indent=2),
        file_name="model_behavior_evaluation.json",
        mime="application/json",
    )
