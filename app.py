import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import streamlit as st
from openai import APIConnectionError, APIError, AuthenticationError, OpenAI, RateLimitError


MODEL = "gpt-5.6-luna"
REASONING_EFFORT = "low"
MAX_OUTPUT_TOKENS = 700
PROMPT_VERSION_A = "v1-general"
PROMPT_VERSION_B = "v1-operational"
CONFIGURATION_A = "Configuration A — General Analysis"
CONFIGURATION_B = "Configuration B — Operational Readiness Analysis"
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


def friendly_api_error(error):
    """Return a fixed message so exception details cannot expose credentials."""
    if isinstance(error, AuthenticationError):
        return "Authentication failed. Check the OPENAI_API_KEY environment variable."
    if isinstance(error, RateLimitError):
        return "The API rate or spend limit was reached. Check account limits and try again later."
    if isinstance(error, APIConnectionError):
        return "The OpenAI API could not be reached. Check the network connection and try again."
    if isinstance(error, APIError):
        return "The OpenAI API returned an unexpected error. Try again later."
    return "An unexpected error occurred while running the evaluation."


def run_configuration(client, name, prompt, scenario):
    started = time.perf_counter()
    response = client.responses.create(
        model=MODEL,
        instructions=prompt,
        input=scenario,
        reasoning={"effort": REASONING_EFFORT},
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return {
        "name": name,
        "prompt": prompt,
        "response": response.output_text,
        "model": getattr(response, "model", MODEL),
        "latency_seconds": time.perf_counter() - started,
        "token_usage": token_usage(response),
    }


def display_metric_value(value):
    return "Unavailable" if value is None else str(value)


st.set_page_config(page_title="AI Model Behavior Evaluation Lab", page_icon="🔬", layout="wide")
st.title("AI Model Behavior Evaluation Lab")
st.write("Compare two prompt configurations against the same synthetic scenario, then score their behavior using human judgment.")

scenarios = load_scenarios()
scenarios_by_id = {item["id"]: item for item in scenarios}


def change_scenario():
    selected = scenarios_by_id[st.session_state["selected_scenario_id"]]
    st.session_state["scenario_text"] = selected["scenario"]
    st.session_state.pop("evaluation", None)


def clear_evaluation():
    st.session_state.pop("evaluation", None)


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
        results, errors = [], []
        try:
            client = OpenAI()
        except Exception as error:
            errors.append(friendly_api_error(error))
        else:
            for name, prompt in ((CONFIGURATION_A, prompt_a), (CONFIGURATION_B, prompt_b)):
                try:
                    results.append(run_configuration(client, name, prompt, scenario.strip()))
                except Exception as error:
                    errors.append(f"{name}: {friendly_api_error(error)}")

        if results:
            timestamp = datetime.now(timezone.utc)
            st.session_state["evaluation"] = {
                "run_id": f"run-{timestamp.strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}",
                "scenario_id": selected_scenario["id"],
                "scenario_title": selected_scenario["title"],
                "scenario_category": selected_scenario["category"],
                "scenario_risk_level": selected_scenario["risk_level"],
                "expected_decision": selected_scenario["expected_decision"],
                "required_considerations": selected_scenario["required_considerations"],
                "prompt_version_a": PROMPT_VERSION_A,
                "prompt_version_b": PROMPT_VERSION_B,
                "model": MODEL,
                "reasoning_effort": REASONING_EFFORT,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
                "timestamp_utc": timestamp.isoformat(),
                "synthetic_data_statement": selected_scenario["synthetic_data_statement"],
                "scenario": scenario.strip(),
                "results": results,
            }
        else:
            st.session_state.pop("evaluation", None)
        for message in errors:
            st.error(message)

evaluation = st.session_state.get("evaluation")
if evaluation:
    st.divider()
    st.subheader("Evaluation results")
    result_columns = st.columns(2, gap="large")
    scores = {}
    for column, result, config_key in zip(result_columns, evaluation["results"], ("a", "b")):
        with column:
            st.markdown(f"### {result['name']}")
            st.markdown(result["response"])
            st.caption(f"Model: {result['model']}")
            st.metric("Latency", f"{result['latency_seconds']:.2f} seconds")
            usage = result["token_usage"]
            st.write(f"Input tokens: {display_metric_value(usage['input_tokens'])}")
            st.write(f"Output tokens: {display_metric_value(usage['output_tokens'])}")
            st.write(f"Total tokens: {display_metric_value(usage['total_tokens'])}")
            st.markdown("#### Human evaluation scores")
            st.caption("These are evaluator-assigned scores, not automated model scores.")
            scores[result["name"]] = {
                dimension: st.slider(dimension, 1, 5, 3, key=f"score_{config_key}_{dimension.lower().replace(' ', '_')}")
                for dimension in SCORE_DIMENSIONS
            }

    if len(evaluation["results"]) == 2:
        names = [result["name"] for result in evaluation["results"]]
        totals = {name: sum(scores[name].values()) for name in names}
        difference = totals[names[0]] - totals[names[1]]
        st.subheader("Comparison summary")
        summary_columns = st.columns(3)
        summary_columns[0].metric("Configuration A total", totals[names[0]])
        summary_columns[1].metric("Configuration B total", totals[names[1]])
        summary_columns[2].metric("Score difference (A − B)", difference)
        if difference > 0:
            st.success("Configuration A currently has the higher human score.")
        elif difference < 0:
            st.success("Configuration B currently has the higher human score.")
        else:
            st.info("The configurations are tied.")

        export = {
            **evaluation,
            "prompt_configurations": {result["name"]: result["prompt"] for result in evaluation["results"]},
            "responses": {result["name"]: result["response"] for result in evaluation["results"]},
            "latency_metrics_seconds": {result["name"]: result["latency_seconds"] for result in evaluation["results"]},
            "token_usage_metrics": {result["name"]: result["token_usage"] for result in evaluation["results"]},
            "manual_scores": scores,
            "total_scores": totals,
        }
        st.download_button(
            "Download evaluation JSON",
            data=json.dumps(export, indent=2),
            file_name="model_behavior_evaluation.json",
            mime="application/json",
        )
