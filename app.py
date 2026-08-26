import json
import os
import time
from datetime import datetime, timezone

import streamlit as st
from openai import APIConnectionError, APIError, AuthenticationError, OpenAI, RateLimitError


MODEL = "gpt-5.6-luna"
SYNTHETIC_DATA_STATEMENT = "This evaluation uses a synthetic scenario and contains no real company or patient data."
CONFIGURATION_A = "Configuration A — General Analysis"
CONFIGURATION_B = "Configuration B — Operational Readiness Analysis"
SCORE_DIMENSIONS = ("Correctness", "Risk awareness", "Actionability", "Evidence quality")


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
        reasoning={"effort": "low"},
        max_output_tokens=700,
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

st.subheader("Synthetic scenario")
scenario = st.text_area(
    "Scenario to evaluate",
    value="Should a healthcare claims application be approved for production migration when performance under combined workloads has not yet been validated?",
    height=120,
)

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
            st.session_state["evaluation"] = {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "synthetic_data_statement": SYNTHETIC_DATA_STATEMENT,
                "model": MODEL,
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
