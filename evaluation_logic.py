import hashlib
import json
from statistics import mean, median


CONFIGURATION_KEYS = ("A", "B")
ALLOWED_TRIAL_COUNTS = (1, 3, 5)
SCORE_DIMENSIONS = ("Correctness", "Risk awareness", "Actionability", "Evidence quality")
EXPECTED_DECISION_ASSESSMENTS = ("Yes", "No", "Unclear")


def execution_order(trial_number):
    """Return sequential configuration order for a one-based trial number."""
    if not isinstance(trial_number, int) or trial_number < 1:
        raise ValueError("trial_number must be a positive integer")
    return CONFIGURATION_KEYS if trial_number % 2 else tuple(reversed(CONFIGURATION_KEYS))


def expected_request_count(trial_count):
    """Return the number of requests needed for a supported trial count."""
    if trial_count not in ALLOWED_TRIAL_COUNTS:
        raise ValueError(f"trial_count must be one of {ALLOWED_TRIAL_COUNTS}")
    return trial_count * len(CONFIGURATION_KEYS)


def _mean_available(records, field):
    values = [record[field] for record in records if record.get(field) is not None]
    return mean(values) if values else None


def aggregate_trial_results(trial_results):
    """Aggregate successful metrics and failure counts by configuration."""
    aggregates = {}
    for configuration_key in CONFIGURATION_KEYS:
        records = [
            record
            for record in trial_results
            if record["configuration_key"] == configuration_key
        ]
        successful = [record for record in records if record["status"] == "success"]
        latencies = [record["latency_seconds"] for record in successful]
        aggregates[configuration_key] = {
            "successful_trial_count": len(successful),
            "failed_trial_count": len(records) - len(successful),
            "result_basis": (
                "single-run"
                if len(successful) == 1
                else "repeated-trial"
                if len(successful) > 1
                else "no-successful-results"
            ),
            "mean_latency_seconds": mean(latencies) if latencies else None,
            "median_latency_seconds": median(latencies) if latencies else None,
            "minimum_latency_seconds": min(latencies) if latencies else None,
            "maximum_latency_seconds": max(latencies) if latencies else None,
            "mean_input_tokens": _mean_available(successful, "input_tokens"),
            "mean_output_tokens": _mean_available(successful, "output_tokens"),
            "mean_total_tokens": _mean_available(successful, "total_tokens"),
        }
    return aggregates


def input_signature(inputs):
    """Return a deterministic signature for canonical evaluation inputs."""
    canonical = json.dumps(
        inputs,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def results_are_stale(saved_signature, current_inputs):
    return saved_signature != input_signature(current_inputs)


def is_reviewable(result):
    return result.get("status") == "success" and bool(
        str(result.get("response") or "").strip()
    )


def empty_review():
    """Return the JSON representation of a response that has not been reviewed."""
    return {
        "review_status": "not_reviewed",
        "human_scores": {dimension: None for dimension in SCORE_DIMENSIONS},
        "total_human_score": None,
        "expected_decision_match": None,
        "addressed_considerations": None,
        "missed_considerations": None,
        "consideration_coverage": None,
        "reviewer_notes": None,
    }


def consideration_coverage(addressed_considerations, required_considerations):
    """Return covered/required, or None when coverage is not applicable."""
    if not required_considerations:
        return None
    addressed = set(addressed_considerations)
    return len(addressed.intersection(required_considerations)) / len(required_considerations)


def format_consideration_coverage(coverage, addressed_count, required_count):
    if coverage is None:
        return "Not applicable — no required considerations"
    return f"{addressed_count} of {required_count} ({coverage:.0%})"


def format_mean_consideration_coverage(coverage, completed_review_count):
    if coverage is None:
        return "Not applicable" if completed_review_count else "Unavailable"
    return f"{coverage:.0%}"


def build_review(
    human_scores,
    expected_decision_match,
    addressed_considerations,
    required_considerations,
    considerations_reviewed,
    reviewer_notes,
    mark_complete,
):
    """Build a complete or incomplete review without inventing missing values."""
    review = empty_review()
    review["human_scores"] = dict(human_scores)
    review["expected_decision_match"] = expected_decision_match
    review["reviewer_notes"] = reviewer_notes.strip() or None
    if considerations_reviewed:
        addressed = [
            item for item in required_considerations if item in addressed_considerations
        ]
        review["addressed_considerations"] = addressed

    prerequisites_met = (
        set(human_scores) == set(SCORE_DIMENSIONS)
        and all(human_scores[dimension] in range(1, 6) for dimension in SCORE_DIMENSIONS)
        and expected_decision_match in EXPECTED_DECISION_ASSESSMENTS
        and considerations_reviewed
    )
    has_progress = (
        any(score is not None for score in human_scores.values())
        or expected_decision_match is not None
        or considerations_reviewed
        or bool(review["reviewer_notes"])
    )
    if mark_complete and prerequisites_met:
        addressed = review["addressed_considerations"]
        review.update(
            {
                "review_status": "complete",
                "total_human_score": sum(human_scores.values()),
                "missed_considerations": [
                    item for item in required_considerations if item not in addressed
                ],
                "consideration_coverage": consideration_coverage(
                    addressed, required_considerations
                ),
            }
        )
    elif has_progress:
        review["review_status"] = "review_incomplete"
    return review


def aggregate_review_summary(trial_results):
    """Summarize completed human reviews without treating missing reviews as zero."""
    summary = {}
    for configuration_key in CONFIGURATION_KEYS:
        usable = [
            result
            for result in trial_results
            if result.get("configuration_key") == configuration_key
            and is_reviewable(result)
        ]
        completed = [
            result for result in usable if result.get("review_status") == "complete"
        ]
        assessments = [result["expected_decision_match"] for result in completed]
        scores = [result["total_human_score"] for result in completed]
        coverage = [
            result["consideration_coverage"]
            for result in completed
            if result["consideration_coverage"] is not None
        ]
        summary[configuration_key] = {
            "usable_response_count": len(usable),
            "completed_review_count": len(completed),
            "incomplete_review_count": len(usable) - len(completed),
            "expected_decision_matches": assessments.count("Yes"),
            "expected_decision_mismatches": assessments.count("No"),
            "expected_decision_unclear": assessments.count("Unclear"),
            "mean_human_score": mean(scores) if scores else None,
            "mean_consideration_coverage": mean(coverage) if coverage else None,
        }
    return summary
