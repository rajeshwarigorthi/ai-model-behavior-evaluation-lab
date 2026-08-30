from statistics import mean, median


CONFIGURATION_KEYS = ("A", "B")
ALLOWED_TRIAL_COUNTS = (1, 3, 5)


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
