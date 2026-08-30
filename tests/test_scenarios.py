import json
import re
import unittest
from pathlib import Path

from evaluation_logic import (
    aggregate_trial_results,
    execution_order,
    expected_request_count,
)


ROOT = Path(__file__).resolve().parents[1]
SCENARIOS_PATH = ROOT / "data" / "scenarios.json"
REQUIRED_FIELDS = {
    "id",
    "title",
    "category",
    "risk_level",
    "scenario",
    "expected_decision",
    "required_considerations",
    "synthetic_data_statement",
}
ALLOWED_DECISIONS = {"GO", "CONDITIONAL_GO", "NO_GO"}
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{16,}"),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(
        r"(?i)(?:password|passwd|token|secret|api[_-]?key)\s*[:=]\s*['\"]?[^\s<'\"]{4,}"
    ),
)


class ScenarioLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenarios = json.loads(SCENARIOS_PATH.read_text(encoding="utf-8"))

    def test_exactly_five_scenarios(self):
        self.assertEqual(len(self.scenarios), 5)

    def test_scenario_ids_are_unique(self):
        ids = [item["id"] for item in self.scenarios]
        self.assertEqual(len(ids), len(set(ids)))

    def test_every_required_field_exists(self):
        for item in self.scenarios:
            self.assertEqual(REQUIRED_FIELDS - item.keys(), set(), item.get("id"))

    def test_expected_decisions_are_allowed(self):
        for item in self.scenarios:
            self.assertIn(item["expected_decision"], ALLOWED_DECISIONS)

    def test_every_scenario_has_synthetic_statement(self):
        for item in self.scenarios:
            statement = item["synthetic_data_statement"].lower()
            self.assertIn("synthetic", statement)
            self.assertIn("no real", statement)

    def test_repository_contains_no_secret_patterns(self):
        excluded_parts = {".git", ".venv", "__pycache__"}
        text_suffixes = {".py", ".json", ".md", ".txt", ".example", ".gitignore"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or excluded_parts.intersection(path.parts):
                continue
            if path.name != ".gitignore" and path.suffix not in text_suffixes:
                continue
            content = path.read_text(encoding="utf-8")
            for pattern in SECRET_PATTERNS:
                self.assertIsNone(pattern.search(content), str(path.relative_to(ROOT)))

    def test_app_has_valid_python_syntax(self):
        for filename in ("app.py", "evaluation_logic.py"):
            source = (ROOT / filename).read_text(encoding="utf-8")
            compile(source, filename, "exec")

    def test_execution_order_for_trials_one_two_and_three(self):
        self.assertEqual(execution_order(1), ("A", "B"))
        self.assertEqual(execution_order(2), ("B", "A"))
        self.assertEqual(execution_order(3), ("A", "B"))

    def test_expected_request_count(self):
        self.assertEqual(expected_request_count(1), 2)
        self.assertEqual(expected_request_count(3), 6)
        self.assertEqual(expected_request_count(5), 10)

    def test_aggregate_calculations(self):
        records = [
            self._result("A", 2.0, 10, 20, 30),
            self._result("A", 4.0, 14, 24, 38),
            self._result("B", 3.0, 12, 22, 34),
        ]
        aggregate = aggregate_trial_results(records)
        self.assertEqual(aggregate["A"]["successful_trial_count"], 2)
        self.assertEqual(aggregate["A"]["mean_latency_seconds"], 3.0)
        self.assertEqual(aggregate["A"]["median_latency_seconds"], 3.0)
        self.assertEqual(aggregate["A"]["minimum_latency_seconds"], 2.0)
        self.assertEqual(aggregate["A"]["maximum_latency_seconds"], 4.0)
        self.assertEqual(aggregate["A"]["mean_input_tokens"], 12)
        self.assertEqual(aggregate["A"]["mean_output_tokens"], 22)
        self.assertEqual(aggregate["A"]["mean_total_tokens"], 34)
        self.assertEqual(aggregate["B"]["result_basis"], "single-run")

    def test_failed_trials_are_counted_and_excluded_from_metrics(self):
        records = [
            self._result("A", 2.0, 10, 20, 30),
            {
                "configuration_key": "A",
                "status": "failure",
                "latency_seconds": 1.0,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            },
            {
                "configuration_key": "B",
                "status": "failure",
                "latency_seconds": 1.5,
                "input_tokens": None,
                "output_tokens": None,
                "total_tokens": None,
            },
        ]
        aggregate = aggregate_trial_results(records)
        self.assertEqual(aggregate["A"]["successful_trial_count"], 1)
        self.assertEqual(aggregate["A"]["failed_trial_count"], 1)
        self.assertEqual(aggregate["A"]["mean_latency_seconds"], 2.0)
        self.assertEqual(aggregate["B"]["successful_trial_count"], 0)
        self.assertEqual(aggregate["B"]["failed_trial_count"], 1)
        self.assertIsNone(aggregate["B"]["mean_latency_seconds"])

    @staticmethod
    def _result(configuration_key, latency, input_tokens, output_tokens, total_tokens):
        return {
            "configuration_key": configuration_key,
            "status": "success",
            "latency_seconds": latency,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
        }


if __name__ == "__main__":
    unittest.main()
