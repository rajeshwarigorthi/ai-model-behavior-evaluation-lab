import json
import hashlib
import re
import unittest
from pathlib import Path

from evaluation_logic import (
    aggregate_trial_results,
    aggregate_review_summary,
    build_review,
    consideration_coverage,
    empty_review,
    execution_order,
    expected_request_count,
    format_consideration_coverage,
    format_mean_consideration_coverage,
    input_signature,
    is_reviewable,
    results_are_stale,
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
EVIDENCE_HASHES = {
    "model_behavior_evaluation_run_001.json": "EC1EC834679D3E78C374CD54373E57962E50CB503C038C5F88EF0B38671B2580",
    "model_behavior_evaluation_run_002.json": "9CB91EAD3BA99D4C672DF0B94DFC55B2804566323E50B9E1711458F13B6A821A",
    "model_behavior_evaluation_run_003.json": "CEE42531A2345903934EDE2607A0F7A43FF59FBA720D25AA14B4B44D4E20FEE7",
}
# Git checkout may use LF on Linux or CRLF on Windows. Accept only the
# independently verified exact hashes, without rewriting historical evidence.
EVIDENCE_LF_HASHES = {
    "model_behavior_evaluation_run_001.json": "EC1EC834679D3E78C374CD54373E57962E50CB503C038C5F88EF0B38671B2580",
    "model_behavior_evaluation_run_002.json": "3DEB4099DEA2E6AEAFB3E06A3397BA75A4F9B281201789FC79A7CCCEEA4AA772",
    "model_behavior_evaluation_run_003.json": "2AB6473A0567547C85B879AF5446517A70B27E71763380140D2A134582478728",
}
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
        text_suffixes = {".py", ".json", ".md", ".txt", ".example", ".gitignore", ".yml", ".yaml"}
        for path in ROOT.rglob("*"):
            if not path.is_file() or excluded_parts.intersection(path.parts):
                continue
            if path.name != ".gitignore" and path.suffix not in text_suffixes:
                continue
            content = path.read_text(encoding="utf-8")
            for pattern in SECRET_PATTERNS:
                self.assertIsNone(pattern.search(content), str(path.relative_to(ROOT)))

    def test_app_has_valid_python_syntax(self):
        for filename in ("app.py", "evaluation_logic.py", "release_logic.py", "workspace_ui.py", "audit_logic.py", "audit_ui.py"):
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

    def test_new_review_is_not_reviewed_and_uses_nulls(self):
        review = empty_review()
        self.assertEqual(review["review_status"], "not_reviewed")
        self.assertTrue(all(score is None for score in review["human_scores"].values()))
        self.assertIsNone(review["total_human_score"])
        self.assertIsNone(review["expected_decision_match"])
        self.assertIsNone(review["addressed_considerations"])
        self.assertIn('"total_human_score": null', json.dumps(review))

    def test_incomplete_review_has_no_total(self):
        review = build_review(
            {"Correctness": 5, "Risk awareness": None, "Actionability": 4, "Evidence quality": 4},
            "Yes",
            ["Rollback"],
            ["Rollback", "Monitoring"],
            True,
            "Partial review",
            True,
        )
        self.assertEqual(review["review_status"], "review_incomplete")
        self.assertIsNone(review["total_human_score"])
        self.assertIsNone(review["missed_considerations"])
        self.assertIsNone(review["consideration_coverage"])

    def test_failed_and_empty_responses_are_not_reviewable(self):
        self.assertFalse(is_reviewable({"status": "failure", "response": "output"}))
        self.assertFalse(is_reviewable({"status": "success", "response": "  "}))
        self.assertTrue(is_reviewable({"status": "success", "response": "usable"}))

    def test_complete_review_calculates_total_and_coverage(self):
        review = self._complete_review("Yes", ["Rollback"], ["Rollback", "Monitoring"])
        self.assertEqual(review["review_status"], "complete")
        self.assertEqual(review["total_human_score"], 14)
        self.assertEqual(review["missed_considerations"], ["Monitoring"])
        self.assertEqual(review["consideration_coverage"], 0.5)

    def test_zero_required_considerations_are_not_applicable_but_completable(self):
        review = self._complete_review("Yes", [], [])
        self.assertEqual(review["review_status"], "complete")
        self.assertEqual(review["addressed_considerations"], [])
        self.assertEqual(review["missed_considerations"], [])
        self.assertIsNone(review["consideration_coverage"])
        self.assertEqual(
            format_consideration_coverage(review["consideration_coverage"], 0, 0),
            "Not applicable — no required considerations",
        )

    def test_not_applicable_coverage_is_excluded_from_aggregate(self):
        not_applicable = {
            **self._result("A", 1.0, 1, 1, 2),
            "response": "usable",
            **self._complete_review("Yes", [], []),
        }
        applicable = {
            **self._result("A", 1.0, 1, 1, 2),
            "response": "usable",
            **self._complete_review("Yes", ["One"], ["One", "Two"]),
        }
        summary = aggregate_review_summary([not_applicable, applicable])
        self.assertEqual(summary["A"]["mean_consideration_coverage"], 0.5)

    def test_aggregate_is_safe_when_all_coverage_is_not_applicable(self):
        result = {
            **self._result("A", 1.0, 1, 1, 2),
            "response": "usable",
            **self._complete_review("Yes", [], []),
        }
        summary = aggregate_review_summary([result])
        self.assertIsNone(summary["A"]["mean_consideration_coverage"])
        self.assertEqual(
            format_mean_consideration_coverage(
                summary["A"]["mean_consideration_coverage"],
                summary["A"]["completed_review_count"],
            ),
            "Not applicable",
        )

    def test_review_aggregates_exclude_unreviewed_responses(self):
        complete = {
            **self._result("A", 1.0, 1, 1, 2),
            "response": "usable",
            **self._complete_review("Yes", ["Rollback"], ["Rollback"]),
        }
        unreviewed = {
            **self._result("A", 2.0, 1, 1, 2),
            "response": "usable",
            **empty_review(),
        }
        mismatch = {
            **self._result("B", 2.0, 1, 1, 2),
            "response": "usable",
            **self._complete_review("No", [], ["Rollback"]),
        }
        summary = aggregate_review_summary([complete, unreviewed, mismatch])
        self.assertEqual(summary["A"]["completed_review_count"], 1)
        self.assertEqual(summary["A"]["incomplete_review_count"], 1)
        self.assertEqual(summary["A"]["mean_human_score"], 14)
        self.assertEqual(summary["A"]["expected_decision_matches"], 1)
        self.assertEqual(summary["B"]["expected_decision_mismatches"], 1)

    def test_input_signatures_are_deterministic_and_prompt_sensitive(self):
        inputs = {
            "scenario_id": "example",
            "scenario_text": "Synthetic scenario",
            "configuration_a_prompt": "Prompt A",
            "configuration_b_prompt": "Prompt B",
            "requested_trial_count": 1,
            "model": "model",
            "reasoning_effort": "low",
            "max_output_tokens": 700,
        }
        reordered = dict(reversed(list(inputs.items())))
        self.assertEqual(input_signature(inputs), input_signature(reordered))
        changed_a = {**inputs, "configuration_a_prompt": "Changed A"}
        changed_b = {**inputs, "configuration_b_prompt": "Changed B"}
        self.assertNotEqual(input_signature(inputs), input_signature(changed_a))
        self.assertNotEqual(input_signature(inputs), input_signature(changed_b))
        self.assertTrue(results_are_stale(input_signature(inputs), changed_a))
        self.assertFalse(results_are_stale(input_signature(inputs), reordered))

    def test_historical_evaluation_files_are_unchanged(self):
        for filename, expected_hash in EVIDENCE_HASHES.items():
            content = (ROOT / "evaluations" / filename).read_bytes()
            self.assertIn(hashlib.sha256(content).hexdigest().upper(), {expected_hash, EVIDENCE_LF_HASHES[filename]})
            self.assertEqual(hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest().upper(), EVIDENCE_LF_HASHES[filename])

    @staticmethod
    def _complete_review(expected_match, addressed, required):
        return build_review(
            {"Correctness": 5, "Risk awareness": 4, "Actionability": 3, "Evidence quality": 2},
            expected_match,
            addressed,
            required,
            True,
            "",
            True,
        )

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
