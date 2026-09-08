import copy
import unittest
from unittest.mock import patch
from pathlib import Path

from audit_logic import freeze_plan, validate_plan, record_review, paired_analysis
from release_logic import release_report, import_evidence
from test_release_workflow import fixture, CRITERIA


class AuditTests(unittest.TestCase):
    def test_plan_is_immutable_versioned_and_bound_to_inputs(self):
        run = fixture()
        plan = freeze_plan(run["evaluation_inputs"], [], CRITERIA, [])
        validate_plan(plan, run["evaluation_inputs"])
        newer = freeze_plan(run["evaluation_inputs"], [], CRITERIA, [], plan, True)
        self.assertEqual(newer["version"], 2)
        self.assertTrue(newer["post_results"])
        changed = copy.deepcopy(run["evaluation_inputs"])
        changed["requested_trial_count"] = 4
        with self.assertRaises(ValueError):
            validate_plan(plan, changed)
        newer["payload"]["criteria"]["minimum_score"] = 4
        self.assertEqual(plan["payload"]["criteria"]["minimum_score"], 16)
        with self.assertRaises(ValueError):
            validate_plan(newer)

    def test_all_gates_explain_failures_and_reference_evidence(self):
        run = fixture()
        run["individual_trial_results"][1]["unacceptable_behavior"] = True
        row = release_report([run], [run["scenario_id"]], CRITERIA)["rows"][1]
        self.assertFalse(row["Pass"])
        self.assertEqual(len(row["Gates"]), 9)
        self.assertFalse(row["Gates"][-1]["passed"])
        self.assertEqual(row["Evidence"][0]["run_id"], run["run_id"])

    def test_revision_history_and_irreversible_reveal(self):
        result = {}
        review = {"review_status": "complete", "total_human_score": 20}
        record_review(result, review, "reviewer-1", False)
        record_review(result, review, "reviewer-1", False)
        self.assertEqual(len(result["review_history"]), 1)
        record_review(result, {**review, "total_human_score": 19}, "reviewer-2", True)
        record_review(result, review, "reviewer-1", False)
        self.assertEqual(len(result["review_history"]), 3)
        self.assertTrue(result["identity_revealed"])
        self.assertEqual(result["review_history"][0]["snapshot"]["review"]["total_human_score"], 20)

    def test_pairs_deltas_failures_and_incompatible_groups(self):
        run = fixture()
        b = run["individual_trial_results"][1]
        b.update(total_human_score=18, latency_seconds=3, total_tokens=30)
        report = paired_analysis([run], [run["scenario_id"]])
        row = report["pairs"][0]
        self.assertEqual((row["Score B-A"], row["Latency B-A"], row["Tokens B-A"]), (-2, 1, 10))
        self.assertTrue(row["Critical regression"])
        b["status"] = "failure"
        self.assertIsNone(paired_analysis([run])["pairs"][0]["Score B-A"])
        other = fixture()
        other["evaluation_inputs"]["configuration_b_prompt"] = "different"
        self.assertEqual(len(paired_analysis([run, other])["scenarios"]), 2)

    def test_plan_and_provenance_roundtrip(self):
        run = fixture()
        run["evaluation_plan"] = freeze_plan(run["evaluation_inputs"], [], CRITERIA, [])
        record_review(run["individual_trial_results"][0], {"review_status": "complete"}, "reviewer", False)
        restored = import_evidence(run)[run["run_id"]]
        self.assertEqual(restored["evaluation_plan"], run["evaluation_plan"])
        self.assertEqual(restored["individual_trial_results"][0]["review_history"], run["individual_trial_results"][0]["review_history"])

    def test_plan_ui_blocks_execution_without_matching_plan(self):
        from streamlit.testing.v1 import AppTest
        with patch("openai.OpenAI", side_effect=AssertionError("No API client allowed")):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=15).run()
            next(b for b in app.button if b.label == "Run evaluation").click().run()
            self.assertTrue(any("Freeze a matching" in e.value for e in app.error))
            next(b for b in app.button if b.label == "Freeze evaluation plan").click().run()
            self.assertEqual(len(app.session_state["evaluation_plans"]), 1)
            app.text_area(key="prompt_a").set_value("Changed prompt").run()
            self.assertTrue(any("Plan draft changed" in w.value for w in app.warning))
            self.assertFalse(app.exception)

    def test_attributed_review_ui_preserves_revisions_and_reveal(self):
        from streamlit.testing.v1 import AppTest
        with patch("openai.OpenAI", side_effect=AssertionError("No API client allowed")):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=15).run()
            run = import_evidence(fixture())["offline-fixture"]
            app.session_state["saved_runs"] = {run["run_id"]: run}
            app.run()
            next(b for b in app.button if b.label == "Resume selected review").click().run()
            app.text_input(key="response_reviewer_id").set_value("synthetic-reviewer").run()
            result = app.session_state["evaluation"]["individual_trial_results"][0]
            self.assertEqual(result["reviewer_id"], "synthetic-reviewer")
            self.assertEqual(len(result["review_history"]), 1)
            next(c for c in app.checkbox if c.label.startswith("Blind response review")).uncheck().run()
            next(c for c in app.checkbox if c.label.startswith("Blind response review")).check().run()
            result = app.session_state["evaluation"]["individual_trial_results"][0]
            self.assertTrue(result["identity_revealed"])
            self.assertFalse(result["review_history"][0]["snapshot"]["identity_revealed"])
            self.assertFalse(app.exception)
