import copy
import json
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch

from evaluation_logic import SCORE_DIMENSIONS, build_review, input_signature, execution_order
from release_logic import import_evidence, output_status, prompt_identity, release_report


def fixture():
    inputs = dict(scenario_id="minor-monitoring-gap", scenario_text="Synthetic test scenario", configuration_a_prompt="A", configuration_b_prompt="B", requested_trial_count=1, model="gpt-5.6-luna", reasoning_effort="low", max_output_tokens=700, expected_decision="CONDITIONAL_GO", required_considerations=[])
    records = []
    for key, position in (("A", "first"), ("B", "second")):
        records.append(dict(configuration_key=key, configuration_name=key, trial_number=1, execution_position=position, status="success", response="Synthetic output", prompt=key, prompt_version=prompt_identity(key), model=inputs["model"], latency_seconds=2.0, input_tokens=10, output_tokens=10, total_tokens=20, **build_review(dict.fromkeys(SCORE_DIMENSIONS, 5), "Yes", [], [], False, "", True)))
    return dict(run_id="offline-fixture", scenario_title="Synthetic fixture", scenario_category="test", scenario_risk_level="Low", scenario=inputs["scenario_text"], evaluation_inputs=inputs, input_signature=input_signature(inputs), individual_trial_results=records, total_expected_api_request_count=2, synthetic_data_statement="Synthetic offline fixture", **{k: inputs[k] for k in ("scenario_id", "requested_trial_count", "model", "reasoning_effort", "max_output_tokens", "expected_decision", "required_considerations")})


CRITERIA = dict(minimum_trials=1, minimum_score=16, minimum_match_rate=1.0, maximum_failure_rate=0.0, maximum_latency=10.0, maximum_tokens=100)


class ReleaseWorkflowTests(unittest.TestCase):
    def test_zero_requirements_complete_without_confirmation(self):
        review = build_review(dict.fromkeys(SCORE_DIMENSIONS, 5), "Yes", [], [], False, "", True)
        self.assertEqual(review["review_status"], "complete")
        self.assertIsNone(review["consideration_coverage"])

    def test_balanced_order_for_both_random_starts(self):
        for start in ("A", "B"):
            positions = [execution_order(i, start)[0] for i in range(1, 5)]
            self.assertEqual(positions.count("A"), 2)
            self.assertEqual(positions.count("B"), 2)

    def test_output_classification(self):
        for status, text, expected in (("completed", "ok", "success"), ("completed", " ", "empty"), ("incomplete", "partial", "incomplete")):
            self.assertEqual(output_status(SimpleNamespace(status=status, output_text=text)), expected)
        self.assertNotEqual(prompt_identity("a"), prompt_identity("a "))

    def test_import_roundtrip_and_signature_rejection(self):
        run = fixture()
        restored = import_evidence(json.loads(json.dumps({"runs": [run]})))[run["run_id"]]
        self.assertEqual(restored["individual_trial_results"][0]["total_human_score"], 20)
        run["evaluation_inputs"]["configuration_a_prompt"] = "tampered"
        with self.assertRaises(ValueError):
            import_evidence(run)

    def test_release_criteria_detect_blockers_and_missing_coverage(self):
        run = fixture()
        self.assertEqual(release_report([run], [run["scenario_id"]], CRITERIA)["recommendation"], "No clear winner")
        run["individual_trial_results"][1]["unacceptable_behavior"] = True
        self.assertEqual(release_report([run], [run["scenario_id"]], CRITERIA)["recommendation"], "Keep A")
        self.assertEqual(release_report([run], ["missing"], CRITERIA)["recommendation"], "Insufficient evidence")

    def test_import_rejects_inconsistent_evidence(self):
        for field, value in (("response", " "), ("model", "different"), ("trial_number", True)):
            run = fixture()
            run["individual_trial_results"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                import_evidence(run)
        with self.assertRaises(ValueError):
            import_evidence({"runs": [fixture()], "release_decision": "invalid"})

    def test_incomplete_reviews_and_mixed_prompts_block_release(self):
        run = fixture()
        run["individual_trial_results"][1]["review_status"] = "not_reviewed"
        self.assertEqual(release_report([run], [run["scenario_id"]], CRITERIA)["recommendation"], "Insufficient evidence")
        changed = copy.deepcopy(run)
        changed["evaluation_inputs"]["configuration_b_prompt"] = "changed"
        self.assertEqual(release_report([run, changed], [run["scenario_id"]], CRITERIA)["recommendation"], "Insufficient evidence")

    def test_failure_cannot_hide_behind_high_scores(self):
        run = fixture()
        failed = copy.deepcopy(run["individual_trial_results"][1])
        failed["status"] = "incomplete"
        run["individual_trial_results"].append(failed)
        self.assertEqual(release_report([run], [run["scenario_id"]], CRITERIA)["recommendation"], "Keep A")

    def test_offline_resume_and_blind_review(self):
        from streamlit.testing.v1 import AppTest
        with patch("openai.OpenAI", side_effect=AssertionError("API client must not initialize")):
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"), default_timeout=15).run()
            self.assertFalse(app.exception)
            run = import_evidence(fixture())["offline-fixture"]
            app.session_state["saved_runs"] = {run["run_id"]: run}
            app.run()
            next(b for b in app.button if b.label == "Resume selected review").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["evaluation"]["individual_trial_results"][0]["total_human_score"], 20)
            next(c for c in app.checkbox if c.label.startswith("Blind response review")).uncheck().run()
            self.assertFalse(app.exception)
            self.assertTrue(any(b.label == "Record release decision" for b in app.button))
            self.assertTrue(any("Insufficient evidence" in x.value for x in app.info))
            self.assertFalse(any("stale because" in w.value for w in app.warning))
            next(w for w in app.text_input if w.label == "Decision owner").set_value("Synthetic reviewer")
            next(w for w in app.selectbox if w.label == "PM decision").select("Insufficient evidence")
            next(w for w in app.text_area if w.label.startswith("Release decision rationale")).set_value("More scenario evidence is needed.")
            app.run()
            next(b for b in app.button if b.label == "Record release decision").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["release_decision"]["decision"], "Insufficient evidence")
            next(w for w in app.number_input if w.label.startswith("Maximum mean latency")).set_value(9.0).run()
            self.assertTrue(any("decision record is stale" in w.value for w in app.warning))
            next(c for c in app.checkbox if c.label.startswith("Blind response review")).check().run()
            self.assertTrue(any("Response X" in e.label for e in app.expander))
            app.text_area(key="prompt_a").set_value("changed").run()
            self.assertFalse(app.exception)
            self.assertTrue(any("stale because" in w.value for w in app.warning))
            self.assertEqual(app.session_state["evaluation"]["individual_trial_results"][0]["total_human_score"], 20)


if __name__ == "__main__":
    unittest.main()
