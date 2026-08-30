import json
import re
import unittest
from pathlib import Path


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
        source = (ROOT / "app.py").read_text(encoding="utf-8")
        compile(source, "app.py", "exec")


if __name__ == "__main__":
    unittest.main()
