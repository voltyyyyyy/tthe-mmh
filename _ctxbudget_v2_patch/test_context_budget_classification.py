import json
import tempfile
import unittest
from pathlib import Path

from experiments_tau3_mmh_study.full_study import capability_limit_errors, inspect_record


class ContextBudgetClassificationTest(unittest.TestCase):
    def _record(self):
        return {
            "task_name": "banking_knowledge_task_040",
            "reward": 0.0,
            "num_turns": 35,
            "wall_time_s": 1.0,
            "trial_dir": "/tmp/ran",
        }

    def test_response_backed_context_limit_is_scored_capability_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory)
            events = [
                {"type": "model_error", "error": "HTTPStatusError: 400",
                 "http_status_code": 400,
                 "response_body": '{"error":{"message":"maximum context length is 131072"}}'},
                {"type": "grading", "reward": 0.0},
            ]
            (trace / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            error, parsed = inspect_record(self._record(), trace)
            self.assertIsNone(error)
            self.assertEqual(len(capability_limit_errors(parsed)), 1)

    def test_connection_failure_remains_excluded_as_transport(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory)
            events = [
                {"type": "agent_error", "error": "ConnectError: connection refused"},
                {"type": "grading", "reward": 0.0},
            ]
            (trace / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            error, _ = inspect_record(self._record(), trace)
            self.assertIn("ConnectError", error)

    def test_unknown_http_400_remains_excluded_as_infrastructure(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory)
            events = [
                {"type": "model_error", "error": "HTTPStatusError: 400",
                 "http_status_code": 400, "response_body": '{"error":{"message":"bad schema"}}'},
                {"type": "agent_error", "error": "HTTPStatusError: 400"},
                {"type": "grading", "reward": 0.0},
            ]
            (trace / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            error, parsed = inspect_record(self._record(), trace)
            self.assertIn("HTTPStatusError", error)
            self.assertFalse(capability_limit_errors(parsed))

    def test_http_500_remains_excluded_as_infrastructure(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory)
            events = [
                {"type": "model_error", "error": "HTTPStatusError: 500",
                 "http_status_code": 500,
                 "response_body": '{"error":{"message":"maximum context length is 131072"}}'},
                {"type": "agent_error", "error": "HTTPStatusError: 500"},
                {"type": "grading", "reward": 0.0},
            ]
            (trace / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            error, parsed = inspect_record(self._record(), trace)
            self.assertIn("HTTPStatusError", error)
            self.assertFalse(capability_limit_errors(parsed))


if __name__ == "__main__":
    unittest.main()
