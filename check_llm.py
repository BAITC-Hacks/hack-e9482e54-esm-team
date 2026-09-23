"""Explicit live smoke test: bypass cache and print only safe status fields."""
import json
import os
import tempfile

from llm_planner import LLMPlanner


def main():
    with tempfile.TemporaryDirectory() as cache:
        os.environ["AGENT_LLM_CACHE"] = cache
        os.environ["AGENT_LLM_MODE"] = "auto"
        planner = LLMPlanner()
        choices = planner.choose({"phase": "explore", "round": 1, "pilots_left": 20,
                                  "remaining_budget": 100000, "remaining_contacts": 15000,
                                  "observations": [], "candidates": [{
                                      "candidate_id": "smoke_test", "size": 1000,
                                      "revenue": 5000000, "mean": 0.05, "se": 0.15,
                                      "n": 0, "history_count": 100}]})
        event = planner.events[-1] if planner.events else {}
        passed = event.get("source") == "api" and bool(choices)
        print(json.dumps({"model": planner.model, "passed": passed,
                          "source": event.get("source"), "http_status": event.get("http_status"),
                          "error": event.get("reason") if event.get("source") == "fallback" else None}))
        return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
