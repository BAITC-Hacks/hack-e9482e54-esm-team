"""Save the agent's decision trace alongside the official evaluation."""
import json
from pathlib import Path

from agent import Agent
from local_eval import evaluate_agent


if __name__ == "__main__":
    agent = Agent()
    result = evaluate_agent(agent, seed=42)
    Path("run_report.json").write_text(json.dumps(
        {"decisions": agent.report, "result": result}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Decision trace: run_report.json")
