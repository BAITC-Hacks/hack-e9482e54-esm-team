"""Reproducible offline benchmark and hard-constraint audit; no API calls."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

from agent import Agent
from agent_template import Agent as TemplateAgent
from local_eval import evaluate_agent


def audit(result):
    assert result is not None
    assert 1 <= result["n_pilots"] <= 20
    assert 1 <= result["n_campaigns"] - result["n_pilots"] <= 10
    assert result["total_contacts"] <= 15000
    assert result["total_cost"] <= 100000
    for campaign in result["campaigns_detail"]:
        assert campaign["n_contacts"] <= 5000
        assert not any(campaign.get(key, False) for key in (
            "capped_at_campaign_limit", "capped_at_reach_budget", "capped_at_money_budget"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=10)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--output", default="validation_results.json")
    args = parser.parse_args()
    os.environ["AGENT_LLM_MODE"] = "off"
    report = {"mode": "off", "seed_start": args.start, "runs": args.runs, "agents": {}}
    for name, cls in (("agent", Agent), ("template", TemplateAgent)):
        rows = []
        for seed in range(args.start, args.start + args.runs):
            agent = cls()
            started = time.monotonic()
            result = evaluate_agent(agent, seed=seed, verbose=False)
            if name == "agent":
                audit(result)
            rows.append({"seed": seed, "net": result["net_arpu_gain"],
                         "cost": result["total_cost"], "contacts": result["total_contacts"],
                         "unique_customers": result["unique_customers_targeted"],
                         "pilots": result["n_pilots"], "seconds": time.monotonic() - started})
        nets = [row["net"] for row in rows]
        summary = {"median_net": statistics.median(nets), "min_net": min(nets),
                   "max_net": max(nets), "positive_runs": sum(net > 0 for net in nets),
                   "max_seconds": max(row["seconds"] for row in rows)}
        report["agents"][name] = {"summary": summary, "runs": rows}
        print(name, json.dumps(summary), flush=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
