"""Paired online/offline evaluation, with an offline replay check per seed."""
import argparse
import json
import os
from pathlib import Path
import statistics

import pandas as pd

from agent import Agent
from local_eval import evaluate_agent
from make_submission import CAMPAIGN_COLUMNS
from verify_strategy import audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 10, 11])
    parser.add_argument("--output", default="llm_validation_results.json")
    parser.add_argument("--replay-only", action="store_true", help="Use existing cache; never call API")
    args = parser.parse_args()
    report = {"model": os.getenv("OPENAI_MODEL", "gpt-4.1-mini"), "runs": []}
    for seed in args.seeds:
        os.environ["AGENT_LLM_MODE"] = "off"
        offline = evaluate_agent(Agent(), seed=seed, verbose=False)
        audit(offline)
        os.environ["AGENT_LLM_MODE"] = "replay" if args.replay_only else "auto"
        agent = Agent()
        online = evaluate_agent(agent, seed=seed, verbose=False)
        audit(online)
        sources = [event["source"] for event in agent.report["llm"]]
        fully_llm = bool(sources) and all(source in {"api", "cache"} for source in sources)
        os.environ["AGENT_LLM_MODE"] = "replay"
        replay_agent = Agent()
        replay = evaluate_agent(replay_agent, seed=seed, verbose=False)
        audit(replay)
        replay_matches = (replay_agent.report["campaigns"] == agent.report["campaigns"]
                          and replay_agent.report["pilots"] == agent.report["pilots"]
                          and replay["net_arpu_gain"] == online["net_arpu_gain"])
        row = {"seed": seed, "offline_net": offline["net_arpu_gain"],
               "online_net": online["net_arpu_gain"],
               "difference": online["net_arpu_gain"] - offline["net_arpu_gain"],
               "fully_llm": fully_llm, "sources": sources, "replay_matches": replay_matches,
               "online_contacts": online["total_contacts"], "online_cost": online["total_cost"],
               "decisions": agent.report}
        report["runs"].append(row)
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "decisions"}), flush=True)
        if not fully_llm or not replay_matches:
            raise SystemExit("Incomplete LLM/replay verification; partial report saved.")
        if seed == 42:
            pd.DataFrame(agent.report["campaigns"]).reindex(columns=CAMPAIGN_COLUMNS).to_csv(
                "submission_llm.csv", index=False)
    report["summary"] = {
        "offline_median": statistics.median(row["offline_net"] for row in report["runs"]),
        "online_median": statistics.median(row["online_net"] for row in report["runs"]),
        "llm_wins": sum(row["difference"] > 0 for row in report["runs"]),
        "paired_median_difference": statistics.median(row["difference"] for row in report["runs"])}
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
