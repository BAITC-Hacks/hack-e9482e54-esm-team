"""LLM-directed exploration with statistical updates and constrained allocation."""
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from llm_planner import LLMPlanner


class Agent:
    def __init__(self, planner=None):
        self.planner = planner
        self.report = {}

    def act(self, env):
        started = time.monotonic()
        planner = self.planner or LLMPlanner()
        profile = env.customer_profile
        tariffs = sorted(env.tariffs["tariff_plan_code"].astype(str))
        history = self._history()
        cells, candidates = {}, {}
        for (current, segment), group in profile.groupby(
                ["current_tariff", "arpu_segment"], observed=True, sort=True):
            if segment not in {"LOW", "MID", "HIGH"} or len(group) < 10:
                continue
            cell = (str(current), str(segment))
            cells[cell] = group
            for target in tariffs:
                if target == current:
                    continue
                # History is a weak directional prior, never a conversion estimate.
                hist, count = history.get((current, segment, target), (0., 0))
                prior = float(np.clip(hist * 0.10, -0.12, 0.12))
                cid = f"{current}:{segment}:{target}"
                candidates[cid] = dict(candidate_id=cid, cell=cell, target=target,
                                       prior=prior, mean=prior, se=0.15, n=0,
                                       weighted_sum=0., history_count=count,
                                       size=len(group), revenue=float(group.predicted_arpu.sum()))

        observations = []
        for round_index in range(4):
            if not candidates or env.pilots_left <= 0 or time.monotonic() - started > 210:
                break
            phase = "explore" if round_index < 2 else "confirm"
            ranked = self._rank(candidates, phase)
            if not ranked:
                break
            # At most 3 hypotheses per cell in the LLM context.
            shortlist, per_cell = [], {}
            for c in ranked:
                if per_cell.get(c["cell"], 0) >= 3:
                    continue
                shortlist.append(c)
                per_cell[c["cell"]] = per_cell.get(c["cell"], 0) + 1
                if len(shortlist) >= 30:
                    break
            context = {"round": round_index + 1, "phase": phase,
                       "remaining_budget": float(env.remaining_budget),
                       "remaining_contacts": int(env.remaining_contacts),
                       "pilots_left": int(env.pilots_left), "pilot_channel": "sms",
                       "candidates": [{k: c[k] for k in (
                           "candidate_id", "size", "revenue", "mean", "se", "n", "history_count")}
                                      for c in shortlist], "observations": observations}
            choices = planner.choose(context)
            if not choices:
                selected_cells = set()
                for c in ranked:
                    if c["cell"] in selected_cells:
                        continue
                    choices.append((c["candidate_id"], 200))
                    selected_cells.add(c["cell"])
                    if len(choices) == 5:
                        break
            for cid, requested in choices[:5]:
                if time.monotonic() - started > 240:
                    break
                if cid not in candidates or env.pilots_left <= 0:
                    continue
                c = candidates[cid]
                cost = env.channels["sms"]["cost_per_contact"]
                n = min(max(int(requested), 10), 200, c["size"],
                        max(0, int(env.remaining_contacts) - 1000))
                if cost > 0:
                    n = min(n, max(0, int((env.remaining_budget - 10000) // cost)))
                if n < 10:
                    break
                try:
                    result = env.run_pilot(target_tariff=c["target"], channel="sms", n_customers=n,
                                           filter_current_tariff=c["cell"][0],
                                           filter_arpu_segment=c["cell"][1])
                except (RuntimeError, ValueError):
                    break
                actual = int(result["n_customers"])
                observed = float(result["observed_lift_ratio"])
                if actual <= 0 or not math.isfinite(observed):
                    continue
                c["n"] += actual
                c["weighted_sum"] += actual * observed
                precision = 1 / 0.15**2 + c["n"] / 0.804**2
                c["mean"] = (c["prior"] / 0.15**2 + c["weighted_sum"] / 0.804**2) / precision
                c["se"] = math.sqrt(1 / precision)
                observations.append(dict(candidate_id=cid, n=actual, observed=observed, phase=phase))

        campaigns = self._allocate(env, cells, candidates)
        self.report = {"llm": planner.events, "pilots": observations, "campaigns": campaigns,
                       "elapsed_seconds": round(time.monotonic() - started, 3)}
        return campaigns

    @staticmethod
    def _history():
        path = Path(__file__).parent / "data" / "change_tariff.csv"
        if not path.exists():
            return {}
        df = pd.read_csv(path)
        before = pd.to_numeric(df.AVG_ARPU_PREV_3M, errors="coerce")
        after = pd.to_numeric(df.AVG_ARPU_NEXT_3M, errors="coerce")
        df["segment"] = np.where(before < 1000, "LOW", np.where(before <= 5000, "MID", "HIGH"))
        df["ratio"] = ((after - before) / before.where(before >= 100)).clip(-1, 2)
        grouped = df.groupby(["tariff_plan_code_from", "segment", "tariff_plan_code_to"])["ratio"]
        return {key: (float(values.median()), int(values.count()))
                for key, values in grouped if values.count()}

    @staticmethod
    def _rank(candidates, phase="explore"):
        pool = list(candidates.values())
        if phase == "confirm":
            # Reserve half of the experiments for measured contenders. Untested
            # arms cannot displace confirmation merely by having a wide prior.
            pool = [c for c in pool if c["n"] and c["mean"] + 1.64*c["se"] > 0]
        def priority(c):
            if phase == "confirm":
                return c["revenue"] * c["se"] * max(0., c["mean"] + c["se"])
            # Repeated pilots compete with unexplored alternatives, with diminishing returns.
            return c["revenue"] * max(0., c["mean"] + 1.5*c["se"]) / (1 + c["n"] / 150)
        return sorted(pool, key=lambda c: (-priority(c), c["candidate_id"]))

    @staticmethod
    def _allocate(env, cells, candidates):
        options = []
        sms_multiplier = env.channels["sms"]["conversion_multiplier"]
        for c in candidates.values():
            if not c["n"]:
                continue
            group = cells[c["cell"]]
            # Filters must describe the full selected audience, never arbitrary IDs.
            pieces = [(None, group)] if len(group) <= 5000 else list(group.groupby("data_segment"))
            for data_segment, piece in pieces:
                if not 0 < len(piece) <= 5000:
                    continue
                for channel, config in env.channels.items():
                    multiplier = config["conversion_multiplier"] / sms_multiplier
                    if channel == "call":
                        # Unknown conversion saturation: conservative extrapolation.
                        multiplier = min(multiplier, 1 / sms_multiplier)
                    conservative = c["mean"] - 1.64*c["se"]
                    gain = conservative * multiplier * float(piece.predicted_arpu.sum())
                    cost = len(piece) * config["cost_per_contact"]
                    options.append(dict(cell=c["cell"], data=data_segment, n=len(piece),
                                        gain=gain, cost=cost, target=c["target"], channel=channel))
        money, contacts = float(env.remaining_budget), int(env.remaining_contacts)
        selected, used = [], set()
        while len(selected) < 10:
            feasible = [o for o in options if (o["cell"], o["data"]) not in used
                        and o["cost"] <= money and o["n"] <= contacts and o["gain"] > o["cost"]]
            if not feasible:
                break
            # Shadow price reserves money for other campaigns; this is a heuristic, not exact optimization.
            shadow = max(1., contacts * 4 / max(money, 1))
            chosen = max(feasible, key=lambda o: (o["gain"] - shadow*o["cost"], -o["cost"]))
            selected.append(chosen)
            used.add((chosen["cell"], chosen["data"]))
            money -= chosen["cost"]
            contacts -= chosen["n"]
        if not selected:
            # Mandatory nonempty output: available tested cell with least pessimistic push.
            feasible = [o for o in options if o["channel"] == "push" and o["n"] <= contacts]
            if feasible:
                selected = [max(feasible, key=lambda o: o["gain"])]
        result = []
        for i, o in enumerate(selected):
            campaign = dict(campaign_name=f"campaign_{i+1}", filter_current_tariff=o["cell"][0],
                            filter_arpu_segment=o["cell"][1], target_tariff=o["target"], channel=o["channel"])
            if o["data"] is not None:
                campaign["filter_data_segment"] = str(o["data"])
            result.append(campaign)
        return result
