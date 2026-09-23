import io
import json
import os
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

import pandas as pd
from agent import Agent
from llm_planner import LLMPlanner
from mock_environment import make_mock_env
from scoring_core import apply_filters


class PlannerTests(unittest.TestCase):
    def test_authentication_error_is_reported_without_secret(self):
        with tempfile.TemporaryDirectory() as cache, patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-secret", "AGENT_LLM_CACHE": cache, "AGENT_LLM_MODE": "auto"}), patch(
                "urllib.request.urlopen", side_effect=urllib.error.HTTPError(
                    "https://api.openai.com/v1/responses", 401, "test-secret", {}, None)):
            planner = LLMPlanner()
            self.assertEqual(planner.choose({"candidates": []}), [])
            self.assertEqual(planner.events[-1]["http_status"], 401)
            self.assertNotIn("test-secret", json.dumps(planner.events))

    def test_replay_miss_never_calls_network(self):
        with tempfile.TemporaryDirectory() as cache, patch.dict(os.environ, {
            "AGENT_LLM_CACHE": cache, "AGENT_LLM_MODE": "replay"}), patch("urllib.request.urlopen") as api:
            self.assertEqual(LLMPlanner().choose({"candidates": []}), [])
            api.assert_not_called()

    def test_malformed_response_falls_back(self):
        with tempfile.TemporaryDirectory() as cache, patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-only", "AGENT_LLM_CACHE": cache, "AGENT_LLM_MODE": "auto"}), patch(
                "urllib.request.urlopen", return_value=io.BytesIO(b"not json")):
            self.assertEqual(LLMPlanner().choose({"candidates": []}), [])

    def test_api_validation_and_replay(self):
        context = {"candidates": [{"candidate_id": "valid"}]}
        answer = {"experiments": [{"candidate_id": "invented", "n_customers": 150},
                                   {"candidate_id": "valid", "n_customers": 150}], "reason": "test"}
        body = {"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps(answer)}]}]}
        with tempfile.TemporaryDirectory() as cache, patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-only", "AGENT_LLM_CACHE": cache, "AGENT_LLM_MODE": "auto"}):
            with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(body).encode())) as api:
                self.assertEqual(LLMPlanner().choose(context), [("valid", 150)])
                payload = json.loads(api.call_args.args[0].data)
                self.assertTrue(payload["text"]["format"]["strict"])
            with patch.dict(os.environ, {"AGENT_LLM_MODE": "replay"}), patch("urllib.request.urlopen") as api:
                self.assertEqual(LLMPlanner().choose(context), [("valid", 150)])
                api.assert_not_called()

    def test_timeout_disables_further_calls(self):
        with tempfile.TemporaryDirectory() as cache, patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-only", "AGENT_LLM_CACHE": cache, "AGENT_LLM_MODE": "auto"}):
            planner = LLMPlanner()
            with patch("urllib.request.urlopen", side_effect=TimeoutError) as api:
                self.assertEqual(planner.choose({"candidates": []}), [])
                self.assertEqual(planner.choose({"candidates": []}), [])
                self.assertEqual(api.call_count, 1)

    def test_refusal_falls_back(self):
        body = {"status": "completed", "output": [{"type": "message", "content": [
            {"type": "refusal", "refusal": "no"}]}]}
        with tempfile.TemporaryDirectory() as cache, patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-only", "AGENT_LLM_CACHE": cache, "AGENT_LLM_MODE": "auto"}), patch(
                "urllib.request.urlopen", return_value=io.BytesIO(json.dumps(body).encode())):
            self.assertEqual(LLMPlanner().choose({"candidates": []}), [])


class AgentTests(unittest.TestCase):
    def test_confirmation_repeats_measured_candidates(self):
        with patch.dict(os.environ, {"AGENT_LLM_MODE": "off"}):
            env, _ = make_mock_env(seed=42)
            agent = Agent()
            agent.act(env)
        observed = set()
        confirmations = 0
        for pilot in agent.report["pilots"]:
            if pilot["phase"] == "confirm":
                self.assertIn(pilot["candidate_id"], observed)
                confirmations += 1
            observed.add(pilot["candidate_id"])
        self.assertGreater(confirmations, 0)

    def test_api_failure_does_not_change_offline_plan(self):
        with patch.dict(os.environ, {"AGENT_LLM_MODE": "off"}):
            env, _ = make_mock_env(seed=42)
            expected = Agent().act(env)
        with tempfile.TemporaryDirectory() as cache, patch.dict(os.environ, {
            "OPENAI_API_KEY": "test-only", "AGENT_LLM_MODE": "auto", "AGENT_LLM_CACHE": cache}), patch(
                "urllib.request.urlopen", side_effect=TimeoutError):
            env, _ = make_mock_env(seed=42)
            agent = Agent()
            self.assertEqual(agent.act(env), expected)
            self.assertEqual(agent.report["llm"][0]["reason"], "TimeoutError")

    def test_fallback_limits_and_reproducibility(self):
        with patch.dict(os.environ, {"AGENT_LLM_MODE": "off"}):
            env, _ = make_mock_env(seed=42)
            campaigns = Agent().act(env)
            env2, _ = make_mock_env(seed=42)
            self.assertEqual(campaigns, Agent().act(env2))
        self.assertTrue(1 <= len(campaigns) <= 10)
        self.assertTrue(0 < len(env.pilot_history) <= 20)
        cost, contacts, ids = 0, 0, set()
        for campaign in campaigns:
            segment = apply_filters(env.customer_profile, pd.Series(campaign))
            self.assertTrue(0 < len(segment) <= 5000)
            self.assertFalse(ids.intersection(segment.ID_NUMBER))
            ids.update(segment.ID_NUMBER)
            contacts += len(segment)
            cost += len(segment) * env.channels[campaign["channel"]]["cost_per_contact"]
        self.assertLessEqual(cost, env.remaining_budget)
        self.assertLessEqual(contacts, env.remaining_contacts)

    def test_llm_choice_reaches_pilot(self):
        class StubPlanner:
            events = []
            selected = None

            def choose(self, context):
                cid = context["candidates"][-1]["candidate_id"]
                if self.selected is None:
                    self.selected = cid
                return [(cid, 123)]
        planner = StubPlanner()
        env, _ = make_mock_env(seed=42)
        agent = Agent(planner=planner)
        agent.act(env)
        self.assertEqual(agent.report["pilots"][0]["candidate_id"], planner.selected)
        self.assertEqual(agent.report["pilots"][0]["n"], 123)


if __name__ == "__main__":
    unittest.main()
