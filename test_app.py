import os
import io
import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from campaign_app_service import demo_data, identify_uploads, read_csv, run_simulation, validate_data
from llm_planner import LLMPlanner
from campaign_explanations import explanation_context, explain_campaigns, FIELDS


class ServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profile, cls.history, cls.tariffs = demo_data()

    def test_missing_columns_and_duplicate_ids_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "отсутствуют"):
            validate_data(self.profile.drop(columns="predicted_arpu"), self.history, self.tariffs)
        duplicate = pd.concat([self.profile, self.profile.head(1)], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "уникальны"):
            validate_data(duplicate, self.history, self.tariffs)

    def test_batch_recognition_uses_columns_not_names_or_order(self):
        files = [("random1.csv", self.tariffs.to_csv(index=False).encode()),
                 ("random2.csv", self.profile.head(10).to_csv(index=False).encode()),
                 ("random3.csv", self.history.head(10).to_csv(index=False).encode()),
                 ("extra.csv", b"unused,value\na,1\n")]
        found, inventory = identify_uploads(files)
        self.assertEqual(set(found), {"profile", "history", "tariffs"})
        self.assertEqual(len(found["tariffs"]), len(self.tariffs))
        self.assertEqual(inventory[-1]["Назначение"], "Не используется — не совпадают обязательные колонки")
        with self.assertRaisesRegex(ValueError, "несколько таблиц"):
            identify_uploads(files[:1] * 2)
        ambiguous = self.profile.head(1).assign(tariff_plan_code="tariff_1", price_tariff=100)
        with self.assertRaisesRegex(ValueError, "нескольких таблиц"):
            identify_uploads([("ambiguous.csv", ambiguous.to_csv(index=False).encode())])

    def test_invalid_numeric_and_tariff_are_rejected(self):
        profile = self.profile.copy()
        profile.loc[0, "predicted_arpu"] = float("inf")
        with self.assertRaisesRegex(ValueError, "конечные"):
            validate_data(profile, self.history, self.tariffs)
        profile = self.profile.copy()
        profile.loc[0, "current_tariff"] = "unknown_tariff"
        with self.assertRaisesRegex(ValueError, "неизвестный"):
            validate_data(profile, self.history, self.tariffs)

    def test_uploaded_csv_and_history_are_used_without_file_reads(self):
        uploaded = read_csv(self.profile.to_csv(index=False, sep=";").encode())
        self.assertEqual(len(uploaded), len(self.profile))
        original_env = os.environ.get("AGENT_LLM_MODE")
        with patch("pandas.read_csv", side_effect=AssertionError("Unexpected disk read")), patch("urllib.request.urlopen") as network:
            result = run_simulation(self.profile, self.history, self.tariffs)
        network.assert_not_called()
        self.assertEqual(os.environ.get("AGENT_LLM_MODE"), original_env)
        self.assertEqual(result["csv"], Path("submission.csv").read_text())
        self.assertLessEqual(result["score"]["total_cost"], 100000)
        self.assertLessEqual(result["score"]["total_contacts"], 15000)


class InterfaceTests(unittest.TestCase):
    @staticmethod
    def api_response(request, **kwargs):
        context = json.loads(json.loads(request.data)["input"])
        if "campaigns" in context:
            answer = {"campaigns": [{"campaign_id": c["campaign_id"],
                                     **{key: "Тестовое объяснение " + key for key in FIELDS}}
                                    for c in context["campaigns"]]}
        else:
            answer = {"experiments": [{"candidate_id": c["candidate_id"], "n_customers": 100}
                                       for c in context["candidates"][:5]], "reason": "Тестовый ответ API"}
        return io.BytesIO(json.dumps({"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps(answer)}]}]}).encode())

    def test_single_dropzone_recognizes_batch_and_blocks_incomplete_input(self):
        profile, history, tariffs = demo_data()
        files = [("tariffs.csv", tariffs.to_csv(index=False).encode(), "text/csv"),
                 ("people.csv", profile.head(300).to_csv(index=False).encode(), "text/csv"),
                 ("moves.csv", history.to_csv(index=False).encode(), "text/csv")]
        with patch("urllib.request.urlopen", side_effect=self.api_response) as network:
            app = AppTest.from_file("app.py", default_timeout=30).run()
            app.text_input(key="session_api_key").set_value("test-session-key").run()
            app.checkbox(key="api_consent").check().run()
            app.radio[0].set_value("Загрузить CSV").run()
            self.assertEqual(len(app.file_uploader), 1)
            self.assertTrue(app.file_uploader[0].accept_multiple_files)
            app.file_uploader[0].set_value(files).run()
            self.assertFalse(app.exception)
            self.assertFalse(app.button(key="run_campaigns").disabled)
            app.button(key="run_campaigns").click().run()
            self.assertIn("result", app.session_state)
            app.file_uploader[0].set_value(files[:2]).run()
            self.assertTrue(app.button(key="run_campaigns").disabled)
            self.assertNotIn("result", app.session_state)
            self.assertFalse(app.exception)
            self.assertGreater(network.call_count, 0)
            self.assertLessEqual(network.call_count, 5)

    def test_session_key_enables_run_and_can_be_cleared(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), patch("urllib.request.urlopen") as network:
            app = AppTest.from_file("app.py", default_timeout=30).run()
            self.assertFalse(app.selectbox)
            field = app.text_input(key="session_api_key")
            self.assertEqual(field.proto.type, field.proto.PASSWORD)
            app.text_input(key="session_api_key").set_value("test-session-key").run()
            self.assertTrue(app.button(key="run_campaigns").disabled)
            app.checkbox(key="api_consent").check().run()
            self.assertFalse(app.button(key="run_campaigns").disabled)
            self.assertEqual(os.environ["OPENAI_API_KEY"], "")
            second = AppTest.from_file("app.py", default_timeout=30).run()
            self.assertEqual(second.text_input(key="session_api_key").value, "")
            app.button(key="clear_api_key").click().run()
            self.assertEqual(app.text_input(key="session_api_key").value, "")
            self.assertTrue(app.button(key="run_campaigns").disabled)
            self.assertFalse(app.exception)
            network.assert_not_called()

    def test_planner_uses_session_key_without_persisting_it(self):
        answer = {"experiments": [{"candidate_id": "session-test", "n_customers": 100}], "reason": "test"}
        body = {"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps(answer)}]}]}
        with tempfile.TemporaryDirectory() as cache, patch.dict(os.environ, {"OPENAI_API_KEY": "server-test-key"}), patch(
                "urllib.request.urlopen", return_value=io.BytesIO(json.dumps(body).encode())) as network:
            planner = LLMPlanner(mode="auto", cache_dir=cache, api_key="session-test-key")
            self.assertEqual(planner.choose({"candidates": [{"candidate_id": "session-test"}]}), [("session-test", 100)])
            self.assertEqual(network.call_args.args[0].get_header("Authorization"), "Bearer session-test-key")
            self.assertEqual(os.environ["OPENAI_API_KEY"], "server-test-key")
            self.assertNotIn("session-test-key", json.dumps(planner.events))
            for file in Path(cache).glob("*"):
                self.assertNotIn("session-test-key", file.read_text())

    def test_demo_end_to_end_and_result_invalidation(self):
        with patch("urllib.request.urlopen", side_effect=self.api_response) as network:
            app = AppTest.from_file("app.py", default_timeout=30).run()
            self.assertFalse(app.exception)
            app.text_input(key="session_api_key").set_value("test-session-key").run()
            app.checkbox(key="api_consent").check().run()
            app.button(key="run_campaigns").click().run()
            self.assertFalse(app.exception)
            self.assertTrue(app.session_state["result"]["plan"])
            self.assertGreater(app.session_state["result"]["score"]["n_pilots"], 0)
            self.assertLessEqual(app.session_state["result"]["score"]["n_pilots"], 20)
            self.assertEqual({e["source"] for e in app.session_state["result"]["trace"]["llm"]}, {"api"})
            explanation = app.session_state["result"]["explanation"]
            self.assertEqual(explanation["status"], "api")
            self.assertEqual(len(explanation["campaigns"]), len(app.session_state["result"]["plan"]))
            self.assertIn("Тестовое объяснение client_benefit", [item.value for item in app.text])
            app.number_input[0].set_value(43).run()
            self.assertNotIn("result", app.session_state)
            self.assertGreater(network.call_count, 0)
            self.assertLessEqual(network.call_count, 5)

    def test_explanation_failure_keeps_plan(self):
        def response(request, **kwargs):
            if json.loads(request.data)["text"]["format"]["name"] == "campaign_explanations":
                raise TimeoutError("private request details")
            return self.api_response(request, **kwargs)
        with patch("urllib.request.urlopen", side_effect=response) as network:
            result = run_simulation(*demo_data(), mode="api", api_key="test-key")
        self.assertTrue(result["plan"])
        self.assertTrue(result["csv"])
        self.assertEqual(result["explanation"]["status"], "unavailable")
        self.assertNotIn("private request details", json.dumps(result))
        self.assertLessEqual(network.call_count, 5)

    def test_explanation_context_excludes_ids_and_simulator_effects(self):
        profile, history, tariffs = demo_data()
        result = run_simulation(profile, history, tariffs)
        profile["private_note"] = "private marker"
        tariffs["private_note"] = "private marker"
        tariffs = tariffs.drop(columns=["Data_in_PKG"])
        context = explanation_context(result["plan"], profile, history, tariffs)
        serialized = json.dumps(context, allow_nan=False)
        for excluded in ("ID_NUMBER", "private marker", "Результат симуляции", "gross_lift"):
            self.assertNotIn(excluded, serialized)
        self.assertIsNone(context["campaigns"][0]["target_package"]["Data_in_PKG"])
        self.assertGreater(context["campaigns"][0]["pilot_contacts"], 0)

    def test_explanation_rejects_wrong_campaign(self):
        context = {"campaigns": [{"campaign_id": "expected"}]}
        body = {"status": "completed", "output": [{"type": "message", "content": [
            {"type": "output_text", "text": json.dumps({"campaigns": [
                {"campaign_id": "wrong", **{key: "text" for key in FIELDS}}]})}]}]}
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(body).encode())):
            self.assertEqual(explain_campaigns(context, "test-key")["status"], "unavailable")

    def test_api_mode_ignores_saved_answers_and_handles_failure(self):
        context = {"candidates": [{"candidate_id": "api-only-test"}]}
        with tempfile.TemporaryDirectory() as cache, patch("urllib.request.urlopen", side_effect=self.api_response) as network:
            # The first call creates an exact matching cache entry.
            for _ in range(2):
                planner = LLMPlanner(mode="api", cache_dir=cache, api_key="test-key")
                self.assertTrue(planner.choose(context))
                self.assertEqual(planner.events[0]["source"], "api")
            self.assertEqual(network.call_count, 2)
            network.side_effect = TimeoutError()
            self.assertEqual(planner.choose(context), [])
            self.assertEqual(planner.events[-1]["source"], "fallback")

    def test_missing_uploads_disable_run(self):
        app = AppTest.from_file("app.py", default_timeout=30).run()
        app.radio[0].set_value("Загрузить CSV").run()
        self.assertFalse(app.exception)
        self.assertTrue(app.button[0].disabled)

    def test_live_mode_requires_key_and_consent(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), patch("urllib.request.urlopen") as network:
            app = AppTest.from_file("app.py", default_timeout=30).run()
            self.assertFalse(app.selectbox)
            self.assertFalse(app.exception)
            app.checkbox(key="api_consent").check().run()
            self.assertTrue(app.button(key="run_campaigns").disabled)
            network.assert_not_called()


if __name__ == "__main__":
    unittest.main()
