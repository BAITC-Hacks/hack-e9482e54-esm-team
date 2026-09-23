"""Application adapter: validates uploads and runs the explicit demo simulator.

Simulation effects are used here only to create/evaluate env, never by Agent.
"""
import csv
from io import StringIO
from pathlib import Path
import re
import tempfile
import time

import numpy as np
import pandas as pd

from agent import Agent
from campaign_explanations import explanation_context, explain_campaigns
from environment import make_environment
from llm_planner import LLMPlanner
from make_submission import CAMPAIGN_COLUMNS
from mock_environment import _mock_impact_model, _mock_fallback
from scoring_core import CHANNELS, score_campaigns, validate_strategy

ROOT = Path(__file__).resolve().parent
PROFILE_COLUMNS = {"ID_NUMBER", "current_tariff", "arpu_segment", "data_segment", "call_segment", "predicted_arpu"}
HISTORY_COLUMNS = {"ID_NUMBER", "tariff_plan_code_from", "tariff_plan_code_to", "AVG_ARPU_PREV_3M", "AVG_ARPU_NEXT_3M"}
TARIFF_COLUMNS = {"tariff_plan_code", "price_tariff"}
UPLOAD_SCHEMAS = {
    "profile": ("Профиль абонентов", PROFILE_COLUMNS),
    "history": ("История переходов", HISTORY_COLUMNS),
    "tariffs": ("Справочник тарифов", TARIFF_COLUMNS),
}
KNOWN_COLUMNS = PROFILE_COLUMNS | HISTORY_COLUMNS | TARIFF_COLUMNS | {
    "TIME_KEY", "Data_in_PKG", "Min_another_operator_in_PKG",
    "Min_another_operator_and_city_in_PKG",
}


def normalize_column(name):
    cleaned = str(name).replace("\ufeff", "").strip()
    key = re.sub(r"[\s_-]+", "_", cleaned).casefold()
    return next((column for column in KNOWN_COLUMNS if column.casefold() == key), cleaned)


def read_csv(data):
    if len(data) > 20 * 1024 * 1024:
        raise ValueError("Размер одного CSV не должен превышать 20 МБ.")
    try:
        if data.startswith((b'\xff\xfe', b'\xfe\xff')):
            text = data.decode("utf-16")
        else:
            try:
                text = data.decode("utf-8-sig")
            except UnicodeDecodeError:
                text = data.decode("cp1251")
        text = text.lstrip("\r\n")
        separator = None
        first, _, rest = text.partition("\n")
        if re.fullmatch(r"sep=[,;\t|]", first.strip(), re.IGNORECASE):
            separator, text = first.strip()[4:], rest
        if separator is None:
            try:
                separator = csv.Sniffer().sniff(text[:65536], delimiters=",;\t|").delimiter
            except csv.Error:
                separator = csv.Sniffer().sniff(text.splitlines()[0], delimiters=",;\t|").delimiter
        header = next(csv.reader(StringIO(text), delimiter=separator))
        columns = [normalize_column(name) for name in header]
        if len(set(columns)) != len(columns):
            raise ValueError("CSV: названия колонок дублируются после удаления пробелов и нормализации регистра.")
        frame = pd.read_csv(StringIO(text), sep=separator, skipinitialspace=True)
        if len(frame.columns) != len(columns):
            raise ValueError("CSV: число колонок заголовка не совпадает с данными.")
        frame.columns = columns
        return frame
    except (UnicodeError, csv.Error, IndexError, StopIteration, pd.errors.ParserError) as exc:
        raise ValueError("Не удалось прочитать CSV. Поддерживаются UTF-8, UTF-16 и Windows-1251; разделители: запятая, точка с запятой, табуляция и |.") from exc


def demo_data():
    return (pd.read_csv(ROOT / "customer_profile.csv"), pd.read_csv(ROOT / "data/change_tariff.csv"),
            pd.read_csv(ROOT / "data/dict_tariff.csv"))


def identify_uploads(files, assignments=None, column_maps=None):
    """Match tables by columns, independently of file names and upload order."""
    found, inventory = {}, []
    assignments, column_maps = assignments or {}, column_maps or {}
    for index, (name, data) in enumerate(files):
        assigned = assignments.get(index)
        if assigned == "ignore":
            inventory.append({"Файл": name, "Назначение": "Пропущен пользователем", "Строк": None})
            continue
        frame = read_csv(data)
        mapping = column_maps.get(index, {})
        if mapping:
            if any(source not in frame.columns for source in mapping.values()):
                raise ValueError(f"Файл «{name}»: выбранная колонка не найдена.")
            if len(set(mapping.values())) != len(mapping):
                raise ValueError(f"Файл «{name}»: одну колонку нельзя использовать для нескольких полей.")
            frame = frame.rename(columns={source: target for target, source in mapping.items()})
            if frame.columns.duplicated().any():
                raise ValueError(f"Файл «{name}»: сопоставление создаёт повторяющиеся колонки.")
        matches = [role for role, (_, required) in UPLOAD_SCHEMAS.items()
                   if required.issubset(frame.columns)]
        if assigned in UPLOAD_SCHEMAS:
            missing = UPLOAD_SCHEMAS[assigned][1] - set(frame.columns)
            if missing:
                inventory.append({"Файл": name, "Назначение": "Не используется — сопоставьте колонки: " + ", ".join(sorted(missing)), "Строк": len(frame)})
                continue
            matches = [assigned]
        if len(matches) > 1:
            raise ValueError(f"Файл «{name}» подходит сразу для нескольких таблиц. Разделите данные по назначению.")
        if not matches:
            inventory.append({"Файл": name, "Назначение": "Не используется — не совпадают обязательные колонки", "Строк": len(frame)})
            continue
        role = matches[0]
        if role in found:
            raise ValueError(f"Загружено несколько таблиц «{UPLOAD_SCHEMAS[role][0]}». Оставьте одну.")
        found[role] = frame
        inventory.append({"Файл": name, "Назначение": UPLOAD_SCHEMAS[role][0], "Строк": len(frame)})
    return found, inventory


def validate_data(profile, history, tariffs):
    frames = [profile.copy(), history.copy(), tariffs.copy()]
    for frame, columns, label in zip(frames, [PROFILE_COLUMNS, HISTORY_COLUMNS, TARIFF_COLUMNS],
                                      ["Профиль", "История", "Тарифы"]):
        missing = columns - set(frame.columns)
        if missing:
            raise ValueError(f"{label}: отсутствуют колонки {', '.join(sorted(missing))}.")
        if frame.empty or len(frame) > 100000:
            raise ValueError(f"{label}: требуется от 1 до 100 000 строк.")
    profile, history, tariffs = frames
    for frame, columns in ((profile, ["current_tariff", "arpu_segment", "data_segment", "call_segment"]),
                           (history, ["tariff_plan_code_from", "tariff_plan_code_to"]),
                           (tariffs, ["tariff_plan_code"])):
        for column in columns:
            frame[column] = frame[column].astype("string").str.strip()
            if column.endswith("_segment"):
                frame[column] = frame[column].str.upper()
    for frame, columns, label in ((profile, ["predicted_arpu"], "Профиль"),
                                  (history, ["AVG_ARPU_PREV_3M", "AVG_ARPU_NEXT_3M"], "История"),
                                  (tariffs, ["price_tariff"], "Тарифы")):
        for column in columns:
            if not pd.api.types.is_numeric_dtype(frame[column]):
                frame[column] = (frame[column].astype("string")
                    .str.replace("\u00a0", "", regex=False).str.replace("\u202f", "", regex=False)
                    .str.replace(r"\s+", "", regex=True).str.replace(",", ".", regex=False))
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
            if not np.isfinite(frame[column].to_numpy(dtype=float, na_value=np.nan)).all():
                raise ValueError(f"{label}: {column} должна содержать конечные числа.")
            if label != "История" and (frame[column] < 0).any():
                raise ValueError(f"{label}: {column} не должна содержать отрицательные числа.")
    if profile.ID_NUMBER.isna().any() or profile.ID_NUMBER.duplicated().any():
        raise ValueError("Профиль: ID_NUMBER должны быть заполнены и уникальны.")
    if tariffs.tariff_plan_code.isna().any() or tariffs.tariff_plan_code.duplicated().any() or len(tariffs) < 2:
        raise ValueError("Нужны минимум два тарифа с уникальными заполненными кодами.")
    codes = set(tariffs.tariff_plan_code.astype(str))
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", code) for code in codes):
        raise ValueError("Коды тарифов: используйте латинские буквы, цифры, дефис и подчёркивание.")
    for frame, column in ((profile, "current_tariff"), (history, "tariff_plan_code_from"), (history, "tariff_plan_code_to")):
        if (column != "current_tariff" and frame[column].isna().any()) or not set(frame[column].dropna()).issubset(codes):
            raise ValueError(f"Колонка {column} содержит пустой или неизвестный код тарифа.")
    notices = []
    if profile.current_tariff.isna().any():
        notices.append(f"Текущий тариф не заполнен у {int(profile.current_tariff.isna().sum())} абонентов; агент не выбирает их для кампаний.")
    if (history.AVG_ARPU_PREV_3M < 100).any():
        notices.append(f"История: {int((history.AVG_ARPU_PREV_3M < 100).sum())} строк с ARPU до перехода менее 100 не используются для оценки эффекта.")
    for column, allowed in (("arpu_segment", {"LOW", "MID", "HIGH"}),
                             ("data_segment", {"NON_USER", "LITE", "HEAVY"}),
                             ("call_segment", {"LOW", "MEDIUM", "HIGH"})):
        if not set(profile[column].dropna()).issubset(allowed):
            raise ValueError(f"Профиль: неизвестные значения {column}.")
        if profile[column].isna().any():
            notices.append(f"{column}: пропусков {int(profile[column].isna().sum())}; такие строки не участвуют в соответствующей сегментации.")
    sizes = profile.groupby(["current_tariff", "arpu_segment"], observed=True).size()
    if sizes.empty or sizes.max() < 10:
        raise ValueError("Нужна хотя бы одна группа текущий тариф × ARPU-сегмент с 10 абонентами.")
    if not (history.AVG_ARPU_PREV_3M >= 100).any():
        raise ValueError("Для демонстрационной модели нужна история с ARPU до перехода не менее 100.")
    return profile, history, tariffs, notices


def run_simulation(profile, history, tariffs, mode="replay", seed=42, on_pilot=None, api_key=None):
    started = time.monotonic()
    if mode not in {"replay", "auto", "off", "api"}:
        raise ValueError("Неизвестный режим агента.")
    profile, history, tariffs, notices = validate_data(profile, history, tariffs)
    model = _mock_impact_model(history)
    env, records = make_environment(profile, model, tariffs, CHANNELS, 100000, 15000, _mock_fallback, seed=seed)
    # Upload-derived request/response caches are isolated and removed after the run.
    with tempfile.TemporaryDirectory() as cache:
        planner = LLMPlanner(mode=mode, cache_dir=cache, api_key=api_key)
        agent = Agent(planner=planner, history=history, on_pilot=on_pilot)
        campaigns = agent.act(env)
    if campaigns:
        validate_strategy(pd.DataFrame(campaigns), tariffs)
    all_campaigns = pd.DataFrame(records.executed_pilot_campaigns() + campaigns)
    result = score_campaigns(all_campaigns, profile, model, tariffs,
                             float(profile.predicted_arpu.sum()), _mock_fallback, team_id="demo_app")
    result["n_pilots"] = len(env.pilot_history)
    details = result["campaigns_detail"][len(env.pilot_history):]
    estimates = {r["candidate_id"]: r for r in agent.report["estimates"]}
    plan_rows = []
    for campaign, detail in zip(campaigns, details):
        cid = f"{campaign['filter_current_tariff']}:{campaign['filter_arpu_segment']}:{campaign['target_tariff']}"
        estimate = estimates[cid]
        plan_rows.append({"Кампания": campaign["campaign_name"], "Текущий тариф": campaign["filter_current_tariff"],
            "ARPU-сегмент": campaign["filter_arpu_segment"], "Интернет-сегмент": campaign.get("filter_data_segment", "Все"),
            "Целевой тариф": campaign["target_tariff"], "Канал": campaign["channel"],
            "Контакты": detail["n_contacts"], "Расходы, у.е.": detail["cost"],
            "SMS-эффект по пилотам, %": round(estimate["mean"] * 100, 2),
            "Ошибка оценки, п.п.": round(estimate["se"] * 100, 2), "Контакты в проверках": estimate["n"],
            "Результат симуляции, у.е.": round(detail["gross_lift"] - detail["cost"], 2)})
    csv = pd.DataFrame(campaigns).reindex(columns=CAMPAIGN_COLUMNS).to_csv(index=False)
    explanation = {"status": "skipped", "campaigns": []}
    if mode == "api":
        if planner.disabled or time.monotonic() - started > 540:
            explanation = {"status": "unavailable", "reason": "api_unavailable_or_time_limit", "campaigns": []}
        else:
            try:
                explanation = explain_campaigns(explanation_context(plan_rows, profile, history, tariffs), api_key)
            except Exception:
                # Explanation failure must never discard a valid campaign plan.
                explanation = {"status": "unavailable", "reason": "context_error", "campaigns": []}
    return {"score": result, "plan": plan_rows, "trace": agent.report,
            "explanation": explanation,
            "pilot_history": env.pilot_history, "notices": notices, "csv": csv,
            "seed": seed, "mode": mode, "audience_size": len(profile)}
