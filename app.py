"""Analyst workspace. Run: streamlit run app.py"""
import hashlib
import json
import os

import pandas as pd
import streamlit as st

from campaign_app_service import demo_data, identify_uploads, run_simulation, validate_data, UPLOAD_SCHEMAS

st.set_page_config(page_title="Планировщик тарифных кампаний", page_icon="🐝", layout="wide")
st.markdown("""<style>
/* Native theme also styles canvas tables, menus and inputs; keep CSS scoped. */
[data-testid="stMetric"] {
    background: var(--secondary-background-color, #FFFFFF);
    border: 1px solid var(--border-color, #DCDAD2);
    border-top: 3px solid #FFD633;
    border-radius: 12px;
    padding: 18px;
}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    color: var(--text-color, #24262D);
}
[data-testid="stMetricValue"] {font-weight: 650;}
[data-testid="stMain"] [data-testid="stCaptionContainer"] {color: #666A73;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color: #BDC0C9;}
button[kind="primary"] {
    background: #FFD633;
    color: #24262D;
    border: 1px solid #FFD633;
    font-weight: 600;
}
button[kind="primary"]:hover:not(:disabled) {
    background: #F0C31A;
    border-color: #F0C31A;
    color: #24262D;
}
button[kind="primary"]:disabled {
    background: #EEE8CF;
    border-color: #DDD5B6;
    color: #716B58;
    opacity: 1;
}
[data-testid="stMain"] [data-testid="stDownloadButton"] button {
    background: #24262D;
    border-color: #24262D;
    color: #FFFFFF;
}
[data-testid="stMain"] [data-testid="stDownloadButton"] button:hover:not(:disabled) {
    background: #3A3D47;
    border-color: #3A3D47;
    color: #FFFFFF;
}
[data-testid="stMain"] [data-testid="stDownloadButton"] button:disabled {
    background: #E7E5DE;
    border-color: #DCDAD2;
    color: #706E65;
    opacity: 1;
}
button:focus-visible {outline: 2px solid #A07800; outline-offset: 3px;}
[data-testid="stSidebar"] button:focus-visible {outline-color: #FFD633;}
::selection {background: #FFE990; color: #24262D;}
</style>""", unsafe_allow_html=True)
st.caption("BEELINE CASE · РАБОЧЕЕ МЕСТО АНАЛИТИКА")
st.title("От аудитории — к плану кампаний")
st.write("Загрузите данные, проверьте гипотезы пилотами и получите сегменты, тарифы, каналы и распределение бюджета.")
st.info("Демонстрационная среда: пилоты и эффекты симулируются. SMS, реклама и звонки реальным абонентам не отправляются.")


def clear_session_key():
    st.session_state["session_api_key"] = ""
    st.session_state["api_consent"] = False
    st.session_state.pop("result", None)


with st.sidebar:
    st.header("Настройки исследования")
    source = st.radio("Данные", ["Пример из кейса", "Загрузить CSV"])
    mode = "api"
    st.write("**LLM через OpenAI API**")
    entered_key = st.text_input("Ключ OpenAI API", type="password", key="session_api_key",
                                placeholder="Вставьте API-ключ")
    session_api_key = entered_key.strip() or None
    api_ready = bool(session_api_key or os.environ.get("OPENAI_API_KEY"))
    st.caption("Введённый ключ хранится только в текущей сессии и не записывается в файлы или отчёты.")
    if session_api_key:
        st.button("Очистить введённый ключ", on_click=clear_session_key, key="clear_api_key")
    elif api_ready:
        st.caption("Используется ключ, настроенный на сервере.")
    st.caption("До 5 платных запросов за запуск: 4 для выбора пилотов и 1 для объяснения плана. Передаются агрегаты сегментов, история переходов, параметры тарифов и результаты пилотов, без ID абонентов.")
    consent = st.checkbox("Разрешаю передачу этих агрегатов в OpenAI для этого запуска", key="api_consent")
    if not api_ready:
        st.warning("Введите API-ключ в поле выше, чтобы начать исследование.")
    with st.expander("Параметры симуляции"):
        seed = st.number_input("Seed", min_value=0, max_value=2147483647, value=42, step=1)
    st.divider()
    st.write("**Лимиты кейса**")
    st.caption("100 000 у.е. · 15 000 контактов · до 20 пилотов · до 10 финальных кампаний")

profile = history = tariffs = None
input_key = f"demo:{mode}:{seed}"
if source == "Загрузить CSV":
    st.subheader("1. Загрузите файлы вместе")
    st.write("Перетащите все три CSV в одну область. Порядок и названия файлов не важны — приложение распознает таблицы по колонкам.")
    st.markdown("**Нужны:** `customer_profile.csv` — абоненты; `change_tariff.csv` — история переходов; "
                "`dict_tariff.csv` — тарифы и цены.")
    files = st.file_uploader("Профиль, история переходов и тарифы", type="csv",
                             accept_multiple_files=True, key="campaign_files", max_upload_size=20)
    raw = [(file.name, file.getvalue()) for file in files]
    fingerprint = sorted((name, hashlib.sha256(data).hexdigest()) for name, data in raw)
    input_key = "uploads:" + hashlib.sha256(json.dumps(fingerprint).encode()).hexdigest() + f":{mode}:{seed}"
    with st.expander("Формат файлов"):
        st.write("CSV в UTF-8, разделитель — запятая или точка с запятой. До 20 МБ на файл.")
        st.code("Профиль: ID_NUMBER, current_tariff, arpu_segment, data_segment, call_segment, predicted_arpu\n"
                "История: ID_NUMBER, tariff_plan_code_from, tariff_plan_code_to, AVG_ARPU_PREV_3M, AVG_ARPU_NEXT_3M\n"
                "Тарифы: tariff_plan_code, price_tariff")
    if files:
        try:
            found, inventory = identify_uploads(raw)
            st.dataframe(pd.DataFrame(inventory), hide_index=True, width="stretch")
            if any(row["Назначение"].startswith("Не используется") for row in inventory):
                st.warning("Некоторые файлы не распознаны и не будут использованы. Для расчёта нужны три таблицы указанного формата.")
            missing = [label for role, (label, _) in UPLOAD_SCHEMAS.items() if role not in found]
            if missing:
                st.info("Добавьте недостающие таблицы: " + ", ".join(missing) + ".")
            else:
                profile, history, tariffs = (found[role] for role in ("profile", "history", "tariffs"))
        except ValueError as exc:
            st.error(str(exc))
    else:
        st.session_state.pop("result", None)
        st.info("Добавьте файлы в общую область, затем нажмите «Исследовать и собрать план».")
else:
    profile, history, tariffs = demo_data()

if st.session_state.get("input_key") != input_key:
    st.session_state.pop("result", None)
    st.session_state["input_key"] = input_key

valid = False
if profile is not None:
    try:
        profile, history, tariffs, notices = validate_data(profile, history, tariffs)
        valid = True
        for notice in notices:
            st.caption(notice)
    except ValueError as exc:
        st.error(str(exc))
    if valid:
        metrics = st.columns(3)
        metrics[0].metric("Абонентов", f"{len(profile):,}".replace(",", " "))
        metrics[1].metric("Исторических переходов", f"{len(history):,}".replace(",", " "))
        metrics[2].metric("Доступных тарифов", len(tariffs))
        with st.expander("Аудитория по ARPU-сегментам"):
            segment_counts = profile.groupby("arpu_segment", observed=True).size().rename("Абоненты")
            st.bar_chart(segment_counts)

blocked = not valid or not api_ready or not consent
if st.button("Исследовать и собрать план", type="primary", disabled=blocked, key="run_campaigns"):
    st.session_state.pop("result", None)
    with st.status("Проверяем гипотезы…", expanded=True) as status:
        counter = [0]
        progress = st.progress(0.)
        latest = st.empty()
        def on_pilot(observation):
            counter[0] += 1
            progress.progress(min(counter[0] / 20, 1.))
            latest.write(f"Пилот {counter[0]}: {observation['candidate_id']} · "
                         f"{observation['n']} контактов · наблюдение {observation['observed']:+.1%}")
        try:
            st.session_state["result"] = run_simulation(
                profile, history, tariffs, mode, int(seed), on_pilot, api_key=session_api_key)
            progress.progress(1.)
            status.update(label="Исследование завершено", state="complete", expanded=False)
        except Exception:
            status.update(label="Не удалось завершить исследование", state="error")
            st.error("Проверьте формат данных и повторите запуск. План не опубликован.")

result = st.session_state.get("result")
if result:
    score = result["score"]
    st.subheader("План кампаний")
    summary = st.columns(4)
    summary[0].metric("Финальных кампаний", len(result["plan"]))
    summary[1].metric("Расходы с пилотами, у.е.", f"{score['total_cost']:,.0f}")
    summary[2].metric("Контакты с пилотами", f"{score['total_contacts']:,}")
    summary[3].metric("Чистый эффект симуляции, у.е.", f"{score['net_arpu_gain']:+,.0f}")
    st.caption("Эффект рассчитан симулятором, это не прогноз реальной выручки. Оценки из пилотов шумные.")
    if score["net_arpu_gain"] <= 0:
        st.warning("В этой симуляции получен убыток. Такой план требует пересмотра перед реальным запуском.")
    if not result["plan"]:
        st.warning("Финальные кампании не сформированы. Проверьте размер и состав сегментов.")
    else:
        st.dataframe(pd.DataFrame(result["plan"]), hide_index=True, width="stretch")
    remaining = 100000 - score["total_cost"]
    pilot_cost = sum(p["cost"] for p in result["pilot_history"])
    st.write(f"**Бюджет:** пилоты {pilot_cost:,.0f} · финальные кампании {score['total_cost']-pilot_cost:,.0f} · остаток {remaining:,.0f} у.е.")
    sources = [event["source"] for event in result["trace"]["llm"]]
    st.caption(f"Выбор пилотов: API — {sources.count('api')}, "
               f"события перехода на статистику — {sources.count('fallback')}. Пилотов: {score['n_pilots']}.")
    if "fallback" in sources:
        st.warning("Часть решений принята резервным статистическим алгоритмом: API не вернул пригодный ответ. Подробности — в объяснениях ниже.")
    with st.expander("Почему переход может быть выгоден клиенту и компании", expanded=True):
        explanation = result.get("explanation", {})
        if explanation.get("status") == "api":
            st.caption("Объяснения сгенерированы LLM по итоговому плану, параметрам тарифов и агрегатам данных. Предположения требуют проверки.")
            by_name = {row["campaign_id"]: row for row in explanation["campaigns"]}
            for campaign in result["plan"]:
                row = by_name[campaign["Кампания"]]
                st.write(f"**{campaign['Кампания']}: {campaign['Текущий тариф']} → {campaign['Целевой тариф']}**")
                for label, field in (("Интерес клиента", "client_benefit"),
                                     ("Эффект для компании", "business_case"),
                                     ("На чём основан вывод", "evidence"),
                                     ("Риски и следующая проверка", "risks")):
                    st.write(f"**{label}**")
                    st.text(row[field])
        elif not result["plan"]:
            st.info("Нет итоговых кампаний для объяснения.")
        else:
            st.warning("Объяснение LLM не получено. Рассчитанный план сохранён; повторный запуск исследования попробует получить объяснения заново.")
    with st.expander("Как выбирались пилоты"):
        st.write("Код сравнивает проверенные варианты по эффекту с поправкой на неопределённость, цене канала и доступным лимитам. "
                 "Столбцы SMS-эффекта и ошибки оценки относятся к пилотному каналу, даже если финальный канал отличается.")
        for index, event in enumerate(result["trace"]["llm"], 1):
            st.text(f"Решение {index} · {event['source']}: {event.get('reason', '')}")
    with st.expander("История пилотов"):
        st.dataframe(pd.DataFrame(result["trace"]["pilots"]), hide_index=True, width="stretch")
    exports = st.columns(2)
    exports[0].download_button("Скачать план CSV", result["csv"].encode("utf-8-sig"), "campaign_plan.csv", "text/csv", disabled=not result["plan"])
    exports[1].download_button("Скачать отчёт JSON", json.dumps(result, ensure_ascii=False, indent=2), "campaign_report.json", "application/json")
    st.caption("Для официальной сдачи submission.csv по-прежнему создаётся командой python make_submission.py.")
