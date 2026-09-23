"""Analyst workspace. Run: streamlit run app.py"""
import hashlib
import json
import os

import pandas as pd
import streamlit as st

from campaign_app_service import demo_data, identify_uploads, read_csv, run_simulation, validate_data, UPLOAD_SCHEMAS

st.set_page_config(page_title="Планировщик тарифных кампаний", page_icon="🐝", layout="wide")
st.markdown("""<style>
/* Native theme also styles canvas tables, menus and inputs; keep CSS scoped. */
[data-testid="stMainBlockContainer"] {max-width: 1440px; padding-top: 4.5rem; padding-bottom: 3rem;}
[data-testid="stHeader"] {background: #FAF9F5;}
[data-testid="stMain"] h2 {font-size: 1.55rem; letter-spacing: -.035em;}
[data-testid="stMain"] h3 {font-size: 1.1rem;}
.hero {position: relative; overflow: hidden; padding: 34px 38px; border-radius: 20px;
    background: #24262D; color: #F5F4EF; margin-bottom: 6px;}
.hero::after {content: ''; position: absolute; width: 230px; height: 230px; right: -60px; top: -80px;
    border: 36px solid #FFD633; border-radius: 50%; opacity: .9; pointer-events: none;}
.eyebrow {font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; font-weight: 650;}
.hero .eyebrow {color: #FFE477;}
.hero h1 {color: #FFFFFF; font-size: clamp(1.8rem, 3vw, 2.65rem); font-weight: 700;
    line-height: 1.15; letter-spacing: -.045em; padding: 16px 0 12px; max-width: 740px; position: relative; z-index: 1;}
.hero p {color: #D1D2D8; max-width: 700px; line-height: 1.65; margin: 0; position: relative; z-index: 1;}
.hero-tags {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 22px;}
.hero-tags span {border: 1px solid #555862; border-radius: 50px; padding: 5px 12px;
    color: #E9E8E2; font-size: .76rem;}
.demo-note {font-size: .81rem; color: #666A73; padding: 4px 0 12px; line-height: 1.6;}
.demo-note strong {color: #735800;}
.steps {display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 4px 0 18px;}
.step {padding: 14px 16px; border: 1px solid #DCDAD2; border-radius: 12px; background: #FFFFFF; color: #666A73; font-size: .85rem;}
.step b {display: inline-flex; align-items: center; justify-content: center; width: 26px; height: 26px;
    border-radius: 50%; background: #EEECE4; color: #5B5D64; margin-right: 9px; font-size: .75rem;}
.step.active {border-color: #CEAB2A; background: #FFF8DB; color: #24262D;}
.step.active b {background: #FFD633; color: #24262D;}
.step.done b {background: #EAF4EE; color: #245D42;}
.side-brand {display: flex; align-items: center; gap: 10px; color: #F5F4EF; font-weight: 700; margin-bottom: 22px;}
.brand-mark {width: 28px; height: 28px; border-radius: 50%; background: repeating-linear-gradient(0deg, #FFD633 0 6px, #24262D 6px 9px);}
.side-brand small {display: block; font-size: .68rem; color: #BDC0C9; font-weight: 400; letter-spacing: .08em;}
.budget-bar {height: 12px; display: flex; overflow: hidden; border-radius: 10px; background: #EEECE4; margin: 10px 0 12px;}
.budget-bar .pilot {background: #8D7843;}
.budget-bar .campaign {background: #FFD633;}
.budget-legend {display: flex; flex-wrap: wrap; gap: 12px 26px; color: #666A73; font-size: .83rem; margin-bottom: 12px;}
.budget-legend strong {color: #24262D;}
[data-testid="stMain"] [data-testid="stExpander"] {background: #FFFFFF; border-radius: 12px;}
[data-testid="stMain"] [data-testid="stFileUploaderDropzone"] {border: 1px dashed #B8AD85; background: #FFFDF5; border-radius: 14px; padding: 24px;}
[data-testid="stMain"] [data-testid="stTabs"] [role="tablist"] {gap: 20px; border-bottom: 1px solid #DCDAD2;}
[data-testid="stMain"] [role="tab"] {padding: 12px 2px; font-weight: 600;}
[data-testid="stMain"] [role="tab"][aria-selected="true"] {color: #735800;}
[data-testid="stMain"] [data-testid="stText"] {font-family: inherit; line-height: 1.65; font-size: .94rem;}
[data-testid="stMetric"] {
    background: var(--secondary-background-color, #FFFFFF);
    border: 1px solid var(--border-color, #DCDAD2);
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 3px 12px #24262D04;
}
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
    color: var(--text-color, #24262D);
}
[data-testid="stMetricValue"] {font-weight: 650; letter-spacing: -.04em; font-variant-numeric: tabular-nums;}
[data-testid="stMain"] [data-testid="stCaptionContainer"] {color: #666A73;}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {color: #BDC0C9;}
[data-testid="stCaptionContainer"] p {color: inherit; opacity: 1;}
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
@media (max-width: 700px) {
    .hero {padding: 26px 22px;}
    .hero::after {opacity: .18;}
    .steps {gap: 6px;}
    .step {padding: 10px 8px; font-size: .72rem;}
    .step b {display: flex; margin-bottom: 6px;}
    [data-testid="stMainBlockContainer"] {padding-left: 1rem; padding-right: 1rem;}
}
</style>""", unsafe_allow_html=True)
st.markdown('''<div class="hero">
<div class="eyebrow">Beeline Case · Рабочее место аналитика</div>
<h1>От аудитории —<br>к плану кампаний</h1>
<p>Проверьте гипотезы небольшими пилотами. Получите план переходов на новые тарифы
и объяснение выгоды для клиента и компании.</p>
<div class="hero-tags"><span>100 000 у.е. бюджета</span><span>До 20 пилотов</span><span>До 10 кампаний</span></div>
</div><div class="demo-note"><strong>Демонстрация на синтетических данных.</strong>
Пилоты и эффекты симулируются. Сообщения, реклама и звонки абонентам не отправляются.</div>''', unsafe_allow_html=True)
steps = st.empty()


def format_number(value):
    return f"{value:,.0f}".replace(",", " ")


def clear_session_key():
    st.session_state["session_api_key"] = ""
    st.session_state["api_consent"] = False
    st.session_state.pop("result", None)


with st.sidebar:
    st.markdown('<div class="side-brand"><span class="brand-mark" aria-hidden="true"></span>'
                '<div>Campaign Planner<small>BEELINE CASE</small></div></div>', unsafe_allow_html=True)
    st.subheader("Настройки исследования")
    source = st.radio("Данные", ["Пример из кейса", "Загрузить CSV"])
    mode = "api"
    st.divider()
    st.write("**Подключение LLM**")
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
        st.caption("Меняет случайные результаты пилотов. Используйте разные значения для проверки устойчивости плана.")
    st.divider()
    st.write("**Лимиты кейса**")
    st.caption("100 000 у.е. · 15 000 контактов · до 20 пилотов · до 10 финальных кампаний")

profile = history = tariffs = None
input_key = f"demo:{mode}:{seed}"
st.subheader("01 · Данные аудитории")
if source == "Загрузить CSV":
    st.write("Перетащите все три CSV в одну область. Порядок и названия файлов не важны — приложение распознает таблицы по колонкам.")
    st.markdown("**Нужны:** `customer_profile.csv` — абоненты; `change_tariff.csv` — история переходов; "
                "`dict_tariff.csv` — тарифы и цены.")
    files = st.file_uploader("Профиль, история переходов и тарифы", type="csv",
                             accept_multiple_files=True, key="campaign_files", max_upload_size=20)
    raw = [(file.name, file.getvalue()) for file in files]
    fingerprint = sorted((name, hashlib.sha256(data).hexdigest()) for name, data in raw)
    input_key = "uploads:" + hashlib.sha256(json.dumps(fingerprint).encode()).hexdigest() + f":{mode}:{seed}"
    with st.expander("Формат файлов"):
        st.write("CSV в UTF-8, UTF-16 или Windows-1251. Разделители: запятая, точка с запятой, табуляция или |. До 20 МБ на файл. Пробелы и регистр названий колонок исправляются автоматически.")
        st.code("Профиль: ID_NUMBER, current_tariff, arpu_segment, data_segment, call_segment, predicted_arpu\n"
                "История: ID_NUMBER, tariff_plan_code_from, tariff_plan_code_to, AVG_ARPU_PREV_3M, AVG_ARPU_NEXT_3M\n"
                "Тарифы: tariff_plan_code, price_tariff")
    if files:
        assignments, column_maps = {}, {}
        with st.expander("Настроить распознавание файлов и колонок"):
            st.caption("Если таблица не распознана, выберите её назначение и укажите, какие колонки содержат нужные данные. Изменять CSV не требуется.")
            roles = ["auto", *UPLOAD_SCHEMAS, "ignore"]
            labels = {"auto": "Определить автоматически", "ignore": "Пропустить файл",
                      **{role: label for role, (label, _) in UPLOAD_SCHEMAS.items()}}
            for index, (name, data) in enumerate(raw):
                widget_key = f"upload_{index}_{hashlib.sha256(data).hexdigest()}"
                role = st.selectbox(f"Назначение файла «{name}»", roles,
                                    format_func=labels.get, key=widget_key + "_role")
                assignments[index] = role
                if role in UPLOAD_SCHEMAS:
                    try:
                        preview = read_csv(data)
                        st.dataframe(preview.head(3), hide_index=True, width="stretch")
                        mapping = {}
                        for field in sorted(UPLOAD_SCHEMAS[role][1]):
                            options = [None, *preview.columns]
                            selected = st.selectbox(f"{name} · {field}", options,
                                index=options.index(field) if field in options else 0,
                                format_func=lambda value: "Выберите колонку" if value is None else value,
                                key=widget_key + "_" + role + "_" + field)
                            if selected is not None:
                                mapping[field] = selected
                        column_maps[index] = mapping
                    except ValueError as exc:
                        st.error(str(exc))
        input_key += ":" + hashlib.sha256(json.dumps([assignments, column_maps], sort_keys=True).encode()).hexdigest()
        try:
            found, inventory = identify_uploads(raw, assignments, column_maps)
            st.dataframe(pd.DataFrame(inventory), hide_index=True, width="stretch")
            if any(row["Назначение"].startswith("Не используется") for row in inventory):
                st.warning("Некоторые файлы ещё не распознаны. Откройте «Настроить распознавание файлов и колонок» выше и сопоставьте нужные поля.")
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
    st.caption("Загружен пример из кейса. Для работы со своими таблицами выберите «Загрузить CSV» в боковой панели.")

if st.session_state.get("input_key") != input_key:
    st.session_state.pop("result", None)
    st.session_state["input_key"] = input_key

valid = False
if profile is not None:
    try:
        profile, history, tariffs, notices = validate_data(profile, history, tariffs)
        valid = True
    except ValueError as exc:
        st.error(str(exc))
    if valid:
        metrics = st.columns(3)
        metrics[0].metric("Абонентов", f"{len(profile):,}".replace(",", " "))
        metrics[1].metric("Исторических переходов", f"{len(history):,}".replace(",", " "))
        metrics[2].metric("Доступных тарифов", len(tariffs))
        if notices:
            with st.expander(f"Замечания к данным · {len(notices)}"):
                for notice in notices:
                    st.write(notice)
        with st.expander("Аудитория по ARPU-сегментам"):
            segment_counts = profile.groupby("arpu_segment", observed=True).size().rename("Абоненты")
            st.bar_chart(segment_counts)

blocked = not valid or not api_ready or not consent
st.subheader("02 · Исследование")
if not valid:
    st.caption("Добавьте все три таблицы и исправьте ошибки данных, чтобы продолжить.")
elif not api_ready:
    st.caption("Данные готовы. Введите ключ OpenAI API в боковой панели.")
elif not consent:
    st.caption("Данные готовы. Разрешите передачу агрегатов в OpenAI в боковой панели.")
else:
    st.caption("Всё готово: агент проверит гипотезы, распределит бюджет и объяснит итоговые кампании.")
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
stage = 3 if result else (2 if valid else 1)
steps.markdown('<div class="steps">' + ''.join(
    f'<div class="step {"done" if i < stage else "active" if i == stage else ""}"><b>{"✓" if i < stage else f"0{i}"}</b>{label}</div>'
    for i, label in enumerate(("Данные аудитории", "Проверка гипотез", "План кампаний"), 1)) + '</div>', unsafe_allow_html=True)
if result:
    score = result["score"]
    st.divider()
    st.subheader("03 · План кампаний")
    summary = st.columns(4)
    summary[0].metric("Финальных кампаний", len(result["plan"]))
    summary[1].metric("Расходы с пилотами, у.е.", format_number(score['total_cost']))
    summary[2].metric("Контакты с пилотами", format_number(score['total_contacts']))
    summary[3].metric("Эффект симуляции, у.е.", ("+" if score['net_arpu_gain'] > 0 else "") + format_number(score['net_arpu_gain']))
    st.caption("Эффект рассчитан симулятором, это не прогноз реальной выручки. Оценки из пилотов шумные.")
    if score["net_arpu_gain"] <= 0:
        st.warning("В этой симуляции получен убыток. Такой план требует пересмотра перед реальным запуском.")
    if not result["plan"]:
        st.warning("Финальные кампании не сформированы. Проверьте размер и состав сегментов.")
    remaining = 100000 - score["total_cost"]
    pilot_cost = sum(p["cost"] for p in result["pilot_history"])
    final_cost = score['total_cost'] - pilot_cost
    sources = [event["source"] for event in result["trace"]["llm"]]
    st.caption(f"Выбор пилотов: API — {sources.count('api')}, "
               f"события перехода на статистику — {sources.count('fallback')}. Пилотов: {score['n_pilots']}.")
    if "fallback" in sources:
        st.warning("Часть решений принята резервным статистическим алгоритмом: API не вернул пригодный ответ. Подробности — во вкладке «Пилоты».")
    plan_tab, explanation_tab, pilots_tab = st.tabs(["Кампании и бюджет", "Объяснения LLM", "Пилоты"])
    with plan_tab:
        st.write("**Распределение бюджета**")
        st.markdown(f'''<div class="budget-bar" role="img" aria-label="Пилоты: {format_number(pilot_cost)}; кампании: {format_number(final_cost)}; остаток: {format_number(remaining)} у.е.">
            <span class="pilot" style="width: {max(0, min(100, pilot_cost / 1000)):.2f}%"></span>
            <span class="campaign" style="width: {max(0, min(100, final_cost / 1000)):.2f}%"></span></div>
            <div class="budget-legend"><span>Пилоты <strong>{format_number(pilot_cost)}</strong></span>
            <span>Кампании <strong>{format_number(final_cost)}</strong></span>
            <span>Остаток <strong>{format_number(remaining)} у.е.</strong></span></div>''', unsafe_allow_html=True)
        if result["plan"]:
            st.dataframe(pd.DataFrame(result["plan"]), hide_index=True, width="stretch")
            st.caption("Подробное обоснование каждого перехода — во вкладке «Объяснения LLM».")
    with explanation_tab:
        st.subheader("Почему переход может быть выгоден клиенту и компании")
        explanation = result.get("explanation", {})
        if explanation.get("status") == "api":
            st.caption("Объяснения сгенерированы LLM по итоговому плану, параметрам тарифов и агрегатам данных. Предположения требуют проверки.")
            by_name = {row["campaign_id"]: row for row in explanation["campaigns"]}
            for campaign in result["plan"]:
                row = by_name[campaign["Кампания"]]
                with st.container(border=True):
                    st.caption(f"{campaign['Кампания']} · {campaign['Канал']} · {format_number(campaign['Контакты'])} контактов")
                    st.subheader(f"{campaign['Текущий тариф']} → {campaign['Целевой тариф']}")
                    benefit_columns = st.columns(2, gap="large")
                    for column, label, field in ((benefit_columns[0], "Интерес клиента", "client_benefit"),
                                                  (benefit_columns[1], "Эффект для компании", "business_case")):
                        with column:
                            st.write(f"**{label}**")
                            st.text(row[field])
                    with st.expander("Факты и риски"):
                        st.write("**На чём основан вывод**")
                        st.text(row["evidence"])
                        st.write("**Риски и следующая проверка**")
                        st.text(row["risks"])
        elif not result["plan"]:
            st.info("Нет итоговых кампаний для объяснения.")
        else:
            st.warning("Объяснение LLM не получено. Рассчитанный план сохранён; повторный запуск исследования попробует получить объяснения заново.")
    with pilots_tab:
        st.caption("Столбцы SMS-эффекта и ошибки оценки относятся к пилотному каналу, даже если финальный канал отличается.")
        st.dataframe(pd.DataFrame(result["trace"]["pilots"]), hide_index=True, width="stretch")
        st.write("**Как выбирались пилоты**")
        for index, event in enumerate(result["trace"]["llm"], 1):
            with st.expander(f"Решение {index} · {event['source']}"):
                st.text(event.get('reason', ''))
    st.divider()
    st.write("**Сохранить результаты**")
    exports = st.columns(2)
    exports[0].download_button("Скачать план CSV", result["csv"].encode("utf-8-sig"), "campaign_plan.csv", "text/csv", disabled=not result["plan"])
    exports[1].download_button("Скачать отчёт JSON", json.dumps(result, ensure_ascii=False, indent=2), "campaign_report.json", "application/json")
    st.caption("Для официальной сдачи submission.csv по-прежнему создаётся командой python make_submission.py.")
