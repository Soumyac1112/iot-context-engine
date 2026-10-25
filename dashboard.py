"""Live dashboard: watch the context engine think, one reading at a time.

    streamlit run dashboard.py
"""
import html
import os
import time

import altair as alt
import pandas as pd
import streamlit as st

from context_engine import SCENARIOS, ContextEngine, make_llm
from context_engine.evaluate import ScenarioResult
from context_engine.safety import HIGH_HR_MARGIN
from context_engine.simulator import generate

st.set_page_config(page_title="Wearable Context Engine", page_icon="⌚", layout="wide")

st.markdown("""
<style>
.watch {background:#111827;border-radius:28px;padding:18px 18px 14px;color:#F9FAFB;
        border:6px solid #374151;min-height:250px;font-family:system-ui,sans-serif}
.watch-time {text-align:center;font-size:1.6rem;font-weight:600;letter-spacing:1px}
.watch-sub {text-align:center;font-size:.8rem;color:#9CA3AF;margin-bottom:10px}
.card {border-radius:14px;padding:10px 12px;margin-top:8px;font-size:.9rem;line-height:1.35}
.card.notification {background:#1F2937}
.card.summary {background:#1E3A5F}
.card.nudge {background:#134E4A}
.card.system {background:#1F2937}
.card.emergency {background:#7F1D1D;font-weight:600}
.card .kind {font-size:.7rem;text-transform:uppercase;color:#D1D5DB;letter-spacing:.06em}
.pills {margin-top:12px;text-align:center}
.pill {display:inline-block;border-radius:999px;padding:2px 9px;margin:2px;font-size:.72rem;
       background:#374151;color:#9CA3AF}
.pill.on {background:#2563EB;color:#fff}
.pill.alert {background:#DC2626;color:#fff}
</style>
""", unsafe_allow_html=True)

SPEEDS = {"1x (real time)": 1, "5x": 5, "10x": 10, "25x": 25, "As fast as possible": 0}


@st.cache_resource
def llm_client():
    return make_llm()


# ---------------- sidebar ----------------
# Shareable demo links: ?story=hr_spike_still&autorun=1&speed=10x
params = st.query_params
story_index = next((i for i, s in enumerate(SCENARIOS) if s.key == params.get("story")), 0)

st.sidebar.title("⌚ Context Engine")
scenario = st.sidebar.selectbox("Story", SCENARIOS, index=story_index, format_func=lambda s: s.title)
st.sidebar.caption(scenario.story)
llm = llm_client()
use_llm = st.sidebar.toggle("Use LLM agents", value=llm is not None, disabled=llm is None,
                            help="Off = the same policy runs as plain rules, fully offline.")
if llm is None:
    st.sidebar.info("LLM switched off (LLM_DISABLED=1), so the agents run on rules."
                    if os.getenv("LLM_DISABLED") == "1" else
                    "No LLM key found in .env, so the agents run on rules.")
speed_label = st.sidebar.select_slider("Speed", options=list(SPEEDS),
                                       value=params.get("speed") if params.get("speed") in SPEEDS else "10x")
show_quiet = st.sidebar.checkbox("Show 'do nothing' decisions", value=False)
run = st.sidebar.button("▶ Run story", type="primary", use_container_width=True)
if params.get("autorun") == "1" and not st.session_state.get("autoran"):
    st.session_state["autoran"] = run = True

with st.sidebar.expander("What should happen"):
    if scenario.must_call:
        st.markdown("**Must do:** " + ", ".join(f"`{t}`" for t in sorted(scenario.must_call)))
    if scenario.must_not_call:
        st.markdown("**Must not do:** " + ", ".join(f"`{t}`" for t in sorted(scenario.must_not_call)))
    if scenario.needs_llm:
        st.markdown("Rules alone are expected to fail this story: the message text has to be understood.")

# ---------------- page ----------------
st.title("IoT Wearable Context Engine")
mode_ph = st.empty()
metrics_ph = st.empty()
left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("Heart rate")
    chart_ph = st.empty()
    situation_ph = st.empty()
with right:
    st.subheader("Watch")
    watch_ph = st.empty()
st.subheader("Action log")
log_ph = st.empty()
result_ph = st.empty()
with st.expander("Latest raw reading (JSON from the watch)"):
    raw_ph = st.empty()


def render(engine: ContextEngine, rows: list[dict], reading, finished: ScenarioResult | None = None) -> None:
    f, s, w, stats = engine.features, engine.state, engine.watch, engine.stats
    mode_ph.caption(f"Mode: **{engine.mode}**")

    with metrics_ph.container():
        c = st.columns(6)
        baseline = engine.memory.resting_hr
        c[0].metric("Heart rate", f"{reading.heart_rate} bpm", f"{reading.heart_rate - baseline:+.0f} vs normal",
                    delta_color="inverse")
        c[1].metric("Activity", s.activity.title() if s else "...")
        c[2].metric("Stress", s.stress.title() if s else "...")
        c[3].metric("Attention", s.attention.title() if s else "...")
        c[4].metric("Battery", f"{reading.battery}%")
        c[5].metric("Agent runs", f"{stats.agent_runs} / {stats.readings}",
                    f"{stats.calls_saved_pct:.0f}% calls saved", delta_color="off")

    chart_ph.altair_chart(_hr_chart(rows[-240:]), use_container_width=True)

    if s:
        situation_ph.info(f"**Situation:** {s.summary}  \n"
                          f"_Profiler: {'LLM' if s.source == 'llm' else 'rules'}, "
                          f"confidence {s.confidence:.0%}_")

    watch_ph.markdown(_watch_face(engine, reading), unsafe_allow_html=True)

    actions = [a for a in engine.actions if show_quiet or a.tool not in ("do_nothing", "log_event")]
    if actions:
        log_ph.dataframe(pd.DataFrame([{
            "time": a.time.strftime("%H:%M:%S"), "by": a.by, "action": a.tool, "reason": a.reason, "result": a.result,
        } for a in reversed(actions)]), hide_index=True, use_container_width=True)
    else:
        log_ph.caption("No actions yet. The engine is watching quietly.")

    raw_ph.json(reading.model_dump(mode="json"))

    if finished:
        r = finished
        used = ", ".join(sorted(r.called - {"do_nothing", "log_event"})) or "nothing (stayed quiet)"
        if r.passed:
            result_ph.success(f"**PASS.** Actions used: {used}. Average decision time {stats.avg_decision_ms:.0f} ms.")
        else:
            problems = []
            if r.missing:
                problems.append("missed " + ", ".join(sorted(r.missing)))
            if r.mistakes:
                problems.append("should not have used " + ", ".join(sorted(r.mistakes)))
            note = " This is expected without the LLM." if r.scenario.needs_llm and not engine.llm else ""
            result_ph.error(f"**FAIL:** {'; '.join(problems)}.{note} Actions used: {used}.")


def _hr_chart(rows: list[dict]) -> alt.Chart:
    df = pd.DataFrame(rows).melt("time", var_name="line", value_name="bpm")
    colors = alt.Scale(domain=["Heart rate", "Normal", "Safety line"], range=["#DC2626", "#2563EB", "#9CA3AF"])
    return alt.Chart(df).mark_line().encode(
        x=alt.X("time:T", title=None, axis=alt.Axis(format="%H:%M:%S")),
        y=alt.Y("bpm:Q", title="bpm", scale=alt.Scale(domain=[40, 170])),
        color=alt.Color("line:N", scale=colors, legend=alt.Legend(orient="bottom", title=None)),
        strokeDash=alt.condition(alt.datum.line == "Heart rate", alt.value([1, 0]), alt.value([5, 4])),
    ).properties(height=260)


def _watch_face(engine: ContextEngine, reading) -> str:
    w = engine.watch
    cards = []
    for item in w.screen[-3:][::-1]:
        cards.append(f'<div class="card {item.kind}"><div class="kind">{item.kind} · {item.time:%H:%M:%S}</div>'
                     f'{html.escape(item.text)}</div>')
    if w.emergency and w.emergency.status == "countdown":
        left = max(0, int((w.emergency.deadline - reading.time).total_seconds()))
        cards.insert(0, f'<div class="card emergency">Are you OK? Calling for help in {left}s</div>')
    if not cards:
        cards.append('<div class="card system">No interruptions</div>')

    def pill(label: str, on: bool, alert: bool = False) -> str:
        return f'<span class="pill {"alert" if alert and on else "on" if on else ""}">{label}</span>'

    emergency = bool(w.emergency and w.emergency.status != "cancelled")
    return (f'<div class="watch"><div class="watch-time">{reading.time:%H:%M:%S}</div>'
            f'<div class="watch-sub">{reading.heart_rate} bpm · {reading.battery}% battery</div>'
            f'{"".join(cards)}<div class="pills">'
            f'{pill("Do Not Disturb", w.dnd)}{pill("Battery saver", w.battery_saver)}'
            f'{pill(f"Held: {len(w.deferred)}", bool(w.deferred))}{pill("Emergency", emergency, alert=True)}'
            f'</div></div>')


def play() -> None:
    engine = ContextEngine(llm=llm if use_llm else None)
    baseline = engine.memory.resting_hr
    rows: list[dict] = []
    delay = 1 / SPEEDS[speed_label] if SPEEDS[speed_label] else 0
    readings = list(generate(scenario))
    for i, reading in enumerate(readings):
        step = engine.process(reading)
        rows.append({"time": reading.time, "Heart rate": reading.heart_rate, "Normal": baseline,
                     "Safety line": baseline + HIGH_HR_MARGIN})
        last = i == len(readings) - 1
        if engine.features and (step.actions or step.changes or i % 3 == 0 or last):
            finished = ScenarioResult(scenario, engine.actions, engine.stats, engine.mode) if last else None
            render(engine, rows, reading, finished)
        if delay:
            time.sleep(delay)
    st.session_state["last_run"] = (scenario.key, engine, rows, readings[-1])


if run:
    play()
elif st.session_state.get("last_run") and st.session_state["last_run"][0] == scenario.key:
    key, engine, rows, reading = st.session_state["last_run"]
    render(engine, rows, reading, ScenarioResult(scenario, engine.actions, engine.stats, engine.mode))
else:
    mode_ph.caption(f"Mode: **{'LLM (' + llm.label + ')' if (llm and use_llm) else 'Rules only (offline)'}**")
    result_ph.info("Pick a story on the left and press **Run story**.")
