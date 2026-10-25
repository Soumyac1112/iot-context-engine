# IoT Wearable "Context" Engine

A smartwatch "brain" that reads live sensor data, works out what the user is doing,
and decides **when to stay quiet, when to nudge, and when to call for help**.
It is built in Python with LLM agents (Groq, or xAI Grok) and a Streamlit dashboard.
A fixed safety rule makes sure critical alerts never depend on the AI.

The plan and the architecture diagram are in `IoT_Context_Engine_Plan.docx` and
`IoT_Context_Engine_Architecture.png`.

## Quick start

```bash
pip install -r requirements.txt
copy .env.example .env        # then paste your key into .env

python run.py                 # play all test stories, print a timeline + scorecard
streamlit run dashboard.py    # live dashboard in the browser
python -m pytest              # run the tests (offline, no API calls)
```

`python run.py --rules` runs fully offline, with no LLM. Use `python run.py --list` to see the stories and
`python run.py -s hr_spike_still` to play one.

To open the dashboard straight into a story, use a demo link:
`http://localhost:8501/?story=hr_spike_still&autorun=1&speed=10x`

## The 8 building blocks, and where each one lives

| # | Block | File | What it does |
|---|-------|------|--------------|
| 1 | Telemetry Simulator | `context_engine/simulator.py`, `scenarios.py` | A fake watch: one JSON reading per second, played from scripted stories |
| 2 | Edge Layer | `context_engine/edge.py` | Plain Python, no AI. Rolling 5-minute window, works out facts, wakes the agents only on a real change |
| 3 | Profiler Agent | `context_engine/profiler.py` | "What is the user doing?" Rules give a first guess, then the fast LLM refines it |
| 4 | Action Agent | `context_engine/action_agent.py` | "Should I interrupt?" The LLM uses tool calling, with the same policy as rules as a fallback |
| 5 | Safety Guard | `context_engine/safety.py` | Fixed rule: heart-rate spike while completely still means an emergency, whatever the AI says |
| 6 | Tools | `context_engine/tools.py` | The actions the watch can take, each with a reason |
| 7 | User Memory | `context_engine/memory.py` | Normal heart rate, home/work places, emergency contact (synthetic demo data) |
| 8 | Dashboard | `dashboard.py` | Live chart, watch screen, current situation, action log |

`context_engine/engine.py` wires the blocks together. Each reading flows like this:

```
reading -> Edge Layer -> (only on change) Profiler Agent -> Action Agent -> Tools -> action log
               \-> Safety Guard (every reading, overrides the agents)
```

## Tools the Action Agent can call

`send_notification`, `defer_notification`, `deliver_deferred`, `enable_dnd`, `disable_dnd`,
`battery_saver_mode`, `suggest_breathing`, `start_emergency_protocol`, `do_nothing`.
The Safety Guard also uses `alert_emergency_contact` and `cancel_emergency`.
Every action is recorded by `log_event` with who decided it and why, in `logs/run-*.jsonl`.

These go slightly beyond the 7 tools in the plan. `disable_dnd`, `deliver_deferred` and
`do_nothing` were added so that every decision, including "stay quiet", is an explicit,
logged tool call.

## Test stories

| Story | What should happen |
|-------|--------------------|
| Low battery while navigating a new city | Battery saver on quietly, promo held, **no** battery warning |
| Heart rate spike while sitting still | Emergency countdown, then the emergency contact is alerted |
| High heart rate while running | Nothing: this is normal exercise |
| Low-priority message while asleep | Held until the user wakes, then delivered as a summary |
| Stressed while commuting | Work email held; on arrival, breathing exercise offered |
| Beads-counting (meditation) app open | Do Not Disturb on, message held, delivered after the session |
| Urgent message while asleep | **Needs the LLM:** labelled "medium" but says "Dad is in hospital", so it must wake the user |
| Relaxed walk (control) | The message is shown straight away |

The urgent-message story shows why the LLM matters. Rules only see the priority label
and hold the message. The LLM reads the text and wakes the user.

## LLM notes

- **Groq** (`gsk_...` key) has a free tier. Defaults: `openai/gpt-oss-20b` for the Profiler and
  `openai/gpt-oss-120b` for the Action Agent.
- **xAI Grok** (`xai-...` key) needs paid credits. Set `LLM_PROVIDER=xai` in `.env` to use it.
- The free tier allows about 8,000 tokens per minute per model. The client waits politely when it
  gets close. If a call still fails, **the rules take over**, so the watch never stops working.
- The agents get pre-computed signals (`focus_app_open`, `navigating`, `held_messages`, ...).
  This keeps the LLM reasoning from the same facts as the rules.
- After the agent changes the watch's mode (DND, battery saver), it gets one follow-up look,
  because one change often unlocks another (DND off -> deliver held messages).

## How we know it works

`python run.py` ends with a scorecard:
- **Correct decisions:** how many stories got the right actions
- **False alarms:** interruptions that should not have happened
- **Decision time:** from reading to decision, not counting rate-limit waits
- **AI calls saved:** agent runs compared with "run both agents on every reading"
