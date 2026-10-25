"""The test stories the simulator plays, each with the behaviour we expect."""
from datetime import datetime

from .schemas import Notification
from .simulator import Phase, Scenario


def _day(hh: int, mm: int) -> datetime:
    return datetime(2026, 10, 25, hh, mm, 0)


SCENARIOS: list[Scenario] = [
    Scenario(
        key="low_battery_navigation",
        title="Low battery while navigating a new city",
        story="Walking through Paris with Maps open. Battery falls from 12% to 4%. A promo message arrives.",
        start=_day(17, 20),
        battery_start=12,
        phases=[
            Phase(seconds=180, hr=96, motion="walking", at="paris", speed_kmh=4.5, heading_deg=40,
                  app="maps", screen_on=True, battery=(12, 4),
                  notifications={100: Notification(id="n-promo", source="ShopApp",
                                                   text="50% off sneakers today only!", priority="low")}),
        ],
        must_call={"battery_saver_mode", "defer_notification"},
        must_not_call={"send_notification", "start_emergency_protocol"},
    ),
    Scenario(
        key="hr_spike_still",
        title="Heart rate spike while sitting still",
        story="Sitting on the sofa. Heart rate jumps from 72 to ~150 with no movement, and the user does not answer.",
        start=_day(21, 5),
        battery_start=64,
        phases=[
            Phase(seconds=60, hr=72, motion="still", at="home"),
            Phase(seconds=90, hr=150, motion="still", at="home", hr_ramp=0.25),
        ],
        must_call={"start_emergency_protocol", "alert_emergency_contact"},
    ),
    Scenario(
        key="hr_high_running",
        title="High heart rate while running",
        story="A warm-up walk, then a run in the park. Heart rate climbs to ~155. This is normal exercise.",
        start=_day(6, 30),
        battery_start=88,
        phases=[
            Phase(seconds=30, hr=95, motion="walking", at="park", speed_kmh=5, heading_deg=0),
            Phase(seconds=150, hr=155, motion="running", at="park", speed_kmh=10, heading_deg=0, hr_ramp=0.06),
        ],
        must_not_call={"start_emergency_protocol", "send_notification", "suggest_breathing"},
    ),
    Scenario(
        key="sleeping_low_priority",
        title="Low-priority message while asleep",
        story="Asleep at home before the alarm. A newsletter arrives. The user wakes up a couple of minutes later.",
        start=_day(6, 55),
        battery_start=81,
        phases=[
            Phase(seconds=150, hr=54, motion="still", at="home",
                  notifications={75: Notification(id="n-news", source="Email",
                                                  text="Weekly newsletter: 10 tips for better sleep", priority="low")}),
            Phase(seconds=60, hr=66, motion="light", at="home", screen_on=True, hr_ramp=0.2),
        ],
        must_call={"defer_notification", "deliver_deferred"},
        must_not_call={"send_notification", "start_emergency_protocol"},
    ),
    Scenario(
        key="stressed_commute",
        title="Stressed while commuting",
        story="Riding to work, heart rate well above normal. A work email arrives on the way. Then arrives at the office.",
        start=_day(8, 40),
        battery_start=70,
        phases=[
            Phase(seconds=150, hr=101, motion="vehicle", at="route_to_work", speed_kmh=38, heading_deg=90,
                  notifications={60: Notification(id="n-email", source="Work email",
                                                  text="Can you review the Q3 deck before 10?", priority="medium")}),
            Phase(seconds=30, hr=99, motion="walking", at="work", speed_kmh=4.5),
            Phase(seconds=60, hr=97, motion="light", at="work"),
        ],
        must_call={"defer_notification", "suggest_breathing"},
        must_not_call={"start_emergency_protocol"},
    ),
    Scenario(
        key="meditation_beads",
        title="Beads-counting (meditation) app open",
        story="Opens the beads-counting app at home and meditates. A WhatsApp message arrives. Closes the app afterwards.",
        start=_day(19, 0),
        battery_start=55,
        phases=[
            Phase(seconds=20, hr=72, motion="light", at="home"),
            Phase(seconds=130, hr=64, motion="still", at="home", app="beads_counter", screen_on=True,
                  notifications={50: Notification(id="n-wa", source="WhatsApp",
                                                  text="Are we still on for dinner?", priority="medium")}),
            Phase(seconds=40, hr=68, motion="light", at="home"),
        ],
        must_call={"enable_dnd", "defer_notification", "disable_dnd", "deliver_deferred"},
        must_not_call={"send_notification", "start_emergency_protocol"},
    ),
    Scenario(
        key="urgent_message_asleep",
        title="Urgent message while asleep (needs the LLM)",
        story="Asleep at night. A family message marked only 'medium' priority says Dad is in hospital. "
              "Rules see 'medium' and hold it; reading the text shows it must wake the user.",
        start=_day(2, 10),
        battery_start=60,
        phases=[
            Phase(seconds=120, hr=53, motion="still", at="home",
                  notifications={75: Notification(id="n-urgent", source="Family group",
                                                  text="Dad has been taken to hospital, please call me now",
                                                  priority="medium")}),
        ],
        must_call={"send_notification"},
        must_not_call={"defer_notification"},
        needs_llm=True,
    ),
    Scenario(
        key="free_walk",
        title="Relaxed walk (control case)",
        story="An easy walk near home. A WhatsApp message arrives. Nothing special is going on, so it should be shown.",
        start=_day(16, 0),
        battery_start=76,
        phases=[
            Phase(seconds=120, hr=92, motion="walking", at="near_home", speed_kmh=4.8, heading_deg=200,
                  notifications={60: Notification(id="n-wa2", source="WhatsApp",
                                                  text="Picked up the groceries", priority="medium")}),
        ],
        must_call={"send_notification"},
        must_not_call={"defer_notification", "start_emergency_protocol", "enable_dnd"},
    ),
]

BY_KEY = {s.key: s for s in SCENARIOS}
