"""Safety Guard: fixed rules, no AI. When a rule matches it acts at once,
whatever the agents think. A life-safety alert must never depend on an LLM."""
from .schemas import Features

HIGH_HR_MARGIN = 45   # bpm above the user's normal resting heart rate
SUSTAIN_SECONDS = 10  # the high heart rate must last this long (ignores sensor glitches)
STILL_SECONDS = 20    # ...while the user has not moved at all for this long


class SafetyGuard:
    def __init__(self):
        self.high_run = 0
        self.armed = True

    def check(self, f: Features) -> str | None:
        """Return the reason for an emergency, or None if everything is fine."""
        threshold = f.hr_baseline + HIGH_HR_MARGIN
        self.high_run = self.high_run + 1 if f.hr_latest >= threshold else 0

        if not self.armed:
            # Fire once per episode; re-arm only after the heart rate has recovered.
            if f.hr_zone in ("normal", "low"):
                self.armed = True
            return None

        if self.high_run >= SUSTAIN_SECONDS and f.still_seconds >= STILL_SECONDS:
            self.armed = False
            return (f"Heart rate {f.hr_latest} bpm (normal {f.hr_baseline:.0f}) for {self.high_run}s "
                    f"while completely still for {f.still_seconds}s")
        return None
