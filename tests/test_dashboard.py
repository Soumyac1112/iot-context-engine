from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parent.parent / "dashboard.py"


def test_dashboard_runs_a_story_end_to_end():
    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception
    at.sidebar.select_slider[0].set_value("As fast as possible")
    at.sidebar.button[0].click()
    at.run()
    assert not at.exception
    assert at.success or at.error  # the PASS/FAIL verdict is shown
