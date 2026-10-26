import os
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


def test_dashboard_copies_streamlit_secrets_into_the_environment():
    os.environ.pop("CE_TEST_SECRET", None)
    try:
        at = AppTest.from_file(str(APP), default_timeout=120)
        at.secrets["CE_TEST_SECRET"] = "from-secrets"
        at.run()
        assert not at.exception
        assert os.environ.get("CE_TEST_SECRET") == "from-secrets"
    finally:
        os.environ.pop("CE_TEST_SECRET", None)
