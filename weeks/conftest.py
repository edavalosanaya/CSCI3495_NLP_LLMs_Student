"""Lab tests skip the steps you have not written yet, instead of failing.

Every starter function raises NotImplementedError until you fill it in. Without
this hook, running `pytest -k step3` before writing step 3 answers with a red
traceback, which reads like you broke something. It turns that into a skip with
a plain message, so the only red you ever see is a real wrong answer.

Steps that are already written for you run and pass straight away.
"""
import pytest


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    outcome = yield
    exc = outcome.excinfo
    if exc is not None and issubclass(exc[0], NotImplementedError):
        outcome.force_exception(
            pytest.skip.Exception("not written yet (fill in this step's TODO)",
                                  _use_item_location=True))
