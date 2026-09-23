"""Host integration contract for summarizer-budget ownership."""

from hermes_lcm.engine import LCMEngine


def test_lcm_explicitly_owns_summary_request_budget():
    """The host must never infer this safety property from engine type alone."""
    assert LCMEngine.owns_summary_request_budget is True
