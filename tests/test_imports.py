"""Basic import smoke tests for core modules."""


def test_core_modules_import():
    import agents.fundamentals_agent  # noqa: F401
    import agents.news_sentiment  # noqa: F401
    import agents.portfolio_decision  # noqa: F401
    import agents.risk_style  # noqa: F401
    import agents.technical_forecaster  # noqa: F401
    import backend.log_parser  # noqa: F401
    import backend.server  # noqa: F401
    import orchestrator.daily_pipeline  # noqa: F401
    import risk.risk_manager  # noqa: F401

    assert True
