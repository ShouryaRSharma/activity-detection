from activity_detection.logging_config import setup_logger


def test_setup_logger_does_not_duplicate_handlers():
    name = "test_idempotent_logger"
    first = setup_logger(name)
    handler_count = len(first.handlers)

    second = setup_logger(name)

    assert second is first
    assert len(second.handlers) == handler_count
    assert second.propagate is False
