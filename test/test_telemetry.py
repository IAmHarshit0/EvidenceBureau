from src.evidence_bureau.telemetry import start_trace, start_timer, elapsed_ms


def test_start_trace():
    trace = start_trace()

    assert "trace_id" in trace
    assert "started_at" in trace
    assert "_start_time" in trace


def test_timer():
    timer = start_timer()

    elapsed = elapsed_ms(timer)

    assert elapsed >= 0