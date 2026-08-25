import json
import time
import uuid
import traceback

from pathlib import Path
from datetime import datetime, timezone


TELEMETRY_DIR = Path("data/telemetry")
TELEMETRY_FILE = TELEMETRY_DIR / "events.jsonl"


def start_trace() -> dict:
    """Create a new investigation trace."""

    return {
        "trace_id": str(uuid.uuid4()),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "_start_time": time.perf_counter(),
        "status": "in_progress",
    }


def finish_trace(trace: dict) -> dict:
    """Finish the trace and calculate total runtime."""

    trace["total_ms"] = round(
        (time.perf_counter() - trace["_start_time"]) * 1000,
        2,
    )

    trace["finished_at"] = datetime.now(timezone.utc).isoformat()

    if trace["status"] == "in_progress":
        trace["status"] = "success"

    # Internal timer should not be saved.
    trace.pop("_start_time", None)

    return trace


def record_error(
    trace: dict,
    error: Exception,
    stage: str,
) -> dict:
    """Mark a trace as failed and attach error details."""

    trace["status"] = "error"
    trace["failed_stage"] = stage

    trace["error"] = {
        "type": type(error).__name__,
        "message": str(error),
        "traceback": traceback.format_exc(),
    }

    # Finish the trace so failed requests also
    # contain total runtime and finished timestamp.
    finish_trace(trace)

    return trace


def save_trace(trace: dict) -> bool:
    """
    Append a completed trace to the JSONL telemetry file.

    Returns True on success and False if telemetry itself
    fails to write. A telemetry failure should never crash
    the application.
    """

    try:
        TELEMETRY_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        with open(
            TELEMETRY_FILE,
            "a",
            encoding="utf-8",
        ) as file:

            file.write(
                json.dumps(
                    trace,
                    ensure_ascii=False,
                    default=str,
                )
                + "\n"
            )

        return True

    except Exception as error:
        print(
            f"[telemetry] failed to save trace "
            f"{trace.get('trace_id', '?')}: {error}"
        )

        return False


def start_timer() -> float:
    """Start a high-resolution timer."""

    return time.perf_counter()


def elapsed_ms(start_time: float) -> float:
    """Return elapsed time in milliseconds."""

    return round(
        (time.perf_counter() - start_time) * 1000,
        2,
    )