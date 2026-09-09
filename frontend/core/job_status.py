"""Small status helpers shared by training and extraction jobs.

Callers retain their own state dictionaries, locks and execution flow.
"""

import time


def append_job_log(status: dict, message: str) -> None:
    """Append one timestamped message and retain the most recent 100 entries."""
    logs = status["logs"]
    logs.append(f"[{time.strftime('%H:%M:%S')}] {message}")
    del logs[:-100]


def finish_job(status: dict, running_field: str, message: str, *, error=None, result=None) -> None:
    """Record completion without changing job-specific progress fields."""
    status[running_field] = False
    status["status_message"] = message
    if error is not None:
        status["error_message"] = error
    if result is not None:
        status["result"] = result
