"""Run one isolated test process and reap its remaining process group.

Linux GPU workers inherit the group's ID. Cleanup targets only the group
created here, so unrelated services and other users' GPU work are untouched.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def live_group_members(group):
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            # comm can contain spaces and parentheses; fields after its final
            # ')' begin with state, ppid, pgrp.
            fields = (entry / "stat").read_text(errors="replace").rsplit(")", 1)[1].split()
            if int(fields[2]) == group and fields[0] != "Z":
                members.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return members


def cleanup_group(group, timeout):
    remaining = live_group_members(group)
    if not remaining:
        return []
    for sig, grace in [(signal.SIGTERM, timeout), (signal.SIGKILL, 5.0)]:
        try:
            os.killpg(group, sig)
        except ProcessLookupError:
            return []
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            remaining = live_group_members(group)
            if not remaining:
                return []
            time.sleep(0.1)
    return remaining


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result")
    parser.add_argument("--cleanup-timeout", type=float, default=10.0)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command or args.cleanup_timeout < 0:
        parser.error("provide a command and a non-negative cleanup timeout")
    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    started = time.monotonic()
    process = subprocess.Popen(command, start_new_session=True)
    interrupted = False
    try:
        code = process.wait()
    except KeyboardInterrupt:
        interrupted = True
        code = 130
    finally:
        cleanup_started = time.monotonic()
        remaining = cleanup_group(process.pid, args.cleanup_timeout)
        if interrupted:
            process.wait(timeout=5)
    row = {
        "command": command,
        "exit_code": code,
        "seconds": round(time.monotonic() - started, 3),
        "cleanup_seconds": round(time.monotonic() - cleanup_started, 3),
        "remaining_pids": remaining,
    }
    if args.result:
        Path(args.result).write_text(json.dumps(row, indent=2) + "\n")
    print("PROCESS " + json.dumps(row), flush=True)
    if remaining:
        print(f"Test workers did not exit: {remaining}", file=sys.stderr)
        return 1
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    raise SystemExit(main())
