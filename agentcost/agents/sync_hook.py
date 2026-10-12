"""
The hook that runs when a coding agent finishes a turn or starts a session.

It starts ``agentcost sync`` in a process of its own and returns at once. The
sync cannot run inside the hook: an agent kills its hooks when it exits, and a
headless run exits the moment its turn ends, so the session's last turn would
never be sent by the session that made it.

Run by file path, and standard library only, so the agent waits on little more
than an interpreter starting.
"""

import os
import subprocess
import sys


def main() -> int:
    command = [sys.executable, "-m", "agentcost.cli", "sync", "--quiet"]
    quiet = dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        if os.name != "nt":
            subprocess.Popen(command, start_new_session=True, **quiet)
            return 0
        detached = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        try:
            # Out of the agent's job object too, where the job allows it.
            subprocess.Popen(
                command, creationflags=detached | subprocess.CREATE_BREAKAWAY_FROM_JOB, **quiet
            )
        except OSError:
            subprocess.Popen(command, creationflags=detached, **quiet)
    except Exception:  # noqa: BLE001
        pass  # a hook must never fail the agent's turn
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
