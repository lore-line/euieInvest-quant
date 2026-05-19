"""Parse ntfy /json stream events into one-line notifications for Claude Monitor.

Read JSON-per-line from stdin (output of `curl -sN https://ntfy.sh/<topic>/json`),
filter to message events, emit "title - body[:80]" per line with line-buffering.
"""

import json
import sys


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("event") != "message":
            continue
        title = event.get("title") or "(no title)"
        body = (event.get("message") or "").replace("\n", " ").replace("\r", " ")
        if len(body) > 80:
            body = body[:80] + "..."
        print(f"{title} - {body}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
