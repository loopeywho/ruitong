#!/usr/bin/env bash
# Watches this repo's git HEAD and fires a macOS notification whenever a new
# commit lands (e.g. a Kimi/RedStar round). NOTIFY ONLY — approved scope,
# 2026-07-28. Does not switch models, does not launch Claude, does not touch
# the repo. It exists because no layer of this stack (hooks, plugins, cron,
# scheduled tasks) can change a running session's model — see AUDIT_ROUND /
# LESSONS discussion the same day. This sidesteps that by staying a human
# reminder rather than an automated switch.
#
# Runs as a plain bash loop entirely OUTSIDE any Claude Code session — costs
# zero tokens while it waits, same principle as tools/grab_mi300x.sh. Dies if
# the Mac sleeps/reboots, or when you kill it:
#   pkill -f watch_redstar_commits.sh
#
# Usage:
#   nohup tools/watch_redstar_commits.sh > /tmp/redstar_watch.log 2>&1 &

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_FILE="/tmp/.redstar_last_sha"
POLL="${1:-300}"   # default: check every 5 minutes

cd "$REPO"

LAST=$(cat "$STATE_FILE" 2>/dev/null || git rev-parse HEAD)
echo "$LAST" > "$STATE_FILE"
echo "[$(date +%H:%M:%S)] watching $(basename "$REPO") from $LAST, poll ${POLL}s"

while :; do
  CUR=$(git rev-parse HEAD 2>/dev/null || echo "$LAST")
  if [ "$CUR" != "$LAST" ]; then
    SHA_SHORT=$(git rev-parse --short "$CUR")
    SUBJECT=$(git log -1 --format='%s' "$CUR")
    echo "[$(date +%H:%M:%S)] new commit $SHA_SHORT: $SUBJECT"

    # Best-effort — a missing osascript/notification permission must not
    # kill the loop. The log line above is the fallback record either way.
    osascript -e "display notification \"${SUBJECT//\"/\\\"}\" with title \"Ruitong: new commit $SHA_SHORT\" subtitle \"Check if this is a Kimi round — switch to Opus to audit\" sound name \"Glass\"" \
      >/dev/null 2>&1 || true

    LAST="$CUR"
    echo "$LAST" > "$STATE_FILE"
  fi
  sleep "$POLL"
done
