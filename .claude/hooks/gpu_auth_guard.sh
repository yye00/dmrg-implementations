#!/bin/bash
# Path B GPU authorization guard.
#
# Blocks any Bash command that would touch the MI300X host (Hot Aisle)
# unless PATH_B_GPU_AUTH=1 is exported in the session environment.
#
# Purpose: MI300X time costs money. Prevent agents or automation from
# kicking off a campaign by accident. The human exports the variable
# when a Hot Aisle window has been scheduled.
#
# This is a PreToolUse hook for the Bash tool. It reads the tool input
# JSON on stdin and emits a JSON decision on stdout.

set -u

input=$(cat)

# Extract the command field from the tool input.
cmd=$(printf '%s' "$input" | jq -r '.tool_input.command // ""' 2>/dev/null || echo "")

# Patterns that reach the MI300X host.
if printf '%s' "$cmd" | grep -qE '(ssh[[:space:]]+hotaisle@|tmux[[:space:]]+send-keys[[:space:]]+-t[[:space:]]+test_remote|tmux[[:space:]]+attach[[:space:]]+-t[[:space:]]+test_remote)'; then
  if [ "${PATH_B_GPU_AUTH:-0}" != "1" ]; then
    cat <<'EOF'
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "deny",
    "permissionDecisionReason": "MI300X access blocked. Export PATH_B_GPU_AUTH=1 in your shell before launching Claude Code to authorize Hot Aisle operations (these consume GPU-hours)."
  }
}
EOF
    exit 0
  fi
fi

exit 0
