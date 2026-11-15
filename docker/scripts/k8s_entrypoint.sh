#!/bin/bash
set -e

echo "[$(date)] Starting Claude agent execution"
echo "[$(date)] Execution ID: ${EXECUTION_ID}"
echo "[$(date)] Agent: ${AGENT_NAME}"

# Run Claude agent and capture output.
# The call is PIPED into tee, so `set -e` never sees claude's status -- the
# pipeline's status is tee's, which is 0 even when `claude` is missing. That is
# how this script printed "completed successfully" after running nothing at all.
# Read claude's own status out of PIPESTATUS instead. (Deliberately NOT a global
# `set -o pipefail`: this keeps the change to the one pipeline that needs it and
# cannot alter the status of anything else in the script.)
echo "[$(date)] Running Claude Code CLI..."
claude_status=0
claude --print "${AGENT_PROMPT}" | tee /workspace/output.txt || true
claude_status=${PIPESTATUS[0]}

# Check if execution was successful
if [ "${claude_status}" -eq 0 ]; then
    echo "[$(date)] Claude Code CLI execution completed successfully"
else
    echo "[$(date)] Claude Code CLI execution failed with exit code ${claude_status}"
    exit 1
fi

# Collect and upload artifacts to MinIO if configured
if [[ -n "${MINIO_ENDPOINT}" ]]; then
    echo "[$(date)] Uploading artifacts to MinIO..."
    # `set -e` would abort here on a non-zero exit, so the `if` below could
    # never reach its else branch -- capture the status instead, which is what
    # the "don't fail the overall job" intent below actually requires.
    upload_status=0
    /usr/local/bin/collect_artifacts.sh || upload_status=$?
    if [ "${upload_status}" -eq 0 ]; then
        echo "[$(date)] Artifact upload completed successfully"
    else
        echo "[$(date)] Artifact upload failed with exit code ${upload_status}"
        # Don't fail the overall job for artifact upload issues
    fi
fi

echo "[$(date)] Job execution complete"
exit 0
