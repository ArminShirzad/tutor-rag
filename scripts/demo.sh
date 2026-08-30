#!/usr/bin/env bash
# Start the demo server for an interview. Prints the URL and warms the models
# so the first question is not the slow one.
set -e
cd "$(dirname "$0")/.."
PY=".venv/Scripts/python.exe"
[ -x "$PY" ] || PY=".venv/bin/python"
[ -x "$PY" ] || PY="python"
echo "starting tutor-rag on http://localhost:8000 ..."
exec "$PY" -m uvicorn app.api:app --host 0.0.0.0 --port 8000
