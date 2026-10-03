#!/usr/bin/env bash
# One script: setup, start, stop (also restart/status/logs) for the CGNF
# Recommender's Flask backend and Vite frontend. `start` runs setup first
# automatically. Run from anywhere; paths below are resolved relative to
# this script.
set -euo pipefail

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPTS_DIR/.." && pwd)"
RUN_DIR="$PROJECT_ROOT/.run"
mkdir -p "$RUN_DIR"

BACKEND_PORT="${BACKEND_PORT:-5003}"
FRONTEND_PORT="${FRONTEND_PORT:-5174}"

backend_pid_file="$RUN_DIR/backend.pid"
frontend_pid_file="$RUN_DIR/frontend.pid"
backend_log="$RUN_DIR/backend.log"
frontend_log="$RUN_DIR/frontend.log"

python_bin() {
  if [ -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    echo "$PROJECT_ROOT/.venv/bin/python"
  else
    command -v python3
  fi
}

setup_backend() {
  if [ ! -x "$PROJECT_ROOT/.venv/bin/python" ]; then
    echo "creating venv at $PROJECT_ROOT/.venv ..."
    python3 -m venv "$PROJECT_ROOT/.venv"
  fi
  local py; py="$(python_bin)"
  if "$py" -c "import flask, flask_cors, numpy, scipy, pandas" 2>/dev/null; then
    echo "python deps already installed, skipping"
  else
    echo "installing python deps..."
    "$py" -m pip install -q --upgrade pip
    "$py" -m pip install -q -r "$PROJECT_ROOT/backend/requirements.txt"
  fi
  if [ ! -f "$PROJECT_ROOT/input/movielens-1m/ratings.dat" ]; then
    echo "WARNING: $PROJECT_ROOT/input/movielens-1m/ratings.dat not found." >&2
  fi
}

setup_frontend() {
  if [ -d "$PROJECT_ROOT/frontend/node_modules" ]; then
    echo "frontend deps already installed, skipping"
  else
    echo "installing frontend deps..."
    ( cd "$PROJECT_ROOT/frontend" && npm install )
  fi
}

setup_all() { setup_backend; setup_frontend; }

port_listener_pids() { lsof -ti ":$1" -sTCP:LISTEN 2>/dev/null || true; }
is_running() { [ -n "$(port_listener_pids "$1")" ]; }
wait_for_port() {
  local port="$1" tries=60
  while [ "$tries" -gt 0 ]; do
    [ -n "$(port_listener_pids "$port")" ] && return 0
    sleep 0.5
    tries=$((tries - 1))
  done
  return 1
}

start_backend() {
  if is_running "$BACKEND_PORT"; then
    echo "backend  already running on port $BACKEND_PORT"
    return
  fi
  local py; py="$(python_bin)"
  cd "$PROJECT_ROOT/backend"
  CGNF_PORT="$BACKEND_PORT" nohup "$py" -m app.api >"$backend_log" 2>&1 &
  if wait_for_port "$BACKEND_PORT"; then
    echo "backend  started: http://127.0.0.1:$BACKEND_PORT  (log $backend_log)"
    port_listener_pids "$BACKEND_PORT" >"$backend_pid_file"
  else
    echo "backend  failed to start — last log lines:"; tail -n 30 "$backend_log" || true
    exit 1
  fi
}

start_frontend() {
  if is_running "$FRONTEND_PORT"; then
    echo "frontend already running on port $FRONTEND_PORT"
    return
  fi
  cd "$PROJECT_ROOT/frontend"
  VITE_API_BASE="http://127.0.0.1:$BACKEND_PORT" nohup npm run dev -- --port "$FRONTEND_PORT" --strictPort >"$frontend_log" 2>&1 &
  if wait_for_port "$FRONTEND_PORT"; then
    echo "frontend started: http://localhost:$FRONTEND_PORT  (log $frontend_log)"
    port_listener_pids "$FRONTEND_PORT" >"$frontend_pid_file"
  else
    echo "frontend failed to start — last log lines:"; tail -n 30 "$frontend_log" || true
    exit 1
  fi
}

stop_one() {
  local name="$1" port="$2" pid_file="$3"
  local pids; pids="$(port_listener_pids "$port")"
  if [ -n "$pids" ]; then
    echo "$pids" | xargs kill 2>/dev/null || true
    sleep 1
    pids="$(port_listener_pids "$port")"
    [ -n "$pids" ] && echo "$pids" | xargs kill -9 2>/dev/null || true
    echo "$name stopped (port $port)"
  else
    echo "$name not running"
  fi
  rm -f "$pid_file"
}

status_one() {
  local name="$1" port="$2" url="$3"
  if is_running "$port"; then echo "$name: running on port $port — $url"; else echo "$name: stopped"; fi
}

case "${1:-}" in
  setup)
    case "${2:-}" in
      backend) setup_backend ;;
      frontend) setup_frontend ;;
      "" ) setup_all ;;
      *) echo "usage: $0 setup [backend|frontend]" >&2; exit 1 ;;
    esac
    ;;
  start)
    case "${2:-}" in
      backend) setup_backend; start_backend ;;
      frontend) setup_frontend; start_frontend ;;
      "" ) setup_all; start_backend; start_frontend ;;
      *) echo "usage: $0 start [backend|frontend]" >&2; exit 1 ;;
    esac
    ;;
  stop)
    case "${2:-}" in
      backend) stop_one backend "$BACKEND_PORT" "$backend_pid_file" ;;
      frontend) stop_one frontend "$FRONTEND_PORT" "$frontend_pid_file" ;;
      "" ) stop_one backend "$BACKEND_PORT" "$backend_pid_file"; stop_one frontend "$FRONTEND_PORT" "$frontend_pid_file" ;;
      *) echo "usage: $0 stop [backend|frontend]" >&2; exit 1 ;;
    esac
    ;;
  restart)
    "$0" stop "${2:-}"
    "$0" start "${2:-}"
    ;;
  status)
    status_one backend "$BACKEND_PORT" "http://127.0.0.1:$BACKEND_PORT"
    status_one frontend "$FRONTEND_PORT" "http://localhost:$FRONTEND_PORT"
    ;;
  logs)
    case "${2:-}" in
      backend) tail -n 80 -f "$backend_log" ;;
      frontend) tail -n 80 -f "$frontend_log" ;;
      *) echo "usage: $0 logs backend|frontend" >&2; exit 1 ;;
    esac
    ;;
  *)
    echo "usage: $0 {setup|start|stop|restart|status|logs} [backend|frontend]" >&2
    exit 1
    ;;
esac
