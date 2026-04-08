#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$ROOT_DIR/.env"
ENV_EXAMPLE="$ROOT_DIR/.env.example"
PROXY_SERVICE_NAME="completions-proxy"
WEB_SERVICE_NAME="nextchat-web"

DEFAULT_APP_HOST="127.0.0.1"
DEFAULT_APP_PORT="18010"
DEFAULT_WEB_PORT="38080"
DEFAULT_WEB_CODE=""
DEFAULT_UPSTREAM_BASE_URL="https://sub.lnz-study.com"
DEFAULT_UPSTREAM_MODE="responses"
DEFAULT_UPSTREAM_STREAMING_ENABLED="true"
DEFAULT_UPSTREAM_MESSAGES_API_VERSION="2023-06-01"
DEFAULT_UPSTREAM_TIMEOUT_SECONDS="120"
DEFAULT_UPSTREAM_MIN_REQUEST_INTERVAL_SECONDS="10"
DEFAULT_UPSTREAM_FALLBACK_MODEL=""
DEFAULT_DEFAULT_UPSTREAM_MODEL="gpt-5.4"
DEFAULT_DEFAULT_REASONING_EFFORT="medium"
DEFAULT_DEFAULT_UPSTREAM_SPEED="fast"
DEFAULT_RAW_IO_LOG_ENABLED="false"
DEFAULT_RAW_IO_LOG_PATH="logs/raw_io.jsonl"
DEFAULT_RAW_IO_LOG_MAX_CHARS="120000"
DEFAULT_RAW_IO_LOG_KEEP_REQUESTS="10"

info() {
  printf '[info] %s\n' "$*"
}

warn() {
  printf '[warn] %s\n' "$*" >&2
}

die() {
  printf '[error] %s\n' "$*" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

require_prerequisites() {
  command_exists docker || die "Docker is required."
  docker compose version >/dev/null 2>&1 || die "'docker compose' is required."
  docker info >/dev/null 2>&1 || die "Docker daemon is not reachable."
}

compose() {
  (
    cd "$ROOT_DIR"
    docker compose "$@"
  )
}

git_repo_exists() {
  git -C "$ROOT_DIR" rev-parse --is-inside-work-tree >/dev/null 2>&1
}

git_worktree_clean() {
  [ -z "$(git -C "$ROOT_DIR" status --porcelain)" ]
}

get_env_value() {
  local key="$1"
  local source_file="$2"
  if [ ! -f "$source_file" ]; then
    return 1
  fi
  awk -F= -v key="$key" '
    $0 ~ "^[[:space:]]*" key "=" {
      sub(/^[^=]*=/, "", $0)
      print $0
      found = 1
      exit
    }
    END {
      if (!found) {
        exit 1
      }
    }
  ' "$source_file"
}

current_or_default() {
  local key="$1"
  local fallback="$2"
  local value=""
  if value="$(get_env_value "$key" "$ENV_FILE" 2>/dev/null)"; then
    printf '%s' "$value"
    return 0
  fi
  if value="$(get_env_value "$key" "$ENV_EXAMPLE" 2>/dev/null)"; then
    printf '%s' "$value"
    return 0
  fi
  printf '%s' "$fallback"
}

mask_secret() {
  local value="$1"
  if [ -z "$value" ]; then
    printf '%s' ""
    return 0
  fi
  local length="${#value}"
  if [ "$length" -le 8 ]; then
    printf '%s' '********'
    return 0
  fi
  printf '%s...%s' "${value:0:4}" "${value:length-4:4}"
}

prompt_text() {
  local label="$1"
  local default_value="$2"
  local reply=""

  if [ -n "$default_value" ]; then
    read -r -p "$label [$default_value]: " reply || true
  else
    read -r -p "$label: " reply || true
  fi

  if [ "$reply" = "-" ]; then
    printf '%s' ""
  elif [ -n "$reply" ]; then
    printf '%s' "$reply"
  else
    printf '%s' "$default_value"
  fi
}

prompt_secret() {
  local label="$1"
  local default_value="$2"
  local masked_default=""
  local reply=""

  masked_default="$(mask_secret "$default_value")"

  if [ -n "$masked_default" ]; then
    read -r -s -p "$label [$masked_default]: " reply || true
  else
    read -r -s -p "$label: " reply || true
  fi
  printf '\n' >&2

  if [ -n "$reply" ]; then
    printf '%s' "$reply"
  else
    printf '%s' "$default_value"
  fi
}

validate_required() {
  local name="$1"
  local value="$2"
  [ -n "$value" ] || die "$name cannot be empty."
}

write_env_file() {
  local tmp_file
  tmp_file="$(mktemp "${TMPDIR:-/tmp}/proxy-env.XXXXXX")"

  cat >"$tmp_file" <<EOF
APP_HOST=$APP_HOST
APP_PORT=$APP_PORT
WEB_PORT=$WEB_PORT
WEB_CODE=$WEB_CODE
UPSTREAM_BASE_URL=$UPSTREAM_BASE_URL
UPSTREAM_MODE=$UPSTREAM_MODE
UPSTREAM_STREAMING_ENABLED=$UPSTREAM_STREAMING_ENABLED
UPSTREAM_API_KEY=$UPSTREAM_API_KEY
UPSTREAM_MESSAGES_API_VERSION=$UPSTREAM_MESSAGES_API_VERSION
UPSTREAM_TIMEOUT_SECONDS=$UPSTREAM_TIMEOUT_SECONDS
UPSTREAM_MIN_REQUEST_INTERVAL_SECONDS=$UPSTREAM_MIN_REQUEST_INTERVAL_SECONDS
UPSTREAM_FALLBACK_MODEL=$UPSTREAM_FALLBACK_MODEL
USE_FORCE_MODEL=$USE_FORCE_MODEL
# FORCE_UPSTREAM_MODEL=gemini-3.1-pro-high,claude-opus-4-6,gpt-5.3-codex
# FORCE_UPSTREAM_MODEL=["gemini-3.1-pro-high","claude-opus-4-6","gpt-5.3-codex"]
FORCE_UPSTREAM_MODEL=$FORCE_UPSTREAM_MODEL
DEFAULT_UPSTREAM_MODEL=$DEFAULT_UPSTREAM_MODEL
DEFAULT_REASONING_EFFORT=$DEFAULT_REASONING_EFFORT
DEFAULT_UPSTREAM_SPEED=$DEFAULT_UPSTREAM_SPEED

# Raw IO debug logging
RAW_IO_LOG_ENABLED=$RAW_IO_LOG_ENABLED
RAW_IO_LOG_PATH=$RAW_IO_LOG_PATH
RAW_IO_LOG_MAX_CHARS=$RAW_IO_LOG_MAX_CHARS
RAW_IO_LOG_KEEP_REQUESTS=$RAW_IO_LOG_KEEP_REQUESTS
EOF

  if [ -f "$ENV_FILE" ]; then
    cp "$ENV_FILE" "$ENV_FILE.bak.$(date +%Y%m%d-%H%M%S)"
  fi

  mv "$tmp_file" "$ENV_FILE"
  chmod 600 "$ENV_FILE" 2>/dev/null || true
  info "Wrote $ENV_FILE"
}

collect_config() {
  info "Interactive configuration"
  info "Only essential fields are prompted."
  info "Advanced settings can be edited directly in .env."

  APP_HOST="$(current_or_default APP_HOST "$DEFAULT_APP_HOST")"
  APP_PORT="$(current_or_default APP_PORT "$DEFAULT_APP_PORT")"
  WEB_PORT="$(prompt_text "Web UI port" "$(current_or_default WEB_PORT "$DEFAULT_WEB_PORT")")"
  WEB_CODE="$(current_or_default WEB_CODE "$DEFAULT_WEB_CODE")"
  UPSTREAM_BASE_URL="$(prompt_text "Upstream base URL" "$(current_or_default UPSTREAM_BASE_URL "$DEFAULT_UPSTREAM_BASE_URL")")"
  UPSTREAM_MODE="$(prompt_text "Upstream mode (responses/messages)" "$(current_or_default UPSTREAM_MODE "$DEFAULT_UPSTREAM_MODE")")"
  UPSTREAM_API_KEY="$(prompt_secret "Upstream API key" "$(current_or_default UPSTREAM_API_KEY "")")"
  DEFAULT_UPSTREAM_MODEL="$(prompt_text "Default upstream model" "$(current_or_default DEFAULT_UPSTREAM_MODEL "$DEFAULT_DEFAULT_UPSTREAM_MODEL")")"

  UPSTREAM_STREAMING_ENABLED="$(current_or_default UPSTREAM_STREAMING_ENABLED "$DEFAULT_UPSTREAM_STREAMING_ENABLED")"
  UPSTREAM_MESSAGES_API_VERSION="$(current_or_default UPSTREAM_MESSAGES_API_VERSION "$DEFAULT_UPSTREAM_MESSAGES_API_VERSION")"
  UPSTREAM_TIMEOUT_SECONDS="$(current_or_default UPSTREAM_TIMEOUT_SECONDS "$DEFAULT_UPSTREAM_TIMEOUT_SECONDS")"
  UPSTREAM_MIN_REQUEST_INTERVAL_SECONDS="$(current_or_default UPSTREAM_MIN_REQUEST_INTERVAL_SECONDS "$DEFAULT_UPSTREAM_MIN_REQUEST_INTERVAL_SECONDS")"
  UPSTREAM_FALLBACK_MODEL="$(current_or_default UPSTREAM_FALLBACK_MODEL "$DEFAULT_UPSTREAM_FALLBACK_MODEL")"
  DEFAULT_REASONING_EFFORT="$(current_or_default DEFAULT_REASONING_EFFORT "$DEFAULT_DEFAULT_REASONING_EFFORT")"
  DEFAULT_UPSTREAM_SPEED="$(current_or_default DEFAULT_UPSTREAM_SPEED "$DEFAULT_DEFAULT_UPSTREAM_SPEED")"
  USE_FORCE_MODEL="$(current_or_default USE_FORCE_MODEL "false")"
  FORCE_UPSTREAM_MODEL="$(current_or_default FORCE_UPSTREAM_MODEL "")"
  RAW_IO_LOG_ENABLED="$(current_or_default RAW_IO_LOG_ENABLED "$DEFAULT_RAW_IO_LOG_ENABLED")"
  RAW_IO_LOG_PATH="$(current_or_default RAW_IO_LOG_PATH "$DEFAULT_RAW_IO_LOG_PATH")"
  RAW_IO_LOG_MAX_CHARS="$(current_or_default RAW_IO_LOG_MAX_CHARS "$DEFAULT_RAW_IO_LOG_MAX_CHARS")"
  RAW_IO_LOG_KEEP_REQUESTS="$(current_or_default RAW_IO_LOG_KEEP_REQUESTS "$DEFAULT_RAW_IO_LOG_KEEP_REQUESTS")"

  validate_required "APP_HOST" "$APP_HOST"
  validate_required "APP_PORT" "$APP_PORT"
  validate_required "WEB_PORT" "$WEB_PORT"
  validate_required "UPSTREAM_BASE_URL" "$UPSTREAM_BASE_URL"
  validate_required "UPSTREAM_MODE" "$UPSTREAM_MODE"
  validate_required "UPSTREAM_API_KEY" "$UPSTREAM_API_KEY"
  validate_required "DEFAULT_UPSTREAM_MODEL" "$DEFAULT_UPSTREAM_MODEL"
}

wait_for_health() {
  local port
  port="$(current_or_default APP_PORT "$DEFAULT_APP_PORT")"
  local url="http://127.0.0.1:${port}/healthz"
  local attempt

  for attempt in $(seq 1 30); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      info "Service is healthy at $url"
      return 0
    fi
    sleep 1
  done

  warn "Health check did not succeed within 30 seconds: $url"
  return 1
}

docker_up() {
  require_prerequisites
  [ -f "$ENV_FILE" ] || die "Missing $ENV_FILE. Run './proxy.sh setup' first."
  info "Recreating $PROXY_SERVICE_NAME and $WEB_SERVICE_NAME"
  compose rm -sf "$PROXY_SERVICE_NAME" "$WEB_SERVICE_NAME" >/dev/null 2>&1 || true
  compose up -d --build "$PROXY_SERVICE_NAME" "$WEB_SERVICE_NAME"
  wait_for_health || true
  local web_port
  web_port="$(current_or_default WEB_PORT "$DEFAULT_WEB_PORT")"
  info "Proxy: http://127.0.0.1:$(current_or_default APP_PORT "$DEFAULT_APP_PORT")"
  info "Web UI: http://127.0.0.1:${web_port}"
}

docker_down() {
  require_prerequisites
  compose stop "$PROXY_SERVICE_NAME" "$WEB_SERVICE_NAME" || true
}

docker_logs() {
  require_prerequisites
  compose logs -f "$PROXY_SERVICE_NAME" "$WEB_SERVICE_NAME"
}

docker_status() {
  require_prerequisites
  compose ps
  wait_for_health || true
}

run_setup() {
  require_prerequisites
  collect_config
  write_env_file
  docker_up
}

run_config() {
  require_prerequisites
  collect_config
  write_env_file
  docker_up
}

run_update() {
  require_prerequisites

  if git_repo_exists; then
    if git_worktree_clean; then
      local current_branch
      current_branch="$(git -C "$ROOT_DIR" symbolic-ref --quiet --short HEAD 2>/dev/null || true)"
      if [ -n "$current_branch" ] && git -C "$ROOT_DIR" rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' >/dev/null 2>&1; then
        info "Pulling latest code on branch '$current_branch'"
        git -C "$ROOT_DIR" pull --rebase --autostash
      else
        warn "Skipping git pull because this branch has no upstream."
      fi
    else
      warn "Skipping git pull because the worktree has local changes."
    fi
  fi

  docker_up
}

show_help() {
  cat <<'EOF'
Usage:
  ./proxy.sh setup    # ask only web port / base URL / API key / upstream mode / default model, then start
  ./proxy.sh config   # same minimal reconfigure flow, then recreate the containers
  ./proxy.sh up       # recreate/start the proxy and web ui using the current .env
  ./proxy.sh down     # stop the proxy and web ui
  ./proxy.sh logs     # tail proxy and web ui logs
  ./proxy.sh status   # show compose status and health check
  ./proxy.sh update   # try git pull (if safe), then recreate the containers
  ./proxy.sh help
EOF
}

main() {
  local command="${1:-help}"

  case "$command" in
    setup)
      run_setup
      ;;
    config)
      run_config
      ;;
    up|start|restart)
      docker_up
      ;;
    down|stop)
      docker_down
      ;;
    logs)
      docker_logs
      ;;
    status)
      docker_status
      ;;
    update)
      run_update
      ;;
    help|-h|--help)
      show_help
      ;;
    *)
      die "Unknown command: $command. Run './proxy.sh help'."
      ;;
  esac
}

main "$@"
