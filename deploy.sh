#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="pi@192.168.1.155"

rm -rf squirrel-daemon/.venv
ssh "$REMOTE_HOST" "killall python3" || true

# Park recordings outside the application directory before replacing it. If a
# deployment is interrupted, the backup remains at this path for recovery.
ssh "$REMOTE_HOST" bash -s <<'REMOTE_BACKUP'
set -euo pipefail

app_dir=/home/pi/squirrel-daemon
recordings_dir=/home/pi/squirrel-daemon/static/recordings
backup_dir=/home/pi/.squirrel-recordings-deploy-backup

if [ -d "$recordings_dir" ]; then
  if [ -e "$backup_dir" ]; then
    echo "Refusing to deploy: recordings backup already exists at $backup_dir" >&2
    echo "Restore or remove that backup before retrying." >&2
    exit 1
  fi
  mkdir "$backup_dir"
  mv "$recordings_dir" "$backup_dir/recordings"
fi

rm -rf "$app_dir"/*
REMOTE_BACKUP

scp -r squirrel-daemon "$REMOTE_HOST":/home/pi/

ssh "$REMOTE_HOST" bash -s <<'REMOTE_RESTORE'
set -euo pipefail

app_dir=/home/pi/squirrel-daemon
recordings_dir=/home/pi/squirrel-daemon/static/recordings
backup_dir=/home/pi/.squirrel-recordings-deploy-backup

if [ -d "$backup_dir" ]; then
  if [ -d "$backup_dir/recordings" ]; then
    mkdir -p "$app_dir/static"
    if [ -d "$recordings_dir" ]; then
      if ! rmdir "$recordings_dir"; then
        echo "Refusing to overwrite non-empty recordings directory: $recordings_dir" >&2
        echo "Preserved recordings remain at $backup_dir/recordings" >&2
        exit 1
      fi
    elif [ -e "$recordings_dir" ]; then
      echo "Refusing to overwrite non-directory path: $recordings_dir" >&2
      echo "Preserved recordings remain at $backup_dir/recordings" >&2
      exit 1
    fi
    mv "$backup_dir/recordings" "$recordings_dir"
  fi
  rmdir "$backup_dir"
fi

# uv is installed by pipx and added through the Pi user's login-shell PATH.
exec bash -lc 'cd /home/pi/squirrel-daemon && ./run.sh'
REMOTE_RESTORE
