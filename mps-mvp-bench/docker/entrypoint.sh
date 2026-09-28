#!/bin/sh
# Signal-forwarding entrypoint.
#
# Why this exists: the F3 safe-exit case requires SIGTERM/SIGINT to reach the CUDA
# worker. A shell that runs the worker without `exec` (or that ignores signals)
# would swallow them and make the test meaningless.
#
# Strategy: run the worker in the background, forward SIGTERM/SIGINT/SIGHUP to it,
# then wait for its real exit status. `tini -g` (the image ENTRYPOINT) reaps any
# remaining children, so we do not implement reaping here.
set -eu

child=""

forward() {
  sig="$1"
  if [ -n "$child" ]; then
    # Signal the process group so helper subprocesses are included.
    kill -"$sig" "$child" 2>/dev/null || true
  fi
}

trap 'forward TERM' TERM
trap 'forward INT' INT
trap 'forward HUP' HUP

# The worker writes its status/evidence under /results; make sure it exists even
# when no bind mount was supplied.
mkdir -p /results 2>/dev/null || true

if [ "$#" -eq 0 ]; then
  echo "entrypoint: 未提供命令" >&2
  exit 64
fi

"$@" &
child=$!

# `wait` returns early when a trapped signal arrives; loop until the child is
# actually gone so we report its true exit status.
status=0
while :; do
  if wait "$child"; then
    status=0
    break
  else
    status=$?
    # 128+n means we were interrupted by signal n; keep waiting for the child to
    # finish its own fence -> drain -> exit sequence.
    if [ "$status" -gt 128 ] && kill -0 "$child" 2>/dev/null; then
      continue
    fi
    break
  fi
done

exit "$status"
