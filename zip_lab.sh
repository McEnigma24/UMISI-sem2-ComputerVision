#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

list_labs() {
  local d
  for d in "$ROOT"/lab*/; do
    [[ -d "$d" ]] && basename "$d"
  done
}

usage() {
  echo "Użycie: $(basename "$0") <katalog_lab>" >&2
  echo "" >&2
  echo "Dostępne katalogi:" >&2
  list_labs
}

if [[ $# -ne 1 ]]; then
  usage
  exit 1
fi

LAB="$1"

if [[ ! -d "$LAB" ]]; then
  echo "Błąd: nie ma katalogu „$LAB”." >&2
  usage
  exit 1
fi

if [[ ! "$LAB" =~ ^lab ]]; then
  echo "Błąd: oczekuję nazwy typu lab01, lab02, …" >&2
  exit 1
fi

OUT="${LAB}.zip"
if [[ -f "$OUT" ]]; then
  echo "Usuwam istniejący: $OUT"
  rm -f "$OUT"
fi

zip -r "$OUT" "$LAB" \
  -x "${LAB}/.venv/*" \
  -x "${LAB}/.venv/**" \
  -x "*/__pycache__/*" \
  -x "*__pycache__*" \
  -x "*.pyc" \
  -x "*models/yolo/*" \
  -x "*.pt*"

echo "Utworzono: $ROOT/$OUT"
