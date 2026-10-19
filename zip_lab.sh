#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

SMALL=0
SPLIT=""

list_labs() {
  local d
  for d in "$ROOT"/lab*/; do
    [[ -d "$d" ]] && basename "$d"
  done
}

split_size_for_split() {
  # GNU split: 95m -> 95M
  local s="$1"
  if [[ "$s" =~ ^([0-9]+)[mM]$ ]]; then
    echo "${BASH_REMATCH[1]}M"
  elif [[ "$s" =~ ^([0-9]+)[kK]$ ]]; then
    echo "${BASH_REMATCH[1]}K"
  elif [[ "$s" =~ ^([0-9]+)[gG]$ ]]; then
    echo "${BASH_REMATCH[1]}G"
  else
    echo "$s"
  fi
}

usage() {
  cat >&2 <<EOF
Użycie: $(basename "$0") [opcje] <katalog_lab>

Opcje:
  --small       bez .venv, __pycache__, models/yolo/, *.pyc, *.pt
  --split SIZE  podziel na części (np. 95m) -> lab02.01.zip, lab02.02.zip, …
  -h, --help    ta pomoc

Domyślnie: cały katalog lab (nic nie pomijamy), jeden plik labNN.zip.

Przykłady:
  $(basename "$0") lab02
  $(basename "$0") --small lab02
  $(basename "$0") --small --split 95m lab02

Składanie po --split: cat lab02.0*.zip > lab02.zip

Dostępne katalogi:
EOF
  list_labs >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --small)
      SMALL=1
      shift
      ;;
    --split)
      [[ $# -ge 2 ]] || {
        echo "Błąd: --split wymaga rozmiaru (np. 95m)." >&2
        exit 1
      }
      SPLIT="$2"
      shift 2
      ;;
    --split=*)
      SPLIT="${1#*=}"
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    -*)
      echo "Błąd: nieznana opcja: $1" >&2
      usage
      exit 1
      ;;
    *)
      break
      ;;
  esac
done

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
BASE="${OUT%.zip}"

# poprzednie archiwum (stary format .z01 też sprzątamy)
shopt -s nullglob
OLD=( "$OUT" "${BASE}".z* "${BASE}".[0-9][0-9].zip "${BASE}.__part"* )
shopt -u nullglob
if [[ ${#OLD[@]} -gt 0 ]]; then
  echo "Usuwam poprzednie archiwum:"
  printf '  %s\n' "${OLD[@]}"
  rm -f "${OLD[@]}"
fi

if [[ $SMALL -eq 1 ]]; then
  echo "Tryb: --small (bez .venv, __pycache__, models/yolo, *.pyc, *.pt)"
  mapfile -t FILES < <(
    find "$LAB" \
      \( -path "${LAB}/.venv" -o -path "${LAB}/.venv/*" \) -prune -o \
      \( -path "*/__pycache__" -o -path "*/__pycache__/*" \) -prune -o \
      \( -path "${LAB}/models/yolo" -o -path "${LAB}/models/yolo/*" \) -prune -o \
      -type f ! -name '*.pyc' ! -name '*.pt' -print
  )
  if [[ ${#FILES[@]} -eq 0 ]]; then
    echo "Błąd: brak plików do spakowania w $LAB" >&2
    exit 1
  fi
  zip -r "$OUT" "${FILES[@]}"
  COUNT=${#FILES[@]}
else
  echo "Tryb: pełny (cały katalog $LAB)"
  zip -r "$OUT" "$LAB"
  COUNT="$(find "$LAB" -type f | wc -l)"
fi

if [[ -n "$SPLIT" ]]; then
  SPLIT_B="$(split_size_for_split "$SPLIT")"
  echo "Dzielę $OUT na części po max ~$SPLIT_B (${BASE}.01.zip, ${BASE}.02.zip, …)"
  TMP_PREFIX="${BASE}.__part"
  split -b "$SPLIT_B" "$OUT" "$TMP_PREFIX"
  rm -f "$OUT"
  i=1
  shopt -s nullglob
  for f in "${TMP_PREFIX}"*; do
    printf -v dest '%s.%02d.zip' "$BASE" "$i"
    mv "$f" "$dest"
    i=$((i + 1))
  done
  shopt -u nullglob
fi

echo "Gotowe:"
shopt -s nullglob
if compgen -G "${BASE}.[0-9][0-9].zip" >/dev/null; then
  PARTS=( "${BASE}".[0-9][0-9].zip )
else
  PARTS=( "$OUT" )
fi
shopt -u nullglob
for p in "${PARTS[@]}"; do
  [[ -f "$p" ]] && echo "  $p ($(du -h "$p" | cut -f1))"
done
echo "Łącznie plików w labie (szac.): $COUNT"
