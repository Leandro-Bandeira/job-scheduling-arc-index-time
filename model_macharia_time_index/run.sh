#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<EOF
Uso:
  $0 --dt YYYY-MM-DD --day-start "YYYY-MM-DD HH:MM" [--time-step MIN=5]

Exemplos:
  $0 --dt 2025-10-12 --day-start "2025-10-13 00:00" --time-step 5
  $0 --dt 2025-10-12 --day-start "2025-10-13 00:00"
EOF
}

# Defaults
TIME_STEP_MIN=5

# Parse args
DT=""
DAY_START=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dt) DT="$2"; shift 2 ;;
    --day-start) DAY_START="$2"; shift 2 ;;
    --time-step) TIME_STEP_MIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Argumento desconhecido: $1"; usage; exit 1 ;;
  esac
done

# Validação mínima
if [[ -z "${DT:-}" || -z "${DAY_START:-}" ]]; then
  echo "ERRO: --dt e --day-start são obrigatórios."; usage; exit 1
fi

# Descobre runner
if command -v poetry >/dev/null 2>&1; then
  RUNNER=(poetry run python)
else
  PYBIN="$(command -v python3 || command -v python)"
  RUNNER=("$PYBIN")
fi

# Raiz do projeto (pasta do script)
SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

# Scripts STD
STD_IN="${PROJECT_ROOT}/src/main/data_input_process.py"
STD_OPT="${PROJECT_ROOT}/src/main/optimize.py"
STD_OUT="${PROJECT_ROOT}/src/main/data_output_process.py"

for f in "$STD_IN" "$STD_OPT" "$STD_OUT"; do
  [[ -f "$f" ]] || { echo "ERRO: script não encontrado: $f"; exit 3; }
done

echo "==> Runner: ${RUNNER[*]}"
echo "==> DT: ${DT}"
echo "==> DAY_START: ${DAY_START}"
echo "==> TIME_STEP_MIN: ${TIME_STEP_MIN}"
echo "============================================================"
echo " Iniciando pipeline (STD)"
echo "============================================================"

echo "------------------------------------------------------------"
echo " [STD] 1/3  main/data_input_process.py  (usa --dt/--day-start/--time-step)"
echo "------------------------------------------------------------"
"${RUNNER[@]}" "$STD_IN" \
  --dt "${DT}" \
  --day-start "${DAY_START}" \
  --time-step "${TIME_STEP_MIN}"

echo "------------------------------------------------------------"
echo " [STD] 2/3  main/optimize.py  (usa --dt)"
echo "------------------------------------------------------------"
"${RUNNER[@]}" "$STD_OPT" \
  --dt "${DT}"

echo "------------------------------------------------------------"
echo " [STD] 3/3  main/data_output_process.py  (usa --dt)"
echo "------------------------------------------------------------"
"${RUNNER[@]}" "$STD_OUT" \
  --dt "${DT}"

echo "✅ Pipeline finalizado com sucesso!"
