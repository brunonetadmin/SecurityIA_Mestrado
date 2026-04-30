#!/usr/bin/env bash
#
# run_all_tests.sh — Executa as 4 análises sequencialmente em background.
#
# CARACTERÍSTICAS:
#   - Imune a SIGHUP (SSH desconectado não interrompe)
#   - Cada análise em SUBPROCESS independente (sem leak TF acumulado)
#   - Falha de uma análise NÃO mata as demais
#   - Logs persistidos em Tests/Logs/<analise>_<timestamp>.log + run_all.log
#
# USO:
#   cd /opt/SecurityIA
#   ./run_all_tests.sh           # inicia em background, libera o terminal
#   tail -f Tests/Logs/run_all.log   # acompanha (opcional)
#
#   # Verificar se ainda está rodando:
#   pgrep -af analise_
#
#   # Conferir progresso:
#   ls -lt Tests/Logs/*.log
#
set -u

PROJECT_DIR="/opt/SecurityIA"
VENV_ACTIVATE="$PROJECT_DIR/.venv/bin/activate"
TESTS_DIR="$PROJECT_DIR/Tests"
LOGS_DIR="$TESTS_DIR/Logs"
RUN_LOG="$LOGS_DIR/run_all.log"

ANALISES=(
  "analise_1_arquiteturas.py"
  "analise_2_balanceamento.py"
  "analise_3_teoria_informacao.py"
  "analise_4_otimizacao_validacao.py"
)

mkdir -p "$LOGS_DIR"

run_in_background() {
  # Roda com nohup + setsid → grupo de processos novo, imune a SIGHUP
  nohup setsid bash -c "$(cat <<'INNER'
set -u
cd "$1"
# shellcheck disable=SC1090
source "$2"

RUN_LOG="$3"
TESTS_DIR="$4"
shift 4
ANALISES=("$@")

ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "===========================================================" >> "$RUN_LOG"
echo "[$(ts)] Iniciando execução em background"                    >> "$RUN_LOG"
echo "PID master: $$"                                              >> "$RUN_LOG"
echo "Python    : $(which python3)"                                >> "$RUN_LOG"
echo "Análises  : ${ANALISES[*]}"                                  >> "$RUN_LOG"
echo "==========================================================="  >> "$RUN_LOG"

for script in "${ANALISES[@]}"; do
  caminho="$TESTS_DIR/$script"
  if [[ ! -f "$caminho" ]]; then
    echo "[$(ts)] PULANDO: $script (não encontrado em $TESTS_DIR)" >> "$RUN_LOG"
    continue
  fi
  echo ""                                                          >> "$RUN_LOG"
  echo "[$(ts)] >>> Iniciando $script"                             >> "$RUN_LOG"

  # Cada script em SUBPROCESS ISOLADO. Falha individual não interrompe a fila.
  if python3 "$caminho" >> "$RUN_LOG" 2>&1; then
    echo "[$(ts)] <<< $script CONCLUIDO"                           >> "$RUN_LOG"
  else
    rc=$?
    echo "[$(ts)] <<< $script FALHOU (exit=$rc) — seguindo próximo" >> "$RUN_LOG"
  fi
done

echo ""                                                            >> "$RUN_LOG"
echo "[$(ts)] FIM da execução conjunta."                           >> "$RUN_LOG"
echo "===========================================================" >> "$RUN_LOG"
INNER
)" "$PROJECT_DIR" "$VENV_ACTIVATE" "$RUN_LOG" "$TESTS_DIR" "${ANALISES[@]}" \
    >/dev/null 2>&1 < /dev/null &

  disown
  local pid=$!
  echo ""
  echo "  ✓ Execução iniciada em background (PID: $pid)"
  echo ""
  echo "  Acompanhe com:"
  echo "    tail -f $RUN_LOG"
  echo ""
  echo "  Verificar se ainda está rodando:"
  echo "    pgrep -af analise_"
  echo ""
  echo "  Logs individuais por análise:"
  echo "    ls -lt $LOGS_DIR/*.log"
  echo ""
  echo "  Você pode encerrar o SSH agora — a execução continua."
  echo ""
}

# Validação rápida antes de iniciar
if [[ ! -f "$VENV_ACTIVATE" ]]; then
  echo "ERRO: venv não encontrado em $VENV_ACTIVATE" >&2
  exit 1
fi

if [[ ! -d "$TESTS_DIR" ]]; then
  echo "ERRO: diretório Tests não encontrado em $TESTS_DIR" >&2
  exit 1
fi

# Aborta se já houver execução em andamento
if pgrep -f "analise_.*\.py" > /dev/null; then
  echo "AVISO: já há análises em execução:" >&2
  pgrep -af "analise_.*\.py" >&2
  echo "" >&2
  echo "Cancele com 'pkill -f analise_' antes de iniciar nova execução." >&2
  exit 2
fi

run_in_background
