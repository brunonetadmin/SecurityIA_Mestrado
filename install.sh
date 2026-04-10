#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# SecurityIA - Installer
# Root expected:
# SecurityIA/
# ├── install.sh
# ├── Base/
# ├── Model/
# ├── IDS/
# └── Tests/
# ==============================================================================

readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$SCRIPT_DIR"
readonly LEGACY_TESTS_DIR="$PROJECT_ROOT/Testes"
readonly TESTS_DIR="$PROJECT_ROOT/Tests"
readonly DATASET_DIR="$PROJECT_ROOT/Base"
readonly MODEL_DIR="$PROJECT_ROOT/Model"
readonly IDS_DIR="$PROJECT_ROOT/IDS"
readonly REPORTS_DIR="$TESTS_DIR/Reports"
readonly VENV_DIR="$PROJECT_ROOT/.venv"
readonly REQ_LOG="$PROJECT_ROOT/.install_packages.log"
readonly MENDELEY_PAGE_URL="https://data.mendeley.com/datasets/29hdbdzx2r/1"
readonly KAGGLE_SLUG="solarmainframe/ids-intrusion-csv"

EXPECTED_CSVS=(
  "Bot.csv"
  "Brute Force -Web.csv"
  "Brute Force -XSS.csv"
  "DDOS attack-HOIC.csv"
  "DDOS attack-LOIC-UDP.csv"
  "DoS attacks-GoldenEye.csv"
  "DoS attacks-Hulk.csv"
  "DoS attacks-SlowHTTPTest.csv"
  "DoS attacks-Slowloris.csv"
  "FTP-BruteForce.csv"
  "Infilteration.csv"
  "SQL Injection.csv"
  "SSH-Bruteforce.csv"
)

if [[ -t 1 ]]; then
  C_RESET='\033[0m'
  C_BOLD='\033[1m'
  C_DIM='\033[2m'
  C_BLUE='\033[1;34m'
  C_GREEN='\033[1;32m'
  C_YELLOW='\033[1;33m'
  C_RED='\033[1;31m'
  C_CYAN='\033[1;36m'
else
  C_RESET=''; C_BOLD=''; C_DIM=''; C_BLUE=''; C_GREEN=''; C_YELLOW=''; C_RED=''; C_CYAN=''
fi

line() { printf '%*s\n' 76 '' | tr ' ' '═'; }
subline() { printf '%*s\n' 76 '' | tr ' ' '─'; }
info() { echo -e "${C_BLUE}ℹ${C_RESET} $*"; }
ok()   { echo -e "${C_GREEN}✓${C_RESET} $*"; }
warn() { echo -e "${C_YELLOW}⚠${C_RESET} $*"; }
fail() { echo -e "${C_RED}✖${C_RESET} $*" >&2; }
step() { echo -e "\n${C_CYAN}${C_BOLD}▶ $*${C_RESET}"; }

on_error() {
  local exit_code=$?
  fail "A instalação foi interrompida por um erro."
  [[ -f "$REQ_LOG" ]] && fail "Consulte o log: $REQ_LOG"
  exit "$exit_code"
}
trap on_error ERR

header() {
  clear 2>/dev/null || true
  line
  echo -e "${C_BOLD} SecurityIA — Instalador de Ambiente${C_RESET}"
  echo " Estrutura alvo: SecurityIA/{Tests,Base,Model,IDS}"
  echo " Diretório do projeto: $PROJECT_ROOT"
  line
}

ask_yes_no() {
  local prompt="$1"
  local default="${2:-N}"
  local answer=""
  local suffix='[y/N]'
  [[ "${default^^}" == "Y" ]] && suffix='[Y/n]'

  while true; do
    read -r -p "$prompt $suffix: " answer || true
    answer="${answer:-$default}"
    case "${answer,,}" in
      y|yes|s|sim) return 0 ;;
      n|no|nao|não) return 1 ;;
      *) warn "Resposta inválida. Digite y/yes/s/sim ou n/no." ;;
    esac
  done
}

require_python() {
  step "Validando Python"

  if ! command -v python3 >/dev/null 2>&1; then
    fail "python3 não foi encontrado no PATH."
    echo "Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
    exit 1
  fi

  local pyver
  pyver="$(python3 - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
PY
)"

  info "Python detectado: $pyver"

  if ! python3 - <<'PY' >/dev/null 2>&1
import sys
assert sys.version_info >= (3, 10)
PY
  then
    fail "É necessário Python 3.10 ou superior."
    exit 1
  fi

  if ! python3 - <<'PY' >/dev/null 2>&1
import venv
PY
  then
    fail "O módulo venv não está disponível."
    echo "Ubuntu/Debian: sudo apt install python3-venv"
    exit 1
  fi

  ok "Python e venv estão prontos."
}

prepare_project_structure() {
  step "Preparando a estrutura de diretórios"

  if [[ -d "$LEGACY_TESTS_DIR" && ! -d "$TESTS_DIR" ]]; then
    warn "Diretório legado encontrado: $LEGACY_TESTS_DIR"
    info "Ele será migrado para o padrão em inglês: $TESTS_DIR"
    mv "$LEGACY_TESTS_DIR" "$TESTS_DIR"
    ok "Migração concluída: Testes → Tests"
  elif [[ -d "$LEGACY_TESTS_DIR" && -d "$TESTS_DIR" ]]; then
    warn "Foram encontrados os diretórios 'Testes' e 'Tests'."
    warn "O instalador usará 'Tests' como diretório oficial."
  fi

  mkdir -p "$TESTS_DIR" "$DATASET_DIR" "$MODEL_DIR" "$IDS_DIR" "$REPORTS_DIR"

  for i in 1 2 3 4; do
    mkdir -p "$REPORTS_DIR/Relatorio_${i}/figuras" "$REPORTS_DIR/Relatorio_${i}/tabelas"
  done

  ok "Estrutura principal garantida."
  echo " • Tests   : $TESTS_DIR"
  echo " • Base    : $DATASET_DIR"
  echo " • Model   : $MODEL_DIR"
  echo " • IDS     : $IDS_DIR"
  echo " • Reports : $REPORTS_DIR"
}

check_project_files() {
  step "Validando arquivos esperados do projeto"

  local missing=0
  for f in config.py menu.py analise_1_arquiteturas.py analise_2_balanceamento.py analise_3_teoria_informacao.py analise_4_otimizacao_validacao.py; do
    if [[ -f "$TESTS_DIR/$f" ]]; then
      ok "Encontrado: Tests/$f"
    else
      warn "Não encontrado: Tests/$f"
      ((missing+=1)) || true
    fi
  done

  if (( missing > 0 )); then
    warn "Há arquivos ausentes no diretório Tests."
    warn "O ambiente será preparado mesmo assim, mas revise a estrutura depois."
  fi
}

patch_config_py() {
  local config_file="$TESTS_DIR/config.py"

  step "Ajustando config.py para a estrutura SecurityIA"

  if [[ ! -f "$config_file" ]]; then
    warn "config.py não encontrado em $TESTS_DIR. Ajuste automático ignorado."
    return 0
  fi

  python3 - "$config_file" <<'PY'
from pathlib import Path
import re
import sys

config_path = Path(sys.argv[1])
text = config_path.read_text(encoding="utf-8")
original = text

paths_block = '''ROOT_DIR    = Path(__file__).resolve().parent.parent   # .../SecurityIA
TESTS_DIR   = ROOT_DIR / "Tests"
DATASET_DIR = ROOT_DIR / "Base"
MODEL_DIR   = ROOT_DIR / "Model"
IDS_DIR     = ROOT_DIR / "IDS"
BASE_DIR    = ROOT_DIR
REPORTS_DIR = TESTS_DIR / "Reports"'''

text = re.sub(
    r'BASE_DIR\s*=\s*Path\("/opt/Testes"\)\s*\n\s*DATASET_DIR\s*=\s*BASE_DIR\s*/\s*"Base"\s*\n\s*REPORTS_DIR\s*=\s*BASE_DIR\s*/\s*"Reports"',
    paths_block,
    text,
    count=1,
)

text = re.sub(
    r'DATASET_FILES\s*=\s*\[(?:.|\n)*?\]',
    '''DATASET_FILES = [
    "Bot.csv",
    "Brute Force -Web.csv",
    "Brute Force -XSS.csv",
    "DDOS attack-HOIC.csv",
    "DDOS attack-LOIC-UDP.csv",
    "DoS attacks-GoldenEye.csv",
    "DoS attacks-Hulk.csv",
    "DoS attacks-SlowHTTPTest.csv",
    "DoS attacks-Slowloris.csv",
    "FTP-BruteForce.csv",
    "Infilteration.csv",
    "SQL Injection.csv",
    "SSH-Bruteforce.csv",
]''',
    text,
    count=1,
)

setup_block = '''def setup_environment() -> None:
    """Cria estrutura de diretórios e configura TF."""
    for d in [ROOT_DIR, TESTS_DIR, DATASET_DIR, MODEL_DIR, IDS_DIR, REPORTS_DIR, *REPORT_DIRS.values()]:
        d.mkdir(parents=True, exist_ok=True)

    for d in REPORT_DIRS.values():
        (d / "figuras").mkdir(exist_ok=True)
        (d / "tabelas").mkdir(exist_ok=True)

    try:
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(TF_INTER_OP_THREADS)
        tf.config.threading.set_intra_op_parallelism_threads(TF_INTRA_OP_THREADS)
        tf.random.set_seed(RANDOM_SEED)
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_visible_devices([], "GPU")
    except ImportError:
        pass'''

text = re.sub(
    r'def setup_environment\(\) -> None:(?:.|\n)*?def apply_plot_style\(\) -> None:',
    setup_block + '\n\n\ndef apply_plot_style() -> None:',
    text,
    count=1,
)

print_block = '''def print_config() -> None:
    sep = "═" * 62
    print(f"\\n{sep}")
    print(" CONFIGURAÇÃO — IDS Bi-LSTM + Atenção | PPGI/UFAL")
    print(sep)
    print(f" Project root : {ROOT_DIR}")
    print(f" Tests dir    : {TESTS_DIR}")
    print(f" Dataset dir  : {DATASET_DIR}")
    print(f" Model dir    : {MODEL_DIR}")
    print(f" IDS dir      : {IDS_DIR}")
    print(f" Reports dir  : {REPORTS_DIR}")
    print(f" Seed         : {RANDOM_SEED}")
    print(f" TF threads   : inter={TF_INTER_OP_THREADS}, intra={TF_INTRA_OP_THREADS}")
    print(f" Features     : {N_FEATURES} | Dropout: {DROPOUT_RATE}")
    print(f" LSTM L1/L2   : {LSTM_UNITS_L1}/{LSTM_UNITS_L2} | Atenção: {ATTENTION_UNITS}u")
    print(f" SMOTE k/ENN  : {SMOTE_K}/{ENN_K} | IG/MI: {IG_WEIGHT}/{MI_WEIGHT}")
    print(f" CV folds / α : {CV_FOLDS} / {ALPHA_SIGNIFICANCE}")
    print(sep + "\\n")'''

text = re.sub(
    r'def print_config\(\) -> None:(?:.|\n)*?(?=# Inicializa diretórios ao importar)',
    print_block + '\n\n',
    text,
    count=1,
)

# Desabilita tentativa de download direto via URL S3 hardcoded, se existir.
text = re.sub(
    r'def _baixar_mendeley\(\) -> bool:(?:.|\n)*?def _baixar_kaggle\(\) -> bool:',
    '''def _baixar_mendeley() -> bool:
    """
    O Mendeley é mantido apenas como referência para download manual,
    pois links diretos hardcoded podem retornar 403.
    """
    print(" Download automático via Mendeley desabilitado.")
    print(" Use o link oficial da página do dataset no navegador:")
    print(f" {DATASET_MENDELEY_URL}")
    print(f" DOI: {DATASET_MENDELEY_DOI}")
    return False


def _baixar_kaggle() -> bool:''',
    text,
    count=1,
)

if text != original:
    backup = config_path.with_suffix(config_path.suffix + '.bak')
    backup.write_text(original, encoding='utf-8')
    config_path.write_text(text, encoding='utf-8')
    print(f'PATCHED:{backup}')
else:
    print('UNCHANGED')
PY
}

show_config_patch_result() {
  local backup_file="$TESTS_DIR/config.py.bak"
  if [[ -f "$backup_file" ]]; then
    ok "config.py ajustado com backup salvo em: $backup_file"
  else
    info "config.py já parecia compatível ou não precisou de alterações adicionais."
  fi
}

setup_venv() {
  step "Criando ambiente virtual Python"

  if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    ok "Ambiente virtual criado em $VENV_DIR"
  else
    info "Ambiente virtual já existe em $VENV_DIR"
  fi

  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  python -m pip install --upgrade pip setuptools wheel >/dev/null
  ok "pip, setuptools e wheel atualizados."
}

install_dependencies() {
  step "Instalando dependências Python"

  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"

  local tf_pkg="tensorflow-cpu"
  [[ "$(uname -s)" == "Darwin" ]] && tf_pkg="tensorflow"

  info "Pacote TensorFlow selecionado: $tf_pkg"
  info "Log completo: $REQ_LOG"

  python -m pip install \
    "$tf_pkg" \
    numpy pandas matplotlib seaborn \
    scikit-learn imbalanced-learn scipy \
    requests tabulate >"$REQ_LOG" 2>&1

  ok "Dependências principais instaladas."

  if python -m pip install scikit-optimize kaggle >>"$REQ_LOG" 2>&1; then
    ok "Dependências opcionais instaladas: scikit-optimize, kaggle"
  else
    warn "Alguma dependência opcional não pôde ser instalada."
    warn "Isso não bloqueia o uso básico do projeto."
  fi

  python - <<'PY'
mods = [
    "numpy", "pandas", "matplotlib", "seaborn", "sklearn",
    "imblearn", "scipy", "requests", "tabulate", "tensorflow"
]
for mod in mods:
    __import__(mod)
print("IMPORTS_OK")
PY

  ok "Validação de imports concluída."
}

dataset_present() {
  shopt -s nullglob
  local csvs=("$DATASET_DIR"/*.csv)
  shopt -u nullglob

  if (( ${#csvs[@]} >= 5 )); then
    return 0
  fi

  local missing=0
  for file in "${EXPECTED_CSVS[@]}"; do
    [[ -f "$DATASET_DIR/$file" ]] || ((missing+=1)) || true
  done

  (( missing == 0 ))
}

count_csvs() {
  shopt -s nullglob
  local csvs=("$DATASET_DIR"/*.csv)
  shopt -u nullglob
  echo "${#csvs[@]}"
}

extract_csvs_from_dir() {
  local src_dir="$1"
  local moved=0
  while IFS= read -r -d '' csv_file; do
    mv -f "$csv_file" "$DATASET_DIR/$(basename "$csv_file")"
    ((moved+=1)) || true
  done < <(find "$src_dir" -type f -iname '*.csv' -print0)
  echo "$moved"
}

show_manual_dataset_instructions() {
  echo
  subline
  echo -e "${C_BOLD}Download manual da base${C_RESET}"
  echo "1) Abra a página oficial do dataset no navegador:"
  echo "   $MENDELEY_PAGE_URL"
  echo "2) Baixe o pacote completo ('Download All')."
  echo "3) Extraia os arquivos CSV para este diretório:"
  echo "   $DATASET_DIR"
  echo "4) Os nomes esperados incluem, por exemplo:"
  echo "   - Bot.csv"
  echo "   - Brute Force -Web.csv"
  echo "   - DDOS attack-HOIC.csv"
  echo "   - DoS attacks-Hulk.csv"
  echo "   - SSH-Bruteforce.csv"
  subline
}

download_dataset_kaggle() {
  local tmp_dir moved

  if ! command -v kaggle >/dev/null 2>&1; then
    warn "Kaggle CLI não encontrado no PATH."
    return 1
  fi

  tmp_dir="$(mktemp -d)"
  info "Tentando download via Kaggle CLI..."

  if ! kaggle datasets download -d "$KAGGLE_SLUG" --path "$tmp_dir" --unzip; then
    warn "Falha no download via Kaggle. Verifique suas credenciais."
    rm -rf "$tmp_dir"
    return 1
  fi

  moved="$(extract_csvs_from_dir "$tmp_dir")"
  rm -rf "$tmp_dir"

  if [[ "$moved" -gt 0 ]]; then
    ok "Dataset obtido via Kaggle: $moved CSV(s) em $DATASET_DIR"
    return 0
  fi

  warn "Nenhum CSV foi encontrado no conteúdo do Kaggle."
  return 1
}

maybe_download_dataset() {
  step "Verificando dataset CSE-CIC-IDS2018"

  if dataset_present; then
    ok "Dataset já está disponível em $DATASET_DIR ($(count_csvs) CSV)."
    return 0
  fi

  warn "Dataset não encontrado em $DATASET_DIR"
  echo "Fonte de referência: $MENDELEY_PAGE_URL"

  if ! ask_yes_no "Deseja obter o dataset agora?" "N"; then
    info "Download ignorado. O projeto poderá usar dados sintéticos quando aplicável."
    return 0
  fi

  echo
  echo "Como deseja obter o dataset?"
  echo "  [1] Tentar download automático via Kaggle CLI (recomendado)"
  echo "  [2] Exibir instruções para download manual"
  echo "  [3] Pular e continuar sem a base real"

  local choice
  read -r -p "Escolha uma opção [1-3]: " choice || true

  case "${choice:-2}" in
    1)
      if download_dataset_kaggle; then
        ok "Dataset obtido com sucesso."
      else
        warn "Falha na tentativa automática via Kaggle."
        show_manual_dataset_instructions
      fi
      ;;
    2)
      show_manual_dataset_instructions
      ;;
    3)
      warn "Continuando sem a base real. Os testes poderão usar dados sintéticos."
      ;;
    *)
      warn "Opção inválida. Exibindo instruções manuais."
      show_manual_dataset_instructions
      ;;
  esac
}

final_summary() {
  step "Instalação concluída"
  echo "Resumo:"
  echo " • Projeto : $PROJECT_ROOT"
  echo " • Tests   : $TESTS_DIR"
  echo " • Base    : $DATASET_DIR"
  echo " • Model   : $MODEL_DIR"
  echo " • IDS     : $IDS_DIR"
  echo " • Venv    : $VENV_DIR"
  echo " • CSVs    : $(count_csvs)"
  echo
  echo "Próximos passos:"
  echo "  1) source .venv/bin/activate"
  if [[ -f "$TESTS_DIR/menu.py" ]]; then
    echo "  2) cd Tests"
    echo "  3) python menu.py"
  else
    echo "  2) Execute seus scripts a partir do diretório Tests"
  fi
  echo
  ok "Ambiente pronto."
}

main() {
  header
  require_python
  prepare_project_structure
  check_project_files
  patch_config_py
  show_config_patch_result
  setup_venv
  install_dependencies
  maybe_download_dataset
  final_summary
}

main "$@"
