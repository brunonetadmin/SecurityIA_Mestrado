#!/usr/bin/env bash
set -Eeuo pipefail

# ==============================================================================
# SecurityIA - Installer
# Root expected:
#   SecurityIA/
#   ├── install.sh
#   ├── Base/
#   ├── Model/
#   ├── IDS/
#   └── Tests/
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
readonly MENDELEY_ZIP_URL="https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com/29hdbdzx2r-1.zip"
readonly KAGGLE_SLUG="solarmainframe/ids-intrusion-csv"

EXPECTED_CSVS=(
  "02-14-2018.csv" "02-15-2018.csv" "02-16-2018.csv"
  "02-20-2018.csv" "02-21-2018.csv" "02-22-2018.csv"
  "02-23-2018.csv" "02-28-2018.csv" "03-01-2018.csv"
  "03-02-2018.csv"
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

line() { printf '%*s\n' 74 '' | tr ' ' '═'; }
subline() { printf '%*s\n' 74 '' | tr ' ' '─'; }
info()    { echo -e "${C_BLUE}ℹ${C_RESET}  $*"; }
ok()      { echo -e "${C_GREEN}✓${C_RESET}  $*"; }
warn()    { echo -e "${C_YELLOW}⚠${C_RESET}  $*"; }
fail()    { echo -e "${C_RED}✖${C_RESET}  $*" >&2; }
step()    { echo -e "\n${C_CYAN}${C_BOLD}▶ $*${C_RESET}"; }

on_error() {
  local exit_code=$?
  fail "A instalação foi interrompida por um erro."
  fail "Consulte o log, se existir: $REQ_LOG"
  exit "$exit_code"
}
trap on_error ERR

header() {
  clear 2>/dev/null || true
  line
  echo -e "${C_BOLD}  SecurityIA — Environment Installer${C_RESET}"
  echo "  Estrutura alvo: SecurityIA/{Tests,Base,Model,IDS}"
  echo "  Instalador localizado em: $PROJECT_ROOT"
  line
}

pause() {
  read -r -p "Pressione ENTER para continuar..." _ || true
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
    echo "  Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
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
    echo "  Ubuntu/Debian: sudo apt install python3-venv"
    exit 1
  fi

  ok "Python e venv estão prontos."
}

prepare_tests_directory() {
  step "Preparando a estrutura de diretórios"

  if [[ -d "$LEGACY_TESTS_DIR" && ! -d "$TESTS_DIR" ]]; then
    warn "Diretório legado encontrado: $LEGACY_TESTS_DIR"
    info "Ele será migrado para o padrão em inglês: $TESTS_DIR"
    mv "$LEGACY_TESTS_DIR" "$TESTS_DIR"
    ok "Migração concluída: Testes → Tests"
  elif [[ -d "$LEGACY_TESTS_DIR" && -d "$TESTS_DIR" ]]; then
    warn "Existem os diretórios 'Testes' e 'Tests'."
    warn "O instalador usará 'Tests' como diretório oficial."
  fi

  mkdir -p "$TESTS_DIR" "$DATASET_DIR" "$MODEL_DIR" "$IDS_DIR" "$REPORTS_DIR"
  for i in 1 2 3 4; do
    mkdir -p "$REPORTS_DIR/Relatorio_${i}/figuras" "$REPORTS_DIR/Relatorio_${i}/tabelas"
  done

  ok "Estrutura principal garantida."
  echo "  • Tests   : $TESTS_DIR"
  echo "  • Base    : $DATASET_DIR"
  echo "  • Model   : $MODEL_DIR"
  echo "  • IDS     : $IDS_DIR"
  echo "  • Reports : $REPORTS_DIR"
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

patch_config_paths() {
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
    r'BASE_DIR\s*=\s*Path\("/opt/Testes"\)\nDATASET_DIR\s*=\s*BASE_DIR\s*/\s*"Base"\nREPORTS_DIR\s*=\s*BASE_DIR\s*/\s*"Reports"',
    paths_block,
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
    r'def setup_environment\(\) -> None:\n(?:    .*\n)+?def apply_plot_style\(\) -> None:',
    setup_block + '\n\n\ndef apply_plot_style() -> None:',
    text,
    count=1,
)

print_block = '''def print_config() -> None:
    sep = "═" * 62
    print(f"\\n{sep}")
    print("  CONFIGURAÇÃO — IDS Bi-LSTM + Atenção | PPGI/UFAL")
    print(sep)
    print(f"  Project root   : {ROOT_DIR}")
    print(f"  Tests dir      : {TESTS_DIR}")
    print(f"  Dataset dir    : {DATASET_DIR}")
    print(f"  Model dir      : {MODEL_DIR}")
    print(f"  IDS dir        : {IDS_DIR}")
    print(f"  Reports dir    : {REPORTS_DIR}")
    print(f"  Seed           : {RANDOM_SEED}")
    print(f"  TF threads     : inter={TF_INTER_OP_THREADS}, intra={TF_INTRA_OP_THREADS}")
    print(f"  Features       : {N_FEATURES} | Dropout: {DROPOUT_RATE}")
    print(f"  LSTM L1/L2     : {LSTM_UNITS_L1}/{LSTM_UNITS_L2} | Atenção: {ATTENTION_UNITS}u")
    print(f"  SMOTE k/ENN k  : {SMOTE_K}/{ENN_K} | IG/MI: {IG_WEIGHT}/{MI_WEIGHT}")
    print(f"  CV folds / α   : {CV_FOLDS} / {ALPHA_SIGNIFICANCE}")
    print(sep + "\\n")'''

text = re.sub(
    r'def print_config\(\) -> None:\n(?:    .*\n)+?(?=# Inicializa diretórios ao importar)',
    print_block + '\n\n',
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
  local config_file="$TESTS_DIR/config.py"
  if [[ -f "$config_file.bak" ]]; then
    ok "config.py ajustado com backup salvo em: $config_file.bak"
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
    warn "Alguma dependência opcional não pôde ser instalada. Isso não bloqueia o uso básico."
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

download_dataset_mendeley() {
  local tmp_dir zip_file moved
  tmp_dir="$(mktemp -d)"
  zip_file="$tmp_dir/cic_ids2018.zip"

  info "Baixando dataset do Mendeley..."
  info "Página de referência: $MENDELEY_PAGE_URL"

  if ! python3 - "$MENDELEY_ZIP_URL" "$zip_file" <<'PY'
import sys
import urllib.request

url = sys.argv[1]
dest = sys.argv[2]

with urllib.request.urlopen(url, timeout=60) as response:
    total = response.headers.get("Content-Length")
    total = int(total) if total else 0
    downloaded = 0
    chunk_size = 1024 * 1024
    with open(dest, "wb") as f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 / total
                print(f"\r  Download: {pct:5.1f}% ({downloaded/1024/1024:.1f} MB)", end="", flush=True)
            else:
                print(f"\r  Baixado: {downloaded/1024/1024:.1f} MB", end="", flush=True)
print()
PY
  then
    warn "Falha no download direto via Mendeley."
    rm -rf "$tmp_dir"
    return 1
  fi

  info "Descompactando e organizando CSVs..."
  if ! python3 - "$zip_file" "$tmp_dir/extracted" <<'PY'
import sys
import zipfile
from pathlib import Path

zip_path = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
out_dir.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(out_dir)
PY
  then
    warn "Falha na descompactação do ZIP baixado."
    rm -rf "$tmp_dir"
    return 1
  fi

  moved="$(extract_csvs_from_dir "$tmp_dir/extracted")"
  rm -rf "$tmp_dir"

  if [[ "$moved" -gt 0 ]]; then
    ok "Dataset obtido com sucesso: $moved CSV(s) em $DATASET_DIR"
    return 0
  fi

  warn "Nenhum CSV foi encontrado após a extração."
  return 1
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
  step "Verificando dataset CIC-IDS2018"

  if dataset_present; then
    ok "Dataset já está disponível em $DATASET_DIR ($(count_csvs) CSV)."
    return 0
  fi

  warn "Dataset não encontrado em $DATASET_DIR"
  echo "  Fonte recomendada: $MENDELEY_PAGE_URL"

  if ! ask_yes_no "Deseja baixar o dataset automaticamente agora?" "N"; then
    info "Download ignorado. O sistema poderá usar dados sintéticos nos testes."
    return 0
  fi

  if download_dataset_mendeley; then
    return 0
  fi

  warn "A tentativa via Mendeley falhou."
  if ask_yes_no "Deseja tentar via Kaggle CLI?" "N"; then
    if download_dataset_kaggle; then
      return 0
    fi
  fi

  warn "O download automático não foi concluído."
  echo "  Download manual: $MENDELEY_PAGE_URL"
  echo "  Destino esperado: $DATASET_DIR"
  return 0
}

show_summary() {
  step "Resumo final"
  line
  echo -e "${C_BOLD}  Instalação concluída${C_RESET}"
  subline
  echo "  Project root : $PROJECT_ROOT"
  echo "  Tests dir    : $TESTS_DIR"
  echo "  Dataset dir  : $DATASET_DIR"
  echo "  Model dir    : $MODEL_DIR"
  echo "  IDS dir      : $IDS_DIR"
  echo "  Reports dir  : $REPORTS_DIR"
  echo "  Virtual env  : $VENV_DIR"
  subline
  echo "  Próximos passos:"
  echo "    1) source \"$VENV_DIR/bin/activate\""
  echo "    2) cd \"$TESTS_DIR\""
  if [[ -f "$TESTS_DIR/menu.py" ]]; then
    echo "    3) python menu.py"
  else
    echo "    3) Execute seus scripts em Tests/ conforme necessário"
  fi
  line
}

main() {
  header
  require_python
  prepare_tests_directory
  check_project_files
  patch_config_paths
  show_config_patch_result
  setup_venv
  install_dependencies
  maybe_download_dataset
  show_summary
}

main "$@"
