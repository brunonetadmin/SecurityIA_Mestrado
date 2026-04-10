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
#

#!/usr/bin/env bash
set -Eeuo pipefail

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
readonly CIC_PAGE_URL="https://www.unb.ca/cic/datasets/ids-2018.html"

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

WRONG_CSVS=(
  "02-14-2018.csv"
  "02-15-2018.csv"
  "02-16-2018.csv"
  "02-20-2018.csv"
  "02-21-2018.csv"
  "02-22-2018.csv"
  "02-23-2018.csv"
  "02-28-2018.csv"
  "03-01-2018.csv"
  "03-02-2018.csv"
)

if [[ -t 1 ]]; then
  C_RESET='\033[0m'; C_BOLD='\033[1m'; C_DIM='\033[2m'
  C_BLUE='\033[1;34m'; C_GREEN='\033[1;32m'; C_YELLOW='\033[1;33m'; C_RED='\033[1;31m'; C_CYAN='\033[1;36m'
else
  C_RESET=''; C_BOLD=''; C_DIM=''; C_BLUE=''; C_GREEN=''; C_YELLOW=''; C_RED=''; C_CYAN=''
fi

line() { printf '%*s\n' 78 '' | tr ' ' '═'; }
subline() { printf '%*s\n' 78 '' | tr ' ' '─'; }
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
  echo " Projeto        : $PROJECT_ROOT"
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
      *) warn "Resposta inválida. Use y/yes/s/sim ou n/no." ;;
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
  step "Preparando a estrutura do projeto"

  if [[ -d "$LEGACY_TESTS_DIR" && ! -d "$TESTS_DIR" ]]; then
    warn "Diretório legado encontrado: $LEGACY_TESTS_DIR"
    info "Ele será migrado para o padrão em inglês: $TESTS_DIR"
    mv "$LEGACY_TESTS_DIR" "$TESTS_DIR"
    ok "Migração concluída: Testes → Tests"
  elif [[ -d "$LEGACY_TESTS_DIR" && -d "$TESTS_DIR" ]]; then
    warn "Foram encontrados os diretórios 'Testes' e 'Tests'. O instalador usará 'Tests'."
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
  step "Validando arquivos esperados"

  local files=(
    config.py menu.py
    analise_1_arquiteturas.py analise_2_balanceamento.py
    analise_3_teoria_informacao.py analise_4_otimizacao_validacao.py
  )
  local missing=0

  for f in "${files[@]}"; do
    if [[ -f "$TESTS_DIR/$f" ]]; then
      ok "Encontrado: Tests/$f"
    else
      warn "Não encontrado: Tests/$f"
      ((missing+=1)) || true
    fi
  done

  if (( missing > 0 )); then
    warn "Há arquivos ausentes em Tests. O ambiente será preparado mesmo assim."
  fi
}

patch_config_py() {
  local config_file="$TESTS_DIR/config.py"

  step "Ajustando Tests/config.py para a estrutura SecurityIA"

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

text = re.sub(
    r'DATASET_KAGGLE_SLUG\s*=\s*.*',
    'DATASET_KAGGLE_SLUG   = ""  # download automático desabilitado para evitar variante incorreta',
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
    r'def setup_environment\(\) -> None:\n(?:    .*\n)+?(?=\n\ndef apply_plot_style)',
    setup_block + '\n',
    text,
    count=1,
)

baixar_mendeley_block = '''def _baixar_mendeley() -> bool:
    """
    O Mendeley é mantido apenas como referência para download manual,
    pois links diretos hardcoded podem retornar 403.
    """
    print("  Download automático via Mendeley desabilitado.")
    print("  Use o link oficial da página do dataset no navegador:")
    print(f"  {DATASET_MENDELEY_URL}")
    print(f"  DOI: {DATASET_MENDELEY_DOI}")
    return False'''

text = re.sub(
    r'def _baixar_mendeley\(\) -> bool:\n(?:    .*\n)+?(?=\n\ndef _instrucoes_manuais)',
    baixar_mendeley_block + '\n',
    text,
    count=1,
)

instrucoes_block = '''def _instrucoes_manuais() -> None:
    print("\\n" + "─" * 60)
    print("  DOWNLOAD MANUAL — BASE ESPERADA PELO PROJETO")
    print("─" * 60)
    print("\\n  1. Mendeley (versão cleaned esperada):")
    print(f"     {DATASET_MENDELEY_URL}")
    print(f"     DOI: {DATASET_MENDELEY_DOI}")
    print("\\n  2. CIC/UNB (fonte original completa):")
    print("     https://www.unb.ca/cic/datasets/ids-2018.html")
    print(f"\\n  Copie os CSVs corretos para: {DATASET_DIR}")
    print("  Esperados: Bot.csv, Brute Force -Web.csv, DDOS attack-HOIC.csv, etc.")
    print("─" * 60)'''

text = re.sub(
    r'def _instrucoes_manuais\(\) -> None:\n(?:    .*\n)+?(?=\n\ndef verificar_dataset)',
    instrucoes_block + '\n',
    text,
    count=1,
)

verificar_block = '''def verificar_dataset(interativo: bool = True) -> bool:
    """
    Verifica se o dataset esperado pelo projeto existe em DATASET_DIR.
    Aceita apenas a variante cleaned por classe (Bot.csv, Brute Force -Web.csv, etc.).
    """
    presente, ausentes = _dataset_presente()

    wrong_files = [
        "02-14-2018.csv", "02-15-2018.csv", "02-16-2018.csv", "02-20-2018.csv",
        "02-21-2018.csv", "02-22-2018.csv", "02-23-2018.csv", "02-28-2018.csv",
        "03-01-2018.csv", "03-02-2018.csv",
    ]
    detectados_errados = [f for f in wrong_files if (DATASET_DIR / f).exists()]

    if detectados_errados:
        print(f"\\n  ⚠ Foi detectada uma variante incorreta da base em: {DATASET_DIR}")
        print("  Estes arquivos por data não são a base esperada por este projeto:")
        for f in detectados_errados:
            print(f"    - {f}")
        print("  Remova-os e use a versão cleaned do Mendeley.")
        if interativo:
            _instrucoes_manuais()
        return False

    if presente:
        csvs = list(DATASET_DIR.glob("*.csv"))
        print(f"  ✓ Dataset encontrado: {len(csvs)} arquivo(s) CSV em {DATASET_DIR}")
        return True

    print(f"\\n  ⚠ Dataset esperado não encontrado em: {DATASET_DIR}")
    if ausentes:
        print(f"  Arquivos ausentes: {len(ausentes)} de {len(DATASET_FILES)}")

    if not interativo:
        print("  Modo não-interativo: prosseguindo com dados sintéticos.")
        return False

    print("\\n  Como deseja continuar?")
    print("  [1] Exibir instruções para download manual da base correta")
    print("  [2] Prosseguir com dados sintéticos")

    while True:
        opcao = input("\\n  Opção [1-2]: ").strip()
        if opcao == "1":
            _instrucoes_manuais()
            return False
        elif opcao == "2":
            print("  Prosseguindo com dados sintéticos.")
            return False
        else:
            print("  Opção inválida. Digite 1 ou 2.")'''

text = re.sub(
    r'def verificar_dataset\(interativo: bool = True\) -> bool:\n(?:    .*\n)+?(?=\n\ndef carregar_dataset_real)',
    verificar_block + '\n',
    text,
    count=1,
)

print_config_block = '''def print_config() -> None:
    sep = "═" * 62
    print(f"\\n{sep}")
    print("  CONFIGURAÇÃO — IDS Bi-LSTM + Atenção | PPGI/UFAL")
    print(sep)
    print(f"  Root dir      : {ROOT_DIR}")
    print(f"  Tests dir     : {TESTS_DIR}")
    print(f"  Dataset dir   : {DATASET_DIR}")
    print(f"  Model dir     : {MODEL_DIR}")
    print(f"  IDS dir       : {IDS_DIR}")
    print(f"  Reports dir   : {REPORTS_DIR}")
    print(f"  Seed          : {RANDOM_SEED}")
    print(f"  TF threads    : inter={TF_INTER_OP_THREADS}, intra={TF_INTRA_OP_THREADS}")
    print(f"  Features      : {N_FEATURES} | Dropout: {DROPOUT_RATE}")
    print(f"  LSTM L1/L2    : {LSTM_UNITS_L1}/{LSTM_UNITS_L2} | Atenção: {ATTENTION_UNITS}u")
    print(f"  SMOTE k/ENN k : {SMOTE_K}/{ENN_K} | IG/MI: {IG_WEIGHT}/{MI_WEIGHT}")
    print(f"  CV folds / α  : {CV_FOLDS} / {ALPHA_SIGNIFICANCE}")
    print(sep + "\\n")'''

text = re.sub(
    r'def print_config\(\) -> None:\n(?:    .*\n)+?(?=\n\n# Inicializa diretórios ao importar)',
    print_config_block + '\n',
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

  ok "config.py ajustado. Um backup .bak foi criado se houve alteração."
}

setup_venv() {
  step "Preparando ambiente virtual Python"

  if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
    ok "Ambiente virtual criado em $VENV_DIR"
  else
    info "Ambiente virtual já existe: $VENV_DIR"
  fi

  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"

  python -m pip install --upgrade pip setuptools wheel >/dev/null

  local tf_pkg="tensorflow-cpu"
  [[ "$(uname -s)" == "Darwin" ]] && tf_pkg="tensorflow"

  info "Instalando dependências principais..."
  python -m pip install \
    "$tf_pkg" \
    numpy pandas matplotlib seaborn \
    scikit-learn imbalanced-learn scipy \
    requests tabulate >"$REQ_LOG" 2>&1 || {
      fail "Falha ao instalar dependências principais."
      fail "Veja o log em: $REQ_LOG"
      exit 1
    }

  info "Instalando dependências opcionais (não bloqueantes)..."
  python -m pip install scikit-optimize >>"$REQ_LOG" 2>&1 || \
    warn "Não foi possível instalar alguma dependência opcional."

  info "Validando imports principais..."
  python - <<'PY'
mods = [
    'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn',
    'imblearn', 'scipy', 'requests', 'tabulate', 'tensorflow'
]
for mod in mods:
    __import__(mod)
print('OK')
PY

  ok "Ambiente Python preparado com sucesso."
}

count_csvs() {
  shopt -s nullglob
  local csvs=("$DATASET_DIR"/*.csv)
  shopt -u nullglob
  echo "${#csvs[@]}"
}

dataset_present_correct() {
  local missing=0
  for file in "${EXPECTED_CSVS[@]}"; do
    [[ -f "$DATASET_DIR/$file" ]] || ((missing+=1))
  done
  (( missing == 0 ))
}

wrong_dataset_present() {
  local found=1
  for file in "${WRONG_CSVS[@]}"; do
    if [[ -f "$DATASET_DIR/$file" ]]; then
      found=0
      break
    fi
  done
  return "$found"
}

list_wrong_dataset_files() {
  for file in "${WRONG_CSVS[@]}"; do
    [[ -f "$DATASET_DIR/$file" ]] && echo "  - $file"
  done
}

remove_wrong_dataset_files() {
  local removed=0
  for file in "${WRONG_CSVS[@]}"; do
    if [[ -f "$DATASET_DIR/$file" ]]; then
      rm -f "$DATASET_DIR/$file"
      ((removed+=1)) || true
    fi
  done
  echo "$removed"
}

extract_expected_csvs_from_dir() {
  local src_dir="$1"
  python3 - "$src_dir" "$DATASET_DIR" <<'PY'
from pathlib import Path
import shutil
import sys

src = Path(sys.argv[1])
out = Path(sys.argv[2])
expected = {
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
}
count = 0
for path in src.rglob("*.csv"):
    if path.name in expected:
        shutil.move(str(path), str(out / path.name))
        count += 1
print(count)
PY
}

extract_zip_by_path() {
  local zip_path="$1"
  local tmp_dir moved

  if [[ ! -f "$zip_path" ]]; then
    warn "Arquivo não encontrado: $zip_path"
    return 1
  fi

  tmp_dir="$(mktemp -d)"
  info "Descompactando: $zip_path"

  if ! python3 - "$zip_path" "$tmp_dir" <<'PY'
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
    warn "Falha ao descompactar o arquivo ZIP."
    rm -rf "$tmp_dir"
    return 1
  fi

  moved="$(extract_expected_csvs_from_dir "$tmp_dir")"
  rm -rf "$tmp_dir"

  if [[ "$moved" -gt 0 ]]; then
    ok "Foram copiados $moved CSV(s) esperados para $DATASET_DIR"
    return 0
  fi

  warn "Nenhum dos CSVs esperados foi encontrado no ZIP informado."
  return 1
}

show_manual_dataset_instructions() {
  echo
  subline
  echo " Base esperada pelo projeto"
  subline
  echo " Fonte principal : $MENDELEY_PAGE_URL"
  echo " Fonte original  : $CIC_PAGE_URL"
  echo
  echo " Copie os seguintes arquivos para: $DATASET_DIR"
  for file in "${EXPECTED_CSVS[@]}"; do
    echo "  - $file"
  done
  echo
  echo " Atenção: esta instalação NÃO aceita a variante por data"
  echo " (02-14-2018.csv, 02-15-2018.csv, ..., 03-02-2018.csv)."
  subline
}

handle_dataset() {
  step "Validando a base de dados"

  if dataset_present_correct; then
    ok "Base correta encontrada em $DATASET_DIR"
    return 0
  fi

  if wrong_dataset_present; then
    warn "Foi detectada uma variante incorreta da base em $DATASET_DIR"
    list_wrong_dataset_files
    warn "Esses arquivos por data não são a base esperada por este projeto."
    if ask_yes_no "Deseja remover esses CSVs incorretos agora?" "Y"; then
      local removed
      removed="$(remove_wrong_dataset_files)"
      ok "$removed arquivo(s) incorreto(s) removido(s)."
    else
      warn "Os arquivos incorretos foram mantidos. A validação da base continuará falhando."
    fi
  fi

  if dataset_present_correct; then
    ok "Base correta encontrada em $DATASET_DIR"
    return 0
  fi

  warn "A base correta ainda não está disponível em $DATASET_DIR"
  echo
  echo "Como deseja continuar?"
  echo "  [1] Informar o caminho de um ZIP já baixado e extrair os CSVs corretos"
  echo "  [2] Exibir instruções para download manual da base correta"
  echo "  [3] Continuar sem a base real (dados sintéticos nos testes compatíveis)"

  local choice zip_path
  while true; do
    read -r -p "Opção [1-3]: " choice || true
    case "$choice" in
      1)
        read -r -p "Caminho completo do arquivo ZIP: " zip_path || true
        if [[ -n "${zip_path:-}" ]] && extract_zip_by_path "$zip_path"; then
          if dataset_present_correct; then
            ok "Base correta preparada com sucesso."
            return 0
          fi
        fi
        warn "Não foi possível preparar a base correta a partir do ZIP informado."
        ;;
      2)
        show_manual_dataset_instructions
        return 0
        ;;
      3)
        warn "Continuando sem a base real."
        return 0
        ;;
      *)
        warn "Opção inválida. Digite 1, 2 ou 3."
        ;;
    esac
  done
}

show_summary() {
  echo
  line
  echo " Instalação concluída"
  line
  echo " Projeto     : $PROJECT_ROOT"
  echo " Tests       : $TESTS_DIR"
  echo " Base        : $DATASET_DIR"
  echo " Model       : $MODEL_DIR"
  echo " IDS         : $IDS_DIR"
  echo " Reports     : $REPORTS_DIR"
  echo " Venv        : $VENV_DIR"
  echo " CSVs na Base: $(count_csvs)"
  echo
  if dataset_present_correct; then
    ok "A base correta está pronta para uso."
  else
    warn "A base correta ainda não foi detectada."
  fi
  echo
  echo "Para ativar o ambiente virtual:"
  echo "  source \"$VENV_DIR/bin/activate\""
  echo
  if [[ -f "$TESTS_DIR/menu.py" ]]; then
    echo "Para executar os testes:"
    echo "  cd \"$TESTS_DIR\" && python menu.py"
  fi
  line
}

main() {
  header
  require_python
  prepare_project_structure
  check_project_files
  patch_config_py
  setup_venv
  handle_dataset
  show_summary
}

main "$@"

main "$@"
