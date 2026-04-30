#!/usr/bin/env python3
"""
app_menu.py — Frontend Unificado do projeto SecurityIA

Unifica:
  - Modo de Testes/Análises acadêmicas
  - Modo Sistema IDS (Collector, Detector, Treinamento, Relatórios, Status)

Objetivo:
  - Centralizar a experiência CLI em um único ponto de entrada
  - Manter compatibilidade com os scripts já existentes
  - Servir de base para evolução incremental do IDS

Uso:
  python3 app_menu.py
"""

from __future__ import annotations

import importlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

# -----------------------------------------------------------------------------
# Ambiente e caminhos
# -----------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

APP_NAME = "SecurityIA"
APP_VERSION = "3.0.0"
APP_SUBTITLE = "Unified CLI for Tests and Intelligent IDS Operations"

VENV_DIR = ROOT_DIR / ".venv"
PROJECT_PYTHON = VENV_DIR / "bin" / "python3"


def using_project_venv() -> bool:
    exe = Path(sys.executable).resolve()
    try:
        return VENV_DIR.resolve() in exe.parents
    except FileNotFoundError:
        return False


def bootstrap_project_python() -> None:
    """
    Reexecuta o app com o Python da .venv quando disponível.
    Evita falhas de importação dos módulos de teste/IDS quando o menu é
    iniciado com o Python do sistema, mas as dependências estão na .venv.
    """
    if os.environ.get("SECURITYIA_SKIP_VENV_REEXEC") == "1":
        return

    if using_project_venv():
        return

    if PROJECT_PYTHON.exists() and os.access(PROJECT_PYTHON, os.X_OK):
        os.environ["SECURITYIA_SKIP_VENV_REEXEC"] = "1"
        os.execv(str(PROJECT_PYTHON), [str(PROJECT_PYTHON), str(Path(__file__).resolve()), *sys.argv[1:]])


bootstrap_project_python()

# -----------------------------------------------------------------------------
# ANSI / UI helpers
# -----------------------------------------------------------------------------

ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
}


def supports_color() -> bool:
    return sys.stdout.isatty() and os.environ.get("TERM", "") != "dumb"


def color(text: str, *styles: str) -> str:
    if not supports_color() or not styles:
        return text
    prefix = "".join(ANSI.get(s, "") for s in styles)
    return f"{prefix}{text}{ANSI['reset']}"


def clear_screen() -> None:
    os.system("cls" if os.name == "nt" else "clear")


def line(char: str = "═", n: int = 86) -> str:
    return char * n


def hline(char: str = "─", n: int = 86) -> str:
    return char * n


def human_size(num: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num)
    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def badge_ok(text: str) -> str:
    return color(f"✔ {text}", "green", "bold")


def badge_warn(text: str) -> str:
    return color(f"⚠ {text}", "yellow", "bold")


def badge_err(text: str) -> str:
    return color(f"✖ {text}", "red", "bold")


def badge_info(text: str) -> str:
    return color(f"ℹ {text}", "cyan", "bold")


def prompt(text: str) -> str:
    return input(color("  » ", "cyan", "bold") + text).strip()


def pause(message: str = "Pressione ENTER para continuar...") -> None:
    input("\n" + color("  ⏎ ", "cyan", "bold") + message)


def confirm(text: str) -> bool:
    resp = prompt(f"{text} [s/N]: ").lower()
    return resp in ("s", "sim", "y", "yes")


def section(title: str, subtitle: str | None = None) -> None:
    print()
    print(line())
    print("  " + color(title, "white", "bold"))
    if subtitle:
        print("  " + color(subtitle, "dim"))
    print(line())


def print_header(context: str = "Main Menu") -> None:
    clear_screen()
    print(line())
    print("  " + color(APP_NAME, "cyan", "bold") + color(f"  v{APP_VERSION}", "white", "bold"))
    print("  " + color(APP_SUBTITLE, "dim"))
    print(hline())
    print(f"  Contexto : {color(context, 'white', 'bold')}")
    print(f"  Projeto  : {ROOT_DIR}")
    print(f"  Python   : {sys.executable}")
    print(hline())


def menu_choice(valid: Iterable[str], title: str = "Opção: ") -> str:
    allowed = {str(v) for v in valid}
    while True:
        choice = prompt(title)
        if choice in allowed:
            return choice
        print("  " + badge_err(f"Opção inválida. Escolha entre: {', '.join(sorted(allowed))}"))


def print_simple_table(headers: list[str], rows: list[list[str]], title: str = "") -> None:
    """Tabela ASCII simples para listagens no menu."""
    if title:
        print(f"\n  {color(title, 'white', 'bold')}")
        print("  " + hline(n=72))
    if not rows:
        print("  " + color("(nenhum registro)", "dim"))
        return
    col_widths = [max(len(h), max((len(str(r[i])) for r in rows), default=0))
                  for i, h in enumerate(headers)]
    hdr = "  ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    print(f"  {color(hdr, 'cyan', 'bold')}")
    print("  " + "─" * sum(col_widths + [2 * (len(headers) - 1)]))
    for row in rows:
        line_str = "  ".join(f"{str(row[i]):<{col_widths[i]}}" for i in range(len(headers)))
        print(f"  {line_str}")


# -----------------------------------------------------------------------------
# Safe imports
# -----------------------------------------------------------------------------


def safe_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def safe_import_attr(module_name: str, attr_name: str) -> Any | None:
    try:
        mod = importlib.import_module(module_name)
        return getattr(mod, attr_name)
    except Exception:
        return None


CONFIG = safe_import("config")


def _config_candidates() -> list[Any]:
    candidates: list[Any] = []
    if CONFIG is None:
        return candidates

    candidates.append(CONFIG)
    for attr in ("Config", "IDSConfig"):
        obj = getattr(CONFIG, attr, None)
        if obj is not None and obj not in candidates:
            candidates.append(obj)
    return candidates


def get_config_attr(*names: str, default: Any = None) -> Any:
    for candidate in _config_candidates():
        for name in names:
            if hasattr(candidate, name):
                return getattr(candidate, name)
    return default


def get_config_callable(*names: str) -> Callable[..., Any] | None:
    value = get_config_attr(*names, default=None)
    return value if callable(value) else None


# -----------------------------------------------------------------------------
# Test/analysis integration (sem alterações)
# -----------------------------------------------------------------------------

TEST_MODULES: Dict[int, Tuple[str, str]] = {
    1: ("analise_1_arquiteturas", "Análise Comparativa de Arquiteturas de Redes Neurais"),
    2: ("analise_2_balanceamento", "Análise de Estratégias de Balanceamento de Classes"),
    3: ("analise_3_teoria_informacao", "Análise da Aplicabilidade da Teoria da Informação"),
    4: ("analise_4_otimizacao_validacao", "Análise de Estratégias de Otimização e Validação"),
}


def get_tests_dir() -> Path:
    tests_dir = get_config_attr("TESTS_DIR", default=None)
    if tests_dir is not None:
        return Path(tests_dir)
    return ROOT_DIR / "Tests"


def ensure_tests_path() -> Path:
    tests_dir = get_tests_dir()
    if tests_dir.exists():
        resolved = str(tests_dir.resolve())
        sys.path[:] = [p for p in sys.path if p != resolved]
        sys.path.insert(0, resolved)
    return tests_dir


def get_reports_dir() -> Path:
    report_dirs = get_config_attr("REPORT_DIRS", default=None)
    if report_dirs:
        try:
            sample = next(iter(report_dirs.values()))
            return Path(sample).parent
        except Exception:
            pass

    explicit = get_config_attr("TEST_REPORTS_DIR", default=None)
    if explicit is not None:
        return Path(explicit)

    tests_dir = get_config_attr("TESTS_DIR", default=None)
    if tests_dir is not None:
        return Path(tests_dir) / "Reports"

    legacy = get_config_attr("REPORTS_DIR", default=None)
    if legacy is not None:
        legacy_path = Path(legacy)
        if legacy_path.name.lower() == "reports" and legacy_path.parent.name == "Tests":
            return legacy_path

    return get_tests_dir() / "Reports"


def get_dataset_dir() -> Path:
    dataset_dir = get_config_attr("DATA_DIR", "DATASET_DIR", default=None)
    if dataset_dir is not None:
        return Path(dataset_dir)
    return ROOT_DIR / "Base" / "CSE-CIC-IDS2018"


def get_model_dir() -> Path:
    model_dir = get_config_attr("MODEL_DIR", default=None)
    if model_dir is not None:
        return Path(model_dir)
    return ROOT_DIR / "Model"


def get_logs_dir() -> Path:
    logs_dir = get_config_attr("LOGS_DIR", default=None)
    if logs_dir is not None:
        return Path(logs_dir)
    return ROOT_DIR / "Logs"


def get_temp_dir() -> Path:
    temp_dir = get_config_attr("TEMP_DIR", default=None)
    if temp_dir is not None:
        return Path(temp_dir)
    return ROOT_DIR / "Temp"


def dataset_csv_files() -> list[Path]:
    dataset_dir = get_dataset_dir()
    if not dataset_dir.exists():
        return []
    return sorted(p for p in dataset_dir.glob("*.csv") if p.is_file())


EXPECTED_DATASET_FILES = [
    "bot.csv",
    "brute force -web.csv",
    "brute force -xss.csv",
    "ddos attack-hoic.csv",
    "ddos attack-loic-udp.csv",
    "ddos attacks-loic-http.csv",
    "dos attacks-goldeneye.csv",
    "dos attacks-hulk.csv",
    "dos attacks-slowhttptest.csv",
    "dos attacks-slowloris.csv",
    "ftp-bruteforce.csv",
    "infilteration.csv",
    "sql injection.csv",
    "ssh-bruteforce.csv",
]


def dataset_validation() -> tuple[bool, list[str], list[Path]]:
    csvs = dataset_csv_files()
    names = {p.name.strip().lower() for p in csvs}
    missing = [name for name in EXPECTED_DATASET_FILES if name not in names]
    return len(missing) == 0, missing, csvs


def dataset_available() -> bool:
    ok, _, csvs = dataset_validation()
    return ok or bool(csvs)


def dataset_status() -> str:
    ok, missing, csvs = dataset_validation()
    if ok:
        return badge_ok(f"Dataset validado ({len(csvs)} CSVs)")
    if csvs:
        return badge_warn(f"Dataset parcial ({len(csvs)} CSVs, faltando {len(missing)})")
    return badge_warn("Dataset real não encontrado")


def report_status(analysis_id: int) -> str:
    reports_dir = get_reports_dir()
    report_dirs = get_config_attr("REPORT_DIRS", default=None)
    rdir = reports_dir / f"Relatorio_{analysis_id}_{TEST_MODULES[analysis_id][0].split('_', 2)[-1].title()}"
    if report_dirs:
        try:
            rdir = Path(report_dirs[analysis_id])
        except Exception:
            pass
    md = rdir / f"Relatorio_{analysis_id}.md"
    figs = list((rdir / "figuras").glob("*.png")) if (rdir / "figuras").exists() else []
    tabs = list((rdir / "tabelas").glob("*.csv")) if (rdir / "tabelas").exists() else []
    if md.exists():
        return badge_ok(f"{len(figs)} figura(s) | {len(tabs)} tabela(s)")
    return badge_warn("Não gerado")


def execute_test_analysis(analysis_id: int) -> bool:
    dataset_flag = dataset_available()
    mod_name, label = TEST_MODULES[analysis_id]
    print_header("Tests > Execution")
    section(f"Executando análise {analysis_id}", label)
    if not dataset_flag:
        print("  " + badge_warn("Dataset real não detectado. A execução poderá usar dados sintéticos."))
    print(f"  Módulo   : {mod_name}")
    print(f"  Dataset  : {get_dataset_dir()}")
    print(f"  Relatório: {get_reports_dir()}")
    print(f"  Python   : {sys.executable}")

    try:
        tests_dir = ensure_tests_path()
        mod = importlib.import_module(mod_name)
        importlib.reload(mod)
        fn = getattr(mod, "executar")
    except Exception as exc:
        print("\n  " + badge_err(f"Falha ao carregar o módulo {mod_name}: {exc}"))
        print(f"  Tests dir : {tests_dir if 'tests_dir' in locals() else get_tests_dir()}")
        print(f"  Python    : {sys.executable}")
        print(f"  sys.path  : {sys.path[:6]}")
        if isinstance(exc, ModuleNotFoundError) and getattr(exc, "name", "") in {"numpy", "pandas", "tensorflow", "matplotlib", "seaborn", "sklearn", "scipy", "imblearn"}:
            print("  " + badge_warn("Dependência Python não encontrada no interpretador atual."))
            print("  " + badge_info(f"O app tentará usar automaticamente a .venv do projeto quando existir: {VENV_DIR}"))
        traceback.print_exc()
        return False

    try:
        started = time.perf_counter()
        fn(dataset_disponivel=dataset_flag)
        elapsed = time.perf_counter() - started
        print("\n  " + badge_ok(f"Concluído em {elapsed:.1f}s ({elapsed/60:.1f} min)."))
        return True
    except Exception as exc:
        print("\n  " + badge_err(f"Erro durante a execução: {exc}"))
        traceback.print_exc()
        return False


def execute_test_in_background(analysis_id: int) -> None:
    """
    Executa UMA análise em background imune a SIGHUP. Mesma mecânica de
    execute_all_tests, mas para uma única análise — útil para repetir
    apenas a que falhou sem refazer todas.
    """
    import shlex
    import textwrap

    print_header(f"Tests > Análise {analysis_id} em background")

    mod_name, label = TEST_MODULES[analysis_id]
    project_dir = ROOT_DIR
    venv_python = (VENV_DIR / "bin" / "python3") if VENV_DIR.exists() else Path(sys.executable)
    tests_dir = get_tests_dir()
    logs_dir = tests_dir / "Logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_log = logs_dir / f"run_{mod_name}.log"
    script_path = tests_dir / f"{mod_name}.py"

    if not script_path.exists():
        print("  " + badge_err(f"Script não encontrado: {script_path}"))
        pause()
        return

    # Aborta se a mesma análise já está rodando
    try:
        check = subprocess.run(
            ["pgrep", "-f", f"{mod_name}.py"],
            capture_output=True, text=True, check=False,
        )
        if check.returncode == 0 and check.stdout.strip():
            print("  " + badge_warn(f"Já existe execução de {mod_name} em andamento."))
            print(f"    PIDs: {check.stdout.strip()}")
            pause()
            return
    except FileNotFoundError:
        pass

    bash_runner = textwrap.dedent(f"""\
        set -u
        cd {shlex.quote(str(project_dir))}
        RUN_LOG={shlex.quote(str(run_log))}
        ts() {{ date '+%Y-%m-%d %H:%M:%S'; }}
        {{
          echo "==========================================================="
          echo "[$(ts)] Iniciando {mod_name} em background"
          echo "PID: $$"
          echo "==========================================================="
        }} >> "$RUN_LOG"
        if {shlex.quote(str(venv_python))} {shlex.quote(str(script_path))} >> "$RUN_LOG" 2>&1; then
          echo "[$(ts)] CONCLUIDO" >> "$RUN_LOG"
        else
          echo "[$(ts)] FALHOU (exit=$?)" >> "$RUN_LOG"
        fi
    """)

    try:
        proc = subprocess.Popen(
            ["nohup", "setsid", "bash", "-c", bash_runner],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
            cwd=str(project_dir),
        )
    except Exception as exc:
        print("  " + badge_err(f"Erro ao disparar execução: {exc}"))
        pause()
        return

    section(f"Análise {analysis_id} em background — {label}")
    print(f"  PID             : {proc.pid}")
    print(f"  Log mestre      : {run_log}")
    print(f"  Log detalhado   : {logs_dir}/analise_{analysis_id}_<timestamp>.log")
    print()
    print("  Acompanhar em outro terminal:")
    print(f"    tail -f {run_log}")
    print(f"    pgrep -af {mod_name}")
    print()


def execute_all_tests() -> None:
    """
    Executa as 4 análises em SEQUÊNCIA, em PROCESSO DE FUNDO IMUNE A SIGHUP.

    Comportamento:
      - Cada análise em subprocess Python independente (sem leak de TF acumulado
        entre análises, sem importlib.reload).
      - Falha de uma análise NÃO interrompe a fila — segue para a próxima.
      - O processo mestre roda em novo grupo de sessão (setsid + nohup),
        portanto sobrevive ao encerramento do SSH e do próprio app_menu.
      - Logs persistidos em Tests/Logs/ (run_all.log + um log por análise).
      - O usuário recebe o terminal de volta imediatamente.
    """
    import shlex
    import textwrap

    print_header("Tests > Execução conjunta em background")

    project_dir = ROOT_DIR
    venv_python = (VENV_DIR / "bin" / "python3") if VENV_DIR.exists() else Path(sys.executable)
    tests_dir = get_tests_dir()
    logs_dir = tests_dir / "Logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    run_log = logs_dir / "run_all.log"

    # Aborta se já existe execução em andamento
    try:
        check = subprocess.run(
            ["pgrep", "-f", r"analise_.*\.py"],
            capture_output=True, text=True, check=False,
        )
        if check.returncode == 0 and check.stdout.strip():
            print("  " + badge_warn("Já há análises em execução:"))
            for line in check.stdout.strip().splitlines():
                print(f"    PID {line}")
            print()
            print("  Para cancelar e reiniciar:  pkill -f analise_")
            pause()
            return
    except FileNotFoundError:
        # pgrep ausente — segue sem checar concorrência
        pass

    # Resolve caminhos das análises a partir de TEST_MODULES
    scripts: List[Path] = []
    for analysis_id in sorted(TEST_MODULES):
        mod_name, _ = TEST_MODULES[analysis_id]
        script_path = tests_dir / f"{mod_name}.py"
        if not script_path.exists():
            print("  " + badge_err(f"Não encontrado: {script_path}"))
            pause()
            return
        scripts.append(script_path)

    # Constroi o "runner" inline em bash. Cada análise como subprocess Python
    # independente. Saídas anexadas em run_log.
    quoted_scripts = " \\\n        ".join(shlex.quote(str(s)) for s in scripts)
    bash_runner = textwrap.dedent(f"""\
        set -u
        cd {shlex.quote(str(project_dir))}
        RUN_LOG={shlex.quote(str(run_log))}
        PYBIN={shlex.quote(str(venv_python))}
        ts() {{ date '+%Y-%m-%d %H:%M:%S'; }}
        {{
          echo "==========================================================="
          echo "[$(ts)] Iniciando execução conjunta em background"
          echo "PID master: $$"
          echo "Python    : $PYBIN"
          echo "==========================================================="
        }} >> "$RUN_LOG"
        for script in \\
            {quoted_scripts}; do
          {{
            echo ""
            echo "[$(ts)] >>> Iniciando $(basename "$script")"
          }} >> "$RUN_LOG"
          if "$PYBIN" "$script" >> "$RUN_LOG" 2>&1; then
            echo "[$(ts)] <<< $(basename "$script") CONCLUIDO" >> "$RUN_LOG"
          else
            rc=$?
            echo "[$(ts)] <<< $(basename "$script") FALHOU (exit=$rc)" >> "$RUN_LOG"
          fi
        done
        {{
          echo ""
          echo "[$(ts)] FIM da execução conjunta."
          echo "==========================================================="
        }} >> "$RUN_LOG"
    """)

    # Dispara em novo grupo de sessão, imune a SIGHUP
    try:
        proc = subprocess.Popen(
            ["nohup", "setsid", "bash", "-c", bash_runner],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            start_new_session=True,
            cwd=str(project_dir),
        )
    except FileNotFoundError as exc:
        print("  " + badge_err(f"Falha ao iniciar (binário ausente): {exc}"))
        pause()
        return
    except Exception as exc:
        print("  " + badge_err(f"Erro ao disparar execução: {exc}"))
        pause()
        return

    section("Execução iniciada em background")
    print(f"  PID master      : {proc.pid}")
    print(f"  Log mestre      : {run_log}")
    print(f"  Logs detalhados : {logs_dir}/analise_<n>_<timestamp>.log")
    print()
    print("  A execução é IMUNE ao encerramento do SSH e do app_menu.")
    print("  Você pode fechar o terminal — as 4 análises continuarão.")
    print()
    print("  Acompanhar progresso (em outro terminal):")
    print(f"    tail -f {run_log}")
    print()
    print("  Verificar se ainda está rodando:")
    print("    pgrep -af analise_")
    print()
    print("  Cancelar a execução:")
    print("    pkill -f analise_")
    print()
    pause()


def show_reports() -> None:
    print_header("Tests > Reports")
    section("Status dos relatórios gerados")
    for analysis_id in sorted(TEST_MODULES):
        print(f"  [{analysis_id}] {TEST_MODULES[analysis_id][1]}")
        print(f"      Status: {report_status(analysis_id)}")
        print()

    reports_dir = get_reports_dir()
    print(hline())
    print(f"  Diretório base: {reports_dir}")
    if reports_dir.exists():
        files = [p for p in reports_dir.rglob("*") if p.is_file()]
        print(f"  Arquivos encontrados: {len(files)}")
        for f in sorted(files)[:30]:
            rel = f.relative_to(reports_dir)
            print(f"    • {rel}  ({human_size(f.stat().st_size)})")
        if len(files) > 30:
            print(f"    ... e mais {len(files) - 30} arquivo(s).")
    else:
        print("  " + badge_warn("Diretório de relatórios ainda não existe."))
    pause()


def show_environment_config() -> None:
    print_header("Tests > Environment")
    section("Configuração ativa do ambiente de testes")

    print_config_fn = get_config_callable("print_config")
    summary_fn = get_config_callable("summary")

    if print_config_fn:
        try:
            print_config_fn()
        except Exception as exc:
            print("  " + badge_warn(f"Não foi possível exibir a configuração detalhada: {exc}"))
            if summary_fn:
                try:
                    print(summary_fn())
                except Exception:
                    pass
    elif summary_fn:
        try:
            print(summary_fn())
        except Exception as exc:
            print("  " + badge_warn(f"Não foi possível exibir o resumo da configuração: {exc}"))
    else:
        print(f"  Tests dir   : {get_tests_dir()}")
        print(f"  Dataset dir : {get_dataset_dir()}")
        print(f"  Reports dir : {get_reports_dir()}")
        print(f"  Model dir   : {get_model_dir()}")
        print(f"  Logs dir    : {get_logs_dir()}")

    print("\n  Dependências opcionais:")
    for pkg in ["scikit-optimize", "tabulate", "scipy"]:
        mod_name = pkg.replace("-", "_")
        try:
            __import__(mod_name)
            print(f"    {badge_ok(pkg)}")
        except ImportError:
            print(f"    {badge_warn(pkg + ' não instalado')}  | pip install {pkg}")
    pause()


def manage_dataset() -> None:
    while True:
        print_header("Tests > Dataset")
        section("Gerenciamento do dataset CSE-CIC-IDS2018")
        print(f"  Status   : {dataset_status()}")
        print(f"  Diretório: {get_dataset_dir()}")
        print(hline())
        print("  [1] Verificar / preparar dataset")
        print("  [2] Ver instruções manuais")
        print("  [3] Listar arquivos CSV atuais")
        print("  [0] Voltar")
        print()
        choice = menu_choice(["0", "1", "2", "3"])

        if choice == "0":
            return
        if choice == "1":
            verify_fn = get_config_callable("verificar_dataset")
            if verify_fn:
                try:
                    available = verify_fn(interativo=True)
                    print()
                    print("  " + (badge_ok("Dataset validado com sucesso.") if available else badge_warn("Dataset não validado.")))
                except Exception as exc:
                    print("  " + badge_err(f"Falha ao verificar dataset: {exc}"))
                    traceback.print_exc()
            else:
                ok, missing, csvs = dataset_validation()
                print()
                if ok:
                    print("  " + badge_ok(f"Dataset validado com sucesso. {len(csvs)} CSV(s) encontrados."))
                elif csvs:
                    print("  " + badge_warn(f"Dataset parcial detectado. {len(csvs)} CSV(s) encontrados."))
                    print("  Arquivos ausentes esperados:")
                    for name in missing:
                        print(f"    • {name}")
                else:
                    print("  " + badge_warn("Nenhum CSV encontrado no dataset."))
            pause()
        elif choice == "2":
            manual_fn = get_config_callable("_instrucoes_manuais")
            if manual_fn:
                try:
                    manual_fn()
                except Exception as exc:
                    print("  " + badge_err(f"Falha ao exibir instruções: {exc}"))
            else:
                print("  " + badge_info("Baixe o dataset correto e extraia os CSVs para a pasta Base/CSE-CIC-IDS2018/."))
            pause()
        elif choice == "3":
            csvs = dataset_csv_files()
            if not csvs:
                print("\n  " + badge_warn("Nenhum CSV encontrado."))
            else:
                print()
                for csv in csvs:
                    print(f"  • {csv.name:<36} {human_size(csv.stat().st_size):>10}")
            pause()


def tests_menu() -> None:
    while True:
        print_header("Tests")
        section("Testes e análises acadêmicas", "Execução dos scripts experimentais e gestão de relatórios")
        print(f"  Dataset: {dataset_status()}")
        print(hline())
        for analysis_id in sorted(TEST_MODULES):
            print(f"  [{analysis_id}] {TEST_MODULES[analysis_id][1]}")
            print(f"      Status: {report_status(analysis_id)}")
            print()
        print("  [5] Executar todas as análises")
        print("  [6] Gerenciar dataset")
        print("  [7] Ver relatórios")
        print("  [8] Configuração do ambiente")
        print("  [0] Voltar")
        print()

        choice = menu_choice(["0", "1", "2", "3", "4", "5", "6", "7", "8"])
        if choice == "0":
            return
        if choice in {"1", "2", "3", "4"}:
            print()
            print("  Modo de execução:")
            print("    [F] Foreground (vê o progresso na tela, depende da sessão SSH)")
            print("    [B] Background (libera o terminal, imune ao SSH)")
            modo = menu_choice(["F", "B", "f", "b"]).upper()
            if modo == "B":
                execute_test_in_background(int(choice))
            else:
                execute_test_analysis(int(choice))
            pause()
        elif choice == "5":
            execute_all_tests()
        elif choice == "6":
            manage_dataset()
        elif choice == "7":
            show_reports()
        elif choice == "8":
            show_environment_config()


# =============================================================================
# IDS — CAMADA DE INTEGRAÇÃO COMPLETA
# =============================================================================
# Integra diretamente as funcionalidades reais de:
#   - ids_collector.py  (captura de pacotes)
#   - ids_detector.py   (detecção de incidentes)
#   - ids_learn.py      (treinamento / fine-tuning)
#   - ids_reports.py    (relatórios HTML/TXT)
# =============================================================================

# ── Helpers de processo em background ─────────────────────────────────────────

_IDS_DIR = ROOT_DIR / "IDS"


def _get_config():
    """Retorna o objeto Config do projeto, ou None."""
    cfg = safe_import("config")
    if cfg is None:
        return None
    return getattr(cfg, "Config", cfg)


def _get_pid_file(name: str) -> Path:
    cfg = _get_config()
    temp_dir = Path(cfg.TEMP_DIR) if cfg and hasattr(cfg, "TEMP_DIR") else get_temp_dir()
    return temp_dir / f".{name}.pid"


def _write_pid(pid_file: Path, pid: int) -> None:
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(pid))


def _read_pid(pid_file: Path) -> Optional[int]:
    try:
        return int(pid_file.read_text().strip())
    except Exception:
        return None


def _is_process_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _proc_status_str(pid_file: Path) -> str:
    pid = _read_pid(pid_file)
    if pid and _is_process_running(pid):
        return badge_ok(f"ATIVO (PID {pid})")
    return color("PARADO", "red")


def _run_background(cmd: list[str], log_file: Path | None = None,
                    cwd: Path | None = None) -> subprocess.Popen:
    """Inicia processo em background com stdout/stderr redirecionados."""
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_file, "a", encoding="utf-8")
    else:
        fh = subprocess.DEVNULL

    proc = subprocess.Popen(
        cmd,
        stdout=fh if log_file else subprocess.DEVNULL,
        stderr=subprocess.STDOUT if log_file else subprocess.DEVNULL,
        cwd=cwd or ROOT_DIR,
        start_new_session=True,
    )
    return proc


def _tail_log(log_file: Path, n: int = 50) -> None:
    """Exibe as últimas n linhas de um arquivo de log."""
    if not log_file.exists():
        print(f"  {color('Log não encontrado.', 'dim')}")
        return
    try:
        lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        for row in lines[-n:]:
            print("  " + row[:140])
    except Exception as exc:
        print("  " + badge_warn(f"Não foi possível ler o log: {exc}"))


# ── Resolução de caminhos IDS via Config ──────────────────────────────────────

def _ids_collector_dir() -> Path:
    cfg = _get_config()
    if cfg and hasattr(cfg, "COLLECTOR_DIR"):
        return Path(cfg.COLLECTOR_DIR)
    return ROOT_DIR / "Collector"


def _ids_reports_dir() -> Path:
    cfg = _get_config()
    if cfg and hasattr(cfg, "REPORTS_DIR"):
        return Path(cfg.REPORTS_DIR)
    return ROOT_DIR / "Reports"


def _ids_staging_dir() -> Path:
    cfg = _get_config()
    if cfg and hasattr(cfg, "RETRAIN_CONFIG"):
        return Path(cfg.RETRAIN_CONFIG.get("staging_dir", get_temp_dir() / "staging"))
    return get_temp_dir() / "staging"


def _ids_log(name: str) -> Path:
    cfg = _get_config()
    attr_map = {
        "collector": "LOG_COLLECTOR",
        "app": "LOG_APP",
        "learn": "LOG_LEARN",
    }
    attr = attr_map.get(name)
    if cfg and attr and hasattr(cfg, attr):
        return Path(getattr(cfg, attr))
    return get_logs_dir() / f"{name.capitalize()}.log"


def _ids_capture_interface() -> str:
    cfg = _get_config()
    if cfg and hasattr(cfg, "CAPTURE_INTERFACE"):
        return cfg.CAPTURE_INTERFACE
    return "eth0"


def _ids_collector_budget() -> float:
    cfg = _get_config()
    if cfg and hasattr(cfg, "COLLECTOR_BUDGET_GB"):
        return cfg.COLLECTOR_BUDGET_GB
    return 10.0


def _ids_model_file() -> Path:
    cfg = _get_config()
    model_dir = get_model_dir()
    if cfg and hasattr(cfg, "MODEL_FILENAME"):
        return model_dir / cfg.MODEL_FILENAME
    return model_dir / "ids_model.keras"


def _ids_model_info_file() -> Path:
    cfg = _get_config()
    model_dir = get_model_dir()
    if cfg and hasattr(cfg, "MODEL_INFO_FILENAME"):
        return model_dir / cfg.MODEL_INFO_FILENAME
    return model_dir / "model_info.json"


def _ids_ensure_dirs() -> None:
    cfg = _get_config()
    if cfg and hasattr(cfg, "ensure_dirs"):
        cfg.ensure_dirs()


# ── Scan de arquivos pendentes ────────────────────────────────────────────────

def _ids_scan_new_files() -> list[Path]:
    """Retorna Parquets no COLLECTOR_DIR ainda não analisados."""
    col_dir = _ids_collector_dir()
    if not col_dir.exists():
        return []

    state_file = get_temp_dir() / "ids_state.json"
    analyzed: set[str] = set()
    if state_file.exists():
        try:
            data = json.loads(state_file.read_text(encoding="utf-8"))
            analyzed = set(data.get("analyzed_files", []))
        except Exception:
            pass

    files = sorted(col_dir.glob("*.parquet"))
    return [f for f in files if f.name not in analyzed]


# =============================================================================
# IDS Submenu — Collector
# =============================================================================

def ids_capture_menu() -> None:
    pid_file = _get_pid_file("collector")

    while True:
        print_header("IDS > Captura de Pacotes")
        pid = _read_pid(pid_file)
        running = pid is not None and _is_process_running(pid)

        section("Gerenciamento do Collector")
        print(f"  Status    : {_proc_status_str(pid_file)}")
        print(f"  Interface : {_ids_capture_interface()}")
        print(f"  Saída     : {_ids_collector_dir()}")
        print(f"  Budget    : {_ids_collector_budget():.1f} GiB/dia")
        print(hline())
        print(f"  [1] {'Parar Collector' if running else 'Iniciar Collector'}")
        print("  [2] Ver log do Collector")
        print("  [3] Listar arquivos capturados")
        print("  [4] Importar PCAP externo")
        print("  [5] Ver estatísticas da interface")
        print("  [0] Voltar")
        print()

        choice = menu_choice(["0", "1", "2", "3", "4", "5"])

        if choice == "0":
            return

        elif choice == "1":
            if running:
                if confirm("Parar o Collector?"):
                    import signal as _signal
                    os.kill(pid, _signal.SIGTERM)
                    pid_file.unlink(missing_ok=True)
                    print(f"  {color('Collector encerrado.', 'yellow')}")
                    time.sleep(1)
            else:
                collector_script = _IDS_DIR / "ids_collector.py"
                if not collector_script.exists():
                    print("  " + badge_err(f"Script não encontrado: {collector_script}"))
                    pause()
                    continue
                print("  Iniciando Collector em background …")
                proc = _run_background(
                    [sys.executable, str(collector_script)],
                    log_file=_ids_log("collector"),
                    cwd=ROOT_DIR,
                )
                _write_pid(pid_file, proc.pid)
                print("  " + badge_ok(f"Collector iniciado — PID {proc.pid}"))
                print(f"  Log: {_ids_log('collector')}")
                pause()

        elif choice == "2":
            print()
            _tail_log(_ids_log("collector"), n=50)
            pause()

        elif choice == "3":
            col_dir = _ids_collector_dir()
            files = sorted(col_dir.glob("*.parquet")) if col_dir.exists() else []
            if files:
                rows = [
                    [f.name, human_size(f.stat().st_size),
                     datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")]
                    for f in files
                ]
                print_simple_table(["Arquivo", "Tamanho", "Modificado"], rows,
                                   title="Arquivos Capturados")
            else:
                print("  " + color("Nenhum arquivo encontrado.", "dim"))
            pause()

        elif choice == "4":
            src = prompt("Caminho completo do arquivo PCAP: ")
            if not src:
                print("  " + badge_warn("Nenhum arquivo informado."))
                pause()
                continue
            src_path = Path(src).expanduser()
            if not src_path.exists():
                print("  " + badge_err("Arquivo não encontrado."))
                pause()
                continue
            dst_dir = _IDS_DIR / "imports"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            print("  " + badge_ok(f"PCAP importado para {dst_path}"))
            pause()

        elif choice == "5":
            _show_interface_stats()


# =============================================================================
# IDS Submenu — Detector
# =============================================================================

def ids_detection_menu() -> None:
    pid_file = _get_pid_file("detector")

    while True:
        print_header("IDS > Detecção")
        new_files = _ids_scan_new_files()
        pid = _read_pid(pid_file)
        running = pid is not None and _is_process_running(pid)

        section("Gerenciamento do Detector")
        print(f"  Status detector : {_proc_status_str(pid_file)}")
        print(f"  Arquivos novos  : {color(str(len(new_files)), 'yellow') if new_files else color('0', 'dim')}")
        print(hline())
        print("  [1] Análise em lote (todos os arquivos pendentes)")
        print("  [2] Monitor contínuo (background)")
        print("  [3] Analisar arquivo específico")
        print(f"  [4] {'Parar monitor' if running else color('Monitor não ativo', 'dim')}")
        print("  [5] Ver log de detecção")
        print("  [0] Voltar")
        print()

        choice = menu_choice(["0", "1", "2", "3", "4", "5"])

        if choice == "0":
            return

        elif choice == "1":
            detector_script = _IDS_DIR / "ids_detector.py"
            if not detector_script.exists():
                print("  " + badge_err(f"Script não encontrado: {detector_script}"))
                pause()
                continue
            try:
                from IDS.ids_detector import cmd_batch
                cmd_batch()
            except ImportError:
                # Fallback: executa como subprocesso
                subprocess.run(
                    [sys.executable, str(detector_script), "batch"],
                    cwd=ROOT_DIR,
                )
            pause()

        elif choice == "2":
            interval = prompt("Intervalo de varredura em segundos [60]: ") or "60"
            try:
                interval_s = int(interval)
            except ValueError:
                print("  " + badge_err("Valor inválido."))
                pause()
                continue
            detector_script = _IDS_DIR / "ids_detector.py"
            if not detector_script.exists():
                print("  " + badge_err(f"Script não encontrado: {detector_script}"))
                pause()
                continue
            proc = _run_background(
                [sys.executable, str(detector_script),
                 "watch", "--interval", str(interval_s)],
                log_file=_ids_log("app"),
                cwd=ROOT_DIR,
            )
            _write_pid(pid_file, proc.pid)
            print("  " + badge_ok(f"Monitor iniciado — PID {proc.pid}"))
            pause()

        elif choice == "3":
            path_str = prompt("Caminho do arquivo .parquet: ")
            if not path_str:
                print("  " + badge_warn("Nenhum caminho informado."))
                pause()
                continue
            fpath = Path(path_str).expanduser()
            if not fpath.exists():
                print("  " + badge_err(f"Arquivo não encontrado: {fpath}"))
                pause()
                continue
            detector_script = _IDS_DIR / "ids_detector.py"
            try:
                from IDS.ids_detector import cmd_file
                cmd_file(fpath)
            except ImportError:
                subprocess.run(
                    [sys.executable, str(detector_script), "file", str(fpath)],
                    cwd=ROOT_DIR,
                )
            pause()

        elif choice == "4":
            if running:
                if confirm("Parar o monitor?"):
                    import signal as _signal
                    os.kill(pid, _signal.SIGTERM)
                    pid_file.unlink(missing_ok=True)
                    print(f"  {color('Monitor encerrado.', 'yellow')}")
            else:
                print(f"  {color('Monitor não está em execução.', 'dim')}")
            pause()

        elif choice == "5":
            print()
            _tail_log(_ids_log("app"), n=80)
            pause()


# =============================================================================
# IDS Submenu — Treinamento
# =============================================================================

def ids_training_menu() -> None:
    while True:
        print_header("IDS > Treinamento")
        section("Gerenciamento de Treinamento")

        # Exibe status do modelo atual
        model_file = _ids_model_file()
        info_file = _ids_model_info_file()
        if model_file.exists():
            version, trained_at = "?", "?"
            if info_file.exists():
                try:
                    info = json.loads(info_file.read_text(encoding="utf-8"))
                    version = info.get("version", "?")
                    trained_at = info.get("trained_at", "?")[:19]
                except Exception:
                    pass
            print(f"  Modelo    : {badge_ok(f'v{version} ({trained_at})')}")
            print(f"  Artefato  : {model_file}")
        else:
            print(f"  Modelo    : {color('Não treinado', 'red')}")

        staging_dir = _ids_staging_dir()
        stg_cnt = len(list(staging_dir.glob("*.parquet"))) if staging_dir.exists() else 0
        print(f"  Staging   : {stg_cnt} arquivo(s) para fine-tuning")
        print(hline())

        print("  [1] Treinamento completo (do zero)")
        print("  [2] Treinamento forçado (limpa cache)")
        stg_label = f"  ({color(f'{stg_cnt} arquivo(s)', 'yellow')})" if stg_cnt else ""
        print(f"  [3] Fine-tuning incremental{stg_label}")
        print("  [4] Avaliar modelo (accuracy, recall, F1, ROC)")
        print("  [5] Ver log de treinamento")
        print("  [6] Listar modelos disponíveis")
        print("  [0] Voltar")
        print()

        choice = menu_choice(["0", "1", "2", "3", "4", "5", "6"])

        if choice == "0":
            return

        elif choice in ("1", "2"):
            force = choice == "2"
            msg = "Treinamento forçado" if force else "Treinamento completo"
            if confirm(f"Iniciar {msg}? (pode demorar 30–120 min)"):
                learn_script = _IDS_DIR / "ids_learn.py"
                if not learn_script.exists():
                    print("  " + badge_err(f"Script não encontrado: {learn_script}"))
                    pause()
                    continue
                extra = ["--force"] if force else []
                proc = _run_background(
                    [sys.executable, str(learn_script), "train"] + extra,
                    log_file=_ids_log("learn"),
                    cwd=ROOT_DIR,
                )
                print("  " + badge_ok(f"Treinamento iniciado — PID {proc.pid}"))
                print(f"  Acompanhe em: {_ids_log('learn')}")
            pause()

        elif choice == "3":
            if stg_cnt == 0:
                print(f"  {color('Nenhum dado de staging disponível.', 'dim')}")
                pause()
                continue
            if confirm(f"Fine-tuning com {stg_cnt} arquivo(s)?"):
                learn_script = _IDS_DIR / "ids_learn.py"
                if not learn_script.exists():
                    print("  " + badge_err(f"Script não encontrado: {learn_script}"))
                    pause()
                    continue
                proc = _run_background(
                    [sys.executable, str(learn_script), "finetune"],
                    log_file=_ids_log("learn"),
                    cwd=ROOT_DIR,
                )
                print("  " + badge_ok(f"Fine-tuning iniciado — PID {proc.pid}"))
            pause()

        elif choice == "4":
            _ids_model_evaluation()

        elif choice == "5":
            print()
            _tail_log(_ids_log("learn"), n=100)
            pause()

        elif choice == "6":
            _list_available_models()


def _ids_model_evaluation() -> None:
    """Avalia modelo atual com métricas padrão do trabalho."""
    print_header("IDS > Avaliação do Modelo")
    section("Avaliação do modelo ativo")

    model_file = _ids_model_file()
    if not model_file.exists():
        print("  " + badge_warn("Nenhum modelo treinado encontrado."))
        pause()
        return

    # Tenta usar ids_learn.cmd_status para exibir métricas
    try:
        from IDS.ids_learn import cmd_status
        cmd_status()
    except ImportError:
        print(f"  Modelo    : {model_file}")
        print(f"  Tamanho   : {human_size(model_file.stat().st_size)}")
        info_file = _ids_model_info_file()
        if info_file.exists():
            try:
                info = json.loads(info_file.read_text(encoding="utf-8"))
                print(f"  Versão    : {info.get('version', '?')}")
                print(f"  Treinado  : {info.get('trained_at', '?')}")
                metrics = info.get("metrics", {})
                if metrics:
                    print(hline())
                    print("  " + color("Métricas:", "cyan", "bold"))
                    for k, v in metrics.items():
                        if isinstance(v, float):
                            print(f"    {k:<30s}: {v:.4f}")
                        else:
                            print(f"    {k:<30s}: {v}")
            except Exception as exc:
                print("  " + badge_warn(f"Erro ao ler metadados: {exc}"))
    except Exception as exc:
        print("  " + badge_err(f"Erro na avaliação: {exc}"))
        traceback.print_exc()
    pause()


def _list_available_models() -> None:
    """Lista todos os artefatos de modelo disponíveis."""
    print_header("IDS > Modelos Disponíveis")
    section("Artefatos de modelo")
    model_dir = get_model_dir()
    candidates: list[Path] = []
    if model_dir.exists():
        for pattern in ["*.keras", "*.h5", "*.joblib", "*.pkl", "*.json", "*.yaml", "*.yml"]:
            candidates.extend(model_dir.rglob(pattern))
    candidates = sorted(set(candidates))

    print(f"  Diretório: {model_dir}")
    if not candidates:
        print("  " + badge_warn("Nenhum artefato de modelo encontrado."))
    else:
        for idx, path in enumerate(candidates, 1):
            print(f"  [{idx:02d}] {path.relative_to(model_dir)}  ({human_size(path.stat().st_size)})")
    pause()


# =============================================================================
# IDS Submenu — Relatórios
# =============================================================================

def ids_reports_menu() -> None:
    while True:
        print_header("IDS > Relatórios")
        section("Relatórios do Sistema IDS")

        reports_dir = _ids_reports_dir()
        html_files = sorted(reports_dir.glob("*.html"), reverse=True) if reports_dir.exists() else []
        jsonl_files = sorted(reports_dir.glob("*.jsonl"), reverse=True) if reports_dir.exists() else []

        print(f"  Diretório : {reports_dir}")
        print(f"  HTML      : {len(html_files)} relatório(s)")
        print(f"  JSONL     : {len(jsonl_files)} arquivo(s) de incidentes")
        print(hline())
        print("  [1] Listar e abrir relatórios HTML")
        print("  [2] Listar incidentes (JSONL)")
        print("  [3] Exportar relatórios")
        print("  [0] Voltar")
        print()

        choice = menu_choice(["0", "1", "2", "3"])

        if choice == "0":
            return

        elif choice == "1":
            if not html_files:
                print(f"  {color('Nenhum relatório HTML encontrado.', 'dim')}")
                pause()
                continue
            rows = [
                [f.stem, human_size(f.stat().st_size),
                 datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")]
                for f in html_files[:20]
            ]
            print_simple_table(["Relatório", "Tamanho", "Gerado"], rows)
            print()
            if confirm("Abrir relatório no browser?"):
                idx_str = prompt(f"Número (1–{min(20, len(html_files))}) [1]: ") or "1"
                try:
                    chosen = html_files[int(idx_str) - 1]
                    opener = "xdg-open" if os.name != "nt" else "start"
                    subprocess.run([opener, str(chosen)], check=False,
                                   stderr=subprocess.DEVNULL)
                    print(f"  Abrindo: {chosen.name}")
                except Exception as e:
                    print("  " + badge_err(f"Erro: {e}"))
            pause()

        elif choice == "2":
            if not jsonl_files:
                print(f"  {color('Nenhum arquivo JSONL encontrado.', 'dim')}")
                pause()
                continue
            rows = [
                [f.name, human_size(f.stat().st_size),
                 datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")]
                for f in jsonl_files[:15]
            ]
            print_simple_table(["Arquivo", "Tamanho", "Gerado"], rows)
            pause()

        elif choice == "3":
            export_dir = ROOT_DIR / "Exports" / time.strftime("reports_%Y%m%d_%H%M%S")
            export_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            if reports_dir.exists():
                for f in reports_dir.rglob("*"):
                    if f.is_file():
                        dest = export_dir / f.name
                        shutil.copy2(f, dest)
                        copied += 1
            print("  " + badge_ok(f"{copied} arquivo(s) exportados para {export_dir}"))
            pause()


# =============================================================================
# IDS Submenu — Status do Sistema
# =============================================================================

def ids_system_status() -> None:
    """Visão geral completa do sistema IDS."""
    print_header("IDS > Status")
    section("Status do Sistema IDS")

    pid_col = _get_pid_file("collector")
    pid_det = _get_pid_file("detector")

    print(f"  {'Collector':<22s}: {_proc_status_str(pid_col)}")
    print(f"  {'Detector':<22s}: {_proc_status_str(pid_det)}")

    # Modelo
    model_file = _ids_model_file()
    info_file = _ids_model_info_file()
    if model_file.exists():
        version, trained_at = "?", "?"
        if info_file.exists():
            try:
                info = json.loads(info_file.read_text(encoding="utf-8"))
                version = info.get("version", "?")
                trained_at = info.get("trained_at", "?")[:19]
            except Exception:
                pass
        print(f"  {'Modelo':<22s}: {color(f'v{version} ({trained_at})', 'cyan')}")
    else:
        print(f"  {'Modelo':<22s}: {color('Não treinado', 'red')}")

    # Arquivos capturados
    col_dir = _ids_collector_dir()
    col_files = list(col_dir.glob("*.parquet")) if col_dir.exists() else []
    col_size = sum(f.stat().st_size for f in col_files)
    print(f"  {'Dados capturados':<22s}: {len(col_files)} arquivo(s) | {human_size(col_size)}")

    # Pendentes
    new_files = _ids_scan_new_files()
    pending_color = "yellow" if new_files else "dim"
    print(f"  {'Arquivos pendentes':<22s}: {color(str(len(new_files)), pending_color)}")

    # Relatórios
    rep_dir = _ids_reports_dir()
    rep_files = list(rep_dir.glob("*.html")) if rep_dir.exists() else []
    print(f"  {'Relatórios gerados':<22s}: {len(rep_files)}")

    # Staging
    staging = _ids_staging_dir()
    stg_files = list(staging.glob("*.parquet")) if staging.exists() else []
    print(f"  {'Staging (re-treino)':<22s}: {len(stg_files)} arquivo(s)")

    print(hline())
    pause()


# =============================================================================
# IDS Submenu — Diagnósticos
# =============================================================================

def _show_interface_stats() -> None:
    iface = prompt("Interface de rede (ex: eth0, ens18): ")
    if not iface:
        print("  " + badge_warn("Nenhuma interface informada."))
        pause()
        return
    cmd = shutil.which("ip")
    if not cmd:
        print("  " + badge_warn("Comando 'ip' não encontrado no sistema."))
        pause()
        return
    try:
        result = subprocess.run([cmd, "-s", "link", "show", iface],
                                capture_output=True, text=True, check=False)
        print()
        if result.returncode == 0 and result.stdout.strip():
            print(result.stdout)
        else:
            print("  " + badge_err(result.stderr.strip() or "Não foi possível consultar a interface."))
    except Exception as exc:
        print("  " + badge_err(f"Falha ao obter estatísticas: {exc}"))
    pause()


def diagnostics_cpu_gpu() -> None:
    print_header("IDS > Diagnostics")
    section("Diagnóstico de CPU / GPU")
    print(f"  Python       : {sys.version.split()[0]}")
    print(f"  CPU cores    : {os.cpu_count()}")
    if hasattr(os, "getloadavg"):
        try:
            a, b, c = os.getloadavg()
            print(f"  Load average : {a:.2f} | {b:.2f} | {c:.2f}")
        except Exception:
            pass
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        print("\n  GPU detectada via nvidia-smi:\n")
        result = subprocess.run(
            [nvidia_smi, "--query-gpu=name,driver_version,memory.total,utilization.gpu",
             "--format=csv,noheader"],
            capture_output=True, text=True, check=False,
        )
        print(result.stdout or result.stderr)
    else:
        print("\n  " + badge_info("GPU não detectada via nvidia-smi ou ambiente CPU-only."))
    pause()


def diagnostics_disk() -> None:
    print_header("IDS > Diagnostics")
    section("Teste simples de escrita/leitura em disco")
    temp_dir = get_temp_dir()
    temp_dir.mkdir(exist_ok=True)
    test_file = temp_dir / "disk_benchmark.bin"
    size_mb = 64
    payload = os.urandom(1024 * 1024)
    try:
        t0 = time.perf_counter()
        with open(test_file, "wb") as fh:
            for _ in range(size_mb):
                fh.write(payload)
        t1 = time.perf_counter()
        with open(test_file, "rb") as fh:
            while fh.read(1024 * 1024):
                pass
        t2 = time.perf_counter()
        write_mbps = size_mb / max(t1 - t0, 1e-6)
        read_mbps = size_mb / max(t2 - t1, 1e-6)
        print(f"  Arquivo teste : {test_file}")
        print(f"  Tamanho       : {size_mb} MB")
        print(f"  Escrita       : {write_mbps:.1f} MB/s")
        print(f"  Leitura       : {read_mbps:.1f} MB/s")
    except Exception as exc:
        print("  " + badge_err(f"Falha no benchmark de disco: {exc}"))
    finally:
        test_file.unlink(missing_ok=True)
    pause()


def ids_diagnostics_menu() -> None:
    while True:
        print_header("IDS > Diagnósticos")
        section("Ferramentas de Diagnóstico")
        print("  [1] Testar performance da CPU/GPU")
        print("  [2] Testar velocidade de leitura do disco")
        print("  [3] Testar throughput da interface de rede")
        print("  [4] Verificar integridade dos datasets")
        print("  [5] Benchmark do modelo atual")
        print("  [0] Voltar")
        print()
        choice = menu_choice(["0", "1", "2", "3", "4", "5"])

        if choice == "0":
            return
        elif choice == "1":
            diagnostics_cpu_gpu()
        elif choice == "2":
            diagnostics_disk()
        elif choice == "3":
            _show_interface_stats()
        elif choice == "4":
            print_header("IDS > Diagnostics")
            section("Integridade de datasets")
            dataset_dir = get_dataset_dir()
            print(f"  Dataset dir : {dataset_dir}")
            if not dataset_dir.exists():
                print("  " + badge_warn("Diretório Base não encontrado."))
            else:
                csvs = dataset_csv_files()
                if not csvs:
                    print("  " + badge_warn("Nenhum CSV encontrado."))
                else:
                    print("  " + badge_ok(f"{len(csvs)} CSV(s) encontrados."))
                    for f in csvs[:25]:
                        print(f"    • {f.name:<34} {human_size(f.stat().st_size):>10}")
            pause()
        elif choice == "5":
            _ids_model_evaluation()


# =============================================================================
# IDS Submenu — Logs
# =============================================================================

def ids_logs_menu() -> None:
    while True:
        print_header("IDS > Logs")
        section("Logs e Auditoria")
        print("  [1] Ver logs do sistema (App.log)")
        print("  [2] Ver logs do Collector")
        print("  [3] Ver logs de treinamento (Learn.log)")
        print("  [4] Exportar logs")
        print("  [5] Limpar logs antigos")
        print("  [0] Voltar")
        print()
        choice = menu_choice(["0", "1", "2", "3", "4", "5"])

        if choice == "0":
            return
        elif choice == "1":
            print()
            _tail_log(_ids_log("app"), n=60)
            pause()
        elif choice == "2":
            print()
            _tail_log(_ids_log("collector"), n=60)
            pause()
        elif choice == "3":
            print()
            _tail_log(_ids_log("learn"), n=100)
            pause()
        elif choice == "4":
            log_dir = get_logs_dir()
            if not log_dir.exists():
                print("  " + badge_warn("Diretório de logs não encontrado."))
                pause()
                continue
            target_dir = ROOT_DIR / "Exports" / time.strftime("logs_%Y%m%d_%H%M%S")
            target_dir.mkdir(parents=True, exist_ok=True)
            copied = 0
            for f in log_dir.rglob("*"):
                if f.is_file():
                    dest = target_dir / f.relative_to(log_dir)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, dest)
                    copied += 1
            print("  " + badge_ok(f"{copied} arquivo(s) exportados para {target_dir}"))
            pause()
        elif choice == "5":
            log_dir = get_logs_dir()
            if not log_dir.exists():
                print("  " + badge_warn("Diretório de logs não encontrado."))
                pause()
                continue
            days = prompt("Remover logs mais antigos que quantos dias? [30]: ") or "30"
            try:
                max_age_days = int(days)
            except ValueError:
                print("  " + badge_err("Valor inválido."))
                pause()
                continue
            cutoff = time.time() - (max_age_days * 86400)
            removed = 0
            for f in log_dir.rglob("*"):
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
                    removed += 1
            print("  " + badge_ok(f"{removed} arquivo(s) removidos."))
            pause()


# =============================================================================
# IDS Submenu — Configurações
# =============================================================================

def ids_settings_menu() -> None:
    while True:
        print_header("IDS > Configurações")
        section("Configurações do Sistema")
        print(f"  Interface captura : {_ids_capture_interface()}")
        print(f"  Budget diário     : {_ids_collector_budget():.1f} GiB")
        print(f"  Collector dir     : {_ids_collector_dir()}")
        print(f"  Model dir         : {get_model_dir()}")
        print(f"  Reports dir       : {_ids_reports_dir()}")
        print(f"  Logs dir          : {get_logs_dir()}")
        print(f"  Staging dir       : {_ids_staging_dir()}")
        print(hline())
        print("  [1] Atualizar dependências")
        print("  [2] Recriar diretórios do projeto")
        print("  [0] Voltar")
        print()

        choice = menu_choice(["0", "1", "2"])

        if choice == "0":
            return
        elif choice == "1":
            print("  Use o instalador do projeto para atualizar o ambiente:")
            print(f"  • {ROOT_DIR / 'install.sh'}")
            print("  Ou execute manualmente o gerenciador de pacotes do seu ambiente virtual.")
            pause()
        elif choice == "2":
            _ids_ensure_dirs()
            print("  " + badge_ok("Diretórios recriados/verificados."))
            pause()


# =============================================================================
# IDS — Menu Principal
# =============================================================================

def ids_menu() -> None:
    while True:
        print_header("IDS")
        section("MENU PRINCIPAL — IDS com IA/ML")

        # Mini-status inline
        pid_col = _get_pid_file("collector")
        pid_det = _get_pid_file("detector")
        new_files = _ids_scan_new_files()

        print(f"  Collector: {_proc_status_str(pid_col)}   "
              f"Detector: {_proc_status_str(pid_det)}")
        if new_files:
            print(f"  {color(f'⚠  {len(new_files)} arquivo(s) pendente(s) para análise', 'yellow', 'bold')}")
        print(hline())

        print("  [1] Captura de Pacotes       (Collector)")
        print("  [2] Detecção de Incidentes   (Detector)")
        print("  [3] Treinamento de Modelos   (Train / Fine-tune)")
        print("  [4] Relatórios               (HTML / JSONL)")
        print("  [5] Status do Sistema")
        print("  [6] Ferramentas de Diagnóstico")
        print("  [7] Logs e Auditoria")
        print("  [8] Configurações")
        print("  [0] Voltar")
        print()
        choice = menu_choice(["0", "1", "2", "3", "4", "5", "6", "7", "8"])

        if choice == "0":
            return
        elif choice == "1":
            ids_capture_menu()
        elif choice == "2":
            ids_detection_menu()
        elif choice == "3":
            ids_training_menu()
        elif choice == "4":
            ids_reports_menu()
        elif choice == "5":
            ids_system_status()
        elif choice == "6":
            ids_diagnostics_menu()
        elif choice == "7":
            ids_logs_menu()
        elif choice == "8":
            ids_settings_menu()


# -----------------------------------------------------------------------------
# Main menu
# -----------------------------------------------------------------------------


def show_startup_status() -> None:
    print(f"  Dataset : {dataset_status()}")
    print(f"  Base    : {get_dataset_dir()}")
    print(f"  Model   : {get_model_dir()}")
    print(f"  Reports : {get_reports_dir()}")
    print(hline())


def main_menu() -> None:
    while True:
        print_header("Main Menu")
        show_startup_status()
        print("  " + color("Nome do APP", "white", "bold") + f" : {APP_NAME}")
        print("  " + color("Versão", "white", "bold") + f"     : {APP_VERSION}")
        print()
        print("  [1] Realizar Testes")
        print("      Acesso ao conjunto de análises experimentais, dataset e relatórios.")
        print()
        print("  [2] Sistema IDS")
        print("      Operação do motor de IDS, treinamento, diagnósticos, logs e gestão.")
        print()
        print("  [0] Sair")
        print()
        choice = menu_choice(["0", "1", "2"])

        if choice == "0":
            print()
            print("  " + badge_info("Encerrando o SecurityIA. Até logo."))
            print()
            break
        elif choice == "1":
            tests_menu()
        elif choice == "2":
            ids_menu()


if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\n  " + badge_warn("Execução interrompida pelo usuário."))
        print()
