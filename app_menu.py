#!/usr/bin/env python3
"""
app_menu.py — Frontend Unificado do projeto SecurityIA

Unifica:
  - Modo de Testes/Análises acadêmicas
  - Modo Sistema IDS

Objetivo:
  - Centralizar a experiência CLI em um único ponto de entrada
  - Manter compatibilidade com os scripts já existentes
  - Servir de base para evolução incremental do IDS

Uso:
  python3 app_menu.py
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import time
import traceback
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
# Test/analysis integration
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


def execute_all_tests() -> None:
    results: Dict[int, Tuple[bool, float]] = {}
    started = time.perf_counter()
    for analysis_id in sorted(TEST_MODULES):
        t0 = time.perf_counter()
        ok = execute_test_analysis(analysis_id)
        results[analysis_id] = (ok, time.perf_counter() - t0)
        print()
    total = time.perf_counter() - started

    print_header("Tests > Summary")
    section("Resumo da execução conjunta")
    for analysis_id, (ok, elapsed) in results.items():
        icon = badge_ok("OK") if ok else badge_err("FALHA")
        print(f"  [{analysis_id}] {TEST_MODULES[analysis_id][1][:56]:<56} {icon}  {elapsed:6.1f}s")
    print(hline())
    print(f"  Tempo total : {total:.1f}s ({total/60:.1f} min)")
    print(f"  Relatórios  : {get_reports_dir()}")
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


# -----------------------------------------------------------------------------
# IDS integration layer
# -----------------------------------------------------------------------------


def import_ids_stack() -> Tuple[dict[str, Any], list[str]]:
    """Importa de forma resiliente o stack do IDS."""
    missing: list[str] = []
    loaded: dict[str, Any] = {}

    candidates = {
        "ids_manager": "ids_manager",
        "ids_config": "ids_config",
    }
    for key, mod_name in candidates.items():
        mod = safe_import(mod_name)
        if mod is None:
            missing.append(mod_name)
        else:
            loaded[key] = mod

    # atributos úteis do ids_manager, quando disponível
    ids_manager = loaded.get("ids_manager")
    if ids_manager:
        for attr in [
            "ManagerState",
            "ModelArtifacts",
            "scan_new_files",
            "op_analyze",
            "op_retrain",
            "op_generate_report",
            "print_terminal_summary",
            "submenu_avaliacao",
            "submenu_relatorios",
            "submenu_configuracoes",
            "menu_status",
            "load_evaluator",
        ]:
            if hasattr(ids_manager, attr):
                loaded[attr] = getattr(ids_manager, attr)
            else:
                missing.append(f"ids_manager.{attr}")

    ids_config = loaded.get("ids_config")
    if ids_config and hasattr(ids_config, "IDSConfig"):
        loaded["IDSConfig"] = ids_config.IDSConfig
    elif ids_config:
        missing.append("ids_config.IDSConfig")

    return loaded, missing


def require_ids_stack() -> dict[str, Any] | None:
    loaded, missing = import_ids_stack()
    if "ids_manager" not in loaded:
        print("\n  " + badge_warn("Stack principal do IDS ainda não está completo nesta instalação."))
        if missing:
            print("  Componentes ausentes:")
            for item in missing[:10]:
                print(f"    • {item}")
        print("  O menu continua disponível, mas algumas operações ficarão em modo informativo.")
        return None
    return loaded


def show_placeholder(title: str, detail: str, tips: list[str] | None = None) -> None:
    print_header(title)
    section(title, "Funcionalidade prevista na arquitetura, pronta para evolução incremental")
    print("  " + badge_info(detail))
    if tips:
        print()
        for tip in tips:
            print(f"  • {tip}")
    pause()


def run_ids_analysis(retrain_after: bool = False) -> None:
    loaded = require_ids_stack()
    if not loaded:
        pause()
        return

    ManagerState = loaded["ManagerState"]
    ModelArtifacts = loaded["ModelArtifacts"]
    scan_new_files = loaded["scan_new_files"]
    op_analyze = loaded["op_analyze"]
    op_retrain = loaded["op_retrain"]
    op_generate_report = loaded["op_generate_report"]
    print_terminal_summary = loaded["print_terminal_summary"]

    try:
        state = ManagerState()
        new_files = scan_new_files(state)
        if not new_files:
            print_header("IDS > Detection")
            section("Detecção em tempo real")
            print("  " + badge_warn("Nenhum arquivo novo encontrado no diretório de coleta."))
            pause()
            return

        arts = ModelArtifacts()
        arts.load()
        version = arts.version_tag()

        results = op_analyze(state, arts, new_files)
        retrained = False

        if retrain_after:
            retrained = op_retrain(results, new_files, state)

        if results:
            print_terminal_summary(results, retrained)
            op_generate_report(results, version, retrained)
    except FileNotFoundError as exc:
        print("\n  " + badge_err(str(exc)))
    except Exception as exc:
        print("\n  " + badge_err(f"Falha ao executar a rotina do IDS: {exc}"))
        traceback.print_exc()
    pause()


def list_available_models() -> None:
    print_header("IDS > Model Management")
    section("Modelos disponíveis")
    model_dir = get_model_dir()
    candidates: list[Path] = []
    if model_dir.exists():
        patterns = ["*.keras", "*.h5", "*.joblib", "*.pkl", "*.json", "*.yaml", "*.yml"]
        for pattern in patterns:
            candidates.extend(model_dir.rglob(pattern))
    candidates = sorted(set(candidates))

    print(f"  Diretório: {model_dir}")
    if not candidates:
        print("  " + badge_warn("Nenhum artefato de modelo encontrado."))
    else:
        for idx, path in enumerate(candidates, 1):
            print(f"  [{idx:02d}] {path.relative_to(model_dir)}  ({human_size(path.stat().st_size)})")
    pause()


def show_logs(kind: str = "all") -> None:
    print_header("IDS > Logs & Audit")
    section("Logs e auditoria")
    log_dir = get_logs_dir()
    if not log_dir.exists():
        print("  " + badge_warn("Diretório de logs não encontrado."))
        print(f"  Caminho esperado: {log_dir}")
        pause()
        return

    patterns = {
        "system": ["*system*.log", "*app*.log", "*.log"],
        "detection": ["*detect*.log", "*ids*.log", "*.log"],
        "training": ["*train*.log", "*learn*.log", "*.log"],
        "all": ["*.log", "*.txt"],
    }
    files: list[Path] = []
    for pattern in patterns.get(kind, ["*.log"]):
        files.extend(log_dir.rglob(pattern))
    files = sorted(set(f for f in files if f.is_file()), reverse=True)

    if not files:
        print("  " + badge_warn("Nenhum arquivo de log encontrado."))
        pause()
        return

    for f in files[:15]:
        print(f"  • {f.relative_to(log_dir)}  ({human_size(f.stat().st_size)})")
    print()
    print("  Últimas linhas do arquivo mais recente:")
    print(hline())
    try:
        tail = files[0].read_text(encoding="utf-8", errors="ignore").splitlines()[-20:]
        for row in tail:
            print("  " + row[:120])
    except Exception as exc:
        print("  " + badge_warn(f"Não foi possível ler o log: {exc}"))
    pause()


def export_logs() -> None:
    print_header("IDS > Logs Export")
    section("Exportar logs")
    log_dir = get_logs_dir()
    if not log_dir.exists():
        print("  " + badge_warn("Diretório de logs não encontrado."))
        pause()
        return

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


def clean_old_logs() -> None:
    print_header("IDS > Logs Cleanup")
    section("Limpeza de logs antigos")
    log_dir = get_logs_dir()
    if not log_dir.exists():
        print("  " + badge_warn("Diretório de logs não encontrado."))
        pause()
        return

    days = prompt("Remover logs mais antigos que quantos dias? [30]: ") or "30"
    try:
        max_age_days = int(days)
    except ValueError:
        print("  " + badge_err("Valor inválido."))
        pause()
        return

    cutoff = time.time() - (max_age_days * 86400)
    removed = 0
    for f in log_dir.rglob("*"):
        if f.is_file() and f.stat().st_mtime < cutoff:
            f.unlink(missing_ok=True)
            removed += 1

    print("  " + badge_ok(f"{removed} arquivo(s) removidos."))
    pause()


def show_interface_stats() -> None:
    print_header("IDS > Capture")
    section("Estatísticas da interface")
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
        result = subprocess.run([cmd, "-s", "link", "show", iface], capture_output=True, text=True, check=False)
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
        result = subprocess.run([nvidia_smi, "--query-gpu=name,driver_version,memory.total,utilization.gpu", "--format=csv,noheader"], capture_output=True, text=True, check=False)
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


def diagnostics_network() -> None:
    print_header("IDS > Diagnostics")
    section("Throughput / counters de interface")
    show_interface_stats()


def diagnostics_datasets() -> None:
    print_header("IDS > Diagnostics")
    section("Integridade de datasets")
    dataset_dir = get_dataset_dir()
    print(f"  Dataset dir : {dataset_dir}")
    if not dataset_dir.exists():
        print("  " + badge_warn("Diretório Base não encontrado."))
        pause()
        return

    csvs = dataset_csv_files()
    if not csvs:
        print("  " + badge_warn("Nenhum CSV encontrado."))
    else:
        print("  " + badge_ok(f"{len(csvs)} CSV(s) encontrados."))
        for f in csvs[:25]:
            print(f"    • {f.name:<34} {human_size(f.stat().st_size):>10}")
        if len(csvs) > 25:
            print(f"    ... e mais {len(csvs)-25} arquivo(s).")
    pause()


def diagnostics_model_benchmark() -> None:
    loaded = require_ids_stack()
    if not loaded:
        pause()
        return

    load_evaluator = loaded.get("load_evaluator")
    if not callable(load_evaluator):
        print("  " + badge_warn("Avaliador do modelo não disponível."))
        pause()
        return

    print_header("IDS > Diagnostics")
    section("Benchmark do modelo atual")
    ev = load_evaluator()
    if ev is None:
        print("  " + badge_warn("Avaliador indisponível."))
        pause()
        return

    version = prompt("Identificador da versão [current]: ") or "current"
    label = prompt("Descrição [benchmark manual]: ") or "benchmark manual"
    notes = prompt("Notas [opcional]: ")
    try:
        ev.evaluate_model(version, label, notes)
    except Exception as exc:
        print("  " + badge_err(f"Falha ao avaliar o modelo: {exc}"))
        traceback.print_exc()
    pause()


def ids_capture_menu() -> None:
    while True:
        print_header("IDS > Packet Capture")
        section("🛰️  Captura de Pacotes")
        print("  [1] Iniciar captura em interface")
        print("  [2] Parar captura")
        print("  [3] Captura com filtros (IP, porta, VLAN)")
        print("  [4] Captura contínua (modo sensor)")
        print("  [5] Importar PCAP externo")
        print("  [6] Ver estatísticas da interface")
        print("  [0] Voltar")
        print()
        choice = menu_choice(["0", "1", "2", "3", "4", "5", "6"])

        if choice == "0":
            return
        elif choice == "1":
            show_placeholder(
                "IDS > Packet Capture",
                "A captura ativa pode ser integrada ao seu coletor atual (ex.: ids_coletor.py).",
                [
                    "Use esta opção como ponto central do CLI para iniciar o coletor quando o script definitivo estiver consolidado.",
                    "Sugestão: integrar subprocess.Popen com PID file em temp/capture.pid.",
                ],
            )
        elif choice == "2":
            show_placeholder(
                "IDS > Packet Capture",
                "Encerramento de captura ainda depende do processo definitivo do coletor.",
                ["Sugestão: encerrar via PID file e registrar a ação em Logs/."],
            )
        elif choice == "3":
            show_placeholder(
                "IDS > Packet Capture",
                "Captura filtrada prevista para BPF/tcpdump/scapy conforme evolução do pipeline.",
                ["Exemplos futuros: host 10.0.0.1, port 443, vlan 200."],
            )
        elif choice == "4":
            show_placeholder(
                "IDS > Packet Capture",
                "Modo sensor previsto para execução contínua do coletor + conversor em pipeline.",
                ["Ideal para integração com serviço systemd ou container dedicado."],
            )
        elif choice == "5":
            src = prompt("Informe o caminho completo do arquivo PCAP: ")
            if not src:
                print("  " + badge_warn("Nenhum arquivo informado."))
                pause()
                continue
            src_path = Path(src).expanduser()
            if not src_path.exists():
                print("  " + badge_err("Arquivo não encontrado."))
                pause()
                continue
            dst_dir = ROOT_DIR / "IDS" / "imports"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_path = dst_dir / src_path.name
            shutil.copy2(src_path, dst_path)
            print("  " + badge_ok(f"PCAP importado para {dst_path}"))
            pause()
        elif choice == "6":
            show_interface_stats()


def ids_processing_menu() -> None:
    while True:
        print_header("IDS > Processing")
        section("🧪  Processamento e Feature Engineering")
        print("  [1] Extrair features de PCAP")
        print("  [2] Normalizar dados")
        print("  [3] Balancear dataset (SMOTE, undersampling)")
        print("  [4] Gerar dataset para ML")
        print("  [5] Visualizar amostras e estatísticas")
        print("  [0] Voltar")
        print()
        choice = menu_choice(["0", "1", "2", "3", "4", "5"])

        if choice == "0":
            return
        if choice == "5":
            diagnostics_datasets()
        else:
            descriptions = {
                "1": "Extração de features prevista para o conversor de fluxo/PCAP do pipeline IDS.",
                "2": "Normalização deve reaproveitar o mesmo scaler/artefatos usados no treinamento.",
                "3": "Balanceamento pode reutilizar a estratégia validada nos scripts de testes (SMOTE-ENN, etc.).",
                "4": "Geração de dataset para ML deve alimentar a trilha de treinamento supervisionado e incremental.",
            }
            show_placeholder("IDS > Processing", descriptions[choice])


def ids_training_menu() -> None:
    while True:
        print_header("IDS > Training")
        section("🤖  Treinamento de Modelos")
        print("  [1] Treinar modelo supervisionado")
        print("  [2] Treinar modelo não supervisionado (anomaly detection)")
        print("  [3] Treinar modelo incremental / online")
        print("  [4] Ajustar hiperparâmetros (grid/random search)")
        print("  [5] Avaliar modelo (accuracy, recall, F1, ROC)")
        print("  [6] Salvar modelo treinado")
        print("  [0] Voltar")
        print()
        choice = menu_choice(["0", "1", "2", "3", "4", "5", "6"])

        if choice == "0":
            return
        elif choice == "3":
            run_ids_analysis(retrain_after=True)
        elif choice == "5":
            loaded = require_ids_stack()
            if loaded and callable(loaded.get("submenu_avaliacao")):
                try:
                    loaded["submenu_avaliacao"]()
                except Exception as exc:
                    print("  " + badge_err(f"Falha ao abrir avaliação do modelo: {exc}"))
                    traceback.print_exc()
                pause("Pressione ENTER para retornar ao treinamento...")
            else:
                show_placeholder("IDS > Training", "Submenu de avaliação ainda não disponível no ambiente atual.")
        elif choice == "6":
            print_header("IDS > Training")
            section("Salvar modelo treinado")
            print("  " + badge_info("No fluxo atual, o salvamento do modelo é gerenciado pelo pipeline de treinamento / re-treinamento."))
            print(f"  Diretório de modelos: {get_model_dir()}")
            pause()
        else:
            descriptions = {
                "1": "Treinamento supervisionado inicial pode ser integrado ao script principal de treinamento do projeto.",
                "2": "Trilha de anomaly detection prevista para Autoencoder, Isolation Forest, One-Class SVM ou similar.",
                "4": "Ajuste de hiperparâmetros pode reutilizar Grid Search, Random Search ou Bayesiana dos estudos experimentais.",
            }
            show_placeholder("IDS > Training", descriptions[choice])


def ids_detection_menu() -> None:
    while True:
        print_header("IDS > Detection")
        section("⚡  Detecção em Tempo Real")
        print("  [1] Iniciar motor de detecção")
        print("  [2] Parar motor de detecção")
        print("  [3] Modo simulação (replay de PCAP)")
        print("  [4] Ajustar sensibilidade do modelo")
        print("  [5] Monitorar alertas em tempo real")
        print("  [0] Voltar")
        print()
        choice = menu_choice(["0", "1", "2", "3", "4", "5"])

        if choice == "0":
            return
        elif choice == "1":
            run_ids_analysis(retrain_after=False)
        elif choice == "5":
            loaded = require_ids_stack()
            if loaded and callable(loaded.get("menu_status")):
                print_header("IDS > Detection")
                section("Monitor de alertas / status")
                try:
                    loaded["menu_status"]()
                except Exception as exc:
                    print("  " + badge_err(f"Falha ao abrir status do sistema: {exc}"))
                    traceback.print_exc()
                pause()
            else:
                show_placeholder("IDS > Detection", "Monitoramento contínuo previsto para dashboards e stream de alertas em tempo real.")
        else:
            descriptions = {
                "2": "Parada do motor será ativada quando o processo de detecção contínua estiver desacoplado em background/service.",
                "3": "Replay de PCAP previsto para simulação controlada de incidentes e testes de tuning.",
                "4": "Ajuste de sensibilidade pode ser ligado a thresholds de confiança e severidade do modelo ativo.",
            }
            show_placeholder("IDS > Detection", descriptions[choice])


def ids_reports_menu() -> None:
    while True:
        print_header("IDS > Reports")
        section("📊  Análise e Relatórios")
        print("  [1] Relatório de alertas")
        print("  [2] Relatório de tráfego")
        print("  [3] Relatório de anomalias detectadas")
        print("  [4] Relatório de performance do modelo")
        print("  [5] Exportar relatórios (JSON/CSV)")
        print("  [0] Voltar")
        print()
        choice = menu_choice(["0", "1", "2", "3", "4", "5"])

        if choice == "0":
            return
        loaded = require_ids_stack()
        if loaded and callable(loaded.get("submenu_relatorios")) and choice in {"1", "2", "3", "4"}:
            try:
                loaded["submenu_relatorios"]()
            except Exception as exc:
                print("  " + badge_err(f"Falha ao abrir o submenu de relatórios: {exc}"))
                traceback.print_exc()
            pause("Pressione ENTER para retornar aos relatórios do IDS...")
        elif choice == "5":
            export_dir = ROOT_DIR / "Exports" / time.strftime("reports_%Y%m%d_%H%M%S")
            export_dir.mkdir(parents=True, exist_ok=True)
            source_dirs = [ROOT_DIR / "IDS" / "reports", get_reports_dir()]
            copied = 0
            for source in source_dirs:
                if not source.exists():
                    continue
                for f in source.rglob("*"):
                    if f.is_file():
                        dest = export_dir / f.name
                        shutil.copy2(f, dest)
                        copied += 1
            print_header("IDS > Reports")
            section("Exportação de relatórios")
            print("  " + badge_ok(f"{copied} arquivo(s) exportados para {export_dir}"))
            pause()
        else:
            show_placeholder("IDS > Reports", "O mecanismo detalhado de relatórios ainda será consolidado com o backend do IDS.")


def ids_model_management_menu() -> None:
    while True:
        print_header("IDS > Model Management")
        section("📦  Gerenciamento de Modelos")
        print("  [1] Listar modelos disponíveis")
        print("  [2] Carregar modelo ativo")
        print("  [3] Remover modelo")
        print("  [4] Comparar modelos")
        print("  [5] Ver metadados do modelo (dataset, métricas, data)")
        print("  [0] Voltar")
        print()
        choice = menu_choice(["0", "1", "2", "3", "4", "5"])

        if choice == "0":
            return
        elif choice == "1":
            list_available_models()
        elif choice == "5":
            print_header("IDS > Model Management")
            section("Metadados do modelo")
            model_dir = get_model_dir()
            info_files = sorted(model_dir.rglob("*.json")) if model_dir.exists() else []
            if not info_files:
                print("  " + badge_warn("Nenhum arquivo de metadados encontrado."))
            else:
                for f in info_files[:10]:
                    print(f"  • {f.relative_to(model_dir)}  ({human_size(f.stat().st_size)})")
            pause()
        else:
            descriptions = {
                "2": "Ativação de modelo prevista para seleção de artefato principal e atualização de ponteiros/symlinks no diretório Model/.",
                "3": "Remoção de modelo deve ser protegida por confirmação e retenção mínima de versões estáveis.",
                "4": "Comparação de modelos pode combinar benchmark offline, métricas históricas e custo computacional.",
            }
            show_placeholder("IDS > Model Management", descriptions[choice])


def ids_system_settings_menu() -> None:
    while True:
        print_header("IDS > Settings")
        section("⚙️  Configurações do Sistema")
        print("  [1] Configurar interface de captura")
        print("  [2] Ajustar diretórios (pcap, modelos, logs)")
        print("  [3] Configurar thresholds de alerta")
        print("  [4] Configurar retenção de dados")
        print("  [5] Atualizar dependências")
        print("  [0] Voltar")
        print()
        choice = menu_choice(["0", "1", "2", "3", "4", "5"])

        if choice == "0":
            return
        elif choice == "2":
            loaded = require_ids_stack()
            if loaded and callable(loaded.get("submenu_configuracoes")):
                try:
                    loaded["submenu_configuracoes"]()
                except Exception as exc:
                    print("  " + badge_err(f"Falha ao abrir o submenu de configurações: {exc}"))
                    traceback.print_exc()
                pause("Pressione ENTER para retornar às configurações do sistema...")
            else:
                show_placeholder("IDS > Settings", "Submenu de configurações ainda não está disponível no stack atual.")
        elif choice == "5":
            print_header("IDS > Settings")
            section("Atualização de dependências")
            print("  Use o instalador do projeto para atualizar o ambiente:")
            print(f"  • {ROOT_DIR / 'install.sh'}")
            print("  Ou execute manualmente o gerenciador de pacotes do seu ambiente virtual.")
            pause()
        else:
            descriptions = {
                "1": "A configuração da interface de captura pode ser persistida em ids_config.py ou em arquivo YAML/JSON de runtime.",
                "3": "Thresholds de alerta devem refletir confiança mínima, criticidade e políticas de resposta.",
                "4": "Retenção de dados prevista para capturas, parquet, relatórios e logs históricos.",
            }
            show_placeholder("IDS > Settings", descriptions[choice])


def ids_diagnostics_menu() -> None:
    while True:
        print_header("IDS > Diagnostics")
        section("🛠️  Ferramentas de Diagnóstico")
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
            diagnostics_network()
        elif choice == "4":
            diagnostics_datasets()
        elif choice == "5":
            diagnostics_model_benchmark()


def ids_logs_menu() -> None:
    while True:
        print_header("IDS > Logs")
        section("📜  Logs e Auditoria")
        print("  [1] Ver logs do sistema")
        print("  [2] Ver logs de detecção")
        print("  [3] Ver logs de treinamento")
        print("  [4] Exportar logs")
        print("  [5] Limpar logs antigos")
        print("  [0] Voltar")
        print()
        choice = menu_choice(["0", "1", "2", "3", "4", "5"])

        if choice == "0":
            return
        elif choice == "1":
            show_logs("system")
        elif choice == "2":
            show_logs("detection")
        elif choice == "3":
            show_logs("training")
        elif choice == "4":
            export_logs()
        elif choice == "5":
            clean_old_logs()


def ids_menu() -> None:
    while True:
        print_header("IDS")
        section("🎯 MENU PRINCIPAL — IDS com IA/ML")
        print("  [1] Captura de Pacotes")
        print("  [2] Processamento e Feature Engineering")
        print("  [3] Treinamento de Modelos")
        print("  [4] Detecção em Tempo Real")
        print("  [5] Análise e Relatórios")
        print("  [6] Gerenciamento de Modelos")
        print("  [7] Configurações do Sistema")
        print("  [8] Ferramentas de Diagnóstico")
        print("  [9] Logs e Auditoria")
        print("  [0] Voltar")
        print()
        choice = menu_choice(["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"])

        if choice == "0":
            return
        elif choice == "1":
            ids_capture_menu()
        elif choice == "2":
            ids_processing_menu()
        elif choice == "3":
            ids_training_menu()
        elif choice == "4":
            ids_detection_menu()
        elif choice == "5":
            ids_reports_menu()
        elif choice == "6":
            ids_model_management_menu()
        elif choice == "7":
            ids_system_settings_menu()
        elif choice == "8":
            ids_diagnostics_menu()
        elif choice == "9":
            ids_logs_menu()


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
