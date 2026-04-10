#!/usr/bin/env python3
"""
IDS/ids_manager.py — Interface de Linha de Comando Principal (SecurityIA)

Menu interativo completo para gerenciamento do sistema IDS:
  1. Collector   — iniciar/parar daemon de captura
  2. Detector    — análise de tráfego (watch/batch/arquivo)
  3. Treinamento — treinar ou fazer fine-tuning do modelo
  4. Avaliação   — benchmark e histórico de métricas
  5. Relatórios  — listar e abrir relatórios gerados
  6. Status      — visão geral do sistema
  0. Sair

Uso:
    python3 IDS/ids_manager.py
    python3 IDS/ids_manager.py --batch        # análise imediata (não-interativo)
    python3 IDS/ids_manager.py --train        # treinar imediatamente
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config
from IDS.modules.utils import (
    get_app_logger,
    print_header, _section, pause, prompt, confirm,
    progress_bar, print_table, format_bytes, format_duration,
    _c, _C, _sep, _WIDTH,
    run_background, is_process_running,
)
from IDS.modules.incident_engine import (
    ModelArtifacts, ManagerState, scan_new_files,
)

log = get_app_logger()

# ─────────────────────────────────────────────────────────────────────────────
# Estado de processos em background
# ─────────────────────────────────────────────────────────────────────────────

_PID_FILE_COLLECTOR = Config.TEMP_DIR / ".collector.pid"
_PID_FILE_DETECTOR  = Config.TEMP_DIR / ".detector.pid"


def _write_pid(pid_file: Path, pid: int) -> None:
    pid_file.parent.mkdir(parents=True, exist_ok=True)
    pid_file.write_text(str(pid))


def _read_pid(pid_file: Path) -> Optional[int]:
    try:
        return int(pid_file.read_text().strip())
    except Exception:
        return None


def _proc_status(pid_file: Path, label: str) -> str:
    pid = _read_pid(pid_file)
    if pid and is_process_running(pid):
        return _c(f"ATIVO  (PID {pid})", _C.GREEN, _C.BOLD)
    return _c("PARADO", _C.RED)


# ─────────────────────────────────────────────────────────────────────────────
# Status geral do sistema
# ─────────────────────────────────────────────────────────────────────────────

def _system_status() -> None:
    _section("STATUS DO SISTEMA")
    state = ManagerState()
    st    = state.stats()

    model_ok = (Config.MODEL_DIR / Config.MODEL_FILENAME).exists()
    version  = "—"
    trained  = "—"
    if model_ok:
        mi = Config.MODEL_DIR / Config.MODEL_INFO_FILENAME
        if mi.exists():
            with open(mi, encoding="utf-8") as f:
                info = json.load(f)
            version = info.get("version", "?")
            trained = info.get("trained_at", "?")[:19]

    new_files = scan_new_files(state)

    print(f"  {'Collector':<22s}: {_proc_status(_PID_FILE_COLLECTOR, 'Collector')}")
    print(f"  {'Detector':<22s}: {_proc_status(_PID_FILE_DETECTOR, 'Detector')}")
    print(f"  {'Modelo':<22s}: {_c(f'v{version} ({trained})', _C.CYAN) if model_ok else _c('Não treinado', _C.RED)}")
    print(f"  {'Arquivos analisados':<22s}: {st['analyzed']:,}")
    print(f"  {'Arquivos pendentes':<22s}: {_c(str(len(new_files)), _C.YELLOW if new_files else _C.DIM)}")
    print(f"  {'Diretório collector':<22s}: {Config.COLLECTOR_DIR}")

    col_files = list(Config.COLLECTOR_DIR.glob("*.parquet")) if Config.COLLECTOR_DIR.exists() else []
    col_size  = sum(f.stat().st_size for f in col_files)
    print(f"  {'Dados capturados':<22s}: {len(col_files)} arquivo(s) | {format_bytes(col_size)}")

    rep_files = list(Config.REPORTS_DIR.glob("*.html")) if Config.REPORTS_DIR.exists() else []
    print(f"  {'Relatórios gerados':<22s}: {len(rep_files)}")

    staging   = Config.RETRAIN_CONFIG["staging_dir"]
    stg_files = list(staging.glob("*.parquet")) if staging.exists() else []
    stg_rows  = sum(1 for _ in stg_files)
    print(f"  {'Staging (re-treino)':<22s}: {stg_rows} arquivo(s)")
    print(_sep("─"))


# ─────────────────────────────────────────────────────────────────────────────
# Submenu — Collector
# ─────────────────────────────────────────────────────────────────────────────

def menu_collector() -> None:
    while True:
        print_header()
        _section("GERENCIAMENTO DO COLLECTOR")
        pid = _read_pid(_PID_FILE_COLLECTOR)
        running = pid and is_process_running(pid)

        print(f"  Status    : {_proc_status(_PID_FILE_COLLECTOR, 'Collector')}")
        print(f"  Interface : {Config.CAPTURE_INTERFACE}")
        print(f"  Saída     : {Config.COLLECTOR_DIR}")
        print(f"  Budget    : {Config.COLLECTOR_BUDGET_GB:.1f} GiB/dia")
        print()
        print(f"  {_c('[1]', _C.CYAN)} {'Parar Collector' if running else 'Iniciar Collector'}")
        print(f"  {_c('[2]', _C.CYAN)} Ver log do Collector")
        print(f"  {_c('[3]', _C.CYAN)} Listar arquivos capturados")
        print(f"  {_c('[0]', _C.DIM)} Voltar")
        print(_sep("─"))

        op = prompt("Opção")
        if op == "1":
            if running:
                if confirm("Parar o Collector?"):
                    import signal as _signal
                    os.kill(pid, _signal.SIGTERM)
                    _PID_FILE_COLLECTOR.unlink(missing_ok=True)
                    print(f"  {_c('Collector encerrado.', _C.YELLOW)}")
                    log.info(f"Collector (PID {pid}) encerrado pelo operador.")
                    time.sleep(1)
            else:
                print(f"  Iniciando Collector em background …")
                proc = run_background(
                    [sys.executable,
                     str(Path(__file__).parent / "ids_collector.py")],
                    log_file=Config.LOG_COLLECTOR,
                    cwd=Path(__file__).parents[1],
                )
                _write_pid(_PID_FILE_COLLECTOR, proc.pid)
                print(f"  {_c(f'Collector iniciado — PID {proc.pid}', _C.GREEN)}")
                print(f"  Log: {Config.LOG_COLLECTOR}")
                log.info(f"Collector iniciado (PID {proc.pid}) via menu.")
                pause()

        elif op == "2":
            if Config.LOG_COLLECTOR.exists():
                print()
                tail = subprocess.run(
                    ["tail", "-n", "50", str(Config.LOG_COLLECTOR)],
                    capture_output=True, text=True,
                )
                print(tail.stdout)
            else:
                print(f"  {_c('Log não encontrado.', _C.DIM)}")
            pause()

        elif op == "3":
            files = sorted(Config.COLLECTOR_DIR.glob("*.parquet")) \
                if Config.COLLECTOR_DIR.exists() else []
            if files:
                rows = [[f.name, format_bytes(f.stat().st_size),
                         datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")]
                        for f in files]
                print_table(["Arquivo", "Tamanho", "Modificado"], rows, title="Arquivos Capturados")
            else:
                print(f"  {_c('Nenhum arquivo encontrado.', _C.DIM)}")
            pause()

        elif op == "0":
            return


# ─────────────────────────────────────────────────────────────────────────────
# Submenu — Detector
# ─────────────────────────────────────────────────────────────────────────────

def menu_detector() -> None:
    while True:
        print_header()
        _section("GERENCIAMENTO DO DETECTOR")
        state    = ManagerState()
        new_files = scan_new_files(state)

        print(f"  Status detector : {_proc_status(_PID_FILE_DETECTOR, 'Detector')}")
        print(f"  Arquivos novos  : {_c(str(len(new_files)), _C.YELLOW if new_files else _C.DIM)}")
        print()
        print(f"  {_c('[1]', _C.CYAN)} Análise em lote (todos os arquivos pendentes)")
        print(f"  {_c('[2]', _C.CYAN)} Monitor contínuo (background)")
        print(f"  {_c('[3]', _C.CYAN)} Analisar arquivo específico")
        print(f"  {_c('[4]', _C.CYAN)} Parar monitor")
        print(f"  {_c('[5]', _C.CYAN)} Ver log de detecção")
        print(f"  {_c('[0]', _C.DIM)} Voltar")
        print(_sep("─"))

        op = prompt("Opção")

        if op == "1":
            from IDS.ids_detector import cmd_batch
            cmd_batch()
            pause()

        elif op == "2":
            interval = int(prompt("Intervalo de varredura (s)", "60"))
            proc = run_background(
                [sys.executable,
                 str(Path(__file__).parent / "ids_detector.py"),
                 "watch", "--interval", str(interval)],
                log_file=Config.LOG_APP,
                cwd=Path(__file__).parents[1],
            )
            _write_pid(_PID_FILE_DETECTOR, proc.pid)
            print(f"  {_c(f'Monitor iniciado — PID {proc.pid}', _C.GREEN)}")
            log.info(f"Detector watch iniciado (PID {proc.pid}).")
            pause()

        elif op == "3":
            path_str = prompt("Caminho do arquivo .parquet")
            path = Path(path_str)
            from IDS.ids_detector import cmd_file
            cmd_file(path)
            pause()

        elif op == "4":
            pid = _read_pid(_PID_FILE_DETECTOR)
            if pid and is_process_running(pid):
                if confirm("Parar o monitor?"):
                    import signal as _signal
                    os.kill(pid, _signal.SIGTERM)
                    _PID_FILE_DETECTOR.unlink(missing_ok=True)
                    print(f"  {_c('Monitor encerrado.', _C.YELLOW)}")
                    log.info(f"Detector (PID {pid}) encerrado pelo operador.")
            else:
                print(f"  {_c('Monitor não está em execução.', _C.DIM)}")
            pause()

        elif op == "5":
            if Config.LOG_APP.exists():
                tail = subprocess.run(
                    ["tail", "-n", "80", str(Config.LOG_APP)],
                    capture_output=True, text=True,
                )
                print("\n" + tail.stdout)
            pause()

        elif op == "0":
            return


# ─────────────────────────────────────────────────────────────────────────────
# Submenu — Treinamento
# ─────────────────────────────────────────────────────────────────────────────

def menu_training() -> None:
    while True:
        print_header()
        _section("GERENCIAMENTO DE TREINAMENTO")

        from IDS.ids_learn import cmd_status
        cmd_status()
        print()
        staging = Config.RETRAIN_CONFIG["staging_dir"]
        stg_cnt = len(list(staging.glob("*.parquet"))) if staging.exists() else 0

        print(f"  {_c('[1]', _C.CYAN)} Treinamento completo (do zero)")
        print(f"  {_c('[2]', _C.CYAN)} Treinamento forçado (limpa cache)")
        print(f"  {_c('[3]', _C.CYAN)} Fine-tuning incremental"
              + (f"  {_c(f'({stg_cnt} arquivo(s) em staging)', _C.YELLOW)}" if stg_cnt else ""))
        print(f"  {_c('[4]', _C.CYAN)} Ver log de treinamento")
        print(f"  {_c('[0]', _C.DIM)} Voltar")
        print(_sep("─"))

        op = prompt("Opção")

        if op in ("1", "2"):
            force = op == "2"
            msg = "Treinamento forçado" if force else "Treinamento completo"
            if confirm(f"Iniciar {msg}? (pode demorar 30–120 min)"):
                print(f"\n  Iniciando em background …")
                extra = ["--force"] if force else []
                proc = run_background(
                    [sys.executable,
                     str(Path(__file__).parent / "ids_learn.py"),
                     "train"] + extra,
                    log_file=Config.LOG_LEARN,
                    cwd=Path(__file__).parents[1],
                )
                print(f"  {_c(f'Treinamento iniciado — PID {proc.pid}', _C.GREEN)}")
                print(f"  Acompanhe em: {Config.LOG_LEARN}")
                log.info(f"Treinamento iniciado (PID {proc.pid}, force={force}).")
            pause()

        elif op == "3":
            if stg_cnt == 0:
                print(f"  {_c('Nenhum dado de staging disponível.', _C.DIM)}")
            elif confirm(f"Fine-tuning com {stg_cnt} arquivo(s)?"):
                proc = run_background(
                    [sys.executable,
                     str(Path(__file__).parent / "ids_learn.py"),
                     "finetune"],
                    log_file=Config.LOG_LEARN,
                    cwd=Path(__file__).parents[1],
                )
                print(f"  {_c(f'Fine-tuning iniciado — PID {proc.pid}', _C.GREEN)}")
                log.info(f"Fine-tuning iniciado (PID {proc.pid}).")
            pause()

        elif op == "4":
            if Config.LOG_LEARN.exists():
                tail = subprocess.run(
                    ["tail", "-n", "100", str(Config.LOG_LEARN)],
                    capture_output=True, text=True,
                )
                print("\n" + tail.stdout)
            pause()

        elif op == "0":
            return


# ─────────────────────────────────────────────────────────────────────────────
# Submenu — Relatórios
# ─────────────────────────────────────────────────────────────────────────────

def menu_reports() -> None:
    print_header()
    _section("RELATÓRIOS GERADOS")

    html_files = sorted(Config.REPORTS_DIR.glob("*.html"), reverse=True) \
        if Config.REPORTS_DIR.exists() else []

    if not html_files:
        print(f"  {_c('Nenhum relatório encontrado em', _C.DIM)} {Config.REPORTS_DIR}")
        pause()
        return

    rows = [
        [f.stem, format_bytes(f.stat().st_size),
         datetime.fromtimestamp(f.stat().st_mtime).strftime("%Y-%m-%d %H:%M")]
        for f in html_files[:20]
    ]
    print_table(["Relatório", "Tamanho", "Gerado"], rows)
    print(f"\n  {_c('Diretório:', _C.DIM)} {Config.REPORTS_DIR}")
    op = prompt("Abrir relatório no browser? [s/N]", "n").lower()
    if op in ("s", "sim", "y", "yes"):
        idx = prompt(f"Número (1–{min(20, len(html_files))})", "1")
        try:
            chosen = html_files[int(idx) - 1]
            subprocess.run(
                ["xdg-open" if os.name != "nt" else "start",
                 str(chosen)],
                check=False, stderr=subprocess.DEVNULL,
            )
            print(f"  Abrindo: {chosen.name}")
        except Exception as e:
            print(f"  Erro: {e}")
    pause()


# ─────────────────────────────────────────────────────────────────────────────
# Menu principal
# ─────────────────────────────────────────────────────────────────────────────

_MENU_ITEMS = [
    ("1", "Collector",    "Iniciar/parar captura de pacotes"),
    ("2", "Detector",     "Analisar tráfego capturado"),
    ("3", "Treinamento",  "Treinar ou fazer fine-tuning do modelo"),
    ("4", "Relatórios",   "Listar e abrir relatórios HTML"),
    ("5", "Status",       "Visão geral do sistema"),
    ("0", "Sair",         ""),
]


def _print_main_menu(new_files: int) -> None:
    state_col = _proc_status(_PID_FILE_COLLECTOR, "Collector")
    state_det = _proc_status(_PID_FILE_DETECTOR,  "Detector")

    print(f"\n  {_c('Collector', _C.DIM)}: {state_col}   "
          f"{_c('Detector', _C.DIM)}: {state_det}\n")

    if new_files > 0:
        print(f"  {_c(f'⚠  {new_files} arquivo(s) pendente(s) para análise', _C.YELLOW, _C.BOLD)}\n")

    for key, label, desc in _MENU_ITEMS:
        label_s = _c(f"[{key}]", _C.CYAN, _C.BOLD)
        desc_s  = _c(f"  {desc}", _C.DIM) if desc else ""
        print(f"  {label_s}  {label}{desc_s}")

    print(_sep("─"))


def main_menu() -> None:
    Config.ensure_dirs()
    log.info("SecurityIA Manager iniciado.")

    while True:
        print_header()
        state     = ManagerState()
        new_files = len(scan_new_files(state))
        _print_main_menu(new_files)

        op = prompt("Opção")

        if op == "1":
            menu_collector()
        elif op == "2":
            menu_detector()
        elif op == "3":
            menu_training()
        elif op == "4":
            menu_reports()
        elif op == "5":
            print_header()
            _system_status()
            pause()
        elif op == "0":
            print(f"\n  {_c('Encerrando SecurityIA. Bons experimentos!', _C.CYAN)}")
            print(f"  Relatórios disponíveis em: {Config.REPORTS_DIR}\n")
            log.info("SecurityIA Manager encerrado.")
            sys.exit(0)
        else:
            print(f"  {_c('Opção inválida.', _C.DIM)}")
            time.sleep(0.8)


# ─────────────────────────────────────────────────────────────────────────────
# CLI direto (não-interativo)
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="SecurityIA — Interface Principal do Sistema IDS",
    )
    p.add_argument("--batch",  action="store_true", help="Análise em lote imediata")
    p.add_argument("--train",  action="store_true", help="Treinar modelo imediatamente")
    p.add_argument("--status", action="store_true", help="Exibir status e sair")
    args = p.parse_args()

    Config.ensure_dirs()

    if args.batch:
        from IDS.ids_detector import cmd_batch
        cmd_batch()
    elif args.train:
        from IDS.ids_learn import cmd_train
        cmd_train()
    elif args.status:
        print_header()
        _system_status()
    else:
        main_menu()


if __name__ == "__main__":
    main()
