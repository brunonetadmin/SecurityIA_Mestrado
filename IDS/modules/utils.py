#!/usr/bin/env python3
"""
IDS/modules/utils.py — Utilitários Centrais do Sistema SecurityIA

Conteúdo:
  - setup_logger()   : configura os 3 loggers independentes (App, Collector, Learn)
  - CLI helpers      : cores ANSI, tabelas, barras de progresso, cabeçalho
  - @timed           : decorator de temporização
  - run_background() : execução de processos longos em segundo plano

Importado por: todos os scripts do IDS/.
"""

import functools
import logging
import signal
import subprocess
import sys
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Callable, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Cores ANSI (desativadas automaticamente fora de TTY)
# ──────────────────────────────────────────────────────────────────────────────
_TTY = sys.stdout.isatty()

class _C:
    """Constantes de cor ANSI."""
    RESET  = "\033[0m"    if _TTY else ""
    BOLD   = "\033[1m"    if _TTY else ""
    DIM    = "\033[2m"    if _TTY else ""
    RED    = "\033[91m"   if _TTY else ""
    GREEN  = "\033[92m"   if _TTY else ""
    YELLOW = "\033[93m"   if _TTY else ""
    BLUE   = "\033[94m"   if _TTY else ""
    CYAN   = "\033[96m"   if _TTY else ""
    WHITE  = "\033[97m"   if _TTY else ""
    ORANGE = "\033[38;5;208m" if _TTY else ""
    GRAY   = "\033[90m"   if _TTY else ""

def _c(text: str, *codes: str) -> str:
    return "".join(codes) + text + _C.RESET


# ──────────────────────────────────────────────────────────────────────────────
# Configuração de logging (3 arquivos independentes + stream)
# ──────────────────────────────────────────────────────────────────────────────
_loggers_initialized: set = set()

def setup_logger(
    name: str,
    log_file: Path,
    level: str = "INFO",
    fmt: str = "%(asctime)s  [%(levelname)-8s]  %(name)-18s  %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
    max_bytes: int = 50 * 1024 * 1024,
    backup_count: int = 5,
    to_stdout: bool = True,
) -> logging.Logger:
    """
    Cria (ou recupera) um logger com RotatingFileHandler + StreamHandler.
    Idempotente: chamadas repetidas retornam o mesmo logger sem duplicar handlers.
    """
    if name in _loggers_initialized:
        return logging.getLogger(name)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(fmt, datefmt=datefmt)

    # Arquivo rotativo
    fh = RotatingFileHandler(
        log_file, maxBytes=max_bytes, backupCount=backup_count,
        encoding="utf-8",
    )
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Console (apenas se TTY ou flag explícita)
    if to_stdout:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(formatter)
        logger.addHandler(sh)

    _loggers_initialized.add(name)
    return logger


def get_app_logger() -> logging.Logger:
    from config import Config
    return setup_logger(
        "app", Config.LOG_APP,
        **{k: Config.LOG_CONFIG[k] for k in
           ("level", "max_bytes", "backup_count")},
    )

def get_collector_logger() -> logging.Logger:
    from config import Config
    return setup_logger(
        "collector", Config.LOG_COLLECTOR,
        **{k: Config.LOG_CONFIG[k] for k in
           ("level", "max_bytes", "backup_count")},
    )

def get_learn_logger() -> logging.Logger:
    from config import Config
    return setup_logger(
        "learn", Config.LOG_LEARN,
        **{k: Config.LOG_CONFIG[k] for k in
           ("level", "max_bytes", "backup_count")},
    )


# ──────────────────────────────────────────────────────────────────────────────
# CLI — cabeçalho e separadores
# ──────────────────────────────────────────────────────────────────────────────
_WIDTH = 72

def _sep(char: str = "═", n: int = _WIDTH) -> str:
    return char * n

def _section(title: str) -> None:
    print(f"\n{_c(_sep('─'), _C.DIM)}")
    print(f"  {_c(title, _C.BOLD, _C.CYAN)}")
    print(_c(_sep('─'), _C.DIM))

def print_header(version: str = "3.0") -> None:
    """Imprime o cabeçalho visual da aplicação."""
    import os
    os.system("clear" if os.name != "nt" else "cls")
    print(_c(_sep(), _C.CYAN, _C.BOLD))
    print(_c(f"  SecurityIA — Sistema Inteligente de Detecção de Anomalias de Rede", _C.WHITE, _C.BOLD))
    print(_c(f"  Bi-LSTM + Atenção de Bahdanau | SMOTE-ENN | Teoria da Informação", _C.DIM))
    print(_c(_sep(), _C.CYAN, _C.BOLD))
    print(f"  {_c('Autor    :', _C.DIM)} Bruno Cavalcante Barbosa — bcb@ic.ufal.br")
    print(f"  {_c('Orient.  :', _C.DIM)} Prof. Dr. André Luiz Lins de Aquino — PPGI/IC/UFAL")
    print(f"  {_c('Versão   :', _C.DIM)} {version}  {_c(datetime.now().strftime('%Y-%m-%d %H:%M'), _C.GRAY)}")
    print(_c(_sep(), _C.CYAN, _C.BOLD))


def progress_bar(
    done: int, total: int, width: int = 30,
    prefix: str = "", suffix: str = "",
) -> str:
    """Barra de progresso inline."""
    if total <= 0:
        pct = 0.0
    else:
        pct = min(done / total, 1.0)
    filled = int(width * pct)
    bar = _c("█" * filled, _C.GREEN) + _c("░" * (width - filled), _C.GRAY)
    return f"{prefix}[{bar}] {pct*100:5.1f}% {suffix}"


def print_table(
    headers: List[str],
    rows: List[List[str]],
    title: str = "",
    col_sep: str = "  │  ",
) -> None:
    """Imprime tabela formatada com alinhamento automático."""
    all_rows = [headers] + rows
    widths = [max(len(str(r[i])) for r in all_rows) for i in range(len(headers))]

    if title:
        print(f"\n  {_c(title, _C.BOLD, _C.WHITE)}")
    print("  " + _c("─" * (sum(widths) + len(col_sep) * (len(headers) - 1) + 2), _C.DIM))

    # Cabeçalho
    header_str = col_sep.join(_c(str(h).ljust(widths[i]), _C.BOLD, _C.CYAN)
                               for i, h in enumerate(headers))
    print(f"  {header_str}")
    print("  " + _c("─" * (sum(widths) + len(col_sep) * (len(headers) - 1) + 2), _C.DIM))

    for row in rows:
        row_str = col_sep.join(str(row[i]).ljust(widths[i]) for i in range(len(headers)))
        print(f"  {row_str}")


def prompt(text: str, default: str = "") -> str:
    """Input formatado com valor padrão."""
    default_hint = f" [{_c(default, _C.DIM)}]" if default else ""
    try:
        val = input(f"  {_c('›', _C.CYAN, _C.BOLD)} {text}{default_hint}: ").strip()
        return val if val else default
    except (KeyboardInterrupt, EOFError):
        print()
        return default


def confirm(text: str, default: bool = True) -> bool:
    """Confirmação Sim/Não."""
    hint = "[S/n]" if default else "[s/N]"
    val = prompt(f"{text} {hint}").lower()
    if val in ("", "s", "sim", "y", "yes"):
        return True
    if val in ("n", "nao", "não", "no"):
        return False
    return default


def pause(msg: str = "Pressione ENTER para continuar...") -> None:
    try:
        input(f"\n  {_c(msg, _C.DIM)}")
    except (KeyboardInterrupt, EOFError):
        pass


# ──────────────────────────────────────────────────────────────────────────────
# Decorator de temporização
# ──────────────────────────────────────────────────────────────────────────────
def timed(label: str = ""):
    """Decorator: mede e loga o tempo de execução de uma função."""
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            result = fn(*args, **kwargs)
            elapsed = time.perf_counter() - t0
            name = label or fn.__name__
            logger = get_app_logger()
            logger.info(f"{name} concluído em {elapsed:.2f}s ({elapsed/60:.1f} min)")
            return result
        return wrapper
    return decorator


# ──────────────────────────────────────────────────────────────────────────────
# Execução em segundo plano
# ──────────────────────────────────────────────────────────────────────────────
def run_background(
    cmd: List[str],
    log_file: Optional[Path] = None,
    cwd: Optional[Path] = None,
) -> subprocess.Popen:
    """
    Executa um comando em segundo plano (não-bloqueante).
    Stdout/stderr são redirecionados para log_file se fornecido.
    Retorna o objeto Popen para controle posterior.
    """
    kwargs = dict(
        cwd=str(cwd) if cwd else None,
        start_new_session=True,  # desconecta do grupo de processo do terminal
    )
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log_file, "a", encoding="utf-8")
        kwargs.update(stdout=fh, stderr=fh)
    else:
        kwargs.update(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    proc = subprocess.Popen(cmd, **kwargs)
    return proc


def is_process_running(pid: int) -> bool:
    """Verifica se um PID está ativo."""
    try:
        import psutil
        return psutil.pid_exists(pid) and psutil.Process(pid).is_running()
    except Exception:
        try:
            import os
            os.kill(pid, 0)
            return True
        except (OSError, ProcessLookupError):
            return False


def format_bytes(n: int) -> str:
    """Formata bytes em unidade legível."""
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PiB"


def format_duration(seconds: float) -> str:
    """Formata duração em h:mm:ss."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"
