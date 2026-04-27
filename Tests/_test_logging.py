"""
Tests/_test_logging.py — Logger compartilhado pelas análises

Escreve em Tests/Logs/<analise>_<YYYYMMDD-HHMMSS>.log INDEPENDENTE de
qualquer redirecionamento do app_menu. Sempre faz flush imediato.
"""

from __future__ import annotations
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path


def get_logger(analise_id: int, name: str) -> logging.Logger:
    """
    Devolve logger que escreve simultaneamente em arquivo e stdout.
    Arquivo: Tests/Logs/analise_<id>_<YYYYMMDD-HHMMSS>.log
    """
    log_dir = Path(__file__).resolve().parent / "Logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = log_dir / f"analise_{analise_id}_{ts}.log"

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)-7s] %(message)s",
        datefmt="%H:%M:%S",
    )

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"Log iniciado em: {log_file}")
    return logger


def log_exception(logger: logging.Logger, ctx: str, exc: BaseException) -> None:
    """Loga exceção com stack trace completo."""
    logger.error(f"FALHA em {ctx}: {type(exc).__name__}: {exc}")
    logger.error("Stack trace:\n" + traceback.format_exc())
