"""
Tests/_test_logging.py — Logger e callback compartilhados.

À PROVA DE FALHAS:
  - Escreve em arquivo INDEPENDENTE de qualquer redirecionamento do app_menu
  - StreamHandler protegido contra OSError [Errno 5] (stdout fechado)
  - EpochLogger substitui verbose=1/2 do Keras (que estoura no subprocess)
"""
from __future__ import annotations
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path

import tensorflow as tf


class _SafeStreamHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            super().emit(record)
        except (OSError, ValueError, BrokenPipeError):
            self.handleError(record)
    def handleError(self, record):
        pass


def get_logger(analise_id: int, name: str) -> logging.Logger:
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

    try:
        sh = _SafeStreamHandler(sys.stdout)
        sh.setLevel(logging.INFO)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    except Exception:
        pass

    logger.info(f"Log iniciado em: {log_file}")
    return logger


def log_exception(logger: logging.Logger, ctx: str, exc: BaseException) -> None:
    try:
        logger.error(f"FALHA em {ctx}: {type(exc).__name__}: {exc}")
        logger.error("Stack trace:\n" + traceback.format_exc())
    except Exception:
        pass


class EpochLogger(tf.keras.callbacks.Callback):
    """Loga uma linha por época no logger; substitui verbose=1/2 do Keras."""
    def __init__(self, logger: logging.Logger, prefix: str = ""):
        super().__init__()
        self.logger = logger
        self.prefix = prefix
        self._epoch_t0 = None

    def on_epoch_begin(self, epoch, logs=None):
        import time
        self._epoch_t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
        import time
        logs = logs or {}
        dt = time.time() - (self._epoch_t0 or time.time())
        loss = logs.get("loss", float("nan"))
        acc = logs.get("accuracy", float("nan"))
        vl = logs.get("val_loss", float("nan"))
        va = logs.get("val_accuracy", float("nan"))
        try:
            self.logger.info(
                f"  {self.prefix}ep{epoch+1:>2d}: loss={loss:.4f} "
                f"acc={acc:.4f} val_loss={vl:.4f} val_acc={va:.4f} ({dt:.1f}s)"
            )
        except Exception:
            pass


def silence_tensorflow():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    try:
        tf.get_logger().setLevel("ERROR")
        tf.autograph.set_verbosity(0)
    except Exception:
        pass
