"""
Tests/_test_logging.py — Logger, callbacks e helpers compartilhados pelas análises.

À PROVA DE FALHAS:
  - Escreve em arquivo INDEPENDENTE de qualquer redirecionamento do app_menu
  - StreamHandler protegido contra OSError [Errno 5] (stdout fechado)
  - EpochLogger substitui verbose=1/2 do Keras
  - safe_run() executa sub-experimentos isoladamente, capturando exceções
"""
from __future__ import annotations
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

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


def safe_run(
    logger: logging.Logger,
    label: str,
    func: Callable[..., Any],
    *args,
    default: Any = None,
    **kwargs,
) -> tuple[bool, Any]:
    """
    Executa func capturando QUALQUER exceção (exceto KeyboardInterrupt).
    Retorna (sucesso, resultado). Permite que cada sub-experimento falhe
    sem matar a análise inteira.
    """
    t0 = time.time()
    logger.info(f">>> {label}")
    try:
        result = func(*args, **kwargs)
        dt = time.time() - t0
        logger.info(f"<<< {label} OK ({dt:.1f}s)")
        return True, result
    except KeyboardInterrupt:
        raise
    except Exception as e:
        dt = time.time() - t0
        logger.error(f"<<< {label} FALHOU após {dt:.1f}s")
        log_exception(logger, label, e)
        return False, default


class EpochLogger(tf.keras.callbacks.Callback):
    """Loga uma linha por época no logger; substitui verbose=1/2 do Keras."""
    def __init__(self, logger: logging.Logger, prefix: str = ""):
        super().__init__()
        self.logger = logger
        self.prefix = prefix
        self._epoch_t0 = None

    def on_epoch_begin(self, epoch, logs=None):
        self._epoch_t0 = time.time()

    def on_epoch_end(self, epoch, logs=None):
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


def stratified_split_3way(X, y, val_frac=0.15, test_frac=0.15, seed=42, logger=None):
    """Split 3-way estratificado com fallback para split simples se necessário."""
    import numpy as np
    from sklearn.model_selection import train_test_split

    try:
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=val_frac + test_frac,
            stratify=y, random_state=seed,
        )
        rel = test_frac / (val_frac + test_frac)
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=rel,
            stratify=y_tmp, random_state=seed,
        )
    except ValueError as e:
        if logger:
            logger.warning(f"Split estratificado falhou ({e}); fallback simples.")
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            X, y, test_size=val_frac + test_frac, random_state=seed,
        )
        rel = test_frac / (val_frac + test_frac)
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=rel, random_state=seed,
        )

    if logger:
        logger.info(f"Split: treino={len(X_tr):,} val={len(X_val):,} teste={len(X_te):,}")
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def fit_scaler_no_leakage(X_tr, X_val, X_te):
    """Ajusta StandardScaler APENAS no treino e aplica nos demais (sem leakage)."""
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    sc = StandardScaler()
    X_tr_s = np.nan_to_num(sc.fit_transform(X_tr), nan=0.0, posinf=0.0, neginf=0.0)
    X_val_s = np.nan_to_num(sc.transform(X_val), nan=0.0, posinf=0.0, neginf=0.0)
    X_te_s = np.nan_to_num(sc.transform(X_te), nan=0.0, posinf=0.0, neginf=0.0)
    return X_tr_s.astype("float32"), X_val_s.astype("float32"), X_te_s.astype("float32"), sc


def metricas_completas(y_true, y_pred, n_classes=None):
    """Conjunto consolidado de métricas para IDS."""
    import numpy as np
    from sklearn.metrics import (
        accuracy_score, balanced_accuracy_score, confusion_matrix,
        f1_score, matthews_corrcoef, precision_score, recall_score,
    )
    if n_classes is None:
        n_classes = max(int(np.max(y_true)), int(np.max(y_pred))) + 1
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    fpr_pc = []
    for c in range(cm.shape[0]):
        fp = cm[:, c].sum() - cm[c, c]
        fn = cm[c, :].sum() - cm[c, c]
        tp = cm[c, c]
        tn = cm.sum() - tp - fp - fn
        fpr_pc.append(fp / (fp + tn) if (fp + tn) else 0.0)
    return {
        "acuracia": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "fpr_macro": float(np.mean(fpr_pc)),
    }
