#!/usr/bin/env python3
"""
reconcile_registry.py
=====================
Reconcilia o triplete M0/Mp/Mc quando o registro automático falhou durante
o treinamento (por bug, interrupção, ou registry corrompido).

O QUE FAZ:
  1. Verifica se o M0 (baseline RandomForest) está registrado. Se NÃO, orienta
     a executar `python3 baseline_rf.py`.
  2. Encontra a pasta `Model/run_<timestamp>/` mais recente (último treino).
  3. Recupera os arrays `y_test*.npy` e `y_pred*.npy` salvos pelo treino.
  4. Recalcula as métricas oficiais (recall_macro, mcc, f1_macro, fpr_macro,
     accuracy, balanced_acc).
  5. Registra a versão Mc com `register_framework_version`.

NÃO retreina. Não toca em pesos. Apenas escreve metadados no registry.

USO:
    cd /opt/SecurityIA
    source .venv/bin/activate
    python3 reconcile_registry.py
"""
from __future__ import annotations
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score,
)

from config import Config
from IDS.modules.model_registry import (
    register_framework_version, _read_index,
)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Métricas oficiais usadas pelo registry."""
    cm = confusion_matrix(y_true, y_pred)
    fpr_pc = []
    for c in range(cm.shape[0]):
        fp = cm[:, c].sum() - cm[c, c]
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        denom = fp + tn
        fpr_pc.append(fp / denom if denom else 0.0)
    return {
        "accuracy":      float(accuracy_score(y_true, y_pred)),
        "balanced_acc":  float(balanced_accuracy_score(y_true, y_pred)),
        "recall_macro":  float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro":      float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted":   float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc":           float(matthews_corrcoef(y_true, y_pred)),
        "fpr_macro":     float(np.mean(fpr_pc)),
        "n_test":        int(len(y_true)),
    }


def _find_latest_run() -> Path | None:
    """Localiza a pasta Model/run_<timestamp> mais recente que contenha
    y_test*.npy e y_pred*.npy."""
    candidates = sorted(
        Config.MODEL_DIR.glob("run_*"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        y_tests = list(run_dir.glob("y_test*.npy"))
        y_preds = list(run_dir.glob("y_pred*.npy"))
        if y_tests and y_preds:
            return run_dir
    return None


def _load_arrays(run_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Carrega o último y_test e y_pred do run_dir (por mtime)."""
    y_test_file = max(run_dir.glob("y_test*.npy"), key=lambda p: p.stat().st_mtime)
    y_pred_file = max(run_dir.glob("y_pred*.npy"), key=lambda p: p.stat().st_mtime)
    return np.load(y_test_file), np.load(y_pred_file), y_test_file, y_pred_file


def main() -> int:
    print("=" * 70)
    print("Reconciliação do triplete M0/Mp/Mc — SecurityIA")
    print("=" * 70)

    # 1) Conferir baseline M0
    idx = _read_index()
    baseline = idx.get("baseline")
    if baseline:
        print(f"  M0 (baseline): REGISTRADO em {baseline['registered_at']}")
        print(f"     accuracy={baseline.get('accuracy', 0):.4f} "
              f"mcc={baseline.get('mcc', 0):.4f}")
    else:
        print("  M0 (baseline): NÃO REGISTRADO")
        print()
        print("  AÇÃO RECOMENDADA: gerar o baseline antes do registro do Mc.")
        print("    cd /opt/SecurityIA")
        print("    source .venv/bin/activate")
        print("    python3 baseline_rf.py")
        print()
        print("  O baseline_rf.py treina um RandomForest sobre o mesmo split")
        print("  estratificado e registra M0 automaticamente. Sem M0, o")
        print("  relatório comparativo M0/Mp/Mc fica incompleto.")
        print()
        ans = input("  Deseja continuar reconciliando apenas o Mc? [s/N]: ").strip().lower()
        if ans not in {"s", "sim", "y", "yes"}:
            print("  Abortado pelo usuário.")
            return 1

    # 2) Encontrar último run com arrays salvos
    print()
    print("Localizando último treino com arrays de teste salvos…")
    run_dir = _find_latest_run()
    if run_dir is None:
        print()
        print("  ERRO: nenhuma pasta Model/run_*/ contém y_test*.npy e y_pred*.npy.")
        print("  Não é possível reconciliar sem retreinar.")
        return 2

    print(f"  Encontrado: {run_dir}")

    # 3) Carregar arrays
    y_true, y_pred, yt_file, yp_file = _load_arrays(run_dir)
    print(f"  y_test : {yt_file.name}  ({len(y_true):,} amostras)")
    print(f"  y_pred : {yp_file.name}  ({len(y_pred):,} amostras)")

    if len(y_true) != len(y_pred):
        print(f"  ERRO: tamanhos divergem ({len(y_true)} vs {len(y_pred)}).")
        return 3

    # 4) Recalcular métricas
    print()
    print("Recalculando métricas oficiais…")
    metrics = _metrics(y_true, y_pred)
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:18s} = {v:.6f}")
        else:
            print(f"  {k:18s} = {v}")

    # 5) Registrar Mc
    print()
    print("Registrando como nova versão Mc…")
    model_path = Config.MODEL_DIR / Config.MODEL_FILENAME
    if not model_path.exists():
        print(f"  AVISO: modelo {model_path.name} não encontrado. Registro prosseguirá")
        print(f"          sem cópia do .keras.")

    entry = register_framework_version(
        model_path=model_path,
        y_true=y_true,
        y_pred=y_pred,
        metrics=metrics,
        source="reconcile",
        extra={
            "reconciled_at": datetime.now().isoformat(),
            "source_run_dir": str(run_dir),
            "loss": Config.MODEL_CONFIG.get("loss_function", "?"),
            "balancing_strategy": Config.BALANCING_CONFIG.get("strategy", "?"),
            "k_features": Config.FEATURE_SELECTION_CONFIG.get("k_best", "?"),
        },
    )

    print()
    print("=" * 70)
    print(f"Mc registrado: {entry['id']}")
    print(f"  recall_macro = {metrics['recall_macro']:.4f}")
    print(f"  mcc          = {metrics['mcc']:.4f}")
    print(f"  f1_macro     = {metrics['f1_macro']:.4f}")
    print(f"  fpr_macro    = {metrics['fpr_macro']:.4f}")
    print("=" * 70)

    if not idx.get("baseline"):
        print()
        print("  Lembrete: M0 ainda não está registrado. Para completar o triplete,")
        print("  execute `python3 baseline_rf.py` após esta reconciliação.")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n  Cancelado pelo usuário.")
        sys.exit(130)
    except Exception as e:
        import traceback
        print(f"\n  ERRO INESPERADO: {type(e).__name__}: {e}")
        print()
        traceback.print_exc()
        sys.exit(99)
