#!/usr/bin/env python3
"""
baseline_rf.py — Treina e registra o classificador Random Forest baseline (M0)

Reproduz EXATAMENTE o split estratificado do cmd_train(), garantindo que
M0 use o mesmo conjunto de treino e o mesmo conjunto de teste que a Bi-LSTM.

Saídas (todas com nome versionado _YYYYMMDD-N):
    Model/baseline_rf.pkl
    Model/baseline_rf_metrics.json
    Reports/baseline_rf_report_<v>.txt
    Reports/baseline_rf_confusion_matrix_<v>.png

Pré-requisitos:
    1. config.py atualizado em uso
    2. Existência do cache Temp/03_X_scaled_unbalanced.pkl (gerado por
       IDS/ids_learn.py train --force)

Uso:
    python3 baseline_rf.py
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import Config

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score,
)
from sklearn.model_selection import train_test_split

from IDS.modules.model_registry import register_baseline
from IDS.modules.full_report import generate as generate_full_report
from IDS.modules.versioning import versioned_path


def split_identical_to_m1():
    """Reproduz o split estratificado de cmd_train() corrigido."""
    cx = Config.TEMP_DIR / "03_X_scaled_unbalanced.pkl"
    cy = Config.TEMP_DIR / "03_y_unbalanced.pkl"

    if not cx.exists() or not cy.exists():
        raise FileNotFoundError(
            "Arquivos de cache não encontrados.\n"
            "Execute primeiro: python3 IDS/ids_learn.py train --force"
        )

    X = joblib.load(cx)
    y = joblib.load(cy)
    print(f"  Cache carregado: X={X.shape} | y={y.shape}")

    cfg = Config.TRAINING_CONFIG
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y,
        test_size=cfg["validation_split"] + cfg["test_split"],
        random_state=cfg["random_state"], stratify=y,
    )
    val_frac = cfg["validation_split"] / (cfg["validation_split"] + cfg["test_split"])
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp,
        test_size=1 - val_frac,
        random_state=cfg["random_state"], stratify=y_tmp,
    )
    print(f"  Treino={X_tr.shape[0]:,} | Val={X_val.shape[0]:,} | Teste={X_te.shape[0]:,}")
    return X_tr, X_val, X_te, y_tr, y_val, y_te


def compute_full_metrics(y_true, y_pred, label_map):
    """Métricas completas: acurácia, F1, MCC, FPR por classe, matriz."""
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]
    fpr_per_class = {}
    for c in range(n_classes):
        fp = cm[:, c].sum() - cm[c, c]
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        denom = fp + tn
        fpr_per_class[label_map.get(c, str(c))] = float(fp / denom) if denom else 0.0

    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = {label_map.get(i, str(i)): float(v) for i, v in enumerate(f1_per)}

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1_per_class": f1_per_class,
        "fpr_per_class": fpr_per_class,
        "fpr_macro": float(np.mean(list(fpr_per_class.values()))),
        "confusion_matrix": cm.tolist(),
    }


def plot_confusion_matrix(y_true, y_pred, label_map, out_path):
    cm = confusion_matrix(y_true, y_pred)
    cm_n = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    labels = [label_map.get(i, str(i)) for i in sorted(label_map)]

    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set(title="Baseline RF — Matriz de Confusão Normalizada",
           xlabel="Classe Predita", ylabel="Classe Real")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=Config.VIZ_CONFIG["dpi"])
    plt.close(fig)


def main():
    print("=" * 62)
    print("  BASELINE RF — Treinamento e Registro como M0")
    print("=" * 62)

    Config.ensure_dirs()

    X_tr, X_val, X_te, y_tr, y_val, y_te = split_identical_to_m1()

    le = joblib.load(Config.MODEL_DIR / Config.LABEL_ENCODER_FILENAME)
    label_map = {i: lbl for i, lbl in enumerate(le.classes_)}
    print(f"  Classes: {len(label_map)}")

    cfg = Config.BASELINE_CONFIG
    print(f"\n  Treinando RF: n_est={cfg['n_estimators']} max_depth={cfg['max_depth']} "
          f"class_weight={cfg['class_weight']}")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        min_samples_split=cfg["min_samples_split"],
        min_samples_leaf=cfg["min_samples_leaf"],
        class_weight=cfg["class_weight"],
        n_jobs=cfg["n_jobs"],
        random_state=cfg["random_state"],
        verbose=0,
    )
    rf.fit(X_tr, y_tr)
    elapsed = time.time() - t0
    print(f"  Treino concluído em {elapsed/60:.1f} min")

    print("\n  Avaliando no conjunto de teste original…")
    y_pred = rf.predict(X_te)
    metrics = compute_full_metrics(y_te, y_pred, label_map)

    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    Config.IDS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = Config.MODEL_DIR / Config.BASELINE_CONFIG["model_filename"]
    joblib.dump(rf, model_path)
    print(f"\n  Modelo salvo: {model_path}")

    metrics_with_meta = {
        **metrics,
        "model": "RandomForest",
        "n_estimators": cfg["n_estimators"],
        "trained_at": datetime.now().isoformat(),
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
        "training_seconds": float(elapsed),
    }
    metrics_path = versioned_path(Config.MODEL_DIR, "baseline_rf_metrics", "json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_with_meta, f, indent=2, ensure_ascii=False)
    print(f"  Métricas salvas: {metrics_path}")

    labels = [label_map.get(i, str(i)) for i in sorted(label_map)]
    report_text = classification_report(y_te, y_pred, target_names=labels)
    report_path = versioned_path(Config.IDS_REPORTS_DIR, "baseline_rf_report", "txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("BASELINE Random Forest — Relatório de Classificação\n")
        f.write("=" * 60 + "\n")
        f.write(f"Treinado em: {datetime.now().isoformat()}\n")
        f.write(f"n_train = {X_tr.shape[0]:,}  n_test = {X_te.shape[0]:,}\n\n")
        f.write(report_text)
        f.write(f"\n\nMCC = {metrics['mcc']:.4f}\n")
        f.write(f"FPR-macro = {metrics['fpr_macro']:.4f}\n")
        f.write("\nFPR por classe:\n")
        for cls, fpr in sorted(metrics["fpr_per_class"].items(), key=lambda x: -x[1]):
            f.write(f"  {cls:<35s}: {fpr:.4f}\n")
    print(f"  Relatório salvo: {report_path}")

    cm_path = versioned_path(Config.IDS_REPORTS_DIR, "baseline_rf_confusion_matrix", "png")
    plot_confusion_matrix(y_te, y_pred, label_map, cm_path)
    print(f"  Matriz de confusão salva: {cm_path}")

    # Registro como M0 imutável
    register_baseline(
        model_pickle_path=model_path,
        y_true=y_te,
        y_pred=y_pred,
        metrics=metrics_with_meta,
        overwrite=False,
    )

    # Relatório completo (apenas M0 disponível por enquanto)
    rep_dir = Config.IDS_REPORTS_DIR / f"full_baseline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    generate_full_report(rep_dir, label_map=label_map)

    print("\n" + "=" * 62)
    print("  RESUMO BASELINE M0 (Random Forest)")
    print("=" * 62)
    print(f"  Acurácia          : {metrics['accuracy']:.4f}")
    print(f"  F1-macro          : {metrics['f1_macro']:.4f}")
    print(f"  F1-weighted       : {metrics['f1_weighted']:.4f}")
    print(f"  Precision-macro   : {metrics['precision_macro']:.4f}")
    print(f"  Recall-macro      : {metrics['recall_macro']:.4f}")
    print(f"  MCC               : {metrics['mcc']:.4f}")
    print(f"  FPR-macro         : {metrics['fpr_macro']:.4f}")
    print("=" * 62)
    print(f"\n  M0 registrado no triplete. Próximo passo:")
    print(f"  python3 IDS/ids_learn.py train --force")


if __name__ == "__main__":
    main()
