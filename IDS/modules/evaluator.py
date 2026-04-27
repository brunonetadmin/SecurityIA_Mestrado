#!/usr/bin/env python3
"""
IDS/modules/evaluator.py — Framework de Avaliação Contínua do Modelo IDS

Versão atualizada com:
  - Métricas operacionais ampliadas (FPR por classe, MCC, alarmes/h)
  - Registro automático no triplete M0/Mp/Mc
  - Geração de relatório completo a cada avaliação
  - Nomes de arquivos versionados via IDS.modules.versioning
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import Config

Config.configure_tensorflow()

import joblib
import tensorflow as tf
from keras.models import load_model as keras_load_model
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, matthews_corrcoef,
    precision_score, recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import chi2

from IDS.modules.versioning import versioned_path


_EVAL_CFG = Config.EVALUATION_CONFIG
_EVAL_DIR       = _EVAL_CFG["eval_dir"]
_BENCHMARK_FILE = _EVAL_DIR / "benchmark_dataset.parquet"
_HISTORY_FILE   = _EVAL_CFG["history_file"]
_ALPHA          = _EVAL_CFG["significance_alpha"]
_BENCH_FRAC     = _EVAL_CFG["benchmark_fraction"]
_BATCH          = _EVAL_CFG["batch_size"]
_DELTA_WARN     = 0.005


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento de artefatos
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_artifacts(model_path: Optional[Path] = None) -> dict:
    """Carrega modelo, scaler, encoder e info."""
    from IDS.ids_learn import BahdanauAttention
    custom = {"BahdanauAttention": BahdanauAttention}

    mp = model_path or (Config.MODEL_DIR / Config.MODEL_FILENAME)
    if not mp.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {mp}")

    # compile=False para tolerar Focal Loss customizada não serializada
    model   = keras_load_model(str(mp), custom_objects=custom, compile=False)
    scaler  = joblib.load(Config.MODEL_DIR / Config.SCALER_FILENAME)
    encoder = joblib.load(Config.MODEL_DIR / Config.LABEL_ENCODER_FILENAME)

    with open(Config.MODEL_DIR / Config.MODEL_INFO_FILENAME, encoding="utf-8") as f:
        info = json.load(f)

    return {
        "model":             model,
        "scaler":            scaler,
        "encoder":           encoder,
        "selected_features": info.get("selected_features", Config.FEATURE_COLUMNS),
        "label_map":         {int(k): v for k, v in info.get("label_mapping", {}).items()},
        "version":           info.get("version", "v?"),
        "trained_at":        info.get("trained_at", "—"),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Inferência
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_np(
    X: np.ndarray,
    artifacts: dict,
    batch_size: int = _BATCH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Executa inferência em lote sobre array numpy."""
    n = len(X)
    classes  = np.empty(n, dtype=np.int32)
    conf_max = np.empty(n, dtype=np.float32)

    first = artifacts["model"].predict(X[:1].reshape(1, -1, 1), verbose=0)
    n_cls = first.shape[1]
    proba = np.empty((n, n_cls), dtype=np.float32)

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        chunk = X[s:e].reshape(e - s, X.shape[1], 1)
        p = artifacts["model"].predict(chunk, verbose=0)
        classes[s:e]  = np.argmax(p, axis=1)
        conf_max[s:e] = np.max(p, axis=1)
        proba[s:e]    = p

    return classes, conf_max, proba


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark congelado
# ─────────────────────────────────────────────────────────────────────────────

def create_benchmark(force: bool = False) -> None:
    """Cria benchmark congelado: amostra estratificada de _BENCH_FRAC."""
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if _BENCHMARK_FILE.exists() and not force:
        df = pd.read_parquet(_BENCHMARK_FILE)
        print(f"  Benchmark existente: {len(df):,} amostras — use force=True para recriar.")
        return

    print("  Criando benchmark congelado …")
    csvs     = list(Config.DATA_DIR.glob("*.csv"))
    parquets = list(Config.DATA_DIR.glob("*.parquet"))

    if not csvs and not parquets:
        raise FileNotFoundError(f"Nenhum dado em {Config.DATA_DIR}")

    frames = [pd.read_csv(f, low_memory=False) for f in csvs]
    frames += [pd.read_parquet(f) for f in parquets]
    df = pd.concat(frames, ignore_index=True)

    meta_drop = [c for c in Config.META_COLUMNS if c in df.columns]
    df.drop(columns=meta_drop, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if "Label" not in df.columns:
        raise ValueError("Coluna 'Label' não encontrada.")

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=_BENCH_FRAC,
        random_state=Config.TRAINING_CONFIG["random_state"],
    )
    _, idx = next(splitter.split(df, df["Label"]))
    bench  = df.iloc[idx].reset_index(drop=True)

    dist = bench["Label"].value_counts()
    print(f"  Benchmark: {len(bench):,} amostras | {len(dist)} classes")
    for cls, cnt in dist.items():
        print(f"    {cls:<40s}: {cnt:>6,}")

    bench.to_parquet(_BENCHMARK_FILE, compression="snappy", index=False)
    print(f"  Benchmark salvo: '{_BENCHMARK_FILE}'")


# ─────────────────────────────────────────────────────────────────────────────
# Avaliação padronizada (com métricas estendidas)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    artifacts: Optional[dict] = None,
    model_path: Optional[Path] = None,
) -> dict:
    """Avalia modelo no benchmark congelado, com métricas estendidas."""
    if not _BENCHMARK_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark não encontrado: {_BENCHMARK_FILE}\nExecute create_benchmark()."
        )

    if artifacts is None:
        artifacts = load_eval_artifacts(model_path)

    print(f"\n  Avaliando modelo {artifacts['version']} …")

    df    = pd.read_parquet(_BENCHMARK_FILE)
    feats = artifacts["selected_features"]
    for f in feats:
        if f not in df.columns:
            df[f] = 0.0

    X_raw    = df[feats].values.astype(np.float32)
    X_scaled = artifacts["scaler"].transform(X_raw)

    enc    = artifacts["encoder"]
    y_true = enc.transform(df["Label"].values)

    print(f"  {len(df):,} amostras | {len(feats)} features | {len(enc.classes_)} classes")
    print("  Executando inferência …")

    y_pred, conf_max, _ = run_inference_np(X_scaled, artifacts)

    label_names = [artifacts["label_map"].get(i, str(i))
                   for i in sorted(artifacts["label_map"])]

    acc        = float(accuracy_score(y_true, y_pred))
    f1_macro   = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro  = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    mcc_v      = float(matthews_corrcoef(y_true, y_pred))

    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = {
        artifacts["label_map"].get(i, str(i)): float(v)
        for i, v in enumerate(f1_per)
    }

    cm = confusion_matrix(y_true, y_pred)

    # Métricas operacionais ampliadas
    fpr_per_class = {}
    tp_fp_fn_tn = {}
    for c in range(cm.shape[0]):
        cls_name = artifacts["label_map"].get(c, str(c))
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        tn = int(cm.sum() - tp - fp - fn)
        denom = fp + tn
        fpr_per_class[cls_name] = float(fp / denom) if denom else 0.0
        tp_fp_fn_tn[cls_name] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn}

    fpr_macro = float(np.mean(list(fpr_per_class.values())))
    lambda_h  = _EVAL_CFG.get("lambda_benign_per_hour", 1_000_000)

    result = {
        "version":          artifacts["version"],
        "trained_at":       artifacts["trained_at"],
        "eval_ts":          datetime.now().isoformat(),
        "n_samples":        int(len(df)),
        "accuracy":         round(acc,         4),
        "f1_macro":         round(f1_macro,    4),
        "f1_weighted":      round(f1_weighted, 4),
        "precision_macro":  round(prec_macro,  4),
        "recall_macro":     round(rec_macro,   4),
        "mcc":              round(mcc_v,       4),
        "fpr_macro":        round(fpr_macro,   4),
        "alarms_per_hour_estimated": round(fpr_macro * lambda_h, 2),
        "per_class_f1":     per_class_f1,
        "fpr_per_class":    fpr_per_class,
        "tp_fp_fn_tn":      tp_fp_fn_tn,
        "confusion_matrix": cm.tolist(),
        "_y_true": y_true.tolist(),
        "_y_pred": y_pred.tolist(),
    }

    print(f"\n  {'Métrica':<22s}  {'Valor':>8s}")
    print("  " + "─" * 32)
    for k, v in [("Acurácia", acc), ("F1-macro", f1_macro),
                 ("F1-ponderado", f1_weighted), ("Precision-macro", prec_macro),
                 ("Recall-macro", rec_macro), ("MCC", mcc_v),
                 ("FPR-macro", fpr_macro)]:
        bar = "█" * max(0, int(v * 20))
        print(f"  {k:<22s}  {v:.4f}  {bar}")

    print(f"\n  F1 por classe:")
    for cls, f1 in sorted(per_class_f1.items(), key=lambda x: -x[1]):
        bar = "█" * max(0, int(f1 * 20))
        status = "✓" if f1 >= 0.90 else ("△" if f1 >= 0.70 else "✗")
        print(f"    {status} {cls:<40s}: {f1:.4f}  {bar}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Histórico e regressão
# ─────────────────────────────────────────────────────────────────────────────

def load_history() -> List[dict]:
    if _HISTORY_FILE.exists():
        with open(_HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(entry: dict) -> None:
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    history = load_history()
    clean = {k: v for k, v in entry.items() if not k.startswith("_")}
    history.append(clean)
    tmp = _HISTORY_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    tmp.replace(_HISTORY_FILE)


def check_regression(current: dict, history: List[dict]) -> List[str]:
    if not history:
        return []
    prev = history[-1]
    warnings = []
    delta_f1 = current["f1_macro"] - prev.get("f1_macro", 0)

    if delta_f1 < -_DELTA_WARN:
        warnings.append(
            f"⚠ REGRESSÃO: F1-macro caiu {abs(delta_f1):.4f} "
            f"({prev['f1_macro']:.4f} → {current['f1_macro']:.4f})"
        )

    for cls, f1 in current["per_class_f1"].items():
        prev_f1 = prev.get("per_class_f1", {}).get(cls)
        if prev_f1 and (f1 - prev_f1) < -_DELTA_WARN * 2:
            warnings.append(f"⚠ REGRESSÃO em '{cls}': {prev_f1:.4f} → {f1:.4f}")

    return warnings


# ─────────────────────────────────────────────────────────────────────────────
# Teste de McNemar
# ─────────────────────────────────────────────────────────────────────────────

def mcnemar_test(
    y_true: List[int],
    y_pred_a: List[int],
    y_pred_b: List[int],
    alpha: float = _ALPHA,
) -> dict:
    """Teste de McNemar com correção de Edwards."""
    yt = np.array(y_true)
    ya = np.array(y_pred_a)
    yb = np.array(y_pred_b)

    correct_a = (ya == yt)
    correct_b = (yb == yt)

    b = int(np.sum(correct_a & ~correct_b))
    c = int(np.sum(~correct_a & correct_b))

    if b + c == 0:
        return {
            "b": 0, "c": 0, "statistic": 0.0, "p_value": 1.0,
            "significant": False,
            "interpretation": "Classificadores idênticos nas amostras de discordância.",
        }

    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    p    = float(1 - chi2.cdf(stat, df=1))

    return {
        "b": b, "c": c,
        "statistic": round(float(stat), 6),
        "p_value":   round(p, 6),
        "alpha":     alpha,
        "significant": p < alpha,
        "interpretation": (
            f"H₀ rejeitada (p={p:.4f} < α={alpha}): diferença significativa — "
            f"modelo {'B melhor' if c > b else 'A melhor'}."
            if p < alpha else
            f"H₀ não rejeitada (p={p:.4f} ≥ α={alpha}): sem evidência de diferença."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Métricas de aprendizado contínuo
# ─────────────────────────────────────────────────────────────────────────────

def continual_learning_metrics(history: List[dict]) -> dict:
    """Forward Transfer (FWT) e Backward Transfer (BWT) por classe."""
    if len(history) < 2:
        return {"fwt": {}, "bwt": {}, "note": "Mínimo de 2 avaliações."}

    n_classes  = len(history[0].get("per_class_f1", {}))
    random_b   = 1.0 / max(n_classes, 1)
    first      = history[0].get("per_class_f1", {})
    current    = history[-1].get("per_class_f1", {})

    fwt = {cls: round(f1 - random_b, 4) for cls, f1 in current.items()}
    bwt = {cls: round(current.get(cls, 0) - first.get(cls, 0), 4) for cls in first}

    return {
        "n_versions":      len(history),
        "random_baseline": round(random_b, 4),
        "fwt":             fwt,
        "bwt":             bwt,
        "bwt_mean":        round(float(np.mean(list(bwt.values()))) if bwt else 0.0, 4),
        "fwt_mean":        round(float(np.mean(list(fwt.values()))) if fwt else 0.0, 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fluxo completo
# ─────────────────────────────────────────────────────────────────────────────

def run_full_evaluation(
    model_path: Optional[Path] = None,
    compare_with: Optional[Path] = None,
    save: bool = True,
    generate_full_triplet_report: bool = True,
) -> dict:
    """Executa avaliação completa e gera relatório consolidado M0/Mp/Mc."""
    print("\n" + "═" * 62)
    print("  SecurityIA — Avaliação do Modelo")
    print("═" * 62)

    if not _BENCHMARK_FILE.exists():
        print("  Benchmark não encontrado. Criando …")
        create_benchmark()

    arts    = load_eval_artifacts(model_path)
    result  = evaluate_model(arts)
    history = load_history()

    warnings = check_regression(result, history)
    if warnings:
        print(f"\n  {chr(10).join(warnings)}")
    else:
        print("\n  ✓ Nenhuma regressão detectada em relação à versão anterior.")

    mcnemar_result = None
    if compare_with and history:
        print(f"\n  Comparando com: {compare_with.name} …")
        arts_b = load_eval_artifacts(compare_with)
        df = pd.read_parquet(_BENCHMARK_FILE)
        feats = arts_b["selected_features"]
        for f in feats:
            if f not in df.columns:
                df[f] = 0.0
        X_b = arts_b["scaler"].transform(df[feats].values.astype(np.float32))
        yp_b, _, _ = run_inference_np(X_b, arts_b)

        mcnemar_result = mcnemar_test(
            result["_y_true"], result["_y_pred"], yp_b.tolist(),
        )
        print(f"\n  McNemar: {mcnemar_result['interpretation']}")
        result["mcnemar"] = mcnemar_result

    cl = continual_learning_metrics(history + [result])
    result["continual_learning"] = cl

    if cl.get("bwt_mean", 0) < -0.01:
        print(f"\n  ⚠ Catastrofic forgetting detectado: BWT médio = {cl['bwt_mean']:.4f}")
    elif "bwt_mean" in cl:
        print(f"\n  ✓ Sem esquecimento catastrófico: BWT médio = {cl['bwt_mean']:.4f}")

    if save:
        save_history(result)
        print(f"\n  Resultado salvo em '{_HISTORY_FILE}'")

    # Relatório completo M0/Mp/Mc
    if generate_full_triplet_report:
        try:
            from IDS.modules.full_report import generate as generate_full_report
            rep_dir = Config.IDS_REPORTS_DIR / f"full_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            generate_full_report(rep_dir, label_map=arts["label_map"])
            print(f"\n  Relatório completo M0/Mp/Mc: {rep_dir}")
        except Exception as exc:
            print(f"\n  ⚠ Não foi possível gerar relatório completo: {exc}")

    print("═" * 62)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    p = argparse.ArgumentParser(
        description="SecurityIA — Framework de Avaliação do Modelo IDS",
    )
    p.add_argument("--create-benchmark", action="store_true",
                   help="Cria/recria o benchmark congelado")
    p.add_argument("--force", action="store_true",
                   help="Força recriação do benchmark")
    p.add_argument("--compare", type=Path, default=None,
                   help="Modelo alternativo para teste de McNemar")
    p.add_argument("--no-save", action="store_true",
                   help="Não persiste resultado no histórico")
    p.add_argument("--no-full-report", action="store_true",
                   help="Não gera relatório completo M0/Mp/Mc")
    p.add_argument("--history", action="store_true",
                   help="Exibe histórico de avaliações")
    args = p.parse_args()

    if args.create_benchmark:
        create_benchmark(force=args.force)
        return

    if args.history:
        hist = load_history()
        if not hist:
            print("  Nenhuma avaliação no histórico.")
        else:
            print(f"\n  {'Versão':<20s}  {'Acurácia':>9s}  {'F1-macro':>9s}  "
                  f"{'MCC':>7s}  {'FPR':>7s}  {'Data'}")
            print("  " + "─" * 76)
            for h in hist:
                print(f"  {h.get('version','?'):<20s}  "
                      f"{h.get('accuracy', 0):.4f}     "
                      f"{h.get('f1_macro', 0):.4f}  "
                      f"{h.get('mcc', 0):>7.4f}  "
                      f"{h.get('fpr_macro', 0):>7.4f}  "
                      f"{h.get('eval_ts','?')[:19]}")
        return

    run_full_evaluation(
        compare_with=args.compare,
        save=not args.no_save,
        generate_full_triplet_report=not args.no_full_report,
    )


if __name__ == "__main__":
    main()
