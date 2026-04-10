#!/usr/bin/env python3
"""
IDS/modules/evaluator.py — Framework de Avaliação Contínua do Modelo IDS

Responsabilidades:
  - Benchmark congelado: amostra estratificada do dataset original
    preservada para comparação justa entre versões do modelo
  - Avaliação padronizada com métricas completas (Acurácia, F1-macro,
    F1-ponderado, Precision, Recall por classe)
  - Histórico de avaliações persistido em JSON
  - Teste de McNemar (1947) para significância estatística entre modelos
  - Detecção de regressão: alerta quando F1-macro cai > DELTA_THRESHOLD
  - Aprendizado contínuo: Forward Transfer, Backward Transfer por classe

Importado por: IDS/ids_manager.py
Uso direto:    python3 -m IDS.modules.evaluator
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
    confusion_matrix, f1_score, precision_score, recall_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2


# ─────────────────────────────────────────────────────────────────────────────
# Configuração local de avaliação
# ─────────────────────────────────────────────────────────────────────────────

_EVAL_CFG = Config.EVALUATION_CONFIG
_EVAL_DIR       = _EVAL_CFG["eval_dir"]
_BENCHMARK_FILE = _EVAL_DIR / "benchmark_dataset.parquet"
_HISTORY_FILE   = _EVAL_CFG["history_file"]
_ALPHA          = _EVAL_CFG["significance_alpha"]
_BENCH_FRAC     = _EVAL_CFG["benchmark_fraction"]
_BATCH          = _EVAL_CFG["batch_size"]
_DELTA_WARN     = 0.005   # queda de F1-macro que dispara alerta de regressão
_CUSTOM_OBJECTS = {}      # preenchido em load_eval_artifacts()


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento de artefatos
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_artifacts(model_path: Optional[Path] = None) -> dict:
    """
    Carrega modelo, scaler, encoder e info a partir de Config.MODEL_DIR
    (ou de model_path se fornecido).

    Retorna dict com chaves: model, scaler, encoder, selected_features,
    label_map, version.
    """
    from IDS.ids_learn import BahdanauAttention
    custom = {"BahdanauAttention": BahdanauAttention}

    mp = model_path or (Config.MODEL_DIR / Config.MODEL_FILENAME)
    if not mp.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {mp}")

    model   = keras_load_model(str(mp), custom_objects=custom)
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
# Inferência em numpy (sem Pandas — mais rápido para avaliação)
# ─────────────────────────────────────────────────────────────────────────────

def run_inference_np(
    X: np.ndarray,
    artifacts: dict,
    batch_size: int = _BATCH,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Executa inferência em lote sobre array numpy.
    Retorna (classes_preditas, conf_max, proba_full).
    """
    n        = len(X)
    classes  = np.empty(n, dtype=np.int32)
    conf_max = np.empty(n, dtype=np.float32)

    # Primeiro batch para descobrir n_classes
    first = artifacts["model"].predict(X[:1].reshape(1, -1, 1), verbose=0)
    n_cls = first.shape[1]
    proba = np.empty((n, n_cls), dtype=np.float32)

    for s in range(0, n, batch_size):
        e       = min(s + batch_size, n)
        chunk   = X[s:e].reshape(e - s, X.shape[1], 1)
        p       = artifacts["model"].predict(chunk, verbose=0)
        classes[s:e]  = np.argmax(p, axis=1)
        conf_max[s:e] = np.max(p, axis=1)
        proba[s:e]    = p

    return classes, conf_max, proba


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark congelado
# ─────────────────────────────────────────────────────────────────────────────

def create_benchmark(force: bool = False) -> None:
    """
    Cria o benchmark congelado: amostra estratificada de BENCH_FRAC do dataset.
    Idempotente — não recria se já existir (a menos que force=True).

    Estratificação por classe garante representação de todas as 15 categorias
    do CIC-IDS2018, incluindo classes raras (Heartbleed, Infiltration).
    """
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

    # Remove metadados e colunas problemáticas
    meta_drop = [c for c in Config.META_COLUMNS if c in df.columns]
    df.drop(columns=meta_drop, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    if "Label" not in df.columns:
        raise ValueError("Coluna 'Label' não encontrada no dataset.")

    # Amostragem estratificada
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        test_size=_BENCH_FRAC,
        random_state=Config.TRAINING_CONFIG["random_state"],
    )
    _, idx = next(splitter.split(df, df["Label"]))
    bench  = df.iloc[idx].reset_index(drop=True)

    # Distribuição por classe
    dist = bench["Label"].value_counts()
    print(f"  Benchmark: {len(bench):,} amostras | {len(dist)} classes")
    for cls, cnt in dist.items():
        print(f"    {cls:<40s}: {cnt:>6,}")

    bench.to_parquet(_BENCHMARK_FILE, compression="snappy", index=False)
    print(f"  Benchmark salvo: '{_BENCHMARK_FILE}'")


# ─────────────────────────────────────────────────────────────────────────────
# Avaliação padronizada
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    artifacts: Optional[dict] = None,
    model_path: Optional[Path] = None,
) -> dict:
    """
    Avalia o modelo no benchmark congelado.
    Retorna métricas completas e por classe.

    Parâmetros
    ----------
    artifacts  : dict retornado por load_eval_artifacts() (opcional)
    model_path : alternativo a artifacts — carrega do caminho especificado

    Retorna
    -------
    dict com: version, accuracy, f1_macro, f1_weighted, precision_macro,
    recall_macro, per_class_f1, n_samples, eval_ts, confusion_matrix
    """
    if not _BENCHMARK_FILE.exists():
        raise FileNotFoundError(
            f"Benchmark não encontrado: {_BENCHMARK_FILE}\n"
            f"Execute create_benchmark() primeiro."
        )

    if artifacts is None:
        artifacts = load_eval_artifacts(model_path)

    print(f"\n  Avaliando modelo {artifacts['version']} …")

    df    = pd.read_parquet(_BENCHMARK_FILE)
    feats = artifacts["selected_features"]

    # Garante colunas — preenche ausentes com 0
    for f in feats:
        if f not in df.columns:
            df[f] = 0.0

    X_raw    = df[feats].values.astype(np.float32)
    X_scaled = artifacts["scaler"].transform(X_raw)

    # Labels reais
    enc    = artifacts["encoder"]
    y_true = enc.transform(df["Label"].values)

    print(f"  {len(df):,} amostras | {len(feats)} features | {len(enc.classes_)} classes")
    print("  Executando inferência …")

    y_pred, conf_max, _ = run_inference_np(X_scaled, artifacts)

    label_names = [artifacts["label_map"].get(i, str(i))
                   for i in sorted(artifacts["label_map"])]

    acc        = float(accuracy_score(y_true, y_pred))
    f1_macro   = float(f1_score(y_true, y_pred, average="macro",    zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    prec_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec_macro  = float(recall_score(y_true, y_pred, average="macro",  zero_division=0))

    # F1 por classe
    f1_per = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = {
        artifacts["label_map"].get(i, str(i)): float(v)
        for i, v in enumerate(f1_per)
    }

    cm = confusion_matrix(y_true, y_pred).tolist()

    result = {
        "version":       artifacts["version"],
        "trained_at":    artifacts["trained_at"],
        "eval_ts":       datetime.now().isoformat(),
        "n_samples":     int(len(df)),
        "accuracy":      round(acc,        4),
        "f1_macro":      round(f1_macro,   4),
        "f1_weighted":   round(f1_weighted,4),
        "precision_macro": round(prec_macro, 4),
        "recall_macro":  round(rec_macro,  4),
        "per_class_f1":  per_class_f1,
        "confusion_matrix": cm,
        # Mantém predições para McNemar
        "_y_true": y_true.tolist(),
        "_y_pred": y_pred.tolist(),
    }

    print(f"\n  {'Métrica':<22s}  {'Valor':>8s}")
    print("  " + "─" * 32)
    for k, v in [("Acurácia", acc), ("F1-macro", f1_macro),
                 ("F1-ponderado", f1_weighted), ("Precision-macro", prec_macro),
                 ("Recall-macro", rec_macro)]:
        bar = "█" * int(v * 20)
        print(f"  {k:<22s}  {v:.4f}  {bar}")

    print(f"\n  F1 por classe:")
    for cls, f1 in sorted(per_class_f1.items(), key=lambda x: -x[1]):
        bar = "█" * int(f1 * 20)
        status = "✓" if f1 >= 0.90 else ("△" if f1 >= 0.70 else "✗")
        print(f"    {status} {cls:<40s}: {f1:.4f}  {bar}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Histórico e detecção de regressão
# ─────────────────────────────────────────────────────────────────────────────

def load_history() -> List[dict]:
    if _HISTORY_FILE.exists():
        with open(_HISTORY_FILE, encoding="utf-8") as f:
            return json.load(f)
    return []


def save_history(entry: dict) -> None:
    _EVAL_DIR.mkdir(parents=True, exist_ok=True)
    history = load_history()
    # Remove campos internos antes de persistir
    clean = {k: v for k, v in entry.items() if not k.startswith("_")}
    history.append(clean)
    tmp = _HISTORY_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    tmp.replace(_HISTORY_FILE)


def check_regression(current: dict, history: List[dict]) -> List[str]:
    """
    Verifica regressão em relação à avaliação anterior.
    Retorna lista de alertas (vazia se tudo OK).
    """
    if not history:
        return []

    prev      = history[-1]
    warnings  = []
    delta_f1  = current["f1_macro"] - prev.get("f1_macro", 0)

    if delta_f1 < -_DELTA_WARN:
        warnings.append(
            f"⚠ REGRESSÃO: F1-macro caiu {abs(delta_f1):.4f} "
            f"({prev['f1_macro']:.4f} → {current['f1_macro']:.4f})"
        )

    # Verifica regressão por classe
    for cls, f1 in current["per_class_f1"].items():
        prev_f1 = prev.get("per_class_f1", {}).get(cls)
        if prev_f1 and (f1 - prev_f1) < -_DELTA_WARN * 2:
            warnings.append(
                f"⚠ REGRESSÃO em '{cls}': {prev_f1:.4f} → {f1:.4f}"
            )

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
    """
    Teste de McNemar (1947) para comparação de dois classificadores.

    Hipótese nula (H₀): os dois classificadores têm a mesma taxa de erro.
    Rejeita H₀ se p < alpha — indica que a diferença de performance é
    estatisticamente significativa e não produto de variância amostral.

    Usa correção de continuidade de Edwards quando b + c ≤ 25.

    Parâmetros
    ----------
    y_true   : rótulos verdadeiros
    y_pred_a : predições do modelo A (geralmente baseline)
    y_pred_b : predições do modelo B (nova versão)

    Retorna
    -------
    dict com: b, c, statistic, p_value, significant, interpretation
    """
    yt = np.array(y_true)
    ya = np.array(y_pred_a)
    yb = np.array(y_pred_b)

    # Tabela de contingência 2×2
    correct_a = (ya == yt)
    correct_b = (yb == yt)

    b = int(np.sum(correct_a & ~correct_b))   # A acertou, B errou
    c = int(np.sum(~correct_a & correct_b))   # A errou, B acertou

    if b + c == 0:
        return {
            "b": 0, "c": 0, "statistic": 0.0, "p_value": 1.0,
            "significant": False,
            "interpretation": "Classificadores idênticos nas amostras de discordância.",
        }

    # Estatística com correção de continuidade (Edwards, 1948)
    stat = (abs(b - c) - 1.0) ** 2 / (b + c)
    p    = float(1 - chi2.cdf(stat, df=1))

    return {
        "b":              b,
        "c":              c,
        "statistic":      round(float(stat), 6),
        "p_value":        round(p, 6),
        "alpha":          alpha,
        "significant":    p < alpha,
        "interpretation": (
            f"H₀ rejeitada (p={p:.4f} < α={alpha}): diferença estatisticamente "
            f"significativa — modelo {'B melhor' if c > b else 'A melhor'}."
            if p < alpha else
            f"H₀ não rejeitada (p={p:.4f} ≥ α={alpha}): sem evidência de diferença "
            f"significativa entre os modelos."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Métricas de aprendizado contínuo
# ─────────────────────────────────────────────────────────────────────────────

def continual_learning_metrics(history: List[dict]) -> dict:
    """
    Calcula Forward Transfer (FWT) e Backward Transfer (BWT) por classe.

    FWT(k) = F1_k(versão_atual) − F1_k(random_baseline ≈ 1/n_classes)
    BWT(k) = F1_k(versão_atual) − F1_k(versão_inicial)

    Indica se o modelo melhorou (FWT > 0) e se não esqueceu (BWT ≈ 0 ou > 0).
    """
    if len(history) < 2:
        return {"fwt": {}, "bwt": {}, "note": "Mínimo de 2 avaliações para calcular."}

    n_classes  = len(history[0].get("per_class_f1", {}))
    random_b   = 1.0 / max(n_classes, 1)
    first      = history[0].get("per_class_f1", {})
    current    = history[-1].get("per_class_f1", {})

    fwt = {cls: round(f1 - random_b, 4) for cls, f1 in current.items()}
    bwt = {
        cls: round(current.get(cls, 0) - first.get(cls, 0), 4)
        for cls in first
    }

    return {
        "n_versions":    len(history),
        "random_baseline": round(random_b, 4),
        "fwt":           fwt,
        "bwt":           bwt,
        "bwt_mean":      round(float(np.mean(list(bwt.values()))), 4),
        "fwt_mean":      round(float(np.mean(list(fwt.values()))), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fluxo completo de avaliação
# ─────────────────────────────────────────────────────────────────────────────

def run_full_evaluation(
    model_path: Optional[Path] = None,
    compare_with: Optional[Path] = None,
    save: bool = True,
) -> dict:
    """
    Executa a avaliação completa:
      1. Carrega artefatos e garante benchmark.
      2. Avalia modelo atual.
      3. Verifica regressão vs. histórico.
      4. Se compare_with fornecido, executa McNemar.
      5. Calcula métricas de aprendizado contínuo.
      6. Persiste resultado no histórico.

    Retorna dict com todos os resultados.
    """
    print("\n" + "═" * 62)
    print("  SecurityIA — Avaliação do Modelo")
    print("═" * 62)

    # Garante benchmark
    if not _BENCHMARK_FILE.exists():
        print("  Benchmark não encontrado. Criando …")
        create_benchmark()

    arts    = load_eval_artifacts(model_path)
    result  = evaluate_model(arts)
    history = load_history()

    # Regressão
    warnings = check_regression(result, history)
    if warnings:
        print(f"\n  {chr(10).join(warnings)}")
    else:
        print("\n  ✓ Nenhuma regressão detectada em relação à versão anterior.")

    # McNemar (opcional)
    mcnemar_result = None
    if compare_with and history:
        print(f"\n  Comparando com: {compare_with.name} …")
        arts_b      = load_eval_artifacts(compare_with)
        df          = pd.read_parquet(_BENCHMARK_FILE)
        feats       = arts_b["selected_features"]
        for f in feats:
            if f not in df.columns:
                df[f] = 0.0
        X_b   = arts_b["scaler"].transform(df[feats].values.astype(np.float32))
        yp_b, _, _ = run_inference_np(X_b, arts_b)

        mcnemar_result = mcnemar_test(
            result["_y_true"],
            result["_y_pred"],
            yp_b.tolist(),
        )
        print(f"\n  McNemar: {mcnemar_result['interpretation']}")
        result["mcnemar"] = mcnemar_result

    # Aprendizado contínuo
    cl = continual_learning_metrics(history + [result])
    result["continual_learning"] = cl

    if cl["bwt_mean"] < -0.01:
        print(f"\n  ⚠ Catastrofic forgetting detectado: BWT médio = {cl['bwt_mean']:.4f}")
    else:
        print(f"\n  ✓ Sem esquecimento catastrófico: BWT médio = {cl['bwt_mean']:.4f}")

    if save:
        save_history(result)
        print(f"\n  Resultado salvo em '{_HISTORY_FILE}'")

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
                   help="Força recriação do benchmark mesmo se já existir")
    p.add_argument("--compare", type=Path, default=None,
                   help="Modelo alternativo para teste de McNemar")
    p.add_argument("--no-save", action="store_true",
                   help="Não persiste resultado no histórico")
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
            print(f"\n  {'Versão':<20s}  {'Acurácia':>9s}  {'F1-macro':>9s}  {'Data'}")
            print("  " + "─" * 60)
            for h in hist:
                print(f"  {h.get('version','?'):<20s}  "
                      f"{h.get('accuracy', 0):.4f}  "
                      f"  {h.get('f1_macro', 0):.4f}  "
                      f"  {h.get('eval_ts','?')[:19]}")
        return

    run_full_evaluation(
        compare_with=args.compare,
        save=not args.no_save,
    )


if __name__ == "__main__":
    main()
