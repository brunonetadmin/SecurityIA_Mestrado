#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analise_sensibilidade_amostra.py
================================
Estudo de sensibilidade do TETO de subamostragem (n_amostras_max) do
SecurityIA sobre o CSE-CIC-IDS2018 (versão Mendeley).

Fundamenta a escolha do teto por DUAS evidências, em uma execução:

  (A) FIDELIDADE DISTRIBUCIONAL
      Para cada teto candidato, mede a divergência da distribuição de classes
      da subamostra em relação à base integral (KL e distância L1), além da
      contagem por classe e da menor classe presente. Mostra que a amostragem
      por arquivo preserva o desbalanceamento original.

  (B) SATURAÇÃO DE MÉTRICAS  (ajuste único, sem validação cruzada)
      Para cada teto, treina a configuração final implantada (CatBoost da
      Investigação 4 + SMOTE-ENN da Investigação 2, sobre os k=23 atributos
      IG_MI60/40) e avalia em teste reservado estratificado. Mostra a partir de
      qual teto as métricas estabilizam (saturação amostral).

Juntas, as duas evidências sustentam a fixação do teto: ponto em que o ganho
marginal de dados não melhora as métricas e a distribuição já está preservada.

Saídas (diretório versionado): tabelas CSV + figuras PNG (tons de cinza, sem
título embutido) + resumo JSON. Colunas-padrão de métricas: Recall-macro, MCC,
F1-macro, FPR-macro, Tempo(s).

Execução (servidor, segundo plano imune à sessão SSH):
    cd /opt/SecurityIA
    nohup setsid python3 analise_sensibilidade_amostra.py \
        > Tests/sens_amostra.out 2>&1 &

Os imports do projeto (config/ids_learn) sao tardios — ficam dentro de funcoes —
para permitir teste isolado dos utilitarios puros.
"""
from __future__ import annotations

import gc
import json
import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# ── matplotlib sem display (servidor headless) ──────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("sensibilidade")

# ═══════════════════════════════════════════════════════════════════════════
# PARAMETROS DO ESTUDO
# ═══════════════════════════════════════════════════════════════════════════
CAPS = [200_000, 366_000, 500_000, 750_000, 1_000_000]

# Proporcao do teste reservado (estratificado) para a curva de saturacao.
TEST_SIZE = 0.15

# Hiperparametros do CatBoost — configuracao final da Investigacao 4 (Optuna).
# Valores reais do trial vencedor; mantidos FIXOS em todos os tetos para que a
# unica variavel do estudo seja o tamanho da amostra.
# Para usar Config.TREE_CONFIG em vez destes, defina USE_CONFIG_TREE=True.
USE_CONFIG_TREE = False
CATBOOST_PARAMS_INV4 = dict(
    iterations=400,
    depth=10,
    learning_rate=0.1205712628744377,
)


# ═══════════════════════════════════════════════════════════════════════════
# UTILITARIOS PUROS (testaveis isoladamente, sem dependencia do projeto)
# ═══════════════════════════════════════════════════════════════════════════
def fpr_macro(y_true, y_pred, labels) -> float:
    """Taxa de Falsos Positivos macro a partir da matriz de confusao multiclasse.

    FPR_k = FP_k / (FP_k + TN_k), media uniforme sobre as classes.
    """
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    fprs = []
    for k in range(len(labels)):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        tn = total - tp - fp - fn
        denom = fp + tn
        fprs.append(fp / denom if denom > 0 else 0.0)
    return float(np.mean(fprs))


def compute_metrics(y_true, y_pred) -> dict:
    """Recall-macro, MCC, F1-macro, FPR-macro sobre o conjunto avaliado."""
    from sklearn.metrics import recall_score, matthews_corrcoef, f1_score
    labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
    return {
        "recall_macro": float(recall_score(y_true, y_pred, average="macro",
                                            zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro",
                                   zero_division=0)),
        "fpr_macro": fpr_macro(y_true, y_pred, labels),
    }


def class_proportions(labels_series, support) -> np.ndarray:
    """Vetor de proporcoes de classe alinhado ao suporte 'support' (lista de classes)."""
    counts = Counter(labels_series)
    total = sum(counts.values())
    return np.array([counts.get(c, 0) / total if total else 0.0 for c in support])


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """KL(p || q) com suavizacao — p e a subamostra, q e a base integral."""
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Distancia L1 (soma dos desvios absolutos) entre duas distribuicoes."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.abs(p - q).sum())


def sample_from_frames(frames, cap: int, seed: int):
    """Espelha config.carregar_amostra_df: amostragem estratificada POR ARQUIVO,
    cota = cap // n_arquivos, semente fixa. Opera sobre frames ja lidos (uma
    leitura de disco para todos os tetos)."""
    from sklearn.model_selection import train_test_split
    n_files = len(frames)
    cota = max(1, cap // max(1, n_files))
    out = []
    for df in frames:
        if len(df) > cota:
            try:
                df, _ = train_test_split(
                    df, train_size=cota, stratify=df["Label"], random_state=seed)
            except Exception:
                df = df.sample(cota, random_state=seed)
        out.append(df)
    return pd.concat(out, ignore_index=True)


def balance_smote_enn(X, y, seed: int):
    """SMOTE-ENN fiel ao projeto (Investigacao 2): SMOTE k=5 'auto' + ENN n=3.
    Adapta k apenas quando a classe mais rara o exige (tetos pequenos), o que e
    registrado no 'status' para transparencia. Retorna (Xb, yb, status)."""
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import EditedNearestNeighbours
    from imblearn.combine import SMOTEENN

    nmin = min(Counter(y).values())
    k = 5
    status = "smote_enn(k=5)"
    if nmin <= k:
        k = max(1, nmin - 1)
        status = f"smote_enn(k={k}; nmin={nmin})"
    if nmin < 2:
        # SMOTE impossivel para classe com 1 amostra: estudo registra e treina cru.
        return np.ascontiguousarray(X, dtype=np.float32), y, f"sem_balanceamento(nmin={nmin})"

    smote = SMOTE(sampling_strategy="auto", k_neighbors=k, random_state=seed)
    enn = EditedNearestNeighbours(n_neighbors=3, kind_sel="all", n_jobs=4)
    sampler = SMOTEENN(smote=smote, enn=enn, random_state=seed)
    X = np.ascontiguousarray(X, dtype=np.float32)
    Xb, yb = sampler.fit_resample(X, y)
    return np.asarray(Xb, dtype=np.float32), yb, status


def train_eval_catboost(X_tr, y_tr, X_te, y_te, params, seed, threads) -> dict:
    """Treina CatBoost com 'params' e avalia em teste reservado. Retorna metricas + tempo."""
    from catboost import CatBoostClassifier
    model = CatBoostClassifier(
        iterations=params["iterations"],
        depth=params["depth"],
        learning_rate=params["learning_rate"],
        loss_function="MultiClass",
        random_seed=seed,
        thread_count=threads,
        verbose=False,
    )
    t0 = time.time()
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te).ravel().astype(int)
    elapsed = time.time() - t0
    m = compute_metrics(np.asarray(y_te, dtype=int), y_pred)
    m["tempo_s"] = round(elapsed, 1)
    return m


# ═══════════════════════════════════════════════════════════════════════════
# CARGA DA BASE INTEGRAL (uma unica leitura de disco)
# ═══════════════════════════════════════════════════════════════════════════
def read_full_frames():
    """Le os arquivos do dataset UMA vez, com o mesmo tratamento de colunas/
    rotulo de config.carregar_amostra_df. Retorna (frames, full_labels)."""
    from config import _dataset_csvs, _resolve_label_column

    csvs = _dataset_csvs()
    if not csvs:
        raise FileNotFoundError("Nenhum CSV do dataset encontrado (config._dataset_csvs).")

    frames = []
    for csv in csvs:
        df = pd.read_csv(csv, low_memory=False)
        if df.empty:
            continue
        df.columns = [c.strip() for c in df.columns]
        label_col = _resolve_label_column(df)
        if label_col is None:
            stem = csv.stem.lower()
            df["Label"] = "Benign" if ("benign" in stem or "normal" in stem) \
                else csv.stem.replace("_", " ").title()
        elif label_col != "Label":
            df = df.rename(columns={label_col: "Label"})
        frames.append(df)
        log.info(f"  lido {csv.name}: {len(df):,} linhas")

    if not frames:
        raise RuntimeError("Nenhum frame valido lido.")
    full_labels = pd.concat([f["Label"] for f in frames], ignore_index=True)
    return frames, full_labels


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main():
    from config import Config, RANDOM_SEED, _drop_meta_cols
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.model_selection import train_test_split

    threads = Config.CPU_CONFIG.get("intra_op_threads", -1) if hasattr(Config, "CPU_CONFIG") else -1
    feat_cols = list(Config.FEATURE_COLUMNS)
    params = (dict(iterations=Config.TREE_CONFIG["iterations"],
                   depth=Config.TREE_CONFIG["depth"],
                   learning_rate=Config.TREE_CONFIG["learning_rate"])
              if USE_CONFIG_TREE else CATBOOST_PARAMS_INV4)
    log.info(f"CatBoost fixo: {params}  | k atributos: {len(feat_cols)}")

    # diretorio de saida versionado
    base_out = Config.BASE_DIR / "Tests" / "sensibilidade_amostra"
    v = 1
    while (base_out / f"v{v}").exists():
        v += 1
    out = base_out / f"v{v}"
    (out / "tabelas").mkdir(parents=True, exist_ok=True)
    (out / "figuras").mkdir(parents=True, exist_ok=True)
    log.info(f"Saidas em: {out}")

    # ── leitura unica da base integral ──────────────────────────────────────
    log.info("Lendo base integral (uma vez)…")
    t_read = time.time()
    frames, full_labels = read_full_frames()
    support = sorted(full_labels.unique().tolist())
    p_full = class_proportions(full_labels, support)
    log.info(f"Base integral: {len(full_labels):,} linhas | {len(support)} classes "
             f"| leitura {time.time() - t_read:.1f}s")

    fidelidade_rows = []
    saturacao_rows = []
    dist_wide = {"classe": support,
                 "base_integral": [int(round(p * len(full_labels))) for p in p_full]}

    for cap in CAPS:
        log.info("=" * 70)
        log.info(f"TETO = {cap:,}")
        try:
            df = sample_from_frames(frames, cap, RANDOM_SEED)
            df = _drop_meta_cols(df)
            y_lab = df["Label"].astype(str).str.strip()
            counts = Counter(y_lab)
            p_cap = class_proportions(y_lab, support)

            # ── (A) FIDELIDADE ──────────────────────────────────────────────
            classe_min = min(counts, key=counts.get)
            fidelidade_rows.append({
                "cap": cap,
                "n_total": len(df),
                "n_classes": len(counts),
                "pct_benign": round(100 * counts.get("Benign", 0) / len(df), 2),
                "classe_min": classe_min,
                "n_classe_min": counts[classe_min],
                "kl_vs_integral": round(kl_divergence(p_cap, p_full), 6),
                "l1_vs_integral": round(l1_distance(p_cap, p_full), 6),
            })
            dist_wide[f"cap_{cap}"] = [counts.get(c, 0) for c in support]
            log.info(f"  (A) fidelidade: KL={fidelidade_rows[-1]['kl_vs_integral']} "
                     f"L1={fidelidade_rows[-1]['l1_vs_integral']} "
                     f"Benign={fidelidade_rows[-1]['pct_benign']}% "
                     f"min={classe_min}({counts[classe_min]})")

            # ── (B) SATURACAO (ajuste unico) ────────────────────────────────
            cols_ok = [c for c in feat_cols if c in df.columns]
            if len(cols_ok) < len(feat_cols):
                log.warning(f"  {len(feat_cols) - len(cols_ok)} atributo(s) k=23 "
                            f"ausente(s); usando {len(cols_ok)}.")
            X = df[cols_ok].replace([np.inf, -np.inf], 0).fillna(0).to_numpy("float32")
            y = LabelEncoder().fit_transform(y_lab)

            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_SEED)

            # escalonamento ajustado SO no treino (sem leakage)
            scaler = StandardScaler().fit(X_tr)
            X_tr = scaler.transform(X_tr).astype("float32")
            X_te = scaler.transform(X_te).astype("float32")

            # balanceamento SO no treino
            try:
                X_trb, y_trb, bal_status = balance_smote_enn(X_tr, y_tr, RANDOM_SEED)
            except Exception as exc:
                log.warning(f"  SMOTE-ENN falhou ({exc}); treinando sem balanceamento.")
                X_trb, y_trb, bal_status = X_tr, y_tr, f"falha:{type(exc).__name__}"

            m = train_eval_catboost(X_trb, y_trb, X_te, y_te, params,
                                    RANDOM_SEED, threads)
            saturacao_rows.append({
                "cap": cap,
                "n_total": len(df),
                "n_train": len(X_tr),
                "n_train_balanced": len(X_trb),
                "balanceamento": bal_status,
                "recall_macro": round(m["recall_macro"], 4),
                "mcc": round(m["mcc"], 4),
                "f1_macro": round(m["f1_macro"], 4),
                "fpr_macro": round(m["fpr_macro"], 4),
                "tempo_s": m["tempo_s"],
            })
            log.info(f"  (B) saturacao: recall={m['recall_macro']:.4f} "
                     f"MCC={m['mcc']:.4f} F1={m['f1_macro']:.4f} "
                     f"FPR={m['fpr_macro']:.4f} t={m['tempo_s']}s [{bal_status}]")

            del df, X, y, X_tr, X_te, X_trb, y_trb
            gc.collect()
        except Exception as exc:
            log.error(f"  TETO {cap:,} falhou: {exc}", exc_info=True)

    # ── persistencia ────────────────────────────────────────────────────────
    df_fid = pd.DataFrame(fidelidade_rows)
    df_sat = pd.DataFrame(saturacao_rows)
    df_dist = pd.DataFrame(dist_wide)
    df_fid.to_csv(out / "tabelas" / "fidelidade.csv", index=False)
    df_sat.to_csv(out / "tabelas" / "saturacao.csv", index=False)
    df_dist.to_csv(out / "tabelas" / "distribuicao_por_classe.csv", index=False)
    if not df_fid.empty:
        log.info(f"\nFIDELIDADE\n{df_fid.to_string(index=False)}")
    if not df_sat.empty:
        log.info(f"\nSATURACAO\n{df_sat.to_string(index=False)}")

    # ── figuras (tons de cinza, sem titulo embutido) ────────────────────────
    if not df_fid.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df_fid["cap"], df_fid["kl_vs_integral"], marker="o",
                color="0.15", label="KL(subamostra || integral)")
        ax.plot(df_fid["cap"], df_fid["l1_vs_integral"], marker="s",
                color="0.55", linestyle="--", label="L1")
        ax.set_xlabel("Teto de amostragem (n_amostras_max)")
        ax.set_ylabel("Divergencia vs. base integral")
        ax.legend(); ax.grid(True, color="0.85")
        fig.tight_layout()
        fig.savefig(out / "figuras" / "fidelidade.png", dpi=200)
        plt.close(fig)

    if not df_sat.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        for col, mk, sh in [("recall_macro", "o", "0.15"),
                            ("mcc", "s", "0.40"),
                            ("f1_macro", "^", "0.62")]:
            ax.plot(df_sat["cap"], df_sat[col], marker=mk, color=sh, label=col)
        ax.set_xlabel("Teto de amostragem (n_amostras_max)")
        ax.set_ylabel("Metrica (teste reservado)")
        ax.legend(); ax.grid(True, color="0.85")
        fig.tight_layout()
        fig.savefig(out / "figuras" / "saturacao.png", dpi=200)
        plt.close(fig)

    summary = {
        "caps": CAPS, "test_size": TEST_SIZE,
        "catboost_params": params, "k_features": len(feat_cols),
        "fidelidade": fidelidade_rows, "saturacao": saturacao_rows,
        "output_dir": str(out),
    }
    (out / "resumo.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    log.info(f"\nConcluido. Saidas em {out}")


if __name__ == "__main__":
    main()
