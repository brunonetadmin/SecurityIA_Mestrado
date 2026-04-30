"""
analise_3_teoria_informacao.py
==============================
Compara 5 métodos de seleção de features sobre TODAS as features numéricas
brutas do CSE-CIC-IDS2018 (~80 atributos), usando MLP+BN com perda padrão
(decisão da Análise 2: tratamento de desbalanceamento delegado à perda).

Métodos:
  1. Information_Gain      — Shannon (1948)
  2. Mutual_Information    — sklearn mutual_info_classif (Peng et al., 2005)
  3. ANOVA_F               — F-test linear (controle clássico)
  4. RF_Feature_Importance — impureza Gini sobre Random Forest (Breiman, 2001)
  5. IG_MI_60_40           — combinação ponderada (estratégia oficial do IDS)

Para cada método, varremos uma grade de k = [10, 15, 23, 32, 48, 'all'].

Critério: macro_recall (PRIMÁRIO) e MCC.
Cada (método, k) em safe_run() — falhas individuais não interrompem.
"""
import os, sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    RANDOM_SEED, BATCH_SIZE,
    fig_path, tab_path, Relatorio, apply_plot_style,
    verificar_dataset, carregar_dataset_real,
)
from _test_logging import (
    get_logger, log_exception, safe_run, EpochLogger, silence_tensorflow,
    stratified_split_3way, fit_scaler_no_leakage, metricas_completas,
)

silence_tensorflow()

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    mutual_info_classif, f_classif, SelectKBest,
)
from scipy.stats import entropy as scipy_entropy

tf.keras.utils.set_random_seed(RANDOM_SEED)
apply_plot_style()

ANALISE_ID = 3
EPOCHS = 18
PATIENCE = 5
K_GRID = [10, 15, 23, 32, 48, "all"]
log = get_logger(ANALISE_ID, "analise_3")


# ─── Modelo fixo (MLP+BN) ──────────────────────────────────────────────────

def build_mlp_bn(n_feat, n_cls):
    inp = Input(shape=(n_feat,))
    x = Dense(256)(inp); x = BatchNormalization()(x); x = Activation("relu")(x); x = Dropout(0.3)(x)
    x = Dense(128)(x);   x = BatchNormalization()(x); x = Activation("relu")(x); x = Dropout(0.3)(x)
    x = Dense(64)(x);    x = BatchNormalization()(x); x = Activation("relu")(x); x = Dropout(0.2)(x)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="MLP_BN")
    m.compile(optimizer=Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


# ─── Métodos de seleção ────────────────────────────────────────────────────

def information_gain(X, y, n_bins=10):
    counts = np.bincount(y.astype(int))
    p = counts[counts > 0] / counts.sum()
    h_y = scipy_entropy(p, base=2)
    ig = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        col = X[:, i]
        if col.max() == col.min():
            continue
        bins = np.linspace(col.min(), col.max(), n_bins + 1)
        digi = np.clip(np.digitize(col, bins[:-1]) - 1, 0, n_bins - 1)
        h_c = 0.0
        for b in range(n_bins):
            mask = digi == b
            nb = mask.sum()
            if nb == 0: continue
            yb = y[mask].astype(int)
            pb_y = np.bincount(yb, minlength=len(counts)) / nb
            pb_y = pb_y[pb_y > 0]
            h_c += (nb / len(y)) * scipy_entropy(pb_y, base=2)
        ig[i] = h_y - h_c
    return ig


def score_ig(X, y):
    return information_gain(X, y)


def score_mi(X, y):
    # Subamostra para acelerar
    n_mi = min(20_000, len(X))
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X), size=n_mi, replace=False)
    return mutual_info_classif(X[idx], y[idx], random_state=RANDOM_SEED)


def score_anova(X, y):
    sk = SelectKBest(f_classif, k="all").fit(X, y)
    return np.nan_to_num(sk.scores_, nan=0.0)


def score_rf_importance(X, y):
    n_sub = min(30_000, len(X))
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X), size=n_sub, replace=False)
    rf = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=RANDOM_SEED,
        class_weight="balanced_subsample",
    )
    rf.fit(X[idx], y[idx])
    return rf.feature_importances_


def score_ig_mi_60_40(X, y):
    ig = information_gain(X, y)
    mi = score_mi(X, y)
    eps = 1e-9
    ig_n = (ig - ig.min()) / (ig.max() - ig.min() + eps)
    mi_n = (mi - mi.min()) / (mi.max() - mi.min() + eps)
    return 0.6 * ig_n + 0.4 * mi_n


METODOS = {
    "Information_Gain":      score_ig,
    "Mutual_Information":    score_mi,
    "ANOVA_F":               score_anova,
    "RF_Feature_Importance": score_rf_importance,
    "IG_MI_60_40":           score_ig_mi_60_40,
}


def selecionar_top_k(scores, k, n_features):
    if isinstance(k, str) and k == "all":
        return np.arange(n_features)
    k = min(int(k), n_features)
    return np.argsort(scores)[::-1][:k]


# ─── Avaliação ─────────────────────────────────────────────────────────────

def avaliar_combinacao(metodo_nome, k, scores, X_tr, X_val, X_te,
                        y_tr, y_val, y_te, n_cls):
    K.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    t0 = time.time()
    idx = selecionar_top_k(scores, k, X_tr.shape[1])
    Xtr_s = X_tr[:, idx]
    Xv_s  = X_val[:, idx]
    Xte_s = X_te[:, idx]
    log.info(f"  {metodo_nome} k={k}: {len(idx)} features de {X_tr.shape[1]}")

    m = build_mlp_bn(Xtr_s.shape[1], n_cls)
    cb = [
        EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, monitor="val_loss"),
        EpochLogger(log, prefix=f"{metodo_nome}_k{k} "),
    ]
    m.fit(Xtr_s, y_tr, validation_data=(Xv_s, y_val),
          epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=cb)
    yp = np.argmax(m.predict(Xte_s, verbose=0, batch_size=BATCH_SIZE), axis=1)
    r = metricas_completas(y_te, yp, n_classes=n_cls)
    r.update({
        "metodo": metodo_nome,
        "k": k if isinstance(k, str) else int(k),
        "k_real": int(len(idx)),
        "tempo_s": time.time() - t0,
    })
    return r


# ─── Plot ──────────────────────────────────────────────────────────────────

def plot_comparativo(df):
    # Normaliza coluna k para valor numérico para plot ordenado
    df = df.copy()
    df["k_plot"] = df["k_real"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metr = [("recall_macro", "Recall-macro (PRIMÁRIO)"),
            ("mcc", "MCC"),
            ("f1_macro", "F1-macro"),
            ("fpr_macro", "FPR-macro")]
    for ax, (col, lbl) in zip(axes.flat, metr):
        if col not in df.columns: continue
        for metodo in df["metodo"].unique():
            sub = df[df["metodo"] == metodo].sort_values("k_plot")
            ax.plot(sub["k_plot"], sub[col], marker="o", label=metodo)
        ax.set_title(lbl); ax.set_xlabel("k (features selecionadas)")
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle("Análise 3 — Seleção de Features sobre Universo Completo",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = fig_path(ANALISE_ID, "comparativo_selecao")
    fig.savefig(p, dpi=300); plt.close(fig)
    return p


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    rel = Relatorio(ANALISE_ID)
    log.info("ANÁLISE 3 — Seleção de Features (5 métodos × 6 valores de k)")

    if not verificar_dataset(interativo=False):
        log.error("Dataset real ausente — análise abortada.")
        return

    ok, dados = safe_run(log, "carregar_dataset_real",
                         carregar_dataset_real,
                         n_amostras_max=120_000, select_features=False)
    if not ok or dados is None:
        log.error("Falha ao carregar dataset.")
        return
    X_raw, y, _ = dados
    n_cls = int(np.max(y) + 1)
    n_feat_full = X_raw.shape[1]
    log.info(f"Universo de features: {n_feat_full} atributos numéricos")
    log.info(f"Dataset: {X_raw.shape[0]:,} amostras × {n_cls} classes")

    X_tr_r, X_val_r, X_te_r, y_tr, y_val, y_te = stratified_split_3way(
        X_raw, y, val_frac=0.15, test_frac=0.15, seed=RANDOM_SEED, logger=log,
    )
    X_tr, X_val, X_te, _ = fit_scaler_no_leakage(X_tr_r, X_val_r, X_te_r)

    # Calcula scores de cada método UMA VEZ (apenas no treino — evita leakage)
    scores_cache = {}
    for metodo_nome, score_fn in METODOS.items():
        ok, scores = safe_run(log, f"calcular scores ({metodo_nome})",
                               score_fn, X_tr, y_tr)
        if ok and scores is not None:
            scores_cache[metodo_nome] = scores
        else:
            log.warning(f"Método {metodo_nome} indisponível — pulando.")

    # Avalia cada combinação (método, k)
    res = []
    for metodo_nome, scores in scores_cache.items():
        for k in K_GRID:
            label = f"{metodo_nome}_k{k}"
            ok, r = safe_run(log, label,
                              avaliar_combinacao,
                              metodo_nome, k, scores,
                              X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls)
            if ok and r:
                res.append(r)
                log.info(f"  {label}: recall={r['recall_macro']:.4f} "
                         f"mcc={r['mcc']:.4f} f1={r['f1_macro']:.4f}")

    if not res:
        log.error("Nenhuma combinação produziu resultado válido.")
        return

    df = pd.DataFrame(res)
    csv_path = tab_path(ANALISE_ID, "metricas_selecao")
    safe_run(log, "salvar CSV", df.to_csv, csv_path, index=False)
    log.info(f"Tabela: {csv_path}")

    safe_run(log, "plot_comparativo", plot_comparativo, df)

    # Veredito: melhor combinação por recall_macro
    venc = max(res, key=lambda r: r["recall_macro"])
    log.info("=" * 62)
    log.info(f"MELHOR COMBINAÇÃO: {venc['metodo']} k={venc['k']}  "
             f"recall={venc['recall_macro']:.4f} mcc={venc['mcc']:.4f} "
             f"f1={venc['f1_macro']:.4f} fpr={venc['fpr_macro']:.4f}")
    # Também mostra o IG_MI_60_40 (estratégia oficial) para comparação direta
    igmi = [r for r in res if r["metodo"] == "IG_MI_60_40"]
    if igmi:
        igmi_best = max(igmi, key=lambda r: r["recall_macro"])
        log.info(f"IG_MI_60_40 (oficial) melhor: k={igmi_best['k']}  "
                 f"recall={igmi_best['recall_macro']:.4f} "
                 f"mcc={igmi_best['mcc']:.4f}")
    log.info("=" * 62)

    try:
        rel.secao("Resumo dos Resultados")
        rel.tabela_df(df, "Métricas para cada (método, k)")
        rel.secao("Veredito")
        rel.texto(f"Melhor combinação por recall_macro: **{venc['metodo']} (k={venc['k']})**.")
        rel.salvar()
    except Exception as e:
        log_exception(log, "rel.salvar", e)


def executar(dataset_disponivel: bool = True, **kwargs) -> None:
    try:
        main()
    except Exception as e:
        log_exception(log, "executar", e)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(log, "main", e)
        sys.exit(0)
