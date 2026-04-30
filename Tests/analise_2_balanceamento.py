"""
analise_2_balanceamento.py
==========================
Compara 5 estratégias de tratamento de desbalanceamento, sobre dados reais
(CSE-CIC-IDS2018), com MLP+BatchNorm fixo (vencedor esperado da A1).

Estratégias:
  1. Sem_Tratamento       — baseline (entropia cruzada padrão)
  2. ClassWeight_Balanced — sklearn class_weight='balanced' no fit
  3. FocalLoss            — Focal Loss pura (Lin et al., 2017)
  4. CB_FocalLoss         — Class-Balanced Focal Loss (Cui et al., 2019)
  5. Undersample_Benign   — undersampling do Benign para 5x a 2ª classe maior

Sobre SMOTE/Borderline/SMOTE-ENN: REMOVIDOS desta análise. A iteração
anterior (29/04/2026) demonstrou empiricamente que oversampling sintético
em alta dimensionalidade prejudica o desempenho neste dataset.

Critério: macro_recall (PRIMÁRIO) e MCC.
Cada estratégia em safe_run() — falhas individuais não interrompem.
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
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Activation,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

tf.keras.utils.set_random_seed(RANDOM_SEED)
apply_plot_style()

ANALISE_ID = 2
EPOCHS = 25
PATIENCE = 6
log = get_logger(ANALISE_ID, "analise_2")


# ─── Modelo fixo (MLP+BN — vencedor esperado da A1) ────────────────────────

def build_mlp_bn(n_feat, n_cls, loss):
    inp = Input(shape=(n_feat,))
    x = Dense(256)(inp); x = BatchNormalization()(x); x = Activation("relu")(x); x = Dropout(0.3)(x)
    x = Dense(128)(x);   x = BatchNormalization()(x); x = Activation("relu")(x); x = Dropout(0.3)(x)
    x = Dense(64)(x);    x = BatchNormalization()(x); x = Activation("relu")(x); x = Dropout(0.2)(x)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="MLP_BN")
    m.compile(optimizer=Adam(1e-3), loss=loss, metrics=["accuracy"])
    return m


# ─── Funções de perda ──────────────────────────────────────────────────────

def focal_loss_pura(gamma=2.0):
    """Focal Loss pura (Lin et al., 2017). Sem reponderação por classe."""
    def fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p = tf.gather(y_pred, y_true, axis=1, batch_dims=1)
        return tf.reduce_mean(tf.pow(1.0 - p, gamma) * (-tf.math.log(p)))
    return fn


def cb_focal_loss(class_counts, gamma=2.0, beta=0.9999):
    """Class-Balanced Focal Loss (Cui et al., 2019).
    Pesos baseados no number of effective samples."""
    cc = np.maximum(np.asarray(class_counts, dtype=np.float64), 1.0)
    n_eff = (1.0 - np.power(beta, cc)) / (1.0 - beta)
    w = (1.0 - beta) / np.maximum(n_eff, 1e-12)
    w = w / w.sum() * len(w)
    wt = tf.constant(w, dtype=tf.float32)
    def fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p = tf.gather(y_pred, y_true, axis=1, batch_dims=1)
        cw = tf.gather(wt, y_true)
        return tf.reduce_mean(cw * tf.pow(1.0 - p, gamma) * (-tf.math.log(p)))
    return fn


# ─── Avaliação genérica ────────────────────────────────────────────────────

def avaliar(nome, X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls,
            loss_fn, class_weight=None):
    K.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    t0 = time.time()
    m = build_mlp_bn(X_tr.shape[1], n_cls, loss_fn)
    log.info(f"  fit {nome}: epochs<={EPOCHS} batch={BATCH_SIZE} "
             f"class_weight={'sim' if class_weight else 'nao'}")
    cb = [
        EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, monitor="val_loss"),
        EpochLogger(log, prefix=f"{nome} "),
    ]
    m.fit(X_tr, y_tr, validation_data=(X_val, y_val),
          epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=cb,
          class_weight=class_weight)
    yp = np.argmax(m.predict(X_te, verbose=0, batch_size=BATCH_SIZE), axis=1)
    r = metricas_completas(y_te, yp, n_classes=n_cls)
    r.update({"estrategia": nome, "tempo_s": time.time() - t0})
    return r


# ─── Estratégias ───────────────────────────────────────────────────────────

def estrat_sem_tratamento(X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls):
    return avaliar("Sem_Tratamento", X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls,
                   loss_fn="sparse_categorical_crossentropy")


def estrat_class_weight(X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls):
    classes = np.unique(y_tr)
    weights = compute_class_weight("balanced", classes=classes, y=y_tr)
    cw = {int(c): float(w) for c, w in zip(classes, weights)}
    return avaliar("ClassWeight_Balanced", X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls,
                   loss_fn="sparse_categorical_crossentropy", class_weight=cw)


def estrat_focal_pura(X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls):
    return avaliar("FocalLoss", X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls,
                   loss_fn=focal_loss_pura(gamma=2.0))


def estrat_cb_focal(X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls):
    cc = np.bincount(y_tr.astype(int), minlength=n_cls)
    return avaliar("CB_FocalLoss", X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls,
                   loss_fn=cb_focal_loss(cc, gamma=2.0, beta=0.9999))


def estrat_undersample_benign(X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls):
    """Undersample classe majoritária (Benign assumido como classe 0 ou maior)."""
    dist = Counter(y_tr)
    maj = max(dist, key=dist.get)
    sorted_counts = sorted(dist.values(), reverse=True)
    n_2nd = sorted_counts[1] if len(sorted_counts) > 1 else sorted_counts[0]
    target = max(n_2nd * 5, n_2nd)
    target = min(target, dist[maj])
    rng = np.random.default_rng(RANDOM_SEED)
    keep_idx = []
    for c, n in dist.items():
        idx_c = np.where(y_tr == c)[0]
        if c == maj and n > target:
            idx_c = rng.choice(idx_c, size=target, replace=False)
        keep_idx.append(idx_c)
    keep = np.concatenate(keep_idx)
    rng.shuffle(keep)
    X_tr_u = X_tr[keep]; y_tr_u = y_tr[keep]
    log.info(f"  Undersample: {len(y_tr):,} → {len(y_tr_u):,} amostras  "
             f"(majoritária classe {maj}: {dist[maj]:,} → {target:,})")
    return avaliar("Undersample_Benign", X_tr_u, X_val, X_te, y_tr_u, y_val, y_te, n_cls,
                   loss_fn="sparse_categorical_crossentropy")


ESTRATEGIAS = [
    ("Sem_Tratamento",      estrat_sem_tratamento),
    ("ClassWeight_Balanced", estrat_class_weight),
    ("FocalLoss",            estrat_focal_pura),
    ("CB_FocalLoss",         estrat_cb_focal),
    ("Undersample_Benign",   estrat_undersample_benign),
]


# ─── Plot ──────────────────────────────────────────────────────────────────

def plot_comparativo(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metr = [("recall_macro", "Recall-macro (PRIMÁRIO)"),
            ("mcc", "MCC"),
            ("f1_macro", "F1-macro"),
            ("fpr_macro", "FPR-macro (menor é melhor)")]
    for ax, (col, lbl) in zip(axes.flat, metr):
        if col not in df.columns: continue
        sns.barplot(data=df, x="estrategia", y=col, ax=ax, palette="rocket")
        ax.set_title(lbl); ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f", fontsize=8)
    fig.suptitle("Análise 2 — Tratamento de Desbalanceamento via Perda",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = fig_path(ANALISE_ID, "comparativo_balanceamento")
    fig.savefig(p, dpi=300); plt.close(fig)
    return p


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    rel = Relatorio(ANALISE_ID)
    log.info("ANÁLISE 2 — Tratamento de Desbalanceamento via Perda (5 estratégias)")

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
    log.info(f"Dataset: {X_raw.shape[0]:,} amostras × {X_raw.shape[1]} features × {n_cls} classes")

    X_tr_r, X_val_r, X_te_r, y_tr, y_val, y_te = stratified_split_3way(
        X_raw, y, val_frac=0.15, test_frac=0.15, seed=RANDOM_SEED, logger=log,
    )
    X_tr, X_val, X_te, _ = fit_scaler_no_leakage(X_tr_r, X_val_r, X_te_r)
    log.info(f"Distribuição treino: {dict(sorted(Counter(y_tr).items()))}")

    res = []
    for nome, fn in ESTRATEGIAS:
        ok, r = safe_run(log, nome, fn,
                         X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls)
        if ok and r:
            res.append(r)
            log.info(f"{nome}: recall={r['recall_macro']:.4f} mcc={r['mcc']:.4f} "
                     f"f1={r['f1_macro']:.4f} fpr={r['fpr_macro']:.4f}")

    if not res:
        log.error("Nenhuma estratégia produziu resultado válido.")
        return

    df = pd.DataFrame(res)
    csv_path = tab_path(ANALISE_ID, "metricas_balanceamento")
    safe_run(log, "salvar CSV", df.to_csv, csv_path, index=False)
    log.info(f"Tabela: {csv_path}")

    safe_run(log, "plot_comparativo", plot_comparativo, df)

    venc = max(res, key=lambda r: r["recall_macro"])
    log.info("=" * 62)
    log.info(f"VENCEDOR (recall_macro): {venc['estrategia']}  "
             f"recall={venc['recall_macro']:.4f} mcc={venc['mcc']:.4f}")
    log.info("=" * 62)

    try:
        rel.secao("Resumo dos Resultados")
        rel.tabela_df(df, "Métricas das estratégias avaliadas")
        rel.secao("Veredito")
        rel.texto(f"Vencedor por recall_macro: **{venc['estrategia']}**.")
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
