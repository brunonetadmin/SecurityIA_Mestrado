"""
analise_4_otimizacao_validacao.py
=================================
Compara estratégias de otimização sobre DADOS REAIS (CSE-CIC-IDS2018),
usando MLP+BN fixo e mesmo split estratificado.

Otimizadores: Adam, AdamW, RMSprop, SGD-Momentum
Schedulers:   none, ReduceLROnPlateau, CosineDecay
Validação:    via val_loss em conjunto de validação separado (sem leakage)

Critério primário: macro_recall + MCC. Tempo de convergência (épocas) e
estabilidade (variância de val_loss nas últimas 5 épocas) reportados.

Cada combinação em safe_run() — falhas individuais não interrompem.
"""
import os, sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

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
from tensorflow.keras.optimizers import Adam, AdamW, SGD, RMSprop
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

tf.keras.utils.set_random_seed(RANDOM_SEED)
apply_plot_style()

ANALISE_ID = 4
EPOCHS = 30
PATIENCE = 6
log = get_logger(ANALISE_ID, "analise_4")


def build_mlp_bn(n_feat, n_cls, optimizer):
    inp = Input(shape=(n_feat,))
    x = Dense(256)(inp); x = BatchNormalization()(x); x = Activation("relu")(x); x = Dropout(0.3)(x)
    x = Dense(128)(x);   x = BatchNormalization()(x); x = Activation("relu")(x); x = Dropout(0.3)(x)
    x = Dense(64)(x);    x = BatchNormalization()(x); x = Activation("relu")(x); x = Dropout(0.2)(x)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="MLP_BN")
    m.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def opt_adam(scheduler):
    if scheduler == "cosine":
        return Adam(learning_rate=CosineDecay(1e-3, decay_steps=EPOCHS * 10))
    return Adam(learning_rate=1e-3)

def opt_adamw(scheduler):
    if scheduler == "cosine":
        return AdamW(learning_rate=CosineDecay(1e-3, decay_steps=EPOCHS * 10),
                     weight_decay=1e-4)
    return AdamW(learning_rate=1e-3, weight_decay=1e-4)

def opt_rmsprop(scheduler):
    if scheduler == "cosine":
        return RMSprop(learning_rate=CosineDecay(1e-3, decay_steps=EPOCHS * 10))
    return RMSprop(learning_rate=1e-3)

def opt_sgd(scheduler):
    if scheduler == "cosine":
        return SGD(learning_rate=CosineDecay(1e-2, decay_steps=EPOCHS * 10), momentum=0.9)
    return SGD(learning_rate=1e-2, momentum=0.9)


CONFIGS = [
    ("Adam_none",            opt_adam,    "none"),
    ("Adam_ReduceLR",        opt_adam,    "reduce"),
    ("Adam_Cosine",          opt_adam,    "cosine"),
    ("AdamW_none",           opt_adamw,   "none"),
    ("AdamW_ReduceLR",       opt_adamw,   "reduce"),
    ("AdamW_Cosine",         opt_adamw,   "cosine"),
    ("RMSprop_none",         opt_rmsprop, "none"),
    ("RMSprop_ReduceLR",     opt_rmsprop, "reduce"),
    ("SGD_Momentum_none",    opt_sgd,     "none"),
    ("SGD_Momentum_ReduceLR", opt_sgd,    "reduce"),
]


def avaliar(label, opt_factory, scheduler,
            X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls):
    K.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    t0 = time.time()
    m = build_mlp_bn(X_tr.shape[1], n_cls, opt_factory(scheduler))

    cb = [
        EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_loss"),
        EpochLogger(log, prefix=f"{label} "),
    ]
    if scheduler == "reduce":
        cb.append(ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7,
                                     monitor="val_loss"))

    log.info(f"  fit {label}: epochs<={EPOCHS} batch={BATCH_SIZE}")
    h = m.fit(X_tr, y_tr, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=cb)

    yp = np.argmax(m.predict(X_te, verbose=0, batch_size=BATCH_SIZE), axis=1)
    r = metricas_completas(y_te, yp, n_classes=n_cls)

    val_losses = h.history.get("val_loss", [])
    estabilidade = float(np.std(val_losses[-5:])) if len(val_losses) >= 5 else float("nan")

    r.update({
        "config": label,
        "epocas_executadas": int(len(val_losses)),
        "estabilidade_val_loss": estabilidade,
        "tempo_s": time.time() - t0,
    })
    return r


def plot_comparativo(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metr = [("recall_macro", "Recall-macro (PRIMÁRIO)"),
            ("mcc", "MCC"),
            ("f1_macro", "F1-macro"),
            ("epocas_executadas", "Épocas até convergência")]
    for ax, (col, lbl) in zip(axes.flat, metr):
        if col not in df.columns: continue
        sns.barplot(data=df, x="config", y=col, ax=ax, palette="cubehelix")
        ax.set_title(lbl); ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=40)
        for c in ax.containers:
            fmt = "%.3f" if col != "epocas_executadas" else "%d"
            ax.bar_label(c, fmt=fmt, fontsize=7)
    fig.suptitle("Análise 4 — Otimização e Validação (dados reais)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = fig_path(ANALISE_ID, "comparativo_otimizacao")
    fig.savefig(p, dpi=300); plt.close(fig)
    return p


def main():
    rel = Relatorio(ANALISE_ID)
    log.info("ANÁLISE 4 — Otimização e Validação (dados reais)")

    if not verificar_dataset(interativo=False):
        log.error("Dataset real ausente — análise abortada.")
        return

    ok, dados = safe_run(log, "carregar_dataset_real",
                         carregar_dataset_real,
                         n_amostras_max=80_000, select_features=False)
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

    res = []
    for label, opt_factory, sched in CONFIGS:
        ok, r = safe_run(log, label,
                         avaliar, label, opt_factory, sched,
                         X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls)
        if ok and r:
            res.append(r)
            log.info(f"{label}: recall={r['recall_macro']:.4f} "
                     f"mcc={r['mcc']:.4f} f1={r['f1_macro']:.4f} "
                     f"epocas={r['epocas_executadas']} "
                     f"estab={r['estabilidade_val_loss']:.4f}")

    if not res:
        log.error("Nenhuma configuração produziu resultado válido.")
        return

    df = pd.DataFrame(res)
    csv_path = tab_path(ANALISE_ID, "metricas_otimizacao")
    safe_run(log, "salvar CSV", df.to_csv, csv_path, index=False)
    log.info(f"Tabela: {csv_path}")

    safe_run(log, "plot_comparativo", plot_comparativo, df)

    venc = max(res, key=lambda r: (r["recall_macro"], r["mcc"], -r["epocas_executadas"]))
    log.info("=" * 62)
    log.info(f"VENCEDOR: {venc['config']}  "
             f"recall={venc['recall_macro']:.4f} mcc={venc['mcc']:.4f} "
             f"f1={venc['f1_macro']:.4f} épocas={venc['epocas_executadas']}")
    log.info("=" * 62)

    try:
        rel.secao("Resumo dos Resultados")
        rel.tabela_df(df, "Métricas das configurações de otimização")
        rel.secao("Veredito")
        rel.texto(f"Configuração vencedora: **{venc['config']}**.\n\n"
                  f"recall_macro = {venc['recall_macro']:.4f}, "
                  f"MCC = {venc['mcc']:.4f}, "
                  f"épocas = {venc['epocas_executadas']}.")
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
