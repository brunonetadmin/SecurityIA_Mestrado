"""
analise_1_arquiteturas.py
=========================
Comparação de arquiteturas para detecção multiclasse de intrusões sobre
fluxos do CSE-CIC-IDS2018.

MODELOS AVALIADOS (7):
  Tabulares clássicos:
    1. RandomForest          — baseline forte (Breiman, 2001)
    2. ExtraTrees            — splits aleatórios (Geurts et al., 2006)
    3. XGBoost               — gradient boosting regularizado (Chen & Guestrin, 2016)
    4. CatBoost              — ordered boosting (Prokhorenkova et al., 2018)
  Redes densas:
    5. MLP+BatchNorm         — Ioffe & Szegedy (2015)
    6. ResNet Tabular        — conexões residuais (He et al., 2016)
  Controle (recorrente):
    7. BiLSTM com Atenção    — hipótese secundária

Critério primário: macro_recall. Secundários: MCC, F1-macro, FPR-macro, tempo.
Cada modelo em safe_run() — falha individual não interrompe a fila.
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
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Activation, Add,
    Bidirectional, LSTM, Reshape, Layer,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier

tf.keras.utils.set_random_seed(RANDOM_SEED)
apply_plot_style()

ANALISE_ID = 1
EPOCHS = 30
PATIENCE = 6
log = get_logger(ANALISE_ID, "analise_1")


# ═══════════════════════════════════════════════════════════════════════════
#   MODELOS TABULARES CLÁSSICOS
# ═══════════════════════════════════════════════════════════════════════════

def avaliar_random_forest(X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls):
    log.info("  fit RandomForest: 300 árvores, balanced_subsample")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=RANDOM_SEED,
        class_weight="balanced_subsample", max_depth=None, min_samples_leaf=2,
    )
    rf.fit(X_tr, y_tr)
    y_pred = rf.predict(X_te)
    m = metricas_completas(y_te, y_pred, n_cls)
    m.update(modelo="RandomForest", tempo_s=time.time() - t0, params=0)
    return m


def avaliar_extra_trees(X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls):
    log.info("  fit ExtraTrees: 400 árvores, splits totalmente aleatórios")
    t0 = time.time()
    et = ExtraTreesClassifier(
        n_estimators=400, n_jobs=-1, random_state=RANDOM_SEED,
        class_weight="balanced_subsample", max_depth=None, min_samples_leaf=2,
        bootstrap=False,
    )
    et.fit(X_tr, y_tr)
    y_pred = et.predict(X_te)
    m = metricas_completas(y_te, y_pred, n_cls)
    m.update(modelo="ExtraTrees", tempo_s=time.time() - t0, params=0)
    return m


def avaliar_xgboost(X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls):
    try:
        import xgboost as xgb
    except ImportError as e:
        raise RuntimeError(
            "xgboost não instalado. Execute: pip install xgboost"
        ) from e
    log.info("  fit XGBoost: 400 rounds, depth 8, eta 0.1, hist tree method")
    t0 = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.1,
        objective="multi:softprob", num_class=n_cls,
        tree_method="hist", n_jobs=-1, random_state=RANDOM_SEED,
        eval_metric="mlogloss", verbosity=0,
    )
    clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    y_pred = clf.predict(X_te)
    m = metricas_completas(y_te, y_pred, n_cls)
    m.update(modelo="XGBoost", tempo_s=time.time() - t0, params=0)
    return m


def avaliar_catboost(X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls):
    try:
        from catboost import CatBoostClassifier
    except ImportError as e:
        raise RuntimeError(
            "catboost não instalado. Execute: pip install catboost"
        ) from e
    log.info("  fit CatBoost: 500 iterations, depth 8, ordered boosting")
    t0 = time.time()
    clf = CatBoostClassifier(
        iterations=500, depth=8, learning_rate=0.1,
        loss_function="MultiClass", thread_count=-1,
        random_seed=RANDOM_SEED, verbose=False, allow_writing_files=False,
        auto_class_weights="Balanced",
    )
    clf.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    y_pred = clf.predict(X_te).ravel().astype(int)
    m = metricas_completas(y_te, y_pred, n_cls)
    m.update(modelo="CatBoost", tempo_s=time.time() - t0, params=0)
    return m


# ═══════════════════════════════════════════════════════════════════════════
#   REDES DENSAS
# ═══════════════════════════════════════════════════════════════════════════

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


def build_resnet_tabular(n_feat, n_cls, blocks=4, hidden=256):
    inp = Input(shape=(n_feat,))
    x = Dense(hidden)(inp); x = BatchNormalization()(x); x = Activation("relu")(x)
    for _ in range(blocks):
        s = x
        h = Dense(hidden)(x);  h = BatchNormalization()(h); h = Activation("relu")(h); h = Dropout(0.2)(h)
        h = Dense(hidden)(h);  h = BatchNormalization()(h)
        x = Add()([s, h]); x = Activation("relu")(x)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="ResNet_Tabular")
    m.compile(optimizer=Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


class BahdanauAttention(Layer):
    def __init__(self, units=64, **kw):
        super().__init__(**kw); self.units = units
    def build(self, input_shape):
        self.W = Dense(self.units); self.U = Dense(self.units); self.V = Dense(1)
        super().build(input_shape)
    def call(self, x):
        score = self.V(tf.nn.tanh(self.W(x) + self.U(x)))
        a = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(a * x, axis=1)


def build_bilstm_atencao(n_feat, n_cls, units=128):
    inp = Input(shape=(n_feat,))
    x = Reshape((n_feat, 1))(inp)
    x = Bidirectional(LSTM(units, return_sequences=True, dropout=0.3))(x)
    x = Bidirectional(LSTM(units // 2, return_sequences=True, dropout=0.3))(x)
    x = BahdanauAttention(64)(x)
    x = Dense(64, activation="relu")(x); x = Dropout(0.3)(x)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="BiLSTM_Atencao")
    m.compile(optimizer=Adam(1e-3),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def avaliar_rede(nome_modelo, build_fn, X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls):
    K.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    m = build_fn(X_tr.shape[1], n_cls)
    log.info(f"  fit {nome_modelo}: epochs<={EPOCHS} batch={BATCH_SIZE} params={m.count_params():,}")
    t0 = time.time()
    cb = [
        EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, monitor="val_loss"),
        EpochLogger(log, prefix=f"{nome_modelo} "),
    ]
    m.fit(X_tr, y_tr, validation_data=(X_val, y_val),
          epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=0)
    y_pred = np.argmax(m.predict(X_te, batch_size=BATCH_SIZE, verbose=0), axis=1)
    out = metricas_completas(y_te, y_pred, n_cls)
    out.update(modelo=nome_modelo, tempo_s=time.time() - t0, params=int(m.count_params()))
    return out


# ═══════════════════════════════════════════════════════════════════════════
#   PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def executar(dataset_disponivel: bool = True) -> None:
    log.info("ANÁLISE 1 — Arquiteturas (7 modelos sobre dados reais)")

    dataset = safe_run(log, "carregar_dataset_real", carregar_dataset_real)
    if dataset is None:
        log.error("Sem dataset real. Abortando."); return
    # carregar_dataset_real() retorna (X, y) ou (X, y, label_encoder)
    if isinstance(dataset, tuple) and len(dataset) >= 2:
        Xfull, yfull = dataset[0], dataset[1]
    else:
        log.error(f"Formato inesperado de dataset: {type(dataset)}"); return
    Xfull = np.asarray(Xfull, dtype=np.float32)
    yfull = np.asarray(yfull).astype(np.int64).ravel()
    n_cls = int(np.max(yfull) + 1)
    log.info(f"Dataset: {Xfull.shape[0]:,} amostras × {Xfull.shape[1]} features × {n_cls} classes")

    X_tr_raw, X_val_raw, X_te_raw, y_tr, y_val, y_te = stratified_split_3way(
        Xfull, yfull, val_frac=0.15, test_frac=0.15, seed=RANDOM_SEED,
    )
    X_tr, X_val, X_te = fit_scaler_no_leakage(X_tr_raw, X_val_raw, X_te_raw)
    log.info(f"Split: treino={len(y_tr):,} val={len(y_val):,} teste={len(y_te):,}")

    resultados = []
    avaliacoes = [
        ("RandomForest",   lambda: avaliar_random_forest(X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls)),
        ("ExtraTrees",     lambda: avaliar_extra_trees  (X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls)),
        ("XGBoost",        lambda: avaliar_xgboost      (X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls)),
        ("CatBoost",       lambda: avaliar_catboost     (X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls)),
        ("MLP_BN",         lambda: avaliar_rede("MLP_BN",         build_mlp_bn,         X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls)),
        ("ResNet_Tabular", lambda: avaliar_rede("ResNet_Tabular", build_resnet_tabular, X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls)),
        ("BiLSTM_Atencao_Tabular (hipótese secundária)",
                           lambda: avaliar_rede("BiLSTM_Atencao", build_bilstm_atencao, X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls)),
    ]
    for nome, fn in avaliacoes:
        out = safe_run(log, nome, fn)
        if out is not None:
            resultados.append(out)

    if not resultados:
        log.error("Nenhum modelo concluído com sucesso."); return

    df = pd.DataFrame(resultados)
    csv_path = tab_path(ANALISE_ID, "metricas_arquiteturas")
    safe_run(log, "salvar CSV", lambda: df.to_csv(csv_path, index=False))
    log.info(f"Tabela: {csv_path}")
    safe_run(log, "plot_comparativo", lambda: _plot(df))

    vencedor = df.sort_values("recall_macro", ascending=False).iloc[0]
    log.info("=" * 62)
    log.info(
        f"VENCEDOR (recall_macro): {vencedor['modelo']}  "
        f"recall={vencedor['recall_macro']:.4f} mcc={vencedor['mcc']:.4f} "
        f"f1_macro={vencedor['f1_macro']:.4f} fpr={vencedor['fpr_macro']:.4f}"
    )
    log.info("=" * 62)


def _plot(df: pd.DataFrame) -> None:
    """Painel 2x2: Recall-macro, MCC, F1-macro, FPR-macro.
    Sem título embutido (segue caption do LaTeX). Paleta neutra para impressão."""
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    metricas = [
        ("recall_macro", "Recall-macro", ax[0, 0]),
        ("mcc",          "MCC",          ax[0, 1]),
        ("f1_macro",     "F1-macro",     ax[1, 0]),
        ("fpr_macro",    "FPR-macro",    ax[1, 1]),
    ]
    cores = sns.color_palette("Greys", n_colors=len(df) + 2)[2:]
    for col, titulo, a in metricas:
        sns.barplot(data=df, x="modelo", y=col, ax=a, palette=cores, edgecolor="black")
        a.set_title(titulo, fontsize=13, fontweight="bold")
        a.set_xlabel(""); a.set_ylabel(col)
        for tick in a.get_xticklabels():
            tick.set_rotation(35); tick.set_ha("right")
        for p in a.patches:
            v = p.get_height()
            a.annotate(f"{v:.3f}", (p.get_x() + p.get_width()/2, v),
                       ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(fig_path(ANALISE_ID, "comparativo_arquiteturas"), dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    executar()
