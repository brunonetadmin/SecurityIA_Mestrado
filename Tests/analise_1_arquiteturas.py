"""
analise_1_arquiteturas.py
=========================
Compara 5 arquiteturas para classificação de fluxos tabulares de rede,
sobre dados reais (CSE-CIC-IDS2018) com TODAS as features e split SEM
leakage.

Modelos:
  1. RandomForest         — baseline tabular forte (Breiman, 2001)
  2. XGBoost              — gradient boosting estado da arte (Chen & Guestrin, 2016)
  3. MLP_BN               — Deep MLP com BatchNorm (controle de DL tabular)
  4. ResNet_Tabular       — MLP com conexões residuais (Gorishniy et al., 2021)
  5. BiLSTM_Atencao       — HIPÓTESE SECUNDÁRIA: aplicada a dados tabulares
                            como (n_features,1). Esperado underperformance
                            por viés indutivo errado.

Critério: macro_recall (PRIMÁRIO) e MCC. F1-macro reportado.
Cada modelo é executado dentro de safe_run() — falhas individuais não
interrompem a análise.

Dependências adicionais: xgboost
  pip install xgboost
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
    Input, Dense, Dropout, BatchNormalization, Add, Activation,
    Bidirectional, LSTM,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier

tf.keras.utils.set_random_seed(RANDOM_SEED)
apply_plot_style()

ANALISE_ID = 1
EPOCHS = 30
PATIENCE = 8
log = get_logger(ANALISE_ID, "analise_1")


# ─── Atenção (para o modelo secundário BiLSTM) ─────────────────────────────

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units=64, **kw):
        super().__init__(**kw)
        self.W = Dense(units); self.V = Dense(1)
    def call(self, x):
        s = self.V(tf.nn.tanh(self.W(x)))
        w = tf.nn.softmax(s, axis=1)
        return tf.reduce_sum(w * x, axis=1)


# ─── Builders ──────────────────────────────────────────────────────────────

def build_mlp_bn(n_feat, n_cls):
    inp = Input(shape=(n_feat,))
    x = Dense(256)(inp); x = BatchNormalization()(x); x = Activation("relu")(x); x = Dropout(0.3)(x)
    x = Dense(128)(x);   x = BatchNormalization()(x); x = Activation("relu")(x); x = Dropout(0.3)(x)
    x = Dense(64)(x);    x = BatchNormalization()(x); x = Activation("relu")(x); x = Dropout(0.2)(x)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="MLP_BN")
    m.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def _resnet_block(x, units):
    skip = x
    h = Dense(units)(x); h = BatchNormalization()(h); h = Activation("relu")(h); h = Dropout(0.2)(h)
    h = Dense(units)(h); h = BatchNormalization()(h)
    if skip.shape[-1] != units:
        skip = Dense(units)(skip)
    out = Add()([skip, h]); out = Activation("relu")(out)
    return out


def build_resnet_tabular(n_feat, n_cls):
    inp = Input(shape=(n_feat,))
    x = Dense(256)(inp); x = BatchNormalization()(x); x = Activation("relu")(x)
    x = _resnet_block(x, 256)
    x = _resnet_block(x, 128)
    x = _resnet_block(x, 64)
    x = Dropout(0.3)(x)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="ResNet_Tabular")
    m.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


def build_bilstm_attention(n_feat, n_cls):
    inp = Input(shape=(n_feat, 1))
    x = Bidirectional(LSTM(128, return_sequences=True))(inp); x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x);   x = Dropout(0.5)(x)
    x = BahdanauAttention(64)(x)
    x = Dense(32, activation="relu")(x); x = Dropout(0.25)(x)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="BiLSTM_Atencao_Tabular")
    m.compile(optimizer=Adam(1e-3), loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
    return m


# ─── Treino e avaliação de modelos Keras ───────────────────────────────────

def avaliar_keras(builder, nome, X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls,
                  reshape_3d=False):
    K.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    t0 = time.time()
    m = builder(X_tr.shape[1], n_cls)
    if reshape_3d:
        Xt  = X_tr.reshape(-1, X_tr.shape[1], 1)
        Xv  = X_val.reshape(-1, X_val.shape[1], 1)
        Xte = X_te.reshape(-1, X_te.shape[1], 1)
    else:
        Xt, Xv, Xte = X_tr, X_val, X_te
    log.info(f"  fit {nome}: epochs<={EPOCHS} batch={BATCH_SIZE} params={m.count_params():,}")
    cb = [
        EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=4, min_lr=1e-7, monitor="val_loss"),
        EpochLogger(log, prefix=f"{nome} "),
    ]
    m.fit(Xt, y_tr, validation_data=(Xv, y_val),
          epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0, callbacks=cb)
    yp = np.argmax(m.predict(Xte, verbose=0, batch_size=BATCH_SIZE), axis=1)
    r = metricas_completas(y_te, yp, n_classes=n_cls)
    r.update({"modelo": nome, "tempo_s": time.time() - t0,
              "params": int(m.count_params())})
    return r


def avaliar_rf(X_tr, X_te, y_tr, y_te, n_cls):
    log.info("  fit RandomForest: 300 árvores, balanced_subsample")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=RANDOM_SEED,
        class_weight="balanced_subsample",
        max_samples=min(50_000, len(X_tr)) if len(X_tr) > 50_000 else None,
    )
    rf.fit(X_tr, y_tr)
    yp = rf.predict(X_te)
    r = metricas_completas(y_te, yp, n_classes=n_cls)
    r.update({"modelo": "RandomForest", "tempo_s": time.time() - t0, "params": 0})
    return r


def avaliar_xgb(X_tr, X_te, y_tr, y_te, n_cls):
    try:
        import xgboost as xgb
    except ImportError as e:
        raise RuntimeError("xgboost não instalado. Execute: pip install xgboost") from e
    log.info("  fit XGBoost: 400 rounds, depth 8")
    t0 = time.time()
    clf = xgb.XGBClassifier(
        n_estimators=400, max_depth=8, learning_rate=0.1,
        objective="multi:softprob", num_class=n_cls,
        n_jobs=-1, random_state=RANDOM_SEED,
        tree_method="hist", verbosity=0,
    )
    clf.fit(X_tr, y_tr)
    yp = clf.predict(X_te)
    r = metricas_completas(y_te, yp, n_classes=n_cls)
    r.update({"modelo": "XGBoost", "tempo_s": time.time() - t0, "params": 0})
    return r


# ─── Plot ──────────────────────────────────────────────────────────────────

def plot_comparativo(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metr = [("recall_macro", "Recall-macro (PRIMÁRIO)"),
            ("mcc", "MCC"),
            ("f1_macro", "F1-macro"),
            ("fpr_macro", "FPR-macro")]
    for ax, (col, lbl) in zip(axes.flat, metr):
        if col not in df.columns: continue
        sns.barplot(data=df, x="modelo", y=col, ax=ax, palette="viridis")
        ax.set_title(lbl); ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
        for c in ax.containers:
            ax.bar_label(c, fmt="%.3f", fontsize=8)
    fig.suptitle("Análise 1 — Arquiteturas (CSE-CIC-IDS2018)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = fig_path(ANALISE_ID, "comparativo_arquiteturas")
    fig.savefig(p, dpi=300); plt.close(fig)
    return p


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    rel = Relatorio(ANALISE_ID)
    log.info("ANÁLISE 1 — Arquiteturas (5 modelos sobre dados reais)")

    if not verificar_dataset(interativo=False):
        log.error("Dataset real ausente — análise abortada.")
        return

    # Carrega dataset COMPLETO (sem seleção de features)
    ok, dados = safe_run(log, "carregar_dataset_real",
                         carregar_dataset_real,
                         n_amostras_max=120_000, select_features=False)
    if not ok or dados is None:
        log.error("Falha ao carregar dataset.")
        return
    X_raw, y, _ = dados
    n_cls = int(np.max(y) + 1)
    log.info(f"Dataset: {X_raw.shape[0]:,} amostras × {X_raw.shape[1]} features × {n_cls} classes")

    # Split estratificado + scaler SEM leakage
    X_tr_r, X_val_r, X_te_r, y_tr, y_val, y_te = stratified_split_3way(
        X_raw, y, val_frac=0.15, test_frac=0.15, seed=RANDOM_SEED, logger=log,
    )
    X_tr, X_val, X_te, _ = fit_scaler_no_leakage(X_tr_r, X_val_r, X_te_r)

    res = []

    ok, r = safe_run(log, "RandomForest",
                     avaliar_rf, X_tr, X_te, y_tr, y_te, n_cls)
    if ok and r: res.append(r)

    ok, r = safe_run(log, "XGBoost",
                     avaliar_xgb, X_tr, X_te, y_tr, y_te, n_cls)
    if ok and r: res.append(r)

    ok, r = safe_run(log, "MLP_BN",
                     avaliar_keras, build_mlp_bn, "MLP_BN",
                     X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls,
                     reshape_3d=False)
    if ok and r: res.append(r)

    ok, r = safe_run(log, "ResNet_Tabular",
                     avaliar_keras, build_resnet_tabular, "ResNet_Tabular",
                     X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls,
                     reshape_3d=False)
    if ok and r: res.append(r)

    ok, r = safe_run(log, "BiLSTM_Atencao_Tabular (hipótese secundária)",
                     avaliar_keras, build_bilstm_attention, "BiLSTM_Atencao",
                     X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls,
                     reshape_3d=True)
    if ok and r: res.append(r)

    if not res:
        log.error("Nenhum modelo produziu resultado válido.")
        return

    df = pd.DataFrame(res)

    # Persistência
    csv_path = tab_path(ANALISE_ID, "metricas_arquiteturas")
    safe_run(log, "salvar CSV", df.to_csv, csv_path, index=False)
    log.info(f"Tabela: {csv_path}")

    safe_run(log, "plot_comparativo", plot_comparativo, df)

    # Veredito por critério primário (recall_macro)
    venc = max(res, key=lambda r: r["recall_macro"])
    log.info("=" * 62)
    log.info(f"VENCEDOR (recall_macro): {venc['modelo']}  "
             f"recall={venc['recall_macro']:.4f} mcc={venc['mcc']:.4f} "
             f"f1_macro={venc['f1_macro']:.4f} fpr={venc['fpr_macro']:.4f}")
    log.info("=" * 62)

    # Salva relatório markdown
    try:
        rel.secao("Resumo dos Resultados")
        rel.tabela_df(df, "Métricas dos modelos avaliados")
        rel.secao("Veredito")
        rel.texto(f"Vencedor por recall_macro: **{venc['modelo']}**.\n\n"
                  f"recall_macro = {venc['recall_macro']:.4f}, "
                  f"MCC = {venc['mcc']:.4f}, "
                  f"F1-macro = {venc['f1_macro']:.4f}.")
        rel.salvar()
    except Exception as e:
        log_exception(log, "rel.salvar", e)


def executar(dataset_disponivel: bool = True, **kwargs) -> None:
    """Entry-point do app_menu. NUNCA propaga exceção."""
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
