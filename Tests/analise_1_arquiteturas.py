"""
analise_1_arquiteturas.py — Comparação de arquiteturas (sintético + real).
Versão blindada: verbose=0 + EpochLogger, executar() não lança, dataset
sanitizado, EarlyStopping com paciência adequada.
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
    RANDOM_SEED, N_FEATURES, CLASS_NAMES, CLASS_DIST,
    LSTM_UNITS_L1, LSTM_UNITS_L2, LSTM_DENSE_UNITS, DROPOUT_RATE,
    LEARNING_RATE_INITIAL, BATCH_SIZE, ATTENTION_UNITS,
    fig_path, tab_path, Relatorio, apply_plot_style,
    verificar_dataset, carregar_dataset_real,
)
from _test_logging import get_logger, log_exception, EpochLogger, silence_tensorflow

silence_tensorflow()

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, LSTM, Bidirectional, Dense, Dropout, LayerNormalization,
    MultiHeadAttention, GlobalAveragePooling1D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    confusion_matrix, recall_score,
)
from sklearn.preprocessing import StandardScaler

tf.keras.utils.set_random_seed(RANDOM_SEED)
apply_plot_style()

ANALISE_ID = 1
EPOCHS = 30
PATIENCE = 8
log = get_logger(ANALISE_ID, "analise_1")


# ─── Sanitização ───────────────────────────────────────────────────────────

def sanitize(X):
    """Remove NaN/Inf, clipa outliers extremos."""
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    p99 = np.percentile(np.abs(X), 99.9, axis=0)
    p99 = np.where(p99 < 1e-9, 1.0, p99)
    X = np.clip(X, -p99 * 10, p99 * 10)
    return X.astype(np.float32)


# ─── Atenção ───────────────────────────────────────────────────────────────

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units=ATTENTION_UNITS, **kw):
        super().__init__(**kw)
        self.W = Dense(units); self.V = Dense(1)
    def call(self, x):
        s = self.V(tf.nn.tanh(self.W(x)))
        w = tf.nn.softmax(s, axis=1)
        return tf.reduce_sum(w * x, axis=1)


def build_lstm(n_feat, n_cls):
    inp = Input(shape=(n_feat, 1))
    x = LSTM(LSTM_UNITS_L1, return_sequences=True)(inp)
    x = Dropout(DROPOUT_RATE)(x)
    x = LSTM(LSTM_UNITS_L2)(x)
    x = Dropout(DROPOUT_RATE)(x)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="LSTM")
    m.compile(optimizer=Adam(LEARNING_RATE_INITIAL),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def build_bilstm(n_feat, n_cls):
    inp = Input(shape=(n_feat, 1))
    x = Bidirectional(LSTM(LSTM_UNITS_L1, return_sequences=True))(inp)
    x = Dropout(DROPOUT_RATE)(x)
    x = Bidirectional(LSTM(LSTM_UNITS_L2))(x)
    x = Dropout(DROPOUT_RATE)(x)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="BiLSTM")
    m.compile(optimizer=Adam(LEARNING_RATE_INITIAL),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def build_bilstm_attention(n_feat, n_cls):
    inp = Input(shape=(n_feat, 1))
    x = Bidirectional(LSTM(LSTM_UNITS_L1, return_sequences=True))(inp)
    x = Dropout(DROPOUT_RATE)(x)
    x = Bidirectional(LSTM(LSTM_UNITS_L2, return_sequences=True))(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = BahdanauAttention()(x)
    x = Dense(LSTM_DENSE_UNITS, activation="relu")(x)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="BiLSTM_Atencao")
    m.compile(optimizer=Adam(LEARNING_RATE_INITIAL),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def build_transformer(n_feat, n_cls):
    inp = Input(shape=(n_feat, 1))
    x = Dense(64)(inp)
    a = MultiHeadAttention(num_heads=4, key_dim=16)(x, x)
    x = LayerNormalization()(x + a)
    f = Dense(64, activation="relu")(x)
    f = Dense(64)(f)
    x = LayerNormalization()(x + f)
    x = Dropout(DROPOUT_RATE)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(LSTM_DENSE_UNITS, activation="relu")(x)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="Transformer")
    m.compile(optimizer=Adam(LEARNING_RATE_INITIAL),
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


ARQUITETURAS = {
    "LSTM": build_lstm,
    "BiLSTM": build_bilstm,
    "BiLSTM_Atencao": build_bilstm_attention,
    "Transformer": build_transformer,
}


# ─── Dados ─────────────────────────────────────────────────────────────────

def gerar_sintetico(n=8000):
    rng = np.random.default_rng(RANDOM_SEED)
    counts = (np.array(CLASS_DIST) * n).astype(int)
    counts[counts < 10] = 10
    Xs, ys = [], []
    for c, k in enumerate(counts):
        center = rng.normal(c * 1.5, 0.3, N_FEATURES)
        Xs.append(rng.normal(center, 1.0, (k, N_FEATURES)))
        ys.append(np.full(k, c))
    X = np.vstack(Xs); y = np.concatenate(ys)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ─── Métricas ──────────────────────────────────────────────────────────────

def metricas(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    fpr_pc = []
    for c in range(cm.shape[0]):
        fp = cm[:, c].sum() - cm[c, c]
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fpr_pc.append(fp / (fp + tn) if (fp + tn) else 0.0)
    return {
        "acuracia": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "fpr_macro": float(np.mean(fpr_pc)),
    }


def avaliar(X_tr, X_te, y_tr, y_te, n_cls, builder, nome):
    K.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    t0 = time.time()
    m = builder(X_tr.shape[1], n_cls)
    Xt = X_tr.reshape(-1, X_tr.shape[1], 1)
    Xv = X_te.reshape(-1, X_te.shape[1], 1)
    log.info(f"  fit {nome}: epochs<={EPOCHS} batch={BATCH_SIZE} params={m.count_params():,}")
    cb = [
        EarlyStopping(patience=PATIENCE, restore_best_weights=True,
                       monitor="val_loss"),
        EpochLogger(log, prefix=f"{nome} "),
    ]
    m.fit(Xt, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
          validation_split=0.15, callbacks=cb)
    yp = np.argmax(m.predict(Xv, verbose=0, batch_size=BATCH_SIZE), axis=1)
    r = metricas(y_te, yp)
    r.update({"modelo": nome, "tempo_s": time.time() - t0,
              "params": int(m.count_params())})
    return r


def baseline_rf(X_tr, X_te, y_tr, y_te):
    log.info("  treinando Baseline_RF")
    rf = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=RANDOM_SEED,
        class_weight="balanced_subsample",
        max_samples=min(50_000, len(X_tr)) if len(X_tr) > 50_000 else None,
    )
    rf.fit(X_tr, y_tr)
    yp = rf.predict(X_te)
    r = metricas(y_te, yp)
    r["modelo"] = "Baseline_RF"; r["tempo_s"] = 0.0; r["params"] = 0
    return r


def rodar_dominio(X, y, nome_dominio):
    X = sanitize(X)
    sc = StandardScaler()
    X = sc.fit_transform(X).astype(np.float32)
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=RANDOM_SEED)
    n_cls = int(np.max(y) + 1)
    log.info(f"[{nome_dominio}] Treino={len(X_tr)} Teste={len(X_te)} Classes={n_cls}")
    res = []
    for nome, builder in ARQUITETURAS.items():
        log.info(f"[{nome_dominio}] >>> {nome}")
        try:
            r = avaliar(X_tr, X_te, y_tr, y_te, n_cls, builder, nome)
            r["dominio"] = nome_dominio
            res.append(r)
            log.info(f"[{nome_dominio}] {nome} OK F1={r['f1_macro']:.4f} "
                     f"Recall={r['recall_macro']:.4f} FPR={r['fpr_macro']:.4f} "
                     f"t={r['tempo_s']:.1f}s")
        except Exception as e:
            log_exception(log, f"[{nome_dominio}] {nome}", e)
            res.append({"modelo": nome, "dominio": nome_dominio, "erro": str(e)})
    try:
        rf = baseline_rf(X_tr, X_te, y_tr, y_te)
        rf["dominio"] = nome_dominio
        res.append(rf)
        log.info(f"[{nome_dominio}] Baseline_RF F1={rf['f1_macro']:.4f} "
                 f"Recall={rf['recall_macro']:.4f}")
    except Exception as e:
        log_exception(log, f"[{nome_dominio}] Baseline_RF", e)
    return res


def plot_comparativo(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (col, lbl) in zip(axes.flat,
                              [("f1_macro", "F1-macro"),
                               ("recall_macro", "Recall-macro"),
                               ("fpr_macro", "FPR-macro"),
                               ("mcc", "MCC")]):
        if col not in df.columns: continue
        sns.barplot(data=df, x="modelo", y=col, hue="dominio", ax=ax)
        ax.set_title(lbl); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Análise 1 — Arquiteturas: Sintético vs Real",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = fig_path(ANALISE_ID, "comparativo_arquiteturas")
    fig.savefig(p, dpi=300); plt.close(fig)
    return p


def main():
    Relatorio(ANALISE_ID)
    log.info("ANÁLISE 1 — Arquiteturas (sintético + real)")

    res_sint = []
    try:
        Xs, ys = gerar_sintetico(8000)
        res_sint = rodar_dominio(Xs, ys, "Sintetico")
    except Exception as e:
        log_exception(log, "dominio Sintetico", e)

    res_real = []
    try:
        if verificar_dataset(interativo=False):
            dados = carregar_dataset_real(n_amostras_max=80_000)
            if dados is not None:
                Xr, yr, _ = dados
                res_real = rodar_dominio(Xr, yr, "Real")
            else:
                log.warning("Dataset real não pôde ser carregado.")
        else:
            log.warning("Dataset CSE-CIC-IDS2018 ausente — análise apenas em sintético.")
    except Exception as e:
        log_exception(log, "dominio Real", e)

    df = pd.DataFrame(res_sint + res_real)
    try:
        csv_path = tab_path(ANALISE_ID, "metricas_arquiteturas")
        df.to_csv(csv_path, index=False)
        log.info(f"Tabela: {csv_path}")
    except Exception as e:
        log_exception(log, "salvar CSV", e)

    try:
        df_v = df[df.get("erro", pd.Series([np.nan]*len(df))).isna()] \
               if "erro" in df.columns else df
        if not df_v.empty and "f1_macro" in df_v.columns:
            log.info(f"Figura: {plot_comparativo(df_v)}")
    except Exception as e:
        log_exception(log, "plot", e)

    real_arch = [r for r in res_real if r.get("modelo") != "Baseline_RF"
                 and "recall_macro" in r]
    if real_arch:
        rf_real = next((r for r in res_real if r.get("modelo") == "Baseline_RF"), None)
        venc = max(real_arch, key=lambda r: r["recall_macro"])
        log.info("=" * 62)
        log.info(f"VENCEDOR EM REAL (recall): {venc['modelo']}  "
                 f"F1={venc['f1_macro']:.4f} Recall={venc['recall_macro']:.4f} "
                 f"FPR={venc['fpr_macro']:.4f}")
        if rf_real:
            log.info(f"Baseline RF: F1={rf_real['f1_macro']:.4f} "
                     f"Recall={rf_real['recall_macro']:.4f} FPR={rf_real['fpr_macro']:.4f}")
        if venc["modelo"] == "BiLSTM_Atencao":
            log.info("✓ Justifica adoção de Bi-LSTM+Atenção no IDS.")
        else:
            log.warning(f"Vencedor ({venc['modelo']}) difere do IDS — revisar.")
        log.info("=" * 62)


def executar(dataset_disponivel: bool = True, **kwargs) -> None:
    """Entry-point do app_menu. NUNCA lança exceção."""
    try:
        main()
    except Exception as e:
        log_exception(log, "executar", e)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log_exception(log, "main", e)
        sys.exit(0)  # exit 0 para não confundir o menu
