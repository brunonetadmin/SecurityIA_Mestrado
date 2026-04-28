"""
analise_4_otimizacao_validacao.py — Otimização e Validação (sintético).
Versão blindada: verbose=0+EpochLogger (corrige OSError do app_menu),
KeyError corrigido, executar() à prova de exceção.
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
    BATCH_SIZE, ATTENTION_UNITS,
    fig_path, tab_path, Relatorio, apply_plot_style,
)
from _test_logging import get_logger, log_exception, EpochLogger, silence_tensorflow

silence_tensorflow()

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam, AdamW, SGD, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    confusion_matrix, recall_score,
)

tf.keras.utils.set_random_seed(RANDOM_SEED)
apply_plot_style()

ANALISE_ID = 4
EPOCHS = 40
PATIENCE = 8
log = get_logger(ANALISE_ID, "analise_4")


def sanitize(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    p99 = np.percentile(np.abs(X), 99.9, axis=0)
    p99 = np.where(p99 < 1e-9, 1.0, p99)
    return np.clip(X, -p99 * 10, p99 * 10).astype(np.float32)


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units=ATTENTION_UNITS, **kw):
        super().__init__(**kw)
        self.W = Dense(units); self.V = Dense(1)
    def call(self, x):
        s = self.V(tf.nn.tanh(self.W(x)))
        w = tf.nn.softmax(s, axis=1)
        return tf.reduce_sum(w * x, axis=1)


def build_model(n_feat, n_cls, optimizer):
    inp = Input(shape=(n_feat, 1))
    x = Bidirectional(LSTM(LSTM_UNITS_L1, return_sequences=True))(inp)
    x = Dropout(DROPOUT_RATE)(x)
    x = Bidirectional(LSTM(LSTM_UNITS_L2, return_sequences=True))(x)
    x = Dropout(DROPOUT_RATE)(x)
    x = BahdanauAttention()(x)
    x = Dense(LSTM_DENSE_UNITS, activation="relu")(x)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="BiLSTM_Atencao")
    m.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


OPTIMIZERS = {
    "SGD":     lambda: SGD(learning_rate=1e-2, momentum=0.9),
    "Adam":    lambda: Adam(learning_rate=1e-3),
    "AdamW":   lambda: AdamW(learning_rate=1e-3, weight_decay=1e-4),
    "RMSprop": lambda: RMSprop(learning_rate=1e-3),
}


def gerar_sintetico(n=4000):
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
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "fpr_macro": float(np.mean(fpr_pc)),
    }


def avaliar(X_tr, X_te, y_tr, y_te, n_cls, opt_name, opt_fn, use_lr_sched):
    K.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    t0 = time.time()
    m = build_model(X_tr.shape[1], n_cls, opt_fn())
    cb = [EarlyStopping(patience=PATIENCE, restore_best_weights=True,
                         monitor="val_loss"),
          EpochLogger(log, prefix=f"{opt_name}/{('LR' if use_lr_sched else 'noLR')} ")]
    if use_lr_sched:
        cb.append(ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7,
                                     monitor="val_loss"))
    Xt = X_tr.reshape(-1, X_tr.shape[1], 1)
    Xv = X_te.reshape(-1, X_te.shape[1], 1)
    log.info(f"  fit {opt_name} lr_sched={use_lr_sched} epochs<={EPOCHS}")
    h = m.fit(Xt, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
               validation_split=0.15, callbacks=cb)
    yp = np.argmax(m.predict(Xv, verbose=0, batch_size=BATCH_SIZE), axis=1)
    r = metricas(y_te, yp)
    r["otimizador"] = opt_name
    r["lr_schedule"] = "ReduceLROnPlateau" if use_lr_sched else "Nenhum"
    r["epocas_executadas"] = int(len(h.history.get("loss", [])))
    r["tempo_s"] = time.time() - t0
    return r


def main():
    Relatorio(ANALISE_ID)
    log.info("ANÁLISE 4 — Otimização e Validação (sintético)")

    X, y = gerar_sintetico(4000)
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
    log.info(f"Treino={len(X_tr)} Teste={len(X_te)} Classes={n_cls}")

    res = []
    for opt_name, opt_fn in OPTIMIZERS.items():
        for use_lr in (False, True):
            tag = f"{opt_name}_{'LR' if use_lr else 'noLR'}"
            log.info(f">>> {tag}")
            try:
                r = avaliar(X_tr, X_te, y_tr, y_te, n_cls,
                            opt_name, opt_fn, use_lr)
                res.append(r)
                log.info(f"{tag} OK F1={r['f1_macro']:.4f} Recall={r['recall_macro']:.4f} "
                         f"MCC={r['mcc']:.4f} ep={r['epocas_executadas']} "
                         f"t={r['tempo_s']:.1f}s")
            except Exception as e:
                log_exception(log, tag, e)
                res.append({"otimizador": opt_name,
                            "lr_schedule": "ReduceLROnPlateau" if use_lr else "Nenhum",
                            "erro": str(e)})

    df = pd.DataFrame(res)
    if "otimizador" in df.columns and "lr_schedule" in df.columns:
        df["config"] = df["otimizador"].fillna("?") + " (" + df["lr_schedule"].fillna("?") + ")"
    else:
        df["config"] = "?"
        log.warning("Nenhuma execução produziu resultado válido.")

    try:
        csv_path = tab_path(ANALISE_ID, "metricas_otimizacao")
        df.to_csv(csv_path, index=False)
        log.info(f"Tabela: {csv_path}")
    except Exception as e:
        log_exception(log, "salvar CSV", e)

    try:
        df_v = df[df.get("erro", pd.Series([np.nan]*len(df))).isna()] \
               if "erro" in df.columns else df
        if not df_v.empty and "f1_macro" in df_v.columns:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            for ax, (col, lbl) in zip(axes.flat,
                                      [("f1_macro", "F1-macro"),
                                       ("recall_macro", "Recall-macro"),
                                       ("mcc", "MCC"),
                                       ("epocas_executadas", "Épocas até convergência")]):
                if col not in df_v.columns: continue
                sns.barplot(data=df_v, x="config", y=col, ax=ax, palette="cubehelix")
                ax.set_title(lbl); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=35)
                for c in ax.containers:
                    fmt = "%.3f" if col != "epocas_executadas" else "%d"
                    ax.bar_label(c, fmt=fmt, fontsize=8)
            fig.suptitle("Análise 4 — Otimização e Validação",
                         fontsize=12, fontweight="bold")
            fig.tight_layout()
            p = fig_path(ANALISE_ID, "comparativo_otimizacao")
            fig.savefig(p, dpi=300); plt.close(fig)
            log.info(f"Figura: {p}")
    except Exception as e:
        log_exception(log, "plot", e)

    cands = [r for r in res if "f1_macro" in r and "epocas_executadas" in r]
    if cands:
        for r in cands:
            r["score_composto"] = r["f1_macro"] - 0.005 * r["epocas_executadas"]
        venc = max(cands, key=lambda r: r["score_composto"])
        log.info("=" * 62)
        log.info(f"VENCEDOR: {venc['otimizador']} + {venc['lr_schedule']}  "
                 f"F1={venc['f1_macro']:.4f} Recall={venc['recall_macro']:.4f} "
                 f"épocas={venc['epocas_executadas']}")
        if venc["otimizador"] == "Adam" and venc["lr_schedule"] == "ReduceLROnPlateau":
            log.info("✓ Justifica Adam+ReduceLROnPlateau no IDS.")
        else:
            log.warning("Vencedor difere do IDS — revisar.")
        log.info("=" * 62)


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
