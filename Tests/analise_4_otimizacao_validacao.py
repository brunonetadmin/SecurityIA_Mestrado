"""
analise_4_otimizacao_validacao.py
=================================
Compara 4 otimizadores (SGD, Adam, AdamW, RMSprop) com e sem ReduceLROnPlateau,
sobre dados sintéticos. Justifica Adam+ReduceLROnPlateau+EarlyStopping no IDS.

Saída: TESTES → RESULTADOS → IDS  (Adam, lr=1e-3, ReduceLROnPlateau)
"""

import os, sys, time, warnings
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from config import (
    RANDOM_SEED, N_FEATURES, CLASS_NAMES, CLASS_DIST,
    LSTM_UNITS_L1, LSTM_UNITS_L2, LSTM_DENSE_UNITS, DROPOUT_RATE,
    BATCH_SIZE, ATTENTION_UNITS,
    fig_path, tab_path, Relatorio, apply_plot_style,
)

import tensorflow as tf
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

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
apply_plot_style()

ANALISE_ID = 4
EPOCHS = 25
PATIENCE = 5


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
    m = Model(inp, out)
    m.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])
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
        "acuracia": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "fpr_macro": float(np.mean(fpr_pc)),
    }


def avaliar(X_tr, X_te, y_tr, y_te, n_cls, opt_name, opt_fn, use_lr_sched):
    t0 = time.time()
    m = build_model(X_tr.shape[1], n_cls, opt_fn())
    cbs = [EarlyStopping(patience=PATIENCE, restore_best_weights=True,
                          monitor="val_loss")]
    if use_lr_sched:
        cbs.append(ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7,
                                      monitor="val_loss"))
    Xt = X_tr.reshape(-1, X_tr.shape[1], 1)
    Xv = X_te.reshape(-1, X_te.shape[1], 1)
    h = m.fit(Xt, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
               validation_split=0.15, callbacks=cbs)
    yp = np.argmax(m.predict(Xv, verbose=0), axis=1)
    r = metricas(y_te, yp)
    r["otimizador"] = opt_name
    r["lr_schedule"] = "ReduceLROnPlateau" if use_lr_sched else "Nenhum"
    r["epocas_executadas"] = len(h.history["loss"])
    r["tempo_s"] = time.time() - t0
    return r


def main():
    Relatorio(ANALISE_ID)
    print(f"\n  ANÁLISE 4 — Otimização e Validação (sintético)")

    X, y = gerar_sintetico(4000)
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    n_cls = len(np.unique(y))
    print(f"  Treino={len(X_tr)} Teste={len(X_te)} Classes={n_cls}")

    res = []
    for opt_name, opt_fn in OPTIMIZERS.items():
        for use_lr in (False, True):
            tag = f"{opt_name}_{'LR' if use_lr else 'noLR'}"
            print(f"\n    [{tag}]")
            try:
                r = avaliar(X_tr, X_te, y_tr, y_te, n_cls,
                            opt_name, opt_fn, use_lr)
                res.append(r)
                print(f"      F1={r['f1_macro']:.4f}  Recall={r['recall_macro']:.4f}  "
                      f"MCC={r['mcc']:.4f}  ep={r['epocas_executadas']}  "
                      f"t={r['tempo_s']:.1f}s")
            except Exception as e:
                print(f"      ✗ falhou: {e}")

    df = pd.DataFrame(res)
    df["config"] = df["otimizador"] + " (" + df["lr_schedule"] + ")"
    csv_path = tab_path(ANALISE_ID, "metricas_otimizacao")
    df.to_csv(csv_path, index=False)
    print(f"\n  Tabela: {csv_path}")

    # Figura
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (col, lbl) in zip(axes.flat,
                              [("f1_macro", "F1-macro"),
                               ("recall_macro", "Recall-macro"),
                               ("mcc", "MCC"),
                               ("epocas_executadas", "Épocas até convergência")]):
        sns.barplot(data=df, x="config", y=col, ax=ax, palette="cubehelix")
        ax.set_title(lbl); ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=35)
        for c in ax.containers:
            fmt = "%.3f" if col != "epocas_executadas" else "%d"
            ax.bar_label(c, fmt=fmt, fontsize=8)
    fig.suptitle("Análise 4 — Otimização e Validação",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = fig_path(ANALISE_ID, "comparativo_otimizacao")
    fig.savefig(p, dpi=300); plt.close(fig)
    print(f"  Figura: {p}")

    # Veredito — composto: alto F1 com poucas épocas
    if res:
        # score = f1 - 0.005*epocas (pequena penalidade por custo)
        for r in res:
            r["score_composto"] = r["f1_macro"] - 0.005 * r["epocas_executadas"]
        venc = max(res, key=lambda r: r["score_composto"])
        print("\n" + "=" * 62)
        print(f"  VENCEDOR (F1 com penalidade por épocas):")
        print(f"    {venc['otimizador']} + {venc['lr_schedule']}")
        print(f"    F1={venc['f1_macro']:.4f}  Recall={venc['recall_macro']:.4f}  "
              f"épocas={venc['epocas_executadas']}")
        if venc["otimizador"] == "Adam" and venc["lr_schedule"] == "ReduceLROnPlateau":
            print(f"  ✓ Justifica Adam+ReduceLROnPlateau no IDS.")
        else:
            print(f"  ⚠ Vencedor difere do IDS — revisar.")
        print("=" * 62)


if __name__ == "__main__":
    main()
