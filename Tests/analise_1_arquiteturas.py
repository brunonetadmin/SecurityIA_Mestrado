"""
analise_1_arquiteturas.py
=========================
Compara 4 arquiteturas (LSTM, Bi-LSTM, Bi-LSTM+Atenção, Transformer compacto)
sobre dados sintéticos E sobre o CSE-CIC-IDS2018 real, demonstrando que a
Bi-LSTM+Atenção mantém ranking entre os domínios e justifica seu uso no IDS.

Saída: TESTES → RESULTADOS → IDS  (Bi-LSTM+Atenção em ids_learn.py)
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
    LEARNING_RATE_INITIAL, BATCH_SIZE, ATTENTION_UNITS,
    fig_path, tab_path, Relatorio, apply_plot_style,
    verificar_dataset, carregar_dataset_real,
)

import tensorflow as tf
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

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
apply_plot_style()

ANALISE_ID = 1
EPOCHS = 15
PATIENCE = 4


# ─── Atenção de Bahdanau (idêntica à do IDS) ────────────────────────────────

class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units=ATTENTION_UNITS, **kw):
        super().__init__(**kw)
        self.W = Dense(units)
        self.V = Dense(1)

    def call(self, x):
        score = self.V(tf.nn.tanh(self.W(x)))
        weights = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(weights * x, axis=1)


# ─── Arquiteturas ───────────────────────────────────────────────────────────

def build_lstm(n_feat, n_cls):
    inp = Input(shape=(n_feat, 1))
    x = LSTM(LSTM_UNITS_L1, return_sequences=True)(inp)
    x = Dropout(DROPOUT_RATE)(x)
    x = LSTM(LSTM_UNITS_L2)(x)
    x = Dropout(DROPOUT_RATE)(x)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="LSTM")
    m.compile(optimizer=Adam(LEARNING_RATE_INITIAL), loss="sparse_categorical_crossentropy", metrics=
              ["accuracy"])
    return m

def build_bilstm(n_feat, n_cls):
    inp = Input(shape=(n_feat, 1))
    x = Bidirectional(LSTM(LSTM_UNITS_L1, return_sequences=True))(inp)
    x = Dropout(DROPOUT_RATE)(x)
    x = Bidirectional(LSTM(LSTM_UNITS_L2))(x)
    x = Dropout(DROPOUT_RATE)(x)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="Bi-LSTM")
    m.compile(optimizer=Adam(LEARNING_RATE_INITIAL), loss="sparse_categorical_crossentropy", metrics=
              ["accuracy"])
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
    m = Model(inp, out, name="Bi-LSTM+Atencao")
    m.compile(optimizer=Adam(LEARNING_RATE_INITIAL), loss="sparse_categorical_crossentropy", metrics=
              ["accuracy"])
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
    m.compile(optimizer=Adam(LEARNING_RATE_INITIAL), loss="sparse_categorical_crossentropy", metrics=
              ["accuracy"])
    return m

ARQUITETURAS = {
    "LSTM": build_lstm,
    "Bi-LSTM": build_bilstm,
    "Bi-LSTM+Atencao": build_bilstm_attention,
    "Transformer": build_transformer,
}


# ─── Geração de dados sintéticos ────────────────────────────────────────────

def gerar_sintetico(n=8000):
    rng = np.random.default_rng(RANDOM_SEED)
    n_cls = len(CLASS_NAMES)
    counts = (np.array(CLASS_DIST) * n).astype(int)
    Xs, ys = [], []
    for c, k in enumerate(counts):
        center = rng.normal(c * 1.5, 0.3, N_FEATURES)
        Xc = rng.normal(center, 1.0, (k, N_FEATURES))
        Xs.append(Xc); ys.append(np.full(k, c))
    X = np.vstack(Xs); y = np.concatenate(ys)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# ─── Avaliação ─────────────────────────────────────────────────────────────

def avaliar(X_tr, X_te, y_tr, y_te, n_cls, builder, nome):
    t0 = time.time()
    m = builder(X_tr.shape[1], n_cls)
    Xt = X_tr.reshape(-1, X_tr.shape[1], 1)
    Xv = X_te.reshape(-1, X_te.shape[1], 1)
    m.fit(Xt, y_tr, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
          validation_split=0.15,
          callbacks=[EarlyStopping(patience=PATIENCE, restore_best_weights=True)])
    yp = np.argmax(m.predict(Xv, verbose=0), axis=1)
    elapsed = time.time() - t0
    cm = confusion_matrix(y_te, yp)
    fpr_pc = []
    for c in range(cm.shape[0]):
        fp = cm[:, c].sum() - cm[c, c]
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fpr_pc.append(fp / (fp + tn) if (fp + tn) else 0.0)
    return {
        "modelo": nome,
        "acuracia": accuracy_score(y_te, yp),
        "f1_macro": f1_score(y_te, yp, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_te, yp, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_te, yp, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_te, yp),
        "fpr_macro": float(np.mean(fpr_pc)),
        "tempo_s": elapsed,
        "params": m.count_params(),
    }


def baseline_rf(X_tr, X_te, y_tr, y_te):
    rf = RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                 random_state=RANDOM_SEED,
                                 class_weight="balanced_subsample")
    rf.fit(X_tr, y_tr)
    yp = rf.predict(X_te)
    cm = confusion_matrix(y_te, yp)
    fpr_pc = []
    for c in range(cm.shape[0]):
        fp = cm[:, c].sum() - cm[c, c]
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        fpr_pc.append(fp / (fp + tn) if (fp + tn) else 0.0)
    return {
        "modelo": "Baseline_RF",
        "acuracia": accuracy_score(y_te, yp),
        "f1_macro": f1_score(y_te, yp, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_te, yp, average="weighted", zero_division=0),
        "recall_macro": recall_score(y_te, yp, average="macro", zero_division=0),
        "mcc": matthews_corrcoef(y_te, yp),
        "fpr_macro": float(np.mean(fpr_pc)),
    }


def rodar_dominio(X, y, nome_dominio):
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    n_cls = len(np.unique(y))
    print(f"\n  [{nome_dominio}] Treino={len(X_tr)} Teste={len(X_te)} Classes={n_cls}")
    res = []
    for nome, builder in ARQUITETURAS.items():
        print(f"    treinando {nome} …")
        r = avaliar(X_tr, X_te, y_tr, y_te, n_cls, builder, nome)
        r["dominio"] = nome_dominio
        res.append(r)
        print(f"      F1={r['f1_macro']:.4f} Recall={r['recall_macro']:.4f} "
              f"FPR={r['fpr_macro']:.4f} t={r['tempo_s']:.1f}s")
    print(f"    referência baseline RF …")
    rf = baseline_rf(X_tr, X_te, y_tr, y_te)
    rf["dominio"] = nome_dominio
    rf["tempo_s"] = 0.0
    rf["params"] = 0
    res.append(rf)
    return res


# ─── Visualização ──────────────────────────────────────────────────────────

def plot_comparativo(df):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [("f1_macro", "F1-macro"),
               ("recall_macro", "Recall-macro"),
               ("fpr_macro", "FPR-macro"),
               ("mcc", "MCC")]
    for ax, (col, lbl) in zip(axes.flat, metrics):
        sns.barplot(data=df, x="modelo", y=col, hue="dominio", ax=ax)
        ax.set_title(lbl); ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Análise 1 — Arquiteturas: Sintético vs Real (CSE-CIC-IDS2018)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = fig_path(ANALISE_ID, "comparativo_arquiteturas")
    fig.savefig(p, dpi=300); plt.close(fig)
    return p


def plot_transferencia(df):
    pivot = df.pivot_table(index="modelo", columns="dominio",
                           values="f1_macro", aggfunc="first")
    if "Sintetico" not in pivot.columns or "Real" not in pivot.columns:
        return None
    pivot["delta"] = pivot["Real"] - pivot["Sintetico"]
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot[["Sintetico", "Real"]].plot.bar(ax=ax)
    ax.set_title("Transferência sintético → real por arquitetura")
    ax.set_ylabel("F1-macro"); ax.tick_params(axis="x", rotation=30)
    ax.axhline(y=pivot.loc[pivot.index != "Baseline_RF", "Real"].max(),
               color="red", ls="--", alpha=0.5, label="melhor real")
    ax.legend()
    fig.tight_layout()
    p = fig_path(ANALISE_ID, "transferencia_dominios")
    fig.savefig(p, dpi=300); plt.close(fig)
    return p


# ─── Main ──────────────────────────────────────────────────────────────────

def main():
    Relatorio(ANALISE_ID)  # garante diretórios
    print(f"\n  ANÁLISE 1 — Arquiteturas (sintético + real)")

    # Sintético
    Xs, ys = gerar_sintetico(8000)
    res_sint = rodar_dominio(Xs, ys, "Sintetico")

    # Real (se disponível)
    res_real = []
    if verificar_dataset(interativo=False):
        dados = carregar_dataset_real(n_amostras_max=80_000)
        if dados is not None:
            Xr, yr, _ = dados
            res_real = rodar_dominio(Xr, yr, "Real")
        else:
            print("\n  ⚠ Não foi possível carregar dataset real.")
    else:
        print("\n  ⚠ Dataset CSE-CIC-IDS2018 ausente — análise apenas em sintético.")

    df = pd.DataFrame(res_sint + res_real)
    csv_path = tab_path(ANALISE_ID, "metricas_arquiteturas")
    df.to_csv(csv_path, index=False)
    print(f"\n  Tabela: {csv_path}")

    p1 = plot_comparativo(df); print(f"  Figura: {p1}")
    if res_real:
        p2 = plot_transferencia(df)
        if p2: print(f"  Figura: {p2}")

    # Veredito
    real_arch = [r for r in res_real if r["modelo"] != "Baseline_RF"]
    if real_arch:
        rf_real = next((r for r in res_real if r["modelo"] == "Baseline_RF"), None)
        venc = max(real_arch, key=lambda r: r["recall_macro"])
        print("\n" + "=" * 62)
        print(f"  VENCEDOR EM REAL (por recall): {venc['modelo']}")
        print(f"    F1-macro={venc['f1_macro']:.4f}  Recall={venc['recall_macro']:.4f}  "
              f"FPR={venc['fpr_macro']:.4f}")
        if rf_real:
            print(f"  vs Baseline RF: F1={rf_real['f1_macro']:.4f}  "
                  f"Recall={rf_real['recall_macro']:.4f}  FPR={rf_real['fpr_macro']:.4f}")
        if venc["modelo"] == "Bi-LSTM+Atencao":
            print(f"  ✓ Justifica adoção de Bi-LSTM+Atenção no IDS.")
        else:
            print(f"  ⚠ Vencedor ({venc['modelo']}) difere do IDS — revisar.")
        print("=" * 62)


if __name__ == "__main__":
    main()
