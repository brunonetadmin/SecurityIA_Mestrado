"""
analise_2_balanceamento.py — Estratégias de balanceamento sobre dados reais
"""
import os, sys, time, math, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from collections import Counter

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    RANDOM_SEED, BATCH_SIZE,
    fig_path, tab_path, Relatorio, apply_plot_style,
    verificar_dataset, carregar_dataset_real,
)
from _test_logging import get_logger, log_exception

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    confusion_matrix, recall_score, precision_score,
)
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from imblearn.combine import SMOTEENN

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
apply_plot_style()

ANALISE_ID = 2
EPOCHS = 12
PATIENCE = 4
log = get_logger(ANALISE_ID, "analise_2")


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units=64, **kw):
        super().__init__(**kw)
        self.W = Dense(units); self.V = Dense(1)
    def call(self, x):
        s = self.V(tf.nn.tanh(self.W(x)))
        w = tf.nn.softmax(s, axis=1)
        return tf.reduce_sum(w * x, axis=1)


def focal_loss(class_counts, gamma=2.0, beta=0.9999):
    cc = np.asarray(class_counts, dtype=np.float64)
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


def build_model(n_feat, n_cls, class_counts):
    inp = Input(shape=(n_feat, 1))
    x = Bidirectional(LSTM(128, return_sequences=True))(inp)
    x = Dropout(0.5)(x)
    x = Bidirectional(LSTM(64, return_sequences=True))(x)
    x = Dropout(0.5)(x)
    x = BahdanauAttention(64)(x)
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.25)(x)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="BiLSTM_Atencao")
    m.compile(optimizer=Adam(1e-3),
              loss=focal_loss(class_counts), metrics=["accuracy"])
    return m


# ─── Estratégias de balanceamento (sem n_jobs) ─────────────────────────────

def bal_none(X, y):
    return X, y


def bal_smote(X, y):
    return SMOTE(k_neighbors=5, random_state=RANDOM_SEED).fit_resample(X, y)


def bal_smote_enn(X, y):
    return SMOTEENN(random_state=RANDOM_SEED).fit_resample(X, y)


def bal_borderline_uniform(X, y):
    return BorderlineSMOTE(k_neighbors=5, kind="borderline-2",
                           random_state=RANDOM_SEED).fit_resample(X, y)


def bal_borderline_adaptive(X, y):
    """Estratégia oficial do IDS — Borderline-SMOTE-2 adaptativo."""
    dist = Counter(y)
    n_maj = max(dist.values())
    maj = max(dist, key=dist.get)
    strat, ks = {}, []
    for c, n_c in dist.items():
        if c == maj: continue
        n_t = max(min(5 * n_c, int(0.10 * n_maj)), n_c)
        strat[c] = n_t
        ks.append(min(11, max(1, math.ceil(0.25 * math.sqrt(n_c)))))
    if not strat:
        return X, y
    k_med = max(1, int(np.median(ks)))
    Xr, yr = BorderlineSMOTE(sampling_strategy=strat, k_neighbors=k_med,
                              kind="borderline-2",
                              random_state=RANDOM_SEED).fit_resample(X, y)
    t_maj = min(int(1.5 * max(strat.values())), dist[maj])
    Xr, yr = RandomUnderSampler(sampling_strategy={maj: t_maj},
                                 random_state=RANDOM_SEED).fit_resample(Xr, yr)
    Xr, yr = EditedNearestNeighbours(n_neighbors=3).fit_resample(Xr, yr)
    return Xr, yr


def bal_undersample(X, y):
    dist = Counter(y)
    n_min = min(dist.values())
    return RandomUnderSampler(
        sampling_strategy={c: max(n_min, n // 5) for c, n in dist.items()},
        random_state=RANDOM_SEED,
    ).fit_resample(X, y)


ESTRATEGIAS = {
    "Sem_Balanceamento":      bal_none,
    "SMOTE":                   bal_smote,
    "SMOTE-ENN":               bal_smote_enn,
    "Borderline-SMOTE2_Unif":  bal_borderline_uniform,
    "Borderline-SMOTE2_Adapt": bal_borderline_adaptive,
    "Undersampling":           bal_undersample,
}


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
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "fpr_macro": float(np.mean(fpr_pc)),
    }


def avaliar(X_tr, X_te, y_tr, y_te, n_cls, nome, fn_bal):
    K.clear_session()
    tf.random.set_seed(RANDOM_SEED)
    t0 = time.time()
    log.info(f"  aplicando {nome}…")
    Xb, yb = fn_bal(X_tr, y_tr)
    log.info(f"  pós-balanceamento: {len(yb):,} amostras  dist={dict(Counter(yb))}")
    cc = np.maximum(np.bincount(yb.astype(int), minlength=n_cls), 1)
    m = build_model(Xb.shape[1], n_cls, cc)
    Xt = Xb.reshape(-1, Xb.shape[1], 1)
    Xv = X_te.reshape(-1, X_te.shape[1], 1)
    log.info(f"  fit {nome}: epochs<={EPOCHS} batch={BATCH_SIZE}")
    m.fit(Xt, yb, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2,
          validation_split=0.15,
          callbacks=[EarlyStopping(patience=PATIENCE, restore_best_weights=True)])
    yp = np.argmax(m.predict(Xv, verbose=0), axis=1)
    r = metricas(y_te, yp)
    r["estrategia"] = nome
    r["tempo_s"] = time.time() - t0
    r["amostras_treino"] = int(len(yb))
    return r


def main():
    Relatorio(ANALISE_ID)
    log.info("ANÁLISE 2 — Balanceamento (dados reais)")

    if not verificar_dataset(interativo=False):
        log.error("Dataset real ausente — análise abortada.")
        return
    try:
        dados = carregar_dataset_real(n_amostras_max=120_000)
    except Exception as e:
        log_exception(log, "carregar_dataset_real", e); return
    if dados is None:
        log.error("Falha ao carregar dataset real."); return
    X, y, _ = dados

    sc = StandardScaler()
    X = sc.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED)
    n_cls = len(np.unique(y))
    log.info(f"Treino={len(X_tr)} Teste={len(X_te)} Classes={n_cls}")
    log.info(f"Distribuição treino: {dict(Counter(y_tr))}")

    res = []
    for nome, fn in ESTRATEGIAS.items():
        log.info(f">>> {nome}")
        try:
            r = avaliar(X_tr, X_te, y_tr, y_te, n_cls, nome, fn)
            res.append(r)
            log.info(f"{nome} OK F1={r['f1_macro']:.4f} Recall={r['recall_macro']:.4f} "
                     f"FPR={r['fpr_macro']:.4f} MCC={r['mcc']:.4f} t={r['tempo_s']:.1f}s")
        except Exception as e:
            log_exception(log, nome, e)
            res.append({"estrategia": nome, "erro": str(e)})

    log.info(">>> Baseline_RF (referência)")
    try:
        rf = RandomForestClassifier(n_estimators=300, n_jobs=-1,
                                     random_state=RANDOM_SEED,
                                     class_weight="balanced_subsample")
        rf.fit(X_tr, y_tr)
        yp_rf = rf.predict(X_te)
        r_rf = metricas(y_te, yp_rf); r_rf["estrategia"] = "Baseline_RF"
        res.append(r_rf)
        log.info(f"Baseline_RF F1={r_rf['f1_macro']:.4f} Recall={r_rf['recall_macro']:.4f} "
                 f"FPR={r_rf['fpr_macro']:.4f} MCC={r_rf['mcc']:.4f}")
    except Exception as e:
        log_exception(log, "Baseline_RF", e)
        r_rf = None

    df = pd.DataFrame(res)
    csv_path = tab_path(ANALISE_ID, "metricas_balanceamento")
    df.to_csv(csv_path, index=False)
    log.info(f"Tabela: {csv_path}")

    try:
        df_v = df[df.get("erro", pd.Series([np.nan]*len(df))).isna()] if "erro" in df.columns else df
        if not df_v.empty and "f1_macro" in df_v.columns:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            for ax, (col, lbl) in zip(axes.flat,
                                      [("recall_macro", "Recall-macro (PRIMÁRIO)"),
                                       ("f1_macro", "F1-macro"),
                                       ("fpr_macro", "FPR-macro (menor é melhor)"),
                                       ("mcc", "MCC")]):
                sns.barplot(data=df_v, x="estrategia", y=col, ax=ax, palette="viridis")
                ax.set_title(lbl); ax.set_xlabel("")
                ax.tick_params(axis="x", rotation=35)
                for c in ax.containers:
                    ax.bar_label(c, fmt="%.3f", fontsize=8)
            fig.suptitle("Análise 2 — Balanceamento sobre CSE-CIC-IDS2018",
                         fontsize=12, fontweight="bold")
            fig.tight_layout()
            p = fig_path(ANALISE_ID, "comparativo_balanceamento")
            fig.savefig(p, dpi=300); plt.close(fig)
            log.info(f"Figura: {p}")
    except Exception as e:
        log_exception(log, "plot", e)

    cands = [r for r in res if r.get("estrategia") not in ("Baseline_RF",)
             and "recall_macro" in r]
    if cands:
        venc = max(cands, key=lambda r: r["recall_macro"])
        log.info("=" * 62)
        log.info(f"VENCEDOR (recall): {venc['estrategia']}  "
                 f"F1={venc['f1_macro']:.4f} Recall={venc['recall_macro']:.4f} "
                 f"FPR={venc['fpr_macro']:.4f} MCC={venc['mcc']:.4f}")
        if r_rf:
            log.info(f"Baseline RF: F1={r_rf['f1_macro']:.4f} "
                     f"Recall={r_rf['recall_macro']:.4f} FPR={r_rf['fpr_macro']:.4f}")
        if venc["estrategia"] == "Borderline-SMOTE2_Adapt":
            log.info("✓ Justifica Borderline-SMOTE-2 adaptativo no IDS.")
        else:
            log.warning(f"Vencedor difere do IDS — revisar.")
        log.info("=" * 62)


def executar(dataset_disponivel: bool = True, **kwargs) -> None:
    try:
        main()
    except Exception as e:
        log_exception(log, "main", e); raise


if __name__ == "__main__":
    main()
