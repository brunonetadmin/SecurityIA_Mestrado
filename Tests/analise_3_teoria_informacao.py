"""
analise_3_teoria_informacao.py — Seleção de Features sobre dados reais.
Versão blindada com EpochLogger, executar() à prova de exceção.
"""
import os, sys, time, math, warnings
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
from _test_logging import get_logger, log_exception, EpochLogger, silence_tensorflow

silence_tensorflow()

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import (
    mutual_info_classif, f_classif, chi2, SelectKBest,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import (
    accuracy_score, f1_score, matthews_corrcoef,
    confusion_matrix, recall_score, precision_score,
)
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
from scipy.stats import entropy as scipy_entropy

tf.keras.utils.set_random_seed(RANDOM_SEED)
apply_plot_style()

ANALISE_ID = 3
EPOCHS = 18
PATIENCE = 6
K_SELECT = 23
log = get_logger(ANALISE_ID, "analise_3")


def sanitize(X):
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    p99 = np.percentile(np.abs(X), 99.9, axis=0)
    p99 = np.where(p99 < 1e-9, 1.0, p99)
    return np.clip(X, -p99 * 10, p99 * 10).astype(np.float32)


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
    cc = np.maximum(cc, 1.0)
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


def _safe_k(y, k_default=5):
    counts = list(Counter(y).values())
    return max(1, min(k_default, min(counts) - 1))


def balance_adaptive(X, y):
    """Borderline-SMOTE-2 adaptativo (IDS)."""
    dist = Counter(y)
    n_maj = max(dist.values())
    maj = max(dist, key=dist.get)
    strat, ks = {}, []
    for c, n_c in dist.items():
        if c == maj: continue
        n_t = max(min(10 * n_c, int(0.30 * n_maj)), n_c)
        strat[c] = n_t
        ks.append(min(11, max(1, math.ceil(0.25 * math.sqrt(n_c)))))
    if not strat: return X, y
    k_med = max(1, min(int(np.median(ks)), min(dist.values()) - 1))
    Xr, yr = BorderlineSMOTE(sampling_strategy=strat, k_neighbors=k_med,
                              kind="borderline-2",
                              random_state=RANDOM_SEED).fit_resample(X, y)
    t_maj = min(int(2.0 * max(strat.values())), dist[maj])
    Xr, yr = RandomUnderSampler(sampling_strategy={maj: t_maj},
                                 random_state=RANDOM_SEED).fit_resample(Xr, yr)
    try:
        Xr, yr = EditedNearestNeighbours(n_neighbors=3).fit_resample(Xr, yr)
    except Exception as exc:
        log.warning(f"ENN falhou ({exc}); prosseguindo sem ENN.")
    return Xr, yr


# ─── Estratégias ───────────────────────────────────────────────────────────

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


def sel_none(X, y, k):
    return np.arange(X.shape[1])

def sel_ig(X, y, k):
    ig = information_gain(X, y)
    return np.argsort(ig)[::-1][:min(k, X.shape[1])]

def sel_mi(X, y, k):
    mi = mutual_info_classif(X, y, random_state=RANDOM_SEED)
    return np.argsort(mi)[::-1][:min(k, X.shape[1])]

def sel_chi2(X, y, k):
    Xp = MinMaxScaler().fit_transform(X)
    sk = SelectKBest(chi2, k=min(k, X.shape[1])).fit(Xp, y)
    scores = np.nan_to_num(sk.scores_, nan=0.0)
    return np.argsort(scores)[::-1][:min(k, X.shape[1])]

def sel_anova(X, y, k):
    sk = SelectKBest(f_classif, k=min(k, X.shape[1])).fit(X, y)
    scores = np.nan_to_num(sk.scores_, nan=0.0)
    return np.argsort(scores)[::-1][:min(k, X.shape[1])]

def sel_ig_mi_60_40(X, y, k):
    """Estratégia oficial do IDS — IG+MI 60/40."""
    ig = information_gain(X, y)
    mi = mutual_info_classif(X, y, random_state=RANDOM_SEED)
    eps = 1e-9
    ig_n = (ig - ig.min()) / (ig.max() - ig.min() + eps)
    mi_n = (mi - mi.min()) / (mi.max() - mi.min() + eps)
    s = 0.6 * ig_n + 0.4 * mi_n
    return np.argsort(s)[::-1][:min(k, X.shape[1])]


ESTRATEGIAS = {
    "Sem_Selecao":      sel_none,
    "IG_puro":           sel_ig,
    "MI_puro":           sel_mi,
    "Chi2":              sel_chi2,
    "ANOVA_F":           sel_anova,
    "IG+MI_60_40":       sel_ig_mi_60_40,
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


def avaliar(X_tr, X_te, y_tr, y_te, n_cls, nome, fn_sel, k):
    K.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    t0 = time.time()
    log.info(f"  selecionando features ({nome})…")
    idx = fn_sel(X_tr, y_tr, k)
    log.info(f"  selecionadas {len(idx)} features de {X_tr.shape[1]}")
    Xs_tr = X_tr[:, idx]; Xs_te = X_te[:, idx]
    Xb, yb = balance_adaptive(Xs_tr, y_tr)
    cc = np.maximum(np.bincount(yb.astype(int), minlength=n_cls), 1)
    m = build_model(Xb.shape[1], n_cls, cc)
    Xt = Xb.reshape(-1, Xb.shape[1], 1)
    Xv = Xs_te.reshape(-1, Xs_te.shape[1], 1)
    log.info(f"  fit {nome}: epochs<={EPOCHS} batch={BATCH_SIZE}")
    cb = [
        EarlyStopping(patience=PATIENCE, restore_best_weights=True,
                       monitor="val_loss"),
        EpochLogger(log, prefix=f"{nome} "),
    ]
    m.fit(Xt, yb, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0,
          validation_split=0.15, callbacks=cb)
    yp = np.argmax(m.predict(Xv, verbose=0, batch_size=BATCH_SIZE), axis=1)
    r = metricas(y_te, yp)
    r["estrategia"] = nome
    r["k_features"] = int(len(idx))
    r["tempo_s"] = time.time() - t0
    return r


def main():
    rel = Relatorio(ANALISE_ID)
    log.info(f"ANÁLISE 3 — Seleção de Features (k={K_SELECT})")

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
    log.info(f"Treino={len(X_tr)} Teste={len(X_te)} Classes={n_cls} Features={X.shape[1]}")

    res = []
    for nome, fn in ESTRATEGIAS.items():
        log.info(f">>> {nome}")
        try:
            r = avaliar(X_tr, X_te, y_tr, y_te, n_cls, nome, fn, K_SELECT)
            res.append(r)
            log.info(f"{nome} OK F1={r['f1_macro']:.4f} Recall={r['recall_macro']:.4f} "
                     f"FPR={r['fpr_macro']:.4f} MCC={r['mcc']:.4f} t={r['tempo_s']:.1f}s")
        except Exception as e:
            log_exception(log, nome, e)
            res.append({"estrategia": nome, "erro": str(e)})

    log.info(">>> Baseline_RF (referência)")
    r_rf = None
    try:
        rf = RandomForestClassifier(
            n_estimators=300, n_jobs=-1, random_state=RANDOM_SEED,
            class_weight="balanced_subsample",
            max_samples=min(50_000, len(X_tr)) if len(X_tr) > 50_000 else None,
        )
        rf.fit(X_tr, y_tr)
        yp_rf = rf.predict(X_te)
        r_rf = metricas(y_te, yp_rf); r_rf["estrategia"] = "Baseline_RF"
        res.append(r_rf)
        log.info(f"Baseline_RF F1={r_rf['f1_macro']:.4f} Recall={r_rf['recall_macro']:.4f} "
                 f"FPR={r_rf['fpr_macro']:.4f}")
    except Exception as e:
        log_exception(log, "Baseline_RF", e)

    df = pd.DataFrame(res)
    try:
        csv_path = tab_path(ANALISE_ID, "metricas_selecao")
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
                                      [("recall_macro", "Recall-macro (PRIMÁRIO)"),
                                       ("f1_macro", "F1-macro"),
                                       ("fpr_macro", "FPR-macro"),
                                       ("mcc", "MCC")]):
                sns.barplot(data=df_v, x="estrategia", y=col, ax=ax, palette="rocket")
                ax.set_title(lbl); ax.set_xlabel(""); ax.tick_params(axis="x", rotation=30)
                for c in ax.containers:
                    ax.bar_label(c, fmt="%.3f", fontsize=8)
            fig.suptitle(f"Análise 3 — Seleção de Features (k={K_SELECT})",
                         fontsize=12, fontweight="bold")
            fig.tight_layout()
            p = fig_path(ANALISE_ID, "comparativo_selecao")
            fig.savefig(p, dpi=300); plt.close(fig)
            log.info(f"Figura: {p}")
    except Exception as e:
        log_exception(log, "plot", e)

    cands = [r for r in res if r.get("estrategia") != "Baseline_RF"
             and "recall_macro" in r]
    if cands:
        venc = max(cands, key=lambda r: r["recall_macro"])
        log.info("=" * 62)
        log.info(f"VENCEDOR (recall): {venc['estrategia']}  "
                 f"F1={venc['f1_macro']:.4f} Recall={venc['recall_macro']:.4f} "
                 f"FPR={venc['fpr_macro']:.4f}")
        if r_rf:
            log.info(f"Baseline RF: F1={r_rf['f1_macro']:.4f} Recall={r_rf['recall_macro']:.4f}")
        if venc["estrategia"] == "IG+MI_60_40":
            log.info(f"✓ Justifica IG+MI 60/40 com k={K_SELECT} no IDS.")
        else:
            log.warning(f"Vencedor difere do IDS — revisar.")
        log.info("=" * 62)

    # Persistir relatório markdown (cria Relatorio_N.md, sinalizando "gerado" ao menu)
    try:
        try:
            rel.secao("Resumo dos Resultados")
            if "df" in locals() and not df.empty:
                rel.tabela_df(df, "Métricas consolidadas")
            rel.secao("Log de execução")
            rel.texto(f"Log completo em: `Tests/Logs/analise_{ANALISE_ID}_*.log`")
        except Exception:
            pass
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
