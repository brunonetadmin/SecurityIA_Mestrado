"""
analise_2_balanceamento.py
==========================
Compara estratégias de tratamento de desbalanceamento de classes para
detecção multiclasse de intrusões sobre o CSE-CIC-IDS2018.

ESTRATÉGIAS AVALIADAS (8):
  Tratamento via PERDA (modificam apenas o objetivo de treinamento):
    1. Sem_Tratamento       — entropia cruzada padrão (baseline)
    2. ClassWeight_Balanced — sklearn class_weight='balanced'
    3. FocalLoss            — Focal Loss pura γ=2.0 (Lin et al., 2017)
    4. CB_FocalLoss         — Class-Balanced Focal Loss (Cui et al., 2019)
  Tratamento via REAMOSTRAGEM (modificam o conjunto de treino):
    5. Undersample_Benign   — undersampling da classe majoritária
    6. SMOTE                — oversampling sintético (Chawla et al., 2002)
    7. Borderline_SMOTE     — versão de fronteira (Han et al., 2005)
    8. SMOTE_ENN            — SMOTE seguido de Edited NN (Batista et al., 2004)

REAMOSTRAGEM aplicada SOMENTE no conjunto de treino (após o split).
Validação e teste preservam a distribuição real, evitando vazamento e
métricas otimistas (Kaufman et al., 2012).

Critério primário: macro_recall + MCC (avaliação multimétrica conforme
discutido no artigo). Cada estratégia em safe_run() — falhas individuais
não interrompem.
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
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

tf.keras.utils.set_random_seed(RANDOM_SEED)
apply_plot_style()

ANALISE_ID = 2
EPOCHS = 25
PATIENCE = 5
log = get_logger(ANALISE_ID, "analise_2")


# ═══════════════════════════════════════════════════════════════════════════
#   MODELO FIXO (MLP+BN) — mantém comparabilidade entre estratégias
# ═══════════════════════════════════════════════════════════════════════════

def build_mlp_bn(n_feat, n_cls, loss_fn):
    inp = Input(shape=(n_feat,))
    x = Dense(256)(inp); x = BatchNormalization()(x); x = Activation("relu")(x); x = Dropout(0.3)(x)
    x = Dense(128)(x);   x = BatchNormalization()(x); x = Activation("relu")(x); x = Dropout(0.3)(x)
    x = Dense(64)(x);    x = BatchNormalization()(x); x = Activation("relu")(x); x = Dropout(0.2)(x)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="MLP_BN")
    m.compile(optimizer=Adam(1e-3), loss=loss_fn, metrics=["accuracy"])
    return m


# ═══════════════════════════════════════════════════════════════════════════
#   PERDAS PERSONALIZADAS
# ═══════════════════════════════════════════════════════════════════════════

def focal_loss_pura(gamma=2.0):
    def fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p = tf.gather(y_pred, y_true, axis=1, batch_dims=1)
        return tf.reduce_mean(tf.pow(1.0 - p, gamma) * (-tf.math.log(p)))
    return fn


def cb_focal_loss(class_counts, gamma=2.0, beta=0.9999):
    """Class-Balanced Focal Loss (Cui et al., 2019).
    Pesos baseados no número de amostras efetivas."""
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


# ═══════════════════════════════════════════════════════════════════════════
#   AVALIAÇÃO GENÉRICA
# ═══════════════════════════════════════════════════════════════════════════

def avaliar(nome, X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls,
            loss_fn, class_weight=None):
    K.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    t0 = time.time()
    m = build_mlp_bn(X_tr.shape[1], n_cls, loss_fn)
    log.info(
        f"  fit {nome}: epochs<={EPOCHS} batch={BATCH_SIZE} "
        f"class_weight={'sim' if class_weight else 'nao'} "
        f"n_treino={len(y_tr):,}"
    )
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
    r.update({"estrategia": nome, "tempo_s": time.time() - t0,
              "n_treino": len(y_tr)})
    return r


# ═══════════════════════════════════════════════════════════════════════════
#   ESTRATÉGIAS — TRATAMENTO VIA PERDA
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
#   ESTRATÉGIAS — TRATAMENTO VIA REAMOSTRAGEM
# ═══════════════════════════════════════════════════════════════════════════

def estrat_undersample_benign(X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls):
    """Undersample da classe majoritária para 5x a 2ª classe maior."""
    dist = Counter(y_tr.astype(int))
    maj = max(dist, key=dist.get)
    sorted_counts = sorted(dist.values(), reverse=True)
    n_2nd = sorted_counts[1] if len(sorted_counts) > 1 else sorted_counts[0]
    target = max(n_2nd * 5, n_2nd)
    target = min(target, dist[maj])
    rng = np.random.default_rng(RANDOM_SEED)
    keep = []
    for c, n in dist.items():
        idx_c = np.where(y_tr == c)[0]
        if c == maj and n > target:
            idx_c = rng.choice(idx_c, size=target, replace=False)
        keep.append(idx_c)
    keep = np.concatenate(keep); rng.shuffle(keep)
    X_tr_u, y_tr_u = X_tr[keep], y_tr[keep]
    log.info(f"  Undersample: {len(y_tr):,} → {len(y_tr_u):,}  "
             f"(majoritária {maj}: {dist[maj]:,} → {target:,})")
    return avaliar("Undersample_Benign", X_tr_u, X_val, X_te, y_tr_u, y_val, y_te, n_cls,
                   loss_fn="sparse_categorical_crossentropy")


def _resample_seguro(sampler, nome, X_tr, y_tr):
    """
    Aplica resampler do imbalanced-learn protegendo contra:
    - classes raras com menos amostras que o k_neighbors padrão
    - falhas silenciosas em datasets multiclasse esparsos
    """
    dist = Counter(y_tr.astype(int))
    rare = [c for c, n in dist.items() if n < 6]
    if rare:
        log.info(f"  {nome}: {len(rare)} classes raras (<6 amostras): {rare}")
    log.info(f"  {nome}: aplicando resampler em {len(y_tr):,} amostras...")
    t0 = time.time()
    X_res, y_res = sampler.fit_resample(X_tr, y_tr)
    log.info(f"  {nome}: {len(y_tr):,} → {len(y_res):,} amostras  "
             f"(+{(len(y_res)-len(y_tr))/len(y_tr)*100:.1f}%) em {time.time()-t0:.1f}s")
    return X_res, y_res


def estrat_smote(X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls):
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError as e:
        raise RuntimeError("imblearn não instalado. pip install imbalanced-learn") from e
    # k_neighbors adaptativo — limitado pela menor classe não-singleton
    dist = Counter(y_tr.astype(int))
    min_n = min(n for n in dist.values() if n > 1)
    k = max(1, min(5, min_n - 1))
    sampler = SMOTE(k_neighbors=k, random_state=RANDOM_SEED, n_jobs=-1)
    X_res, y_res = _resample_seguro(sampler, "SMOTE", X_tr, y_tr)
    return avaliar("SMOTE", X_res, X_val, X_te, y_res, y_val, y_te, n_cls,
                   loss_fn="sparse_categorical_crossentropy")


def estrat_borderline_smote(X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls):
    try:
        from imblearn.over_sampling import BorderlineSMOTE
    except ImportError as e:
        raise RuntimeError("imblearn não instalado. pip install imbalanced-learn") from e
    dist = Counter(y_tr.astype(int))
    min_n = min(n for n in dist.values() if n > 1)
    k = max(1, min(5, min_n - 1))
    sampler = BorderlineSMOTE(
        k_neighbors=k, kind="borderline-2",
        random_state=RANDOM_SEED, n_jobs=-1,
    )
    X_res, y_res = _resample_seguro(sampler, "Borderline_SMOTE", X_tr, y_tr)
    return avaliar("Borderline_SMOTE", X_res, X_val, X_te, y_res, y_val, y_te, n_cls,
                   loss_fn="sparse_categorical_crossentropy")


def estrat_smote_enn(X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls):
    try:
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import EditedNearestNeighbours
    except ImportError as e:
        raise RuntimeError("imblearn não instalado. pip install imbalanced-learn") from e
    dist = Counter(y_tr.astype(int))
    min_n = min(n for n in dist.values() if n > 1)
    k_smote = max(1, min(5, min_n - 1))
    k_enn = 3  # padrão da literatura (Batista et al., 2004)
    sampler = SMOTEENN(
        smote=SMOTE(k_neighbors=k_smote, random_state=RANDOM_SEED, n_jobs=-1),
        enn=EditedNearestNeighbours(n_neighbors=k_enn, n_jobs=-1),
        random_state=RANDOM_SEED,
    )
    X_res, y_res = _resample_seguro(sampler, "SMOTE_ENN", X_tr, y_tr)
    # SMOTE-ENN pode remover classes inteiras se ENN for agressivo;
    # resguarda contra crash em treinos posteriores.
    classes_resultantes = set(np.unique(y_res).tolist())
    classes_originais = set(np.unique(y_tr).tolist())
    if classes_resultantes != classes_originais:
        faltantes = classes_originais - classes_resultantes
        log.info(f"  SMOTE_ENN: ENN removeu classes {faltantes}; reinjetando 1 amostra de cada.")
        for c in faltantes:
            idx = np.where(y_tr == c)[0][:1]
            X_res = np.concatenate([X_res, X_tr[idx]])
            y_res = np.concatenate([y_res, y_tr[idx]])
    return avaliar("SMOTE_ENN", X_res, X_val, X_te, y_res, y_val, y_te, n_cls,
                   loss_fn="sparse_categorical_crossentropy")


# ═══════════════════════════════════════════════════════════════════════════
#   PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

ESTRATEGIAS = [
    # tratamento via perda
    ("Sem_Tratamento",       estrat_sem_tratamento),
    ("ClassWeight_Balanced", estrat_class_weight),
    ("FocalLoss",            estrat_focal_pura),
    ("CB_FocalLoss",         estrat_cb_focal),
    # tratamento via reamostragem
    ("Undersample_Benign",   estrat_undersample_benign),
    ("SMOTE",                estrat_smote),
    ("Borderline_SMOTE",     estrat_borderline_smote),
    ("SMOTE_ENN",            estrat_smote_enn),
]


def executar(dataset_disponivel: bool = True) -> None:
    log.info(f"ANÁLISE 2 — Tratamento de Desbalanceamento ({len(ESTRATEGIAS)} estratégias)")

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
    log.info(f"Distribuição treino: {dict(Counter(y_tr.astype(int)))}")

    resultados = []
    for nome, fn in ESTRATEGIAS:
        out = safe_run(
            log, nome,
            lambda f=fn: f(X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls),
        )
        if out is not None:
            resultados.append(out)
            log.info(
                f"{nome}: recall={out['recall_macro']:.4f} mcc={out['mcc']:.4f} "
                f"f1={out['f1_macro']:.4f} fpr={out['fpr_macro']:.4f}"
            )

    if not resultados:
        log.error("Nenhuma estratégia concluída com sucesso."); return

    df = pd.DataFrame(resultados)
    csv_path = tab_path(ANALISE_ID, "metricas_balanceamento")
    safe_run(log, "salvar CSV", lambda: df.to_csv(csv_path, index=False))
    log.info(f"Tabela: {csv_path}")
    safe_run(log, "plot_comparativo", lambda: _plot(df))

    vencedor = df.sort_values("recall_macro", ascending=False).iloc[0]
    log.info("=" * 62)
    log.info(
        f"VENCEDOR (recall_macro): {vencedor['estrategia']}  "
        f"recall={vencedor['recall_macro']:.4f} mcc={vencedor['mcc']:.4f}"
    )
    log.info("=" * 62)


def _plot(df: pd.DataFrame) -> None:
    """Painel 2x2: Recall, MCC, F1, FPR.
    Sem título embutido; paleta neutra para impressão."""
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    metricas = [
        ("recall_macro", "Recall-macro",            ax[0, 0]),
        ("mcc",          "MCC",                     ax[0, 1]),
        ("f1_macro",     "F1-macro",                ax[1, 0]),
        ("fpr_macro",    "FPR-macro (menor=melhor)", ax[1, 1]),
    ]
    cores = sns.color_palette("Greys", n_colors=len(df) + 2)[2:]
    for col, titulo, a in metricas:
        sns.barplot(data=df, x="estrategia", y=col, ax=a, palette=cores, edgecolor="black")
        a.set_title(titulo, fontsize=13, fontweight="bold")
        a.set_xlabel(""); a.set_ylabel(col)
        for tick in a.get_xticklabels():
            tick.set_rotation(35); tick.set_ha("right")
        for p in a.patches:
            v = p.get_height()
            a.annotate(f"{v:.3f}", (p.get_x() + p.get_width()/2, v),
                       ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path(ANALISE_ID, "comparativo_balanceamento"), dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    executar()
