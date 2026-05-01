"""
analise_3_teoria_informacao.py
==============================
Análise da aplicabilidade da Teoria da Informação à seleção de atributos
para detecção de intrusões sobre o CSE-CIC-IDS2018.

A versão anterior comparava apenas critérios de RELEVÂNCIA (IG, MI puros).
Esta versão amplia a investigação para métodos que tratam REDUNDÂNCIA entre
atributos — limitação conhecida de IG/MI univariados (Brown et al., 2012).

MÉTODOS AVALIADOS:
  Baselines (relevância univariada):
    1. Information_Gain      — Shannon (1948)
    2. Mutual_Information    — sklearn mutual_info_classif
    3. ANOVA_F               — F-test linear (controle clássico)
    4. RF_Feature_Importance — impureza Gini sobre RF (Breiman, 2001)
    5. IG_MI_60_40           — combinação ponderada (estratégia anterior)
  Critérios multivariados (relevância + redundância):
    6. mRMR                  — Min Redundancy, Max Relevance (Peng et al., 2005)
    7. JMI                   — Joint Mutual Information (Yang & Moody, 1999;
                                                          Brown et al., 2012)
    8. CMIM                  — Conditional MI Maximisation (Fleuret, 2004)

Grade de k: [10, 15, 23, 32, 48, 'all']  → 8 métodos × 6 valores = 48 combos.
Critério primário: macro_recall. Cada combinação em safe_run().
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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif

tf.keras.utils.set_random_seed(RANDOM_SEED)
apply_plot_style()

ANALISE_ID = 3
EPOCHS = 18
PATIENCE = 5
K_GRID = [10, 15, 23, 32, 48, "all"]
log = get_logger(ANALISE_ID, "analise_3")

def _run(label, fn, *a, **kw):
    """Wrapper: executa `fn` via safe_run e retorna apenas o resultado.
    Se `fn` falhar, devolve None (safe_run já loga a exceção)."""
    ok, res = safe_run(log, label, fn, *a, **kw)
    return res if ok else None




# ═══════════════════════════════════════════════════════════════════════════
#   MODELO FIXO (MLP+BN) — mantém comparabilidade entre métodos
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


# ═══════════════════════════════════════════════════════════════════════════
#   ESCORES UNIVARIADOS  (baselines)
# ═══════════════════════════════════════════════════════════════════════════

def _discretize_quantile(X, n_bins=10):
    """Discretiza X (n×d) em bins por quantil para estimadores baseados em entropia."""
    n, d = X.shape
    Xd = np.empty_like(X, dtype=np.int32)
    for j in range(d):
        col = X[:, j]
        try:
            bins = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)))
            if bins.size < 2:
                Xd[:, j] = 0
            else:
                Xd[:, j] = np.clip(np.digitize(col, bins[1:-1]), 0, n_bins - 1)
        except Exception:
            Xd[:, j] = 0
    return Xd


def _entropy(p):
    p = p[p > 0]
    return -float(np.sum(p * np.log2(p)))


def information_gain(X, y):
    Xd = _discretize_quantile(X)
    n = len(y)
    py = np.bincount(y) / n
    Hy = _entropy(py)
    igs = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        col = Xd[:, j]
        Hyx = 0.0
        for v in np.unique(col):
            mask = col == v
            p_v = mask.mean()
            sub = y[mask]
            Hyx += p_v * _entropy(np.bincount(sub) / sub.size)
        igs[j] = Hy - Hyx
    return igs


def score_mi(X, y):
    return mutual_info_classif(X, y, random_state=RANDOM_SEED, n_neighbors=3)


def score_anova(X, y):
    f, _ = f_classif(X, y)
    return np.nan_to_num(f, nan=0.0)


def score_rf_importance(X, y, n_sub=20000):
    n_sub = min(n_sub, len(X))
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X), size=n_sub, replace=False)
    rf = RandomForestClassifier(
        n_estimators=200, n_jobs=-1, random_state=RANDOM_SEED,
        class_weight="balanced_subsample",
    )
    rf.fit(X[idx], y[idx])
    return rf.feature_importances_


def score_ig_mi_60_40(X, y):
    ig = information_gain(X, y)
    mi = score_mi(X, y)
    eps = 1e-9
    ig_n = (ig - ig.min()) / (ig.max() - ig.min() + eps)
    mi_n = (mi - mi.min()) / (mi.max() - mi.min() + eps)
    return 0.6 * ig_n + 0.4 * mi_n


# ═══════════════════════════════════════════════════════════════════════════
#   CRITÉRIOS MULTIVARIADOS (relevância + redundância)
# ═══════════════════════════════════════════════════════════════════════════
#
#   IMPORTANTE: estes métodos são GULOSOS e dependentes de k. Não devolvem um
#   vetor de escores ordenável — devolvem diretamente uma SEQUÊNCIA ORDENADA
#   de índices selecionados. Por consistência com a interface dos demais
#   métodos, retornamos um "ranking score" sintético = 1/(rank+1).

def _pairwise_mi_matrix(X, n_sub=10000):
    """Matriz d×d de MI entre pares de atributos (estimada por subamostragem)."""
    n_sub = min(n_sub, len(X))
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X), size=n_sub, replace=False)
    Xs = X[idx]
    d = Xs.shape[1]
    M = np.zeros((d, d), dtype=np.float32)
    log.info(f"  pré-computando matriz MI {d}x{d} (subamostra {n_sub})...")
    for j in range(d):
        # MI(x_j ; x_k) para todo k>=j  (simétrico)
        target = Xs[:, j]
        target_disc = pd.qcut(pd.Series(target), q=10, labels=False, duplicates="drop").to_numpy()
        target_disc = np.nan_to_num(target_disc, nan=-1).astype(int)
        if target_disc.max() < 1:
            M[j, j:] = 0.0; continue
        M[j, j:] = mutual_info_classif(
            Xs[:, j:], target_disc, random_state=RANDOM_SEED, n_neighbors=3,
        )
        M[j:, j] = M[j, j:]
    return M


def _greedy_select(rel, MI_pair, k_max, criterio):
    """
    Seleção gulosa multivariada.
      rel       : array (d,) com relevância MI(x_j ; y).
      MI_pair   : matriz (d,d) com MI(x_j ; x_k).
      criterio  : 'mrmr' | 'jmi' | 'cmim'
    Retorna lista ordenada de índices selecionados.
    """
    d = len(rel)
    k_max = min(k_max, d)
    selected = []
    remaining = list(range(d))
    selected.append(int(np.argmax(rel)))
    remaining.remove(selected[0])

    while len(selected) < k_max and remaining:
        best_j, best_score = None, -np.inf
        S = np.array(selected)
        for j in remaining:
            if criterio == "mrmr":
                redund = MI_pair[j, S].mean()
                score = rel[j] - redund
            elif criterio == "jmi":
                # JMI: rel[j] + soma sobre s in S de MI(x_j , x_s ; y)
                # Aproximação clássica: usa (MI(j,y) - MI(j,s)) somada
                score = float(np.sum(rel[j] + rel[S] - MI_pair[j, S]))
            elif criterio == "cmim":
                # CMIM: minimiza max_s MI(x_j ; x_s | y)
                # Aproximação: rel[j] - max_s MI(j, s)
                score = rel[j] - MI_pair[j, S].max()
            else:
                raise ValueError(f"Critério inválido: {criterio}")
            if score > best_score:
                best_score = score; best_j = j
        selected.append(int(best_j))
        remaining.remove(best_j)

    return selected


def _multivariado_factory(criterio: str):
    """Devolve uma função score(X,y) compatível com a interface dos demais métodos."""
    def _score(X, y):
        rel = score_mi(X, y)
        MI_pair = _pairwise_mi_matrix(X)
        order = _greedy_select(rel, MI_pair, k_max=X.shape[1], criterio=criterio)
        # ranking → score sintético (maior = mais bem ranqueado)
        scores = np.zeros(X.shape[1])
        for rank, j in enumerate(order):
            scores[j] = 1.0 / (rank + 1)
        return scores
    return _score


METODOS = {
    # Univariados (baselines)
    "Information_Gain":      information_gain,
    "Mutual_Information":    score_mi,
    "ANOVA_F":               score_anova,
    "RF_Feature_Importance": score_rf_importance,
    "IG_MI_60_40":           score_ig_mi_60_40,
    # Multivariados (relevância + redundância)
    "mRMR":                  _multivariado_factory("mrmr"),
    "JMI":                   _multivariado_factory("jmi"),
    "CMIM":                  _multivariado_factory("cmim"),
}


def selecionar_top_k(scores, k, n_features):
    if isinstance(k, str) and k == "all":
        return np.arange(n_features)
    k = min(int(k), n_features)
    return np.argsort(scores)[::-1][:k]


# ═══════════════════════════════════════════════════════════════════════════
#   AVALIAÇÃO
# ═══════════════════════════════════════════════════════════════════════════

def avaliar_combinacao(metodo_nome, k, scores, X_tr, X_val, X_te,
                        y_tr, y_val, y_te, n_cls):
    K.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    t0 = time.time()
    idx = selecionar_top_k(scores, k, X_tr.shape[1])
    Xtr_s = X_tr[:, idx]
    Xv_s  = X_val[:, idx]
    Xte_s = X_te[:, idx]
    log.info(f"  {metodo_nome} k={k}: {len(idx)} features de {X_tr.shape[1]}")

    m = build_mlp_bn(Xtr_s.shape[1], n_cls)
    cb = [
        EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, monitor="val_loss"),
        EpochLogger(log, prefix=f"{metodo_nome}_k{k} "),
    ]
    m.fit(Xtr_s, y_tr, validation_data=(Xv_s, y_val),
          epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cb, verbose=0)
    y_pred = np.argmax(m.predict(Xte_s, batch_size=BATCH_SIZE, verbose=0), axis=1)
    out = metricas_completas(y_te, y_pred, n_cls)
    out.update(metodo=metodo_nome, k=str(k), k_real=len(idx), tempo_s=time.time() - t0)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#   PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def executar(dataset_disponivel: bool = True) -> None:
    log.info(f"ANÁLISE 3 — Teoria da Informação ({len(METODOS)} métodos × {len(K_GRID)} valores de k)")

    # Carregamento direto (sem safe_run): a função pode retornar
    # (X, y) ou (X, y, label_encoder). O safe_run anterior embrulhava o
    # resultado em uma estrutura aninhada que quebrava np.asarray(y).
    log.info(">>> carregar_dataset_real")
    try:
        _t0 = time.time()
        _ds = carregar_dataset_real()
        log.info(f"<<< carregar_dataset_real OK ({time.time()-_t0:.1f}s)")
    except Exception as _e:
        log_exception(log, "carregar_dataset_real", _e)
        log.error("Sem dataset real. Abortando."); return
    if _ds is None:
        log.error("carregar_dataset_real retornou None. Abortando."); return
    if not isinstance(_ds, (tuple, list)) or len(_ds) < 2:
        log.error(f"Formato inesperado do dataset: type={type(_ds).__name__}, "
                  f"len={len(_ds) if hasattr(_ds, '__len__') else 'N/A'}. Abortando.")
        return
    Xfull, yfull = _ds[0], _ds[1]
    # Garantir tipos antes de QUALQUER operação numérica
    Xfull = np.ascontiguousarray(Xfull, dtype=np.float32)
    yfull = np.ascontiguousarray(yfull).astype(np.int64).ravel()
    log.info(f"Dataset coerido: X={Xfull.shape} y={yfull.shape}")
    n_cls = int(np.max(yfull) + 1)
    log.info(f"Dataset: {Xfull.shape[0]:,} amostras × {Xfull.shape[1]} features × {n_cls} classes")

    X_tr_raw, X_val_raw, X_te_raw, y_tr, y_val, y_te = stratified_split_3way(
        Xfull, yfull, val_frac=0.15, test_frac=0.15, seed=RANDOM_SEED,
    )
    X_tr, X_val, X_te, _scaler = fit_scaler_no_leakage(X_tr_raw, X_val_raw, X_te_raw)
    log.info(f"Split: treino={len(y_tr):,} val={len(y_val):,} teste={len(y_te):,}")

    resultados = []
    for metodo_nome, score_fn in METODOS.items():
        log.info(f">>> Computando scores: {metodo_nome}")
        scores = _run(f"score_{metodo_nome}", lambda f=score_fn: f(X_tr, y_tr))
        if scores is None:
            log.error(f"FALHA em scores de {metodo_nome}. Pulando."); continue
        for k in K_GRID:
            tag = f"{metodo_nome}_k{k}"
            out = _run(
                tag,
                lambda mn=metodo_nome, kk=k, sc=scores:
                    avaliar_combinacao(mn, kk, sc, X_tr, X_val, X_te,
                                       y_tr, y_val, y_te, n_cls),
            )
            if out is not None:
                resultados.append(out)

    if not resultados:
        log.error("Nenhuma combinação concluída com sucesso."); return

    df = pd.DataFrame(resultados)
    csv_path = tab_path(ANALISE_ID, "metricas_selecao")
    _run("salvar CSV", lambda: df.to_csv(csv_path, index=False))
    log.info(f"Tabela: {csv_path}")
    _run("plot_comparativo", lambda: _plot(df))

    vencedor = df.sort_values("recall_macro", ascending=False).iloc[0]
    log.info("=" * 62)
    log.info(
        f"VENCEDOR (recall_macro): {vencedor['metodo']} (k={vencedor['k']})  "
        f"recall={vencedor['recall_macro']:.4f} mcc={vencedor['mcc']:.4f}"
    )
    log.info("=" * 62)


def _plot(df: pd.DataFrame) -> None:
    """Painel 2x2: curvas (recall, MCC, F1, FPR) vs. k para cada método.
    Sem título embutido; paleta neutra."""
    df = df.copy()
    df["k_num"] = df["k_real"].astype(int)
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    metricas = [
        ("recall_macro", "Recall-macro", ax[0, 0]),
        ("mcc",          "MCC",          ax[0, 1]),
        ("f1_macro",     "F1-macro",     ax[1, 0]),
        ("fpr_macro",    "FPR-macro",    ax[1, 1]),
    ]
    metodos = sorted(df["metodo"].unique())
    cores = sns.color_palette("Greys", n_colors=len(metodos) + 2)[2:]
    estilos = ["-", "--", "-.", ":", "-", "--", "-.", ":"][:len(metodos)]
    marcadores = ["o", "s", "^", "D", "v", "P", "X", "*"][:len(metodos)]
    for col, titulo, a in metricas:
        for i, met in enumerate(metodos):
            sub = df[df["metodo"] == met].sort_values("k_num")
            a.plot(sub["k_num"], sub[col],
                   linestyle=estilos[i], marker=marcadores[i],
                   color=cores[i], label=met, linewidth=1.5, markersize=6)
        a.set_title(titulo, fontsize=13, fontweight="bold")
        a.set_xlabel("k (atributos selecionados)"); a.set_ylabel(col)
        a.grid(True, alpha=0.3); a.legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(fig_path(ANALISE_ID, "comparativo_selecao"), dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    executar()
