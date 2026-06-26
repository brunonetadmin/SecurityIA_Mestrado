"""
analise_1_arquiteturas.py
=========================
Comparação de arquiteturas para detecção multiclasse de intrusões sobre
fluxos do CSE-CIC-IDS2018.

MODELOS AVALIADOS (7):
  Tabulares clássicos (árvores):
    1. RandomForest          — baseline forte (Breiman, 2001)
    2. ExtraTrees            — splits aleatórios (Geurts et al., 2006)
    3. XGBoost               — gradient boosting regularizado (Chen & Guestrin, 2016)
    4. CatBoost              — ordered boosting (Prokhorenkova et al., 2018)
  Redes densas:
    5. MLP+BatchNorm         — normalização em lote (Ioffe & Szegedy, 2015)
    6. ResNet Tabular        — conexões residuais (He et al., 2016)
    7. SNN                   — auto-normalização SELU/AlphaDropout (Klambauer et al., 2017)

Critério de seleção: hierárquico (MCC>=0,80; FPR-macro<=0,010; depois maximiza
recall-macro). Secundários reportados: F1-macro, tempo.

CONDIÇÃO JUSTA: todas as 7 arquiteturas são comparadas SEM balanceamento
(cru), de modo que a Inv. 1 isole a capacidade da arquitetura. O tratamento
de desbalanceamento é investigado na Inv. 2 sobre o vencedor; o resultado cru
do vencedor aqui coincide com o 'Sem_Tratamento' dele na Inv. 2.
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
from _pipeline import selecionar_por_criterio, familia_de, salvar_estado

# Desempate final do critério hierárquico na Inv. 1:
#   "recall" → critério da dissertação (vence o RandomForest)
#   "mcc"    → vence o XGBoost
# Este é o ÚNICO ponto que decide o modelo que percorre as Inv. 2-4.
DESEMPATE_INV1 = "recall"

silence_tensorflow()

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    Input, Dense, Dropout, BatchNormalization, Activation, Add, AlphaDropout,
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

def _run(label, fn, *a, **kw):
    """Wrapper: executa `fn` via safe_run e retorna apenas o resultado.
    Se `fn` falhar, devolve None (safe_run já loga a exceção)."""
    ok, res = safe_run(log, label, fn, *a, **kw)
    return res if ok else None




# ═══════════════════════════════════════════════════════════════════════════
#   MODELOS TABULARES CLÁSSICOS
# ═══════════════════════════════════════════════════════════════════════════

def avaliar_random_forest(X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls):
    log.info("  fit RandomForest: 300 árvores (cru — Inv.1 isola a arquitetura)")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300, n_jobs=-1, random_state=RANDOM_SEED,
        class_weight=None, max_depth=None, min_samples_leaf=2,
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
        class_weight=None, max_depth=None, min_samples_leaf=2,
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
        auto_class_weights=None,
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


def build_snn(n_feat, n_cls, hidden=(256, 128, 64), dropout=0.10):
    """Self-Normalizing Network (Klambauer et al., 2017): camadas densas com
    ativação SELU, inicialização lecun_normal e AlphaDropout. Mantém as
    ativações normalizadas SEM BatchNorm — terceiro paradigma denso, distinto
    de MLP+BN (normalização em lote) e ResNet (conexões residuais)."""
    inp = Input(shape=(n_feat,))
    x = inp
    for u in hidden:
        x = Dense(u, activation="selu", kernel_initializer="lecun_normal")(x)
        x = AlphaDropout(dropout)(x)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="SNN")
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
    avaliacoes = [
        ("RandomForest",   lambda: avaliar_random_forest(X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls)),
        ("ExtraTrees",     lambda: avaliar_extra_trees  (X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls)),
        ("XGBoost",        lambda: avaliar_xgboost      (X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls)),
        ("CatBoost",       lambda: avaliar_catboost     (X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls)),
        ("MLP_BN",         lambda: avaliar_rede("MLP_BN",         build_mlp_bn,         X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls)),
        ("ResNet_Tabular", lambda: avaliar_rede("ResNet_Tabular", build_resnet_tabular, X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls)),
        ("SNN",            lambda: avaliar_rede("SNN",            build_snn,            X_tr, X_val, X_te, y_tr, y_val, y_te, n_cls)),
    ]
    for nome, fn in avaliacoes:
        out = _run(nome, fn)
        if out is not None:
            resultados.append(out)

    if not resultados:
        log.error("Nenhum modelo concluído com sucesso."); return

    df = pd.DataFrame(resultados)
    df["familia"] = df["modelo"].map(familia_de)
    # Ordena por família (árvore antes de rede) e recall, para o relatório
    df = df.sort_values(["familia", "recall_macro"], ascending=[True, False]).reset_index(drop=True)
    csv_path = tab_path(ANALISE_ID, "metricas_arquiteturas")
    _run("salvar CSV", lambda: df.to_csv(csv_path, index=False))
    log.info(f"Tabela: {csv_path}")
    _run("plot_comparativo", lambda: _plot(df))

    vencedor = selecionar_por_criterio(df, coluna_id="modelo",
                                       desempate=DESEMPATE_INV1, logger=log)
    fam = familia_de(vencedor["modelo"])
    log.info("=" * 62)
    log.info(
        f"VENCEDOR (critério hierárquico, desempate={DESEMPATE_INV1}): "
        f"{vencedor['modelo']} [{fam}]  "
        f"recall={vencedor['recall_macro']:.4f} mcc={vencedor['mcc']:.4f} "
        f"f1_macro={vencedor['f1_macro']:.4f} fpr={vencedor['fpr_macro']:.4f} "
        f"(passou_criterio={vencedor['passou_criterio']})"
    )
    # Transparência: líder por métrica isolada (pode divergir do vencedor)
    for met in ("recall_macro", "mcc", "f1_macro"):
        lid = df.sort_values(met, ascending=False).iloc[0]
        log.info(f"  líder {met}: {lid['modelo']} ({lid[met]:.4f})")
    # Separação por família: melhor de cada uma sob o critério hierárquico
    log.info("-" * 62)
    log.info("MELHOR POR FAMÍLIA (critério hierárquico):")
    for fam_nome in ("arvore", "rede"):
        sub = df[df["familia"] == fam_nome]
        if sub.empty:
            continue
        v = selecionar_por_criterio(sub, coluna_id="modelo",
                                    desempate=DESEMPATE_INV1)
        log.info(f"  [{fam_nome}] {v['modelo']}: recall={v['recall_macro']:.4f} "
                 f"mcc={v['mcc']:.4f} f1={v['f1_macro']:.4f} fpr={v['fpr_macro']:.4f} "
                 f"(passou_criterio={v['passou_criterio']})")
    log.info("=" * 62)

    # Persiste a decisão para as Investigações 2-4 (encadeamento do pipeline)
    _run("salvar estado", lambda: salvar_estado(
        1, modelo=vencedor["modelo"], familia=fam,
        recall_macro=float(vencedor["recall_macro"]),
        mcc=float(vencedor["mcc"]),
        f1_macro=float(vencedor["f1_macro"]),
        fpr_macro=float(vencedor["fpr_macro"]),
        passou_criterio=bool(vencedor["passou_criterio"]),
        desempate=DESEMPATE_INV1,
    ))


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
