"""
analise_4_otimizacao_validacao.py
==================================
Otimização de hiperparâmetros com Optuna + validação cruzada estratificada
K-fold 5. Esta análise determina os hiperparâmetros que serão usados pelo
IDS em produção, eliminando o viés de um split fixo.

ESPAÇO DE BUSCA (Optuna TPE):
  - learning_rate ∈ [1e-4, 5e-2]   (log-uniforme)
  - batch_size    ∈ {1024, 2048, 4096, 8192}
  - dropout       ∈ [0.10, 0.50]
  - hidden_units  ∈ {64, 128, 256, 512}
  - n_layers      ∈ {2, 3, 4}
  - weight_decay  ∈ [1e-6, 1e-3]   (log-uniforme)
  - optimizer     ∈ {Adam, AdamW, RMSprop}

PROTOCOLO DE AVALIAÇÃO:
  Cada trial Optuna é avaliado por StratifiedKFold(n_splits=5).
  Objetivo: maximizar média(macro_recall) − 0.25 * desvio(macro_recall).
  Penaliza alta variância: hiperparâmetros instáveis são desencorajados.

REPORTE FINAL:
  Tabela com média ± desvio das 5 dobras para todas as métricas
  (macro_recall, MCC, F1-macro, FPR-macro). Conjunto de teste mantido
  reservado: usado apenas para reportar a configuração vencedora,
  nunca para selecioná-la.
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
from tensorflow.keras.optimizers import Adam, AdamW, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

tf.keras.utils.set_random_seed(RANDOM_SEED)
apply_plot_style()

ANALISE_ID = 4
EPOCHS = 30
PATIENCE = 5
N_FOLDS = 5
N_TRIALS_OPTUNA = 30  # ajuste conforme orçamento computacional
log = get_logger(ANALISE_ID, "analise_4")

def _run(label, fn, *a, **kw):
    """Wrapper: executa `fn` via safe_run e retorna apenas o resultado.
    Se `fn` falhar, devolve None (safe_run já loga a exceção)."""
    ok, res = safe_run(log, label, fn, *a, **kw)
    return res if ok else None




# ═══════════════════════════════════════════════════════════════════════════
#   MODELO PARAMETRIZADO
# ═══════════════════════════════════════════════════════════════════════════

def _make_optimizer(name, lr, weight_decay):
    if name == "Adam":
        return Adam(learning_rate=lr)
    if name == "AdamW":
        return AdamW(learning_rate=lr, weight_decay=weight_decay)
    if name == "RMSprop":
        return RMSprop(learning_rate=lr)
    raise ValueError(f"Optimizer desconhecido: {name}")


def build_mlp(n_feat, n_cls, hidden, n_layers, dropout, optimizer):
    inp = Input(shape=(n_feat,))
    x = inp
    units = hidden
    for _ in range(n_layers):
        x = Dense(units)(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
        x = Dropout(dropout)(x)
        units = max(units // 2, 32)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out)
    m.compile(optimizer=optimizer,
              loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


# ═══════════════════════════════════════════════════════════════════════════
#   AVALIAÇÃO POR K-FOLD
# ═══════════════════════════════════════════════════════════════════════════

def avaliar_kfold(params, X_dev, y_dev, n_cls, n_splits=N_FOLDS, log_prefix=""):
    """
    Avalia uma configuração 'params' por StratifiedKFold sobre (X_dev, y_dev).
    Para cada dobra: ajusta StandardScaler apenas no fold de treino, treina,
    avalia no fold de validação. Retorna lista de dicionários (1 por fold).
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    fold_metrics = []
    for fold, (idx_tr, idx_vl) in enumerate(skf.split(X_dev, y_dev), start=1):
        K.clear_session()
        tf.keras.utils.set_random_seed(RANDOM_SEED + fold)
        Xtr, Xvl = X_dev[idx_tr], X_dev[idx_vl]
        ytr, yvl = y_dev[idx_tr], y_dev[idx_vl]
        sc = StandardScaler().fit(Xtr)
        Xtr_s, Xvl_s = sc.transform(Xtr), sc.transform(Xvl)
        opt = _make_optimizer(params["optimizer"], params["lr"], params["weight_decay"])
        m = build_mlp(
            n_feat=Xtr_s.shape[1], n_cls=n_cls,
            hidden=params["hidden"], n_layers=params["n_layers"],
            dropout=params["dropout"], optimizer=opt,
        )
        cb = [EarlyStopping(patience=PATIENCE, restore_best_weights=True, monitor="val_loss")]
        t0 = time.time()
        m.fit(Xtr_s, ytr, validation_data=(Xvl_s, yvl),
              epochs=EPOCHS, batch_size=params["batch_size"],
              callbacks=cb, verbose=0)
        y_pred = np.argmax(m.predict(Xvl_s, batch_size=params["batch_size"], verbose=0), axis=1)
        met = metricas_completas(yvl, y_pred, n_cls)
        met["fold"] = fold
        met["tempo_s"] = time.time() - t0
        fold_metrics.append(met)
        log.info(
            f"  {log_prefix}fold {fold}/{n_splits}: "
            f"recall={met['recall_macro']:.4f} mcc={met['mcc']:.4f} "
            f"f1={met['f1_macro']:.4f} ({met['tempo_s']:.1f}s)"
        )
    return fold_metrics


def _agregar(fold_metrics):
    """Agrega lista de dicts (1 por fold) em médias e desvios."""
    df = pd.DataFrame(fold_metrics)
    cols = ["acuracia", "balanced_acc", "f1_macro", "f1_weighted",
            "precision_macro", "recall_macro", "mcc", "fpr_macro", "tempo_s"]
    agg = {}
    for c in cols:
        if c in df.columns:
            agg[f"{c}_mean"] = float(df[c].mean())
            agg[f"{c}_std"]  = float(df[c].std(ddof=1))
    return agg


# ═══════════════════════════════════════════════════════════════════════════
#   OPTUNA
# ═══════════════════════════════════════════════════════════════════════════

def _objective_factory(X_dev, y_dev, n_cls):
    def objective(trial):
        params = dict(
            lr           = trial.suggest_float("lr", 1e-4, 5e-2, log=True),
            batch_size   = trial.suggest_categorical("batch_size", [1024, 2048, 4096, 8192]),
            dropout      = trial.suggest_float("dropout", 0.10, 0.50),
            hidden       = trial.suggest_categorical("hidden", [64, 128, 256, 512]),
            n_layers     = trial.suggest_int("n_layers", 2, 4),
            weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            optimizer    = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop"]),
        )
        fm = avaliar_kfold(params, X_dev, y_dev, n_cls,
                           log_prefix=f"trial {trial.number} ")
        recalls = [m["recall_macro"] for m in fm]
        mu, sg = float(np.mean(recalls)), float(np.std(recalls, ddof=1))
        score = mu - 0.25 * sg
        log.info(
            f"  trial {trial.number}: recall={mu:.4f}±{sg:.4f}  score={score:.4f}  "
            f"params={params}"
        )
        return score
    return objective


# ═══════════════════════════════════════════════════════════════════════════
#   PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def executar(dataset_disponivel: bool = True) -> None:
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError as e:
        raise RuntimeError(
            "optuna não instalado. Execute: pip install optuna"
        ) from e

    log.info(f"ANÁLISE 4 — Otimização com Optuna (TPE) + StratifiedKFold-{N_FOLDS}")

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

    # Split: dev (85%) para Optuna+CV, teste (15%) reservado para reporte final.
    X_dev_raw, _Xv_raw, X_te_raw, y_dev, _yv, y_te = stratified_split_3way(
        Xfull, yfull, val_frac=0.0001, test_frac=0.15, seed=RANDOM_SEED,
    )
    log.info(f"Split: dev={len(y_dev):,} (CV-{N_FOLDS}) teste={len(y_te):,} (reservado)")

    # Optuna não pode receber X já escalonado (cada fold escalona por conta própria)
    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        _objective_factory(X_dev_raw, y_dev, n_cls),
        n_trials=N_TRIALS_OPTUNA, show_progress_bar=False,
    )

    log.info("=" * 62)
    log.info(f"BEST PARAMS: {study.best_params}")
    log.info(f"BEST SCORE : {study.best_value:.4f}")
    log.info("=" * 62)

    # Reavaliação por K-fold da configuração vencedora (média ± desvio)
    log.info(">>> Reavaliação K-fold da configuração vencedora")
    best_fold_metrics = _run(
        "kfold_vencedor",
        lambda: avaliar_kfold(study.best_params, X_dev_raw, y_dev, n_cls,
                              log_prefix="best "),
    )
    if best_fold_metrics is None:
        log.error("Falha na reavaliação K-fold."); return
    agg_dev = _agregar(best_fold_metrics)

    # Reporte único no conjunto de teste reservado (config. final treinada em dev inteiro)
    log.info(">>> Treino final em dev inteiro + reporte no teste reservado")
    K.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    sc = StandardScaler().fit(X_dev_raw)
    X_dev_s = sc.transform(X_dev_raw); X_te_s = sc.transform(X_te_raw)
    p = study.best_params
    opt = _make_optimizer(p["optimizer"], p["lr"], p["weight_decay"])
    m = build_mlp(X_dev_s.shape[1], n_cls,
                  p["hidden"], p["n_layers"], p["dropout"], opt)
    m.fit(X_dev_s, y_dev,
          epochs=EPOCHS, batch_size=p["batch_size"], verbose=0)
    y_pred = np.argmax(m.predict(X_te_s, batch_size=p["batch_size"], verbose=0), axis=1)
    met_te = metricas_completas(y_te, y_pred, n_cls)
    log.info(
        f"TESTE RESERVADO: recall={met_te['recall_macro']:.4f} "
        f"mcc={met_te['mcc']:.4f} f1={met_te['f1_macro']:.4f} "
        f"fpr={met_te['fpr_macro']:.4f}"
    )

    # ─── persistência ────────────────────────────────────────────────────
    df_trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df_folds = pd.DataFrame(best_fold_metrics)
    csv_trials = tab_path(ANALISE_ID, "metricas_otimizacao")
    csv_folds  = tab_path(ANALISE_ID, "kfold_vencedor")
    _run("salvar trials",  lambda: df_trials.to_csv(csv_trials, index=False))
    _run("salvar folds",   lambda: df_folds.to_csv(csv_folds, index=False))
    log.info(f"Tabela trials: {csv_trials}")
    log.info(f"Tabela folds : {csv_folds}")

    # Resumo agregado para a tabela do artigo
    df_resumo = pd.DataFrame([{
        "config":       "best_optuna",
        "params":       str(study.best_params),
        "n_folds":      N_FOLDS,
        "recall_mean":  agg_dev["recall_macro_mean"],
        "recall_std":   agg_dev["recall_macro_std"],
        "mcc_mean":     agg_dev["mcc_mean"],
        "mcc_std":      agg_dev["mcc_std"],
        "f1_mean":      agg_dev["f1_macro_mean"],
        "f1_std":       agg_dev["f1_macro_std"],
        "fpr_mean":     agg_dev["fpr_macro_mean"],
        "fpr_std":      agg_dev["fpr_macro_std"],
        "tempo_mean_s": agg_dev["tempo_s_mean"],
        "teste_recall": met_te["recall_macro"],
        "teste_mcc":    met_te["mcc"],
        "teste_f1":     met_te["f1_macro"],
        "teste_fpr":    met_te["fpr_macro"],
    }])
    csv_resumo = tab_path(ANALISE_ID, "resumo_kfold_best")
    _run("salvar resumo", lambda: df_resumo.to_csv(csv_resumo, index=False))
    log.info(f"Tabela resumo: {csv_resumo}")

    _run("plot_otimizacao", lambda: _plot(df_trials, df_folds))


def _plot(df_trials: pd.DataFrame, df_folds: pd.DataFrame) -> None:
    """Painel 2x2: (a) histórico de trials, (b) recall por fold (best),
    (c) importância dos hiperparâmetros (correlação Spearman),
    (d) recall vs lr (best params em destaque)."""
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) Histórico Optuna
    df_ok = df_trials.dropna(subset=["value"]).copy()
    df_ok["best_so_far"] = df_ok["value"].cummax()
    ax[0, 0].plot(df_ok["number"], df_ok["value"],
                  color="black", marker="o", linewidth=1, markersize=4, alpha=0.6, label="trial")
    ax[0, 0].plot(df_ok["number"], df_ok["best_so_far"],
                  color="black", linestyle="--", linewidth=2, label="best so far")
    ax[0, 0].set_title("Histórico de otimização (Optuna)", fontsize=12, fontweight="bold")
    ax[0, 0].set_xlabel("Trial"); ax[0, 0].set_ylabel("Score (recall − 0.25·σ)")
    ax[0, 0].grid(True, alpha=0.3); ax[0, 0].legend()

    # (b) Recall por fold (vencedor)
    cores = sns.color_palette("Greys", n_colors=N_FOLDS + 2)[2:]
    sns.barplot(data=df_folds, x="fold", y="recall_macro",
                ax=ax[0, 1], palette=cores, edgecolor="black")
    ax[0, 1].set_title("Recall-macro por dobra (configuração vencedora)",
                       fontsize=12, fontweight="bold")
    ax[0, 1].set_xlabel("Fold"); ax[0, 1].set_ylabel("recall_macro")
    for p in ax[0, 1].patches:
        v = p.get_height()
        ax[0, 1].annotate(f"{v:.3f}", (p.get_x() + p.get_width()/2, v),
                          ha="center", va="bottom", fontsize=9)

    # (c) Importância de hiperparâmetros (correlação |Spearman| de cada param com value)
    param_cols = [c for c in df_trials.columns if c.startswith("params_")]
    if param_cols:
        from scipy.stats import spearmanr
        importancias = []
        for c in param_cols:
            try:
                vals = pd.to_numeric(df_trials[c], errors="coerce")
                if vals.notna().sum() < 5:
                    importancias.append((c.replace("params_", ""), 0.0)); continue
                r, _ = spearmanr(vals, df_trials["value"], nan_policy="omit")
                importancias.append((c.replace("params_", ""), abs(r) if r is not None else 0.0))
            except Exception:
                importancias.append((c.replace("params_", ""), 0.0))
        df_imp = pd.DataFrame(importancias, columns=["param", "|spearman|"]).sort_values("|spearman|")
        ax[1, 0].barh(df_imp["param"], df_imp["|spearman|"],
                      color="dimgray", edgecolor="black")
        ax[1, 0].set_title("Importância de hiperparâmetros (|ρ Spearman|)",
                           fontsize=12, fontweight="bold")
        ax[1, 0].set_xlabel("|correlação de Spearman|")
        ax[1, 0].grid(True, axis="x", alpha=0.3)

    # (d) Trials espalhados: lr vs value
    if "params_lr" in df_trials.columns:
        df_trials["params_lr"] = pd.to_numeric(df_trials["params_lr"], errors="coerce")
        ax[1, 1].scatter(df_trials["params_lr"], df_trials["value"],
                         color="black", alpha=0.6, edgecolor="white", s=40)
        ax[1, 1].set_xscale("log")
        ax[1, 1].set_title("Score vs. learning rate (todos os trials)",
                           fontsize=12, fontweight="bold")
        ax[1, 1].set_xlabel("learning_rate (log)"); ax[1, 1].set_ylabel("Score")
        ax[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_path(ANALISE_ID, "comparativo_otimizacao"), dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    executar()
