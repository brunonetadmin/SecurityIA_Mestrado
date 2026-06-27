"""
analise_4_otimizacao_validacao.py
==================================
Investigação 4 — Otimização de hiperparâmetros (Optuna/TPE) + validação cruzada
estratificada K-fold sobre o MODELO VENCEDOR do pipeline (lido do estado), com
o balanceamento da Inv. 2 e os atributos da Inv. 3. Não otimiza mais um MLP
fixo: o espaço de busca é ESCOLHIDO PELA FAMÍLIA do vencedor
(_models.espaco_busca) — árvore recebe knobs de árvore; rede, de rede.

PROTOCOLO (livre de vazamento):
  - Split dev (85%) / teste reservado (15%), estratificado.
  - Projeção sobre os atributos selecionados na Inv. 3.
  - Cada trial: StratifiedKFold(5) sobre o dev. Em cada dobra, o balanceamento
    da Inv. 2 é aplicado SÓ ao treino da dobra (reamostragem) ou via ponderação
    nativa/perda; o escalonamento (quando família=rede) é ajustado só no treino
    da dobra. Objetivo = média(recall_macro) − 0.25·desvio(recall_macro).
  - Reavaliação K-fold da config vencedora (média ± desvio de todas as métricas).
  - Treino final no dev inteiro + reporte ÚNICO no teste reservado.

Requer: optuna. Execute as Análises 1–3 antes (geram o estado encadeado).
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
    get_logger, log_exception, safe_run, silence_tensorflow,
    stratified_split_3way, metricas_completas,
)
from _pipeline import exigir_estado, salvar_estado, familia_de
from _models import treina_avalia, espaco_busca
from _balance import estrategia_por_nome

silence_tensorflow()

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

apply_plot_style()

ANALISE_ID = 4
N_FOLDS = 5
N_TRIALS_OPTUNA = 30      # ajuste conforme orçamento computacional
LAMBDA_VAR = 0.25         # penalização da variância entre dobras
EPOCHS = 30               # usado apenas se o vencedor for de família 'rede'
PATIENCE = 5
DESEMPATE_INV4 = "recall"
log = get_logger(ANALISE_ID, "analise_4")


def _run(label, fn, *a, **kw):
    ok, res = safe_run(log, label, fn, *a, **kw)
    return res if ok else None


# ═══════════════════════════════════════════════════════════════════════════
#   PREPARO DE DOBRA (escalonamento p/ rede) + BALANCEAMENTO DA INV. 2
# ═══════════════════════════════════════════════════════════════════════════

def _prep_fold(familia, X_tr, X_vl):
    """Escalona só quando o vencedor é rede (árvore é invariante à escala).
    O scaler é ajustado APENAS no treino da dobra — sem vazamento."""
    if familia == "rede":
        sc = StandardScaler().fit(X_tr)
        return sc.transform(X_tr), sc.transform(X_vl)
    return X_tr, X_vl


def _balancear(familia, balanceamento, n_cls, X_tr, y_tr):
    """Aplica a estratégia da Inv. 2 ao treino. Devolve (X, y, kwargs) em que
    kwargs vai para treina_avalia (balanceamento_nativo p/ árvore; loss_fn/
    class_weight p/ rede)."""
    resampler, kwargs = estrategia_por_nome(balanceamento, familia, y_tr, n_cls)
    if resampler is not None:
        X_b, y_b = resampler(X_tr, y_tr, logger=log)
    else:
        X_b, y_b = X_tr, y_tr
    return X_b, y_b, kwargs


def _preparar_dobras(familia, balanceamento, n_cls, X_dev, y_dev):
    """Pré-computa, UMA única vez, as N_FOLDS dobras já preparadas (escala, se
    família=rede) e BALANCEADAS (Inv. 2). O resultado independe dos
    hiperparâmetros do Optuna: o SMOTE-ENN e o scaler dependem apenas dos
    dados/rótulos de treino de cada dobra, que são fixos (mesmas dobras —
    StratifiedKFold com seed fixo — e mesmas features da Inv. 3). Reutilizar
    estas dobras em todos os trials elimina a recomputação do SMOTE-ENN a cada
    trial (29×N_FOLDS reamostragens redundantes) SEM qualquer efeito sobre as
    métricas. Pressupõe que treina_avalia trata X_b/y_b como somente-leitura
    (verdadeiro para CatBoost/árvore; o Pool é construído sem alterar o array)."""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    dobras, t0 = [], time.time()
    for fold, (i_tr, i_vl) in enumerate(skf.split(X_dev, y_dev), start=1):
        X_tr, X_vl = X_dev[i_tr], X_dev[i_vl]
        y_tr, y_vl = y_dev[i_tr], y_dev[i_vl]
        X_tr, X_vl = _prep_fold(familia, X_tr, X_vl)
        X_b, y_b, bkw = _balancear(familia, balanceamento, n_cls, X_tr, y_tr)
        dobras.append(dict(fold=fold, X_b=X_b, y_b=y_b, bkw=bkw,
                           X_vl=X_vl, y_vl=y_vl))
        log.info(f"  dobra {fold}/{N_FOLDS} preparada+balanceada "
                 f"(treino {len(y_b):,} | val {len(y_vl):,})")
    log.info(f"  {N_FOLDS} dobras pré-balanceadas em {time.time()-t0:.1f}s — "
             f"reutilizadas nos {N_TRIALS_OPTUNA} trials")
    return dobras


def avaliar_kfold_cache(modelo, hparams, dobras, n_cls, log_prefix=""):
    """Avalia `hparams` sobre dobras JÁ preparadas e balanceadas por
    _preparar_dobras. Apenas o ajuste do modelo varia entre trials — que é a
    única etapa que de fato depende dos hiperparâmetros. Retorna lista de dicts
    (1 por dobra), idêntica à versão que rebalanceava a cada chamada."""
    folds = []
    for d in dobras:
        out = treina_avalia(
            modelo, d["X_b"], d["y_b"], d["X_vl"], d["y_vl"], n_cls,
            X_val=d["X_vl"], y_val=d["y_vl"], hparams=hparams, logger=None,
            epochs=EPOCHS, patience=PATIENCE,
            batch_size=hparams.get("batch_size", BATCH_SIZE), **d["bkw"],
        )
        out["fold"] = d["fold"]
        folds.append(out)
        log.info(f"  {log_prefix}fold {d['fold']}/{N_FOLDS}: "
                 f"recall={out['recall_macro']:.4f} mcc={out['mcc']:.4f} "
                 f"f1={out['f1_macro']:.4f} fpr={out['fpr_macro']:.4f} "
                 f"({out['tempo_s']:.1f}s)")
    return folds


def avaliar_kfold(modelo, familia, balanceamento, hparams,
                  X_dev, y_dev, n_cls, log_prefix=""):
    """Compat: prepara as dobras e avalia `hparams` numa só chamada. Usado na
    reavaliação ÚNICA do vencedor. No laço do Optuna as dobras são
    pré-computadas uma vez (ver _preparar_dobras) e reutilizadas via
    avaliar_kfold_cache, evitando rebalancear a cada trial."""
    dobras = _preparar_dobras(familia, balanceamento, n_cls, X_dev, y_dev)
    return avaliar_kfold_cache(modelo, hparams, dobras, n_cls, log_prefix=log_prefix)


def _agregar(folds):
    df = pd.DataFrame(folds)
    cols = ["acuracia", "balanced_acc", "f1_macro", "f1_weighted",
            "precision_macro", "recall_macro", "mcc", "fpr_macro", "tempo_s"]
    agg = {}
    for c in cols:
        if c in df.columns:
            agg[f"{c}_mean"] = float(df[c].mean())
            agg[f"{c}_std"]  = float(df[c].std(ddof=1)) if len(df) > 1 else 0.0
    return agg


# ═══════════════════════════════════════════════════════════════════════════
#   OPTUNA — objetivo family-aware
# ═══════════════════════════════════════════════════════════════════════════

def _objective_factory(modelo, dobras, n_cls):
    """Objetivo do Optuna sobre dobras PRÉ-balanceadas (independem dos
    hiperparâmetros). Só o ajuste do modelo varia entre trials."""
    def objective(trial):
        hp = espaco_busca(modelo, trial)
        folds = avaliar_kfold_cache(modelo, hp, dobras, n_cls,
                                    log_prefix=f"trial {trial.number} ")
        rec = np.array([f["recall_macro"] for f in folds], dtype=float)
        return float(rec.mean() - LAMBDA_VAR * rec.std(ddof=1))
    return objective


# ═══════════════════════════════════════════════════════════════════════════
#   PIPELINE PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════

def executar(dataset_disponivel: bool = True) -> None:
    try:
        import optuna
        from optuna.samplers import TPESampler
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        log.error("Optuna ausente. Instale com: pip install optuna"); return

    # ── Estado da Inv. 3: vencedor + balanceamento + atributos selecionados ─
    try:
        est3 = exigir_estado(3)
    except RuntimeError as e:
        log.error(str(e)); return
    modelo = est3["modelo"]
    familia = est3.get("familia") or familia_de(modelo)
    balanceamento = est3["balanceamento"]
    atributos_idx = est3.get("atributos_idx") or []
    log.info(f"ANÁLISE 4 — Otimização (Optuna/TPE) + CV-{N_FOLDS} sobre "
             f"{modelo} [{familia}] | balanceamento={balanceamento} | "
             f"k={len(atributos_idx) or 'todos'}")
    log.info(f"Espaço de busca: {familia} (selecionado pela família do vencedor)")

    # ── Dataset ────────────────────────────────────────────────────────────
    log.info(">>> carregar_dataset_real")
    try:
        _t0 = time.time()
        _ds = carregar_dataset_real()
        log.info(f"<<< carregar_dataset_real OK ({time.time()-_t0:.1f}s)")
    except Exception as _e:
        log_exception(log, "carregar_dataset_real", _e)
        log.error("Sem dataset real. Abortando."); return
    if _ds is None or not isinstance(_ds, (tuple, list)) or len(_ds) < 2:
        log.error("Dataset inválido. Abortando."); return
    Xfull = np.ascontiguousarray(_ds[0], dtype=np.float32)
    yfull = np.ascontiguousarray(_ds[1]).astype(np.int64).ravel()
    n_cls = int(np.max(yfull) + 1)

    # Projeção sobre os atributos da Inv. 3 (se ausente, usa todos).
    if atributos_idx:
        idx = np.asarray(atributos_idx, dtype=int)
        Xfull = Xfull[:, idx]
    log.info(f"Dataset: {Xfull.shape[0]:,} amostras × {Xfull.shape[1]} features "
             f"× {n_cls} classes")

    # Split dev (85%) / teste reservado (15%). O vestígio de 'val' é reincorporado
    # ao dev para mantê-lo íntegro na validação cruzada.
    X_dev, X_tiny, X_te, y_dev, y_tiny, y_te = stratified_split_3way(
        Xfull, yfull, val_frac=0.0001, test_frac=0.15, seed=RANDOM_SEED,
    )
    if len(y_tiny):
        X_dev = np.concatenate([X_dev, X_tiny])
        y_dev = np.concatenate([y_dev, y_tiny])
    log.info(f"Split: dev={len(y_dev):,} (CV-{N_FOLDS}) "
             f"teste={len(y_te):,} (reservado)")

    # ── Pré-balanceamento das dobras (UMA vez; reutilizado nos trials) ─────
    # O SMOTE-ENN/escala dependem só dos dados de cada dobra (fixos), não dos
    # hiperparâmetros. Calcular as dobras balanceadas uma única vez e reutilizá-
    # las nos 30 trials elimina 29×N_FOLDS reamostragens redundantes — sem
    # alterar nenhum resultado. (Custo: mantém N_FOLDS conjuntos balanceados em
    # RAM simultaneamente, ~N_FOLDS× o pico anterior de um conjunto por vez.)
    log.info(">>> Pré-balanceamento das dobras (reutilizado em todos os trials)")
    dobras = _run("preparar_dobras", lambda: _preparar_dobras(
        familia, balanceamento, n_cls, X_dev, y_dev))
    if not dobras:
        log.error("Falha ao preparar as dobras. Abortando."); return

    # ── Otimização ─────────────────────────────────────────────────────────
    sampler = TPESampler(seed=RANDOM_SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    log.info(f">>> Optuna: {N_TRIALS_OPTUNA} trials × CV-{N_FOLDS} "
             f"= {N_TRIALS_OPTUNA * N_FOLDS} ajustes de {modelo} "
             f"(dobras pré-balanceadas — só o ajuste do modelo varia)")
    study.optimize(
        _objective_factory(modelo, dobras, n_cls),
        n_trials=N_TRIALS_OPTUNA, show_progress_bar=False,
    )
    log.info("=" * 62)
    log.info(f"BEST PARAMS: {study.best_params}")
    log.info(f"BEST SCORE : {study.best_value:.4f}")
    log.info("=" * 62)

    # ── Reavaliação K-fold da config vencedora (reusa as dobras) ───────────
    log.info(">>> Reavaliação K-fold da configuração vencedora")
    best_folds = _run("kfold_vencedor", lambda: avaliar_kfold_cache(
        modelo, study.best_params, dobras, n_cls, log_prefix="best "))
    if best_folds is None:
        log.error("Falha na reavaliação K-fold."); return
    agg = _agregar(best_folds)

    # ── Treino final no dev inteiro + reporte no teste reservado ───────────
    log.info(">>> Treino final (dev inteiro) + teste reservado")
    Xd, Xt = _prep_fold(familia, X_dev, X_te)        # escala só se rede
    Xb, yb, bkw = _balancear(familia, balanceamento, n_cls, Xd, y_dev)
    if familia == "rede":
        # carve val p/ early stopping do treino final, sem tocar no teste
        Xtr_f, Xval_f, ytr_f, yval_f = train_test_split(
            Xb, yb, test_size=0.05, random_state=RANDOM_SEED, stratify=yb)
        met_te = treina_avalia(
            modelo, Xtr_f, ytr_f, Xt, y_te, n_cls,
            X_val=Xval_f, y_val=yval_f, hparams=study.best_params,
            epochs=EPOCHS, patience=PATIENCE,
            batch_size=study.best_params.get("batch_size", BATCH_SIZE), **bkw)
    else:
        met_te = treina_avalia(
            modelo, Xb, yb, Xt, y_te, n_cls,
            X_val=None, y_val=None, hparams=study.best_params, **bkw)
    log.info(f"TESTE RESERVADO: recall={met_te['recall_macro']:.4f} "
             f"mcc={met_te['mcc']:.4f} f1={met_te['f1_macro']:.4f} "
             f"fpr={met_te['fpr_macro']:.4f}")

    # ── Persistência ───────────────────────────────────────────────────────
    df_trials = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df_folds = pd.DataFrame(best_folds)
    csv_trials = tab_path(ANALISE_ID, "metricas_otimizacao")
    csv_folds  = tab_path(ANALISE_ID, "kfold_vencedor")
    _run("salvar trials", lambda: df_trials.to_csv(csv_trials, index=False))
    _run("salvar folds",  lambda: df_folds.to_csv(csv_folds, index=False))
    log.info(f"Tabela trials: {csv_trials}")
    log.info(f"Tabela folds : {csv_folds}")

    df_resumo = pd.DataFrame([{
        "modelo": modelo, "familia": familia, "balanceamento": balanceamento,
        "n_atributos": Xfull.shape[1], "params": str(study.best_params),
        "n_folds": N_FOLDS,
        "recall_mean": agg["recall_macro_mean"], "recall_std": agg["recall_macro_std"],
        "mcc_mean": agg["mcc_mean"], "mcc_std": agg["mcc_std"],
        "f1_mean": agg["f1_macro_mean"], "f1_std": agg["f1_macro_std"],
        "fpr_mean": agg["fpr_macro_mean"], "fpr_std": agg["fpr_macro_std"],
        "tempo_mean_s": agg["tempo_s_mean"],
        "teste_recall": met_te["recall_macro"], "teste_mcc": met_te["mcc"],
        "teste_f1": met_te["f1_macro"], "teste_fpr": met_te["fpr_macro"],
    }])
    csv_resumo = tab_path(ANALISE_ID, "resumo_kfold_best")
    _run("salvar resumo", lambda: df_resumo.to_csv(csv_resumo, index=False))
    log.info(f"Tabela resumo: {csv_resumo}")

    _run("plot_otimizacao", lambda: _plot(df_trials, df_folds))

    _run("salvar estado", lambda: salvar_estado(
        4, modelo=modelo, familia=familia, balanceamento=balanceamento,
        atributos_idx=[int(i) for i in atributos_idx],
        hparams={k: (v if isinstance(v, (int, float, str, bool, type(None)))
                     else str(v)) for k, v in study.best_params.items()},
        teste_recall=float(met_te["recall_macro"]), teste_mcc=float(met_te["mcc"]),
        teste_f1=float(met_te["f1_macro"]), teste_fpr=float(met_te["fpr_macro"]),
        cv_recall_mean=agg["recall_macro_mean"], cv_recall_std=agg["recall_macro_std"],
        cv_mcc_mean=agg["mcc_mean"], cv_mcc_std=agg["mcc_std"],
    ))


# ═══════════════════════════════════════════════════════════════════════════
#   FIGURA (family-agnostic)
# ═══════════════════════════════════════════════════════════════════════════

def _plot(df_trials: pd.DataFrame, df_folds: pd.DataFrame) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) Histórico Optuna
    df_ok = df_trials.dropna(subset=["value"]).copy()
    if not df_ok.empty:
        df_ok["best_so_far"] = df_ok["value"].cummax()
        ax[0, 0].plot(df_ok["number"], df_ok["value"], color="black",
                      marker="o", linewidth=1, markersize=4, alpha=0.6, label="trial")
        ax[0, 0].plot(df_ok["number"], df_ok["best_so_far"], color="black",
                      linestyle="--", linewidth=2, label="best so far")
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

    # (c) Importância de hiperparâmetros (|Spearman| de cada param com value)
    param_cols = [c for c in df_trials.columns if c.startswith("params_")]
    top_num_param = None
    if param_cols:
        from scipy.stats import spearmanr
        importancias = []
        for c in param_cols:
            vals = pd.to_numeric(df_trials[c], errors="coerce")
            if vals.notna().sum() < 5:
                importancias.append((c.replace("params_", ""), 0.0, None)); continue
            r, _ = spearmanr(vals, df_trials["value"], nan_policy="omit")
            importancias.append((c.replace("params_", ""),
                                 abs(r) if r is not None else 0.0, c))
        importancias.sort(key=lambda t: t[1])
        df_imp = pd.DataFrame([(n, v) for n, v, _ in importancias],
                              columns=["param", "|spearman|"])
        ax[1, 0].barh(df_imp["param"], df_imp["|spearman|"],
                      color="dimgray", edgecolor="black")
        ax[1, 0].set_title("Importância de hiperparâmetros (|ρ Spearman|)",
                           fontsize=12, fontweight="bold")
        ax[1, 0].set_xlabel("|correlação de Spearman|")
        ax[1, 0].grid(True, axis="x", alpha=0.3)
        if importancias and importancias[-1][1] > 0:
            top_num_param = importancias[-1][2]  # param numérico mais influente

    # (d) Score vs. o hiperparâmetro numérico mais influente (family-agnostic)
    if top_num_param is not None:
        xv = pd.to_numeric(df_trials[top_num_param], errors="coerce")
        ax[1, 1].scatter(xv, df_trials["value"], color="black",
                         alpha=0.6, edgecolor="white", s=40)
        nome = top_num_param.replace("params_", "")
        if (xv.dropna() > 0).all() and xv.dropna().max() / max(xv.dropna().min(), 1e-12) > 50:
            ax[1, 1].set_xscale("log")
        ax[1, 1].set_title(f"Score vs. {nome} (todos os trials)",
                           fontsize=12, fontweight="bold")
        ax[1, 1].set_xlabel(nome); ax[1, 1].set_ylabel("Score")
        ax[1, 1].grid(True, alpha=0.3)
    else:
        ax[1, 1].axis("off")

    plt.tight_layout()
    plt.savefig(fig_path(ANALISE_ID, "comparativo_otimizacao"), dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    executar()
