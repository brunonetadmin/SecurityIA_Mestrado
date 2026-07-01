"""
analise_2_balanceamento.py
==========================
Investigação 2 — Tratamento de desbalanceamento sobre o MODELO VENCEDOR da
Investigação 1 (lido de pipeline_state.json), não mais sobre um MLP fixo.

As estratégias são adaptadas à família do vencedor (_balance.montar_estrategias):
  - Árvore (RF/XGBoost/...): reamostragem model-agnostic (SMOTE, Borderline,
    SMOTE-ENN, undersampling) + ponderação nativa por classe. Loss reweighting
    NÃO se aplica a árvore.
  - Rede (MLP/ResNet): reamostragem + reponderação de perda (Focal, CB-Focal,
    class_weight).

A reamostragem atua só no treino (sem vazamento). A seleção segue o CRITÉRIO
HIERÁRQUICO (MCC>=0,80; FPR<=0,010; depois maximiza recall), não recall puro —
isto evita escolher um balanceamento que sobe recall às custas de MCC/FPR.
O ganho é medido RELATIVO ao baseline 'Sem_Tratamento' deste mesmo modelo.

Requer: imblearn (SMOTE/ENN). Execute a Análise 1 antes (gera o estado).
"""
import os, sys, time, warnings
from collections import Counter
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
    stratified_split_3way, fit_scaler_no_leakage, metricas_completas,
)
from _pipeline import (
    exigir_estado, selecionar_por_criterio, salvar_estado, familia_de,
)
from _models import treina_avalia
from _balance import montar_estrategias

silence_tensorflow()
apply_plot_style()

ANALISE_ID = 2
EPOCHS = 25
PATIENCE = 6
DESEMPATE_INV2 = "recall"   # critério final; consistente com a Inv. 1
log = get_logger(ANALISE_ID, "analise_2")


def _run(label, fn, *a, **kw):
    ok, res = safe_run(log, label, fn, *a, **kw)
    return res if ok else None


def _avaliar_estrategia(nome, resampler, kwargs, modelo,
                        X_tr, y_tr, X_val, y_val, X_te, y_te, n_cls):
    """Aplica (se houver) a reamostragem ao treino e treina/avalia o modelo
    vencedor da Inv. 1 com os kwargs da estratégia."""
    if resampler is not None:
        X_tr2, y_tr2 = resampler(X_tr, y_tr, logger=log)
    else:
        X_tr2, y_tr2 = X_tr, y_tr
    out = treina_avalia(
        modelo, X_tr2, y_tr2, X_te, y_te, n_cls,
        X_val=X_val, y_val=y_val, logger=log,
        epochs=EPOCHS, batch_size=BATCH_SIZE, patience=PATIENCE,
        **kwargs,
    )
    out["estrategia"] = nome
    return out


def executar(dataset_disponivel: bool = True) -> None:
    # ── 1. Estado da Investigação 1 (modelo vencedor + família) ────────────
    try:
        est1 = exigir_estado(1)
    except RuntimeError as e:
        log.error(str(e)); return
    modelo = est1["modelo"]
    familia = est1.get("familia") or familia_de(modelo)
    log.info(f"ANÁLISE 2 — Balanceamento sobre o vencedor da Inv. 1: "
             f"{modelo} [{familia}]")

    # ── 2. Dataset ─────────────────────────────────────────────────────────
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
    log.info(f"Dataset: {Xfull.shape[0]:,} × {Xfull.shape[1]} feats × {n_cls} classes")

    X_tr_raw, X_val_raw, X_te_raw, y_tr, y_val, y_te = stratified_split_3way(
        Xfull, yfull, val_frac=0.15, test_frac=0.15, seed=RANDOM_SEED,
    )
    X_tr, X_val, X_te, _scaler = fit_scaler_no_leakage(X_tr_raw, X_val_raw, X_te_raw)
    log.info(f"Split: treino={len(y_tr):,} val={len(y_val):,} teste={len(y_te):,}")
    log.info(f"Distribuição treino: {dict(Counter(y_tr.astype(int)))}")

    # ── 3. Estratégias por família ─────────────────────────────────────────
    estrategias = montar_estrategias(familia, y_tr, n_cls)
    log.info(f"Estratégias ({familia}): {[e[0] for e in estrategias]}")

    resultados = []
    for nome, resampler, kwargs in estrategias:
        out = _run(nome, lambda n=nome, r=resampler, kw=kwargs:
                   _avaliar_estrategia(n, r, kw, modelo,
                                       X_tr, y_tr, X_val, y_val, X_te, y_te, n_cls))
        if out is not None:
            resultados.append(out)
            log.info(f"{nome}: recall={out['recall_macro']:.4f} mcc={out['mcc']:.4f} "
                     f"f1={out['f1_macro']:.4f} fpr={out['fpr_macro']:.4f}")

    if not resultados:
        log.error("Nenhuma estratégia concluída."); return

    df = pd.DataFrame(resultados)
    csv_path = tab_path(ANALISE_ID, "metricas_balanceamento")
    _run("salvar CSV", lambda: df.to_csv(csv_path, index=False))
    log.info(f"Tabela: {csv_path}")
    fig_file = _run("plot_comparativo", lambda: _plot(df))

    # ── 4. Seleção pelo critério hierárquico + ganho vs baseline ───────────
    venc = selecionar_por_criterio(df, coluna_id="estrategia",
                                   desempate=DESEMPATE_INV2, logger=log)
    base = df[df["estrategia"] == "Sem_Tratamento"]
    delta = ""
    if not base.empty:
        d = venc["recall_macro"] - float(base.iloc[0]["recall_macro"])
        delta = f"  (Δrecall vs Sem_Tratamento = {d:+.4f})"
    log.info("=" * 62)
    log.info(f"VENCEDOR (critério, desempate={DESEMPATE_INV2}): {venc['estrategia']}  "
             f"recall={venc['recall_macro']:.4f} mcc={venc['mcc']:.4f} "
             f"fpr={venc['fpr_macro']:.4f} (passou_criterio={venc['passou_criterio']}){delta}")
    log.info("=" * 62)

    _run("salvar estado", lambda: salvar_estado(
        2, modelo=modelo, familia=familia,
        balanceamento=venc["estrategia"],
        recall_macro=float(venc["recall_macro"]), mcc=float(venc["mcc"]),
        f1_macro=float(venc["f1_macro"]), fpr_macro=float(venc["fpr_macro"]),
        passou_criterio=bool(venc["passou_criterio"]),
        desempate=DESEMPATE_INV2,
    ))

    # Relatório .md versionado
    def _relatorio():
        rel = Relatorio(ANALISE_ID)
        rel.secao("Configuração")
        rel.texto(f"Tratamento de desbalanceamento sobre o vencedor da Inv. 1 "
                  f"({modelo} [{familia}]). Estratégias adaptadas à família; "
                  f"reamostragem apenas no treino (sem vazamento). Critério "
                  f"hierárquico (MCC>=0,80; FPR<=0,010; depois recall).")
        rel.secao("Resultados")
        rel.tabela_df(df, "Métricas por estratégia de balanceamento.")
        if fig_file is not None:
            rel.figura(Path(fig_file).stem, "Comparativo por métrica.")
        rel.secao("Decisão")
        rel.metrica("Balanceamento (Inv. 3–4)",
                    f"{venc['estrategia']} — recall={venc['recall_macro']:.4f}, "
                    f"MCC={venc['mcc']:.4f}, FPR={venc['fpr_macro']:.4f}{delta}")
        return rel.salvar()
    _run("salvar relatorio", _relatorio)


def _plot(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    metricas = [
        ("recall_macro", "Recall-macro",             ax[0, 0]),
        ("mcc",          "MCC",                      ax[0, 1]),
        ("f1_macro",     "F1-macro",                 ax[1, 0]),
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
    fp = fig_path(ANALISE_ID, "comparativo_balanceamento")
    plt.savefig(fp, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return fp


if __name__ == "__main__":
    executar()
