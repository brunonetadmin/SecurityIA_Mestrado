#!/usr/bin/env python3
"""
IDS/modules/full_report.py — Gerador de Relatório Completo SecurityIA

Produz, em uma única chamada, relatório consolidado contendo:
  1. Métricas agregadas e por classe para M0, Mp e Mc
  2. TP, FP, FN, TN por classe (valores absolutos)
  3. FPR, TPR, TNR, F1, taxa projetada de alarmes/h
  4. Comparações pareadas Mc vs M0 e Mc vs Mp (McNemar)
  5. Matrizes de confusão lado a lado (PNG)
  6. Saídas: TXT, JSON, MD, LaTeX, PNG — todas com nome versionado

Nomes de arquivos seguem o padrão <stem>_YYYYMMDD-N.<ext> via
IDS.modules.versioning, garantindo comparação visual entre execuções.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import Config

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, matthews_corrcoef, precision_score, recall_score,
)
from scipy.stats import chi2

from IDS.modules.model_registry import load_triplet
from IDS.modules.versioning import versioned_path


def _per_class_counts(cm: np.ndarray, label_map: dict) -> dict:
    """Calcula TP, FP, FN, TN, FPR, TPR, TNR, precisão, F1 por classe."""
    n = cm.shape[0]
    total = cm.sum()
    out = {}
    for c in range(n):
        tp = int(cm[c, c])
        fp = int(cm[:, c].sum() - tp)
        fn = int(cm[c, :].sum() - tp)
        tn = int(total - tp - fp - fn)
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1_c = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        out[label_map.get(c, str(c))] = {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": float(prec),
            "recall_tpr": float(rec),
            "specificity_tnr": float(tnr),
            "fpr": float(fpr),
            "f1": float(f1_c),
            "support": int(cm[c, :].sum()),
        }
    return out


def _aggregate(y_true, y_pred, label_map: dict, lambda_h: int) -> dict:
    """Métricas agregadas + per-class para um par (y_true, y_pred)."""
    cm = confusion_matrix(y_true, y_pred, labels=sorted(label_map.keys()))
    per = _per_class_counts(cm, label_map)
    fpr_macro = float(np.mean([v["fpr"] for v in per.values()]))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "fpr_macro": fpr_macro,
        "alarms_per_hour_estimated": float(fpr_macro * lambda_h),
        "n_test": int(len(y_true)),
        "per_class": per,
        "confusion_matrix": cm.tolist(),
    }


def _mcnemar(y_true, y_a, y_b, alpha: float = 0.001) -> dict:
    """Teste de McNemar com correção de continuidade (Edwards, 1948)."""
    yt = np.asarray(y_true)
    a_ok = (np.asarray(y_a) == yt)
    b_ok = (np.asarray(y_b) == yt)
    n_a_only = int(np.sum(a_ok & ~b_ok))
    n_b_only = int(np.sum(~a_ok & b_ok))

    if n_a_only + n_b_only == 0:
        return {"n_a_only": 0, "n_b_only": 0, "chi2": 0.0,
                "p_value": 1.0, "significant": False, "winner": "tie"}
    stat = (abs(n_a_only - n_b_only) - 1.0) ** 2 / (n_a_only + n_b_only)
    p = float(1.0 - chi2.cdf(stat, df=1))
    winner = "B" if n_b_only > n_a_only else "A"
    return {
        "n_a_only": n_a_only,
        "n_b_only": n_b_only,
        "chi2": float(stat),
        "p_value": p,
        "significant": p < alpha,
        "winner": winner if p < alpha else "tie",
    }


def _verdict(mcn: dict, delta_f1: float) -> str:
    if not mcn["significant"]:
        return "ESTAGNAÇÃO (sem diferença estatisticamente significativa)"
    if mcn["winner"] == "B" and delta_f1 > 0:
        return "MELHORIA confirmada estatisticamente"
    if mcn["winner"] == "B" and delta_f1 <= 0:
        return "CONFLITO (McNemar favorece, mas F1 caiu — investigar)"
    if mcn["winner"] == "A":
        return "REGRESSÃO confirmada estatisticamente"
    return "INDETERMINADO"


def _render_txt(triplet: dict, comparisons: dict) -> str:
    L = []
    L.append("=" * 78)
    L.append("  SecurityIA — RELATÓRIO COMPLETO DE TREINAMENTO E AVALIAÇÃO")
    L.append("=" * 78)
    L.append(f"  Gerado em: {datetime.now().isoformat()}")
    L.append("")
    L.append("  IDENTIFICAÇÃO DOS MODELOS COMPARADOS")
    L.append("  " + "─" * 60)
    for tag, name in [("M0", "Baseline"), ("Mp", "Anterior"), ("Mc", "Atual")]:
        m = triplet.get(tag)
        if not m:
            L.append(f"  {tag} ({name:<8}): NÃO DISPONÍVEL")
        else:
            L.append(f"  {tag} ({name:<8}): {m['kind']:<22s}  "
                     f"id={m['id']:<25s}  ts={m['registered_at'][:19]}")
    L.append("")

    L.append("  MÉTRICAS AGREGADAS")
    L.append("  " + "─" * 76)
    header = f"  {'Métrica':<24}{'M0 Baseline':>16}{'Mp Anterior':>16}{'Mc Atual':>16}"
    L.append(header)
    L.append("  " + "─" * 76)
    keys = [
        ("accuracy", "Acurácia"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("f1_macro", "F1-macro"),
        ("f1_weighted", "F1-weighted"),
        ("precision_macro", "Precision-macro"),
        ("recall_macro", "Recall-macro (TPR)"),
        ("mcc", "MCC"),
        ("fpr_macro", "FPR-macro"),
        ("alarms_per_hour_estimated", "Alarmes/h (proj.)"),
    ]
    for k, label in keys:
        row = f"  {label:<24}"
        for tag in ["M0", "Mp", "Mc"]:
            m = triplet.get(tag)
            v = m["agg"].get(k) if m else None
            row += f"{v:>16.4f}" if isinstance(v, (int, float)) else f"{'—':>16}"
        L.append(row)
    L.append("")

    if "Mc_vs_M0" in comparisons:
        L.append("  COMPARAÇÃO PAREADA: Mc (Atual) vs M0 (Baseline)")
        L.append("  " + "─" * 60)
        c = comparisons["Mc_vs_M0"]
        if "note" in c:
            L.append(f"    {c['note']}")
        else:
            L.append(f"    Delta F1-macro          : {c['delta_f1_macro']:+.4f}")
            L.append(f"    Delta MCC               : {c['delta_mcc']:+.4f}")
            L.append(f"    Delta FPR-macro         : {c['delta_fpr_macro']:+.4f}  "
                     f"(negativo é melhor)")
            L.append(f"    McNemar chi2            : {c['mcnemar']['chi2']:.2f}")
            L.append(f"    McNemar p-valor         : {c['mcnemar']['p_value']:.2e}")
            L.append(f"    Veredito                : {c['verdict']}")
        L.append("")

    if "Mc_vs_Mp" in comparisons:
        L.append("  COMPARAÇÃO PAREADA: Mc (Atual) vs Mp (Anterior)")
        L.append("  " + "─" * 60)
        c = comparisons["Mc_vs_Mp"]
        if "note" in c:
            L.append(f"    {c['note']}")
        else:
            L.append(f"    Delta F1-macro          : {c['delta_f1_macro']:+.4f}")
            L.append(f"    Delta MCC               : {c['delta_mcc']:+.4f}")
            L.append(f"    Delta FPR-macro         : {c['delta_fpr_macro']:+.4f}  "
                     f"(negativo é melhor)")
            L.append(f"    McNemar chi2            : {c['mcnemar']['chi2']:.2f}")
            L.append(f"    McNemar p-valor         : {c['mcnemar']['p_value']:.2e}")
            L.append(f"    Veredito                : {c['verdict']}")
        L.append("")

    if triplet.get("Mc"):
        L.append("  PER-CLASS — Mc (Atual)")
        L.append("  " + "─" * 76)
        L.append(f"  {'Classe':<25}{'TP':>8}{'FP':>8}{'FN':>8}"
                 f"{'TN':>10}{'FPR':>8}{'F1':>8}")
        L.append("  " + "─" * 76)
        for cls, v in triplet["Mc"]["agg"]["per_class"].items():
            L.append(f"  {cls[:24]:<25}{v['tp']:>8}{v['fp']:>8}{v['fn']:>8}"
                     f"{v['tn']:>10}{v['fpr']:>8.4f}{v['f1']:>8.4f}")
        L.append("")

    L.append("  F1 POR CLASSE — Comparação entre os três modelos")
    L.append("  " + "─" * 64)
    L.append(f"  {'Classe':<28}{'M0':>10}{'Mp':>10}{'Mc':>10}{'Δ Mc-M0':>10}")
    L.append("  " + "─" * 64)
    classes = []
    for tag in ["Mc", "Mp", "M0"]:
        if triplet.get(tag):
            classes = list(triplet[tag]["agg"]["per_class"].keys())
            break

    def _get_f1(tag, cls):
        m = triplet.get(tag)
        if not m:
            return float("nan")
        return m["agg"]["per_class"].get(cls, {}).get("f1", float("nan"))

    for cls in classes:
        f0 = _get_f1("M0", cls)
        fp_ = _get_f1("Mp", cls)
        fc = _get_f1("Mc", cls)
        delta = (fc - f0) if not np.isnan(f0) and not np.isnan(fc) else float("nan")
        def _fmt(x):
            return f"{x:>10.4f}" if not np.isnan(x) else f"{'—':>10}"
        def _fmt_d(x):
            return f"{x:>+10.3f}" if not np.isnan(x) else f"{'—':>10}"
        L.append(f"  {cls[:27]:<28}{_fmt(f0)}{_fmt(fp_)}{_fmt(fc)}{_fmt_d(delta)}")
    L.append("")
    L.append("=" * 78)
    return "\n".join(L)


def _render_md(triplet: dict, comparisons: dict) -> str:
    L = ["# SecurityIA — Relatório Completo de Treinamento e Avaliação", ""]
    L.append(f"_Gerado em: {datetime.now().isoformat()}_")
    L.append("")
    L.append("## Modelos comparados")
    L.append("")
    L.append("| Tag | Papel | Tipo | ID | Registro |")
    L.append("|-----|-------|------|----|---------|")
    for tag, name in [("M0", "Baseline"), ("Mp", "Anterior"), ("Mc", "Atual")]:
        m = triplet.get(tag)
        if m:
            L.append(f"| {tag} | {name} | {m['kind']} | `{m['id']}` | {m['registered_at'][:19]} |")
        else:
            L.append(f"| {tag} | {name} | — | _não disponível_ | — |")
    L.append("")

    L.append("## Métricas agregadas")
    L.append("")
    L.append("| Métrica | M0 Baseline | Mp Anterior | Mc Atual |")
    L.append("|---------|------------:|------------:|---------:|")
    keys = [
        ("accuracy", "Acurácia"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("f1_macro", "F1-macro"),
        ("f1_weighted", "F1-weighted"),
        ("mcc", "MCC"),
        ("fpr_macro", "FPR-macro"),
        ("alarms_per_hour_estimated", "Alarmes/h projetado"),
    ]
    for k, label in keys:
        cells = []
        for tag in ["M0", "Mp", "Mc"]:
            m = triplet.get(tag)
            v = m["agg"].get(k) if m else None
            cells.append(f"{v:.4f}" if isinstance(v, (int, float)) else "—")
        L.append(f"| {label} | {cells[0]} | {cells[1]} | {cells[2]} |")
    L.append("")

    for cmp_key, cmp_label in [("Mc_vs_M0", "Mc vs M0 (Baseline)"),
                                ("Mc_vs_Mp", "Mc vs Mp (Anterior)")]:
        if cmp_key not in comparisons:
            continue
        c = comparisons[cmp_key]
        L.append(f"## Comparação pareada — {cmp_label}")
        L.append("")
        if "note" in c:
            L.append(f"_{c['note']}_")
        else:
            L.append(f"- Delta F1-macro: **{c['delta_f1_macro']:+.4f}**")
            L.append(f"- Delta MCC: **{c['delta_mcc']:+.4f}**")
            L.append(f"- Delta FPR-macro: **{c['delta_fpr_macro']:+.4f}** _(negativo é melhor)_")
            L.append(f"- McNemar χ²: {c['mcnemar']['chi2']:.2f}")
            L.append(f"- p-valor: {c['mcnemar']['p_value']:.2e}")
            L.append(f"- **Veredito: {c['verdict']}**")
        L.append("")

    if triplet.get("Mc"):
        L.append("## Métricas por classe — Mc (Atual)")
        L.append("")
        L.append("| Classe | TP | FP | FN | TN | FPR | Recall (TPR) | F1 | Suporte |")
        L.append("|--------|---:|---:|---:|---:|----:|-------------:|---:|--------:|")
        for cls, v in triplet["Mc"]["agg"]["per_class"].items():
            L.append(f"| {cls} | {v['tp']} | {v['fp']} | {v['fn']} | {v['tn']} | "
                     f"{v['fpr']:.4f} | {v['recall_tpr']:.4f} | {v['f1']:.4f} | {v['support']} |")
        L.append("")

    return "\n".join(L)


def _render_tex(triplet: dict, comparisons: dict) -> str:
    L = []
    L.append(r"\begin{table}[H]")
    L.append(r"\centering")
    L.append(r"\caption{Comparação dos três modelos: $M_0$ (baseline), $M_p$ (anterior) e $M_c$ (atual).}")
    L.append(r"\label{tab:tripla-m0-mp-mc}")
    L.append(r"\small")
    L.append(r"\begin{tabular}{@{}lccc@{}}")
    L.append(r"\toprule")
    L.append(r"Métrica & $M_0$ Baseline & $M_p$ Anterior & $M_c$ Atual \\")
    L.append(r"\midrule")
    keys = [
        ("accuracy", "Acurácia"),
        ("balanced_accuracy", "Balanced Accuracy"),
        ("f1_macro", "F1-macro"),
        ("f1_weighted", "F1-weighted"),
        ("mcc", "MCC"),
        ("fpr_macro", "FPR-macro"),
    ]
    for k, label in keys:
        cells = []
        for tag in ["M0", "Mp", "Mc"]:
            m = triplet.get(tag)
            v = m["agg"].get(k) if m else None
            cells.append(f"{v:.4f}" if isinstance(v, (int, float)) else "---")
        L.append(f"{label} & {cells[0]} & {cells[1]} & {cells[2]} \\\\")
    L.append(r"\midrule")
    if "Mc_vs_M0" in comparisons and "delta_f1_macro" in comparisons["Mc_vs_M0"]:
        c = comparisons["Mc_vs_M0"]
        L.append(f"$\\Delta$F1 ($M_c-M_0$) & --- & --- & {c['delta_f1_macro']:+.4f} \\\\")
        L.append(f"$p$-valor McNemar ($M_c$ vs $M_0$) & --- & --- & {c['mcnemar']['p_value']:.2e} \\\\")
    if "Mc_vs_Mp" in comparisons and "delta_f1_macro" in comparisons["Mc_vs_Mp"]:
        c = comparisons["Mc_vs_Mp"]
        L.append(f"$\\Delta$F1 ($M_c-M_p$) & --- & --- & {c['delta_f1_macro']:+.4f} \\\\")
        L.append(f"$p$-valor McNemar ($M_c$ vs $M_p$) & --- & --- & {c['mcnemar']['p_value']:.2e} \\\\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\fonte{elaboração própria, geração automática pelo módulo \texttt{full\_report}}")
    L.append(r"\end{table}")
    return "\n".join(L)


def _plot_confusion_panel(triplet: dict, label_map: dict, out_path: Path) -> None:
    available = [(t, triplet[t]) for t in ["M0", "Mp", "Mc"] if triplet.get(t)]
    n = len(available)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(8 * n, 8), squeeze=False)
    labels = [label_map.get(i, str(i)) for i in sorted(label_map)]
    for ax, (tag, m) in zip(axes[0], available):
        cm = np.array(m["agg"]["confusion_matrix"], dtype=float)
        cm_n = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
        title = {"M0": "$M_0$ Baseline", "Mp": "$M_p$ Anterior", "Mc": "$M_c$ Atual"}[tag]
        ax.set_title(f"{title}\nF1-macro = {m['agg']['f1_macro']:.4f}")
        ax.set_xlabel("Predita")
        ax.set_ylabel("Verdadeira" if tag == available[0][0] else "")
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
            tick.set_ha("right")
        for tick in ax.get_yticklabels():
            tick.set_rotation(0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=Config.VIZ_CONFIG["dpi"])
    plt.close(fig)


def _plot_metrics_bars(triplet: dict, out_path: Path) -> None:
    """Gráfico de barras agrupadas comparando métricas-chave M0/Mp/Mc."""
    metrics = [("accuracy", "Acurácia"),
               ("f1_macro", "F1-macro"),
               ("f1_weighted", "F1-weighted"),
               ("mcc", "MCC"),
               ("fpr_macro", "FPR-macro")]
    tags = [("M0", "M0 Baseline"), ("Mp", "Mp Anterior"), ("Mc", "Mc Atual")]
    available_tags = [(t, l) for t, l in tags if triplet.get(t)]
    if not available_tags:
        return

    x = np.arange(len(metrics))
    width = 0.8 / len(available_tags)
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, (tag, label) in enumerate(available_tags):
        vals = [triplet[tag]["agg"].get(k, 0.0) for k, _ in metrics]
        ax.bar(x + (i - (len(available_tags) - 1) / 2) * width,
               vals, width, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.set_title("Comparação de Métricas — M0 / Mp / Mc")
    ax.set_ylabel("Valor")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=Config.VIZ_CONFIG["dpi"])
    plt.close(fig)


def generate(out_dir: Path,
             label_map: Optional[dict] = None,
             lambda_h: int = 1_000_000) -> dict:
    """
    Gera relatório completo a partir do triplete registrado.
    Todos os arquivos de saída são versionados via IDS.modules.versioning.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if label_map is None:
        import joblib
        le_path = Config.MODEL_DIR / Config.LABEL_ENCODER_FILENAME
        if le_path.exists():
            le = joblib.load(le_path)
            label_map = {i: lbl for i, lbl in enumerate(le.classes_)}
        else:
            triplet_tmp = load_triplet()
            sample = next((triplet_tmp[t] for t in ["Mc", "Mp", "M0"]
                           if triplet_tmp.get(t)), None)
            if sample is None:
                raise FileNotFoundError("Nenhum modelo no registry e label encoder ausente.")
            n_cls = len(np.unique(sample["y_true"]))
            label_map = {i: f"class_{i}" for i in range(n_cls)}

    triplet = load_triplet()

    for tag in ["M0", "Mp", "Mc"]:
        m = triplet.get(tag)
        if m:
            m["agg"] = _aggregate(m["y_true"], m["y_pred"], label_map, lambda_h)

    comparisons = {}
    if triplet.get("Mc") and triplet.get("M0"):
        if len(triplet["Mc"]["y_true"]) == len(triplet["M0"]["y_true"]):
            mcn = _mcnemar(triplet["Mc"]["y_true"],
                           triplet["M0"]["y_pred"],
                           triplet["Mc"]["y_pred"])
            df1 = triplet["Mc"]["agg"]["f1_macro"] - triplet["M0"]["agg"]["f1_macro"]
            comparisons["Mc_vs_M0"] = {
                "delta_f1_macro": df1,
                "delta_mcc": triplet["Mc"]["agg"]["mcc"] - triplet["M0"]["agg"]["mcc"],
                "delta_fpr_macro": triplet["Mc"]["agg"]["fpr_macro"] - triplet["M0"]["agg"]["fpr_macro"],
                "mcnemar": mcn,
                "verdict": _verdict(mcn, df1),
            }
        else:
            comparisons["Mc_vs_M0"] = {"note": "tamanhos divergentes — comparação não realizada"}

    if triplet.get("Mc") and triplet.get("Mp"):
        if len(triplet["Mc"]["y_true"]) == len(triplet["Mp"]["y_true"]):
            mcn = _mcnemar(triplet["Mc"]["y_true"],
                           triplet["Mp"]["y_pred"],
                           triplet["Mc"]["y_pred"])
            df1 = triplet["Mc"]["agg"]["f1_macro"] - triplet["Mp"]["agg"]["f1_macro"]
            comparisons["Mc_vs_Mp"] = {
                "delta_f1_macro": df1,
                "delta_mcc": triplet["Mc"]["agg"]["mcc"] - triplet["Mp"]["agg"]["mcc"],
                "delta_fpr_macro": triplet["Mc"]["agg"]["fpr_macro"] - triplet["Mp"]["agg"]["fpr_macro"],
                "mcnemar": mcn,
                "verdict": _verdict(mcn, df1),
            }
        else:
            comparisons["Mc_vs_Mp"] = {"note": "tamanhos divergentes — comparação não realizada"}

    txt = _render_txt(triplet, comparisons)
    md = _render_md(triplet, comparisons)
    tex = _render_tex(triplet, comparisons)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "triplet_summary": {
            tag: ({
                "id": triplet[tag]["id"],
                "kind": triplet[tag]["kind"],
                "registered_at": triplet[tag]["registered_at"],
                "agg": triplet[tag]["agg"],
            } if triplet.get(tag) else None)
            for tag in ["M0", "Mp", "Mc"]
        },
        "comparisons": comparisons,
    }

    paths = {
        "txt":  versioned_path(out_dir, "relatorio_completo", "txt"),
        "md":   versioned_path(out_dir, "relatorio_completo", "md"),
        "tex":  versioned_path(out_dir, "relatorio_completo", "tex"),
        "json": versioned_path(out_dir, "relatorio_completo", "json"),
        "png_cm":      versioned_path(out_dir, "matrizes_confusao_panel", "png"),
        "png_metrics": versioned_path(out_dir, "metricas_comparativas", "png"),
    }
    paths["txt"].write_text(txt, encoding="utf-8")
    paths["md"].write_text(md, encoding="utf-8")
    paths["tex"].write_text(tex, encoding="utf-8")
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    _plot_confusion_panel(triplet, label_map, paths["png_cm"])
    _plot_metrics_bars(triplet, paths["png_metrics"])

    print(f"  [full_report] Relatório completo gerado em: {out_dir}")
    for k, p in paths.items():
        print(f"      {k:>10}: {p.name}")

    print()
    print(txt)
    return {k: str(v) for k, v in paths.items()}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SecurityIA — Relatório Completo")
    p.add_argument("--out", default=None, help="diretório de saída")
    p.add_argument("--lambda-h", type=int, default=1_000_000,
                   help="tráfego benigno por hora (proj. de alarmes)")
    args = p.parse_args()

    out_dir = Path(args.out) if args.out else (
        Config.IDS_REPORTS_DIR / f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    generate(out_dir, lambda_h=args.lambda_h)
