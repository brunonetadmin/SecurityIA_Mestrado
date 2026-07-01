"""
_pipeline.py — Estado compartilhado e critério de seleção do SecurityIA.
========================================================================
Encadeia as quatro investigações. Cada análise lê a decisão da anterior
(modelo vencedor, estratégia de balanceamento, conjunto de atributos) de um
JSON único e grava a sua própria decisão. Substitui a seleção por recall
puro pelo CRITÉRIO HIERÁRQUICO da dissertação:

    (i)   MCC        >= 0,80      (descarta discriminação global instável)
    (ii)  FPR-macro  <= 0,010     (descarta custo operacional inaceitável)
    (iii) maximiza recall-macro sob (i) e (ii)
    (iv)  desempate fino: menor tempo de treino

O parâmetro `desempate` permite trocar o critério final (iii) de recall
para MCC — único ponto que decide entre RandomForest (recall) e XGBoost
(MCC) quando ambos passam nas portas (i)/(ii).

A importação de `config` é preguiçosa (dentro das funções de E/S), de modo
que `selecionar_por_criterio` possa ser usada e testada sem carregar o
TensorFlow nem o restante do ambiente.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

# ─── Limiares do critério (idênticos à dissertação) ────────────────────────
MCC_MIN = 0.80
FPR_MAX = 0.010

# ─── Famílias de modelo (definem o que a Inv. 2 e a Inv. 4 podem fazer) ─────
#   arvore: aceita só reamostragem (model-agnostic); espaço de busca de árvore
#   rede  : aceita reamostragem + reponderação de perda; espaço de busca de MLP
_FAMILIA = {
    "RandomForest":   "arvore",
    "ExtraTrees":     "arvore",
    "XGBoost":        "arvore",
    "CatBoost":       "arvore",
    "MLP_BN":         "rede",
    "ResNet_Tabular": "rede",
    "SNN":            "rede",
    "BiLSTM_Atencao": "rede",
}


def familia_de(modelo: str) -> str:
    """Família do modelo. Default 'rede' para nomes desconhecidos."""
    return _FAMILIA.get(modelo, "rede")


# ─── Critério hierárquico de seleção ───────────────────────────────────────

def selecionar_por_criterio(df: pd.DataFrame, coluna_id: str,
                            desempate: str = "recall", logger=None) -> dict:
    """Aplica o critério hierárquico e devolve a linha vencedora como dict.

    Parâmetros
    ----------
    df         : DataFrame com colunas 'mcc', 'fpr_macro', 'recall_macro'
                 (e opcionalmente 'tempo_s' para desempate fino).
    coluna_id  : nome da coluna identificadora ('modelo', 'estrategia',
                 'metodo'...). Usada apenas para o log.
    desempate  : 'recall' (padrão, = critério da dissertação) ou 'mcc'.
    logger     : logger opcional para registrar a decisão.

    Retorno
    -------
    dict da linha vencedora, acrescido de:
      passou_criterio : True se alguma linha satisfez (i) e (ii).
      desempate       : critério final aplicado.

    Se nenhuma linha passa nas portas (i)/(ii), relaxa para o melhor recall
    global e marca passou_criterio=False (com aviso no logger).
    """
    req = {"mcc", "fpr_macro", "recall_macro"}
    faltando = req - set(df.columns)
    if faltando:
        raise ValueError(f"DataFrame sem colunas exigidas: {sorted(faltando)}")
    if desempate not in ("recall", "mcc"):
        raise ValueError("desempate deve ser 'recall' ou 'mcc'")

    aptos = df[(df["mcc"] >= MCC_MIN) & (df["fpr_macro"] <= FPR_MAX)].copy()
    passou = len(aptos) > 0
    base = aptos if passou else df.copy()

    chave = "mcc" if desempate == "mcc" else "recall_macro"
    if "tempo_s" in base.columns:
        base = base.sort_values([chave, "tempo_s"], ascending=[False, True])
    else:
        base = base.sort_values(chave, ascending=False)

    venc = base.iloc[0].to_dict()
    venc["passou_criterio"] = bool(passou)
    venc["desempate"] = desempate

    if logger is not None:
        if not passou:
            logger.warning(
                f"CRITÉRIO: nenhuma configuração satisfez MCC>={MCC_MIN:.2f} e "
                f"FPR<={FPR_MAX:.3f}; relaxado para melhor recall global."
            )
        logger.info(
            f"CRITÉRIO (desempate={desempate}): vencedor="
            f"{venc.get(coluna_id)} recall={venc['recall_macro']:.4f} "
            f"mcc={venc['mcc']:.4f} fpr={venc['fpr_macro']:.4f}"
        )
    return venc


# ─── Estado compartilhado (JSON) ───────────────────────────────────────────

def _state_path() -> Path:
    from config import TEST_REPORTS_DIR  # import preguiçoso
    return Path(TEST_REPORTS_DIR) / "pipeline_state.json"


def carregar_estado() -> dict:
    """Lê o estado do pipeline; dict vazio se ausente ou corrompido."""
    p = _state_path()
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def salvar_estado(stage: int, **dados) -> Path:
    """Atualiza pipeline_state.json com a decisão de uma investigação."""
    estado = carregar_estado()
    estado.setdefault("historico", [])
    registro = {"stage": stage, "ts": datetime.now().isoformat(), **dados}
    estado[f"inv{stage}"] = registro
    estado["historico"].append(registro)
    p = _state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(estado, ensure_ascii=False, indent=2),
                 encoding="utf-8")
    return p


def exigir_estado(stage: int) -> dict:
    """Carrega a decisão da investigação `stage`; erro claro se ausente."""
    est = carregar_estado()
    chave = f"inv{stage}"
    if chave not in est:
        raise RuntimeError(
            f"Estado da Investigação {stage} ausente em {_state_path()}. "
            f"Execute a Análise {stage} antes desta."
        )
    return est[chave]


# ─── Seleção PARCIMONIOSA (Investigação 3) ─────────────────────────────────
# Adicionada para corrigir o furo do funil: `selecionar_por_criterio` maximiza
# recall e, como recall cresce com o nº de atributos, sempre elege k="all" (77)
# — a redução de dimensionalidade nunca acontece e a escolha de k fica sem
# prova. Este seletor mantém o gate hierárquico (MCC>=0,80; FPR<=0,010) e, entre
# os aptos, escolhe o de MENOS atributos cujo recall fica a até `eps_recall` do
# melhor; opcionalmente exige MCC >= um piso (ex.: MCC do conjunto completo, o
# que garante que reduzir não piora a discriminação global). Tudo é derivado dos
# próprios dados da Inv. 3 (validado sobre metricas_selecao real).

def selecionar_parcimonioso(df: pd.DataFrame, coluna_id: str,
                            eps_recall: float = 0.015,
                            exigir_mcc_min: float | None = None,
                            mcc_min: float = MCC_MIN, fpr_max: float = FPR_MAX,
                            coluna_k: str = "k_real", logger=None) -> dict:
    req = {"mcc", "fpr_macro", "recall_macro", coluna_k}
    faltando = req - set(df.columns)
    if faltando:
        raise ValueError(f"DataFrame sem colunas exigidas: {sorted(faltando)}")

    aptos = df[(df["mcc"] >= mcc_min) & (df["fpr_macro"] <= fpr_max)].copy()
    passou = len(aptos) > 0
    base = aptos if passou else df.copy()

    best_recall = float(base["recall_macro"].max())
    cand = base[base["recall_macro"] >= best_recall - eps_recall].copy()
    if exigir_mcc_min is not None:
        c2 = cand[cand["mcc"] >= exigir_mcc_min]
        if not c2.empty:
            cand = c2

    cols, asc = [coluna_k, "mcc"], [True, False]
    if "tempo_s" in cand.columns:
        cols.append("tempo_s"); asc.append(True)
    cand = cand.sort_values(cols, ascending=asc)

    venc = cand.iloc[0].to_dict()
    venc["passou_criterio"] = bool(passou)
    venc["eps_recall"] = float(eps_recall)

    if logger is not None:
        if not passou:
            logger.warning(
                f"PARCIMÔNIA: nenhum combo passou no gate (MCC>={mcc_min:.2f}, "
                f"FPR<={fpr_max:.3f}); relaxado para o espaço completo."
            )
        logger.info(
            f"PARCIMÔNIA (eps_recall={eps_recall:.3f}, "
            f"MCC_piso={exigir_mcc_min}): vencedor={venc.get(coluna_id)} "
            f"k={venc.get(coluna_k)} recall={venc['recall_macro']:.4f} "
            f"mcc={venc['mcc']:.4f} fpr={venc['fpr_macro']:.4f}"
        )
    return venc


# ─── Seleção RECALL-PRIMÁRIA + parcimônia no nível de ruído (Investigação 3) ─
# Substitui, na Inv. 3, a parcimônia-primária por uma regra que respeita o
# CRITÉRIO DECLARADO (recall-macro é a métrica primária). Fluxo:
#   1) gate hierárquico: MCC>=mcc_min e FPR<=fpr_max;
#   2) recall-primário: identifica o MELHOR recall entre os aptos;
#   3) desempate de parcimônia SÓ no nível de ruído: entre as configurações cujo
#      recall fica a até `delta_ruido` do melhor, escolhe a de MENOS atributos.
# `delta_ruido` é calibrado pelo desvio-padrão do recall entre dobras (Inv.4,
# ~0,0074): assim o desempate só troca por um modelo menor quando a diferença de
# recall é indistinguível de ruído, e NUNCA aceita uma perda real de recall.
# Desempates seguintes: MCC desc; preferência de método (ex.: mRMR sobre JMI
# quando as métricas são idênticas); tempo asc.
# Validado sobre metricas_selecao real: para qualquer delta_ruido em [0,002; 0,007]
# o vencedor é mRMR k=32 (que domina o conjunto completo: MCC maior, FPR menor,
# 45 atributos a menos, recall a 0,001 do completo).

def selecionar_recall_parcimonia(df: pd.DataFrame, coluna_id: str,
                                 coluna_metodo: str = "metodo",
                                 delta_ruido: float = 0.005,
                                 mcc_min: float = MCC_MIN, fpr_max: float = FPR_MAX,
                                 coluna_k: str = "k_real",
                                 preferencia_metodo=("mRMR",),
                                 logger=None) -> dict:
    req = {"recall_macro", "mcc", "fpr_macro", coluna_k}
    faltando = req - set(df.columns)
    if faltando:
        raise ValueError(f"DataFrame sem colunas exigidas: {sorted(faltando)}")

    aptos = df[(df["mcc"] >= mcc_min) & (df["fpr_macro"] <= fpr_max)].copy()
    passou = len(aptos) > 0
    base = aptos if passou else df.copy()

    best_recall = float(base["recall_macro"].max())
    tier = base[base["recall_macro"] >= best_recall - delta_ruido].copy()

    # Desempate: menos atributos -> MCC desc -> preferência de método -> tempo asc
    pref = {m: i for i, m in enumerate(preferencia_metodo)}
    if coluna_metodo in tier.columns:
        tier["_pref"] = tier[coluna_metodo].map(lambda m: pref.get(m, len(pref)))
    else:
        tier["_pref"] = len(pref)
    cols, asc = [coluna_k, "mcc", "_pref"], [True, False, True]
    if "tempo_s" in tier.columns:
        cols.append("tempo_s"); asc.append(True)
    tier = tier.sort_values(cols, ascending=asc)

    venc = tier.iloc[0].to_dict()
    venc.pop("_pref", None)
    venc["passou_criterio"] = bool(passou)
    venc["delta_ruido"] = float(delta_ruido)
    venc["recall_melhor"] = best_recall

    if logger is not None:
        if not passou:
            logger.warning(
                f"RECALL-PARCIMÔNIA: nenhum combo passou no gate "
                f"(MCC>={mcc_min:.2f}, FPR<={fpr_max:.3f}); relaxado ao espaço completo."
            )
        logger.info(
            f"RECALL-PARCIMÔNIA (Δ={delta_ruido:.3f}): melhor_recall={best_recall:.4f}; "
            f"vencedor={venc.get(coluna_id)} k={venc.get(coluna_k)} "
            f"recall={venc['recall_macro']:.4f} mcc={venc['mcc']:.4f} "
            f"fpr={venc['fpr_macro']:.4f}"
        )
    return venc
