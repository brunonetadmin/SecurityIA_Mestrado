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
