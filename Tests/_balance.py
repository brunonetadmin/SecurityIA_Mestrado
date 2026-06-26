"""
_balance.py — Estratégias de tratamento de desbalanceamento do SecurityIA.
=========================================================================
Compartilhado entre as Investigações 2 e 3, para que a estratégia escolhida
na Inv. 2 seja reaplicada de forma idêntica na Inv. 3. As estratégias são
adaptadas à FAMÍLIA do modelo vencedor da Inv. 1:

  - Árvore: reamostragem (model-agnostic) + ponderação nativa por classe.
            Loss reweighting NÃO se aplica (é recurso de rede neural).
  - Rede  : reamostragem + reponderação de perda (Focal, CB-Focal, class_weight).

A reamostragem (SMOTE/Borderline/SMOTE-ENN/undersampling) atua só sobre o
TREINO; o teste permanece intacto (sem vazamento). `montar_estrategias` é a
ÚNICA fonte da lista de estratégias; a Inv. 3 a consulta para reproduzir o
vencedor da Inv. 2 sem reimplementar nada.
"""
from __future__ import annotations

from collections import Counter
import numpy as np

try:
    from config import RANDOM_SEED
except Exception:
    RANDOM_SEED = 42


# ─── Reamostragens (model-agnostic) ────────────────────────────────────────

def undersample_benign(X_tr, y_tr, fator=5, logger=None):
    """Reduz a classe majoritária a `fator`× a segunda maior."""
    dist = Counter(y_tr.astype(int))
    maj = max(dist, key=dist.get)
    ordenado = sorted(dist.values(), reverse=True)
    n_2nd = ordenado[1] if len(ordenado) > 1 else ordenado[0]
    target = min(max(n_2nd * fator, n_2nd), dist[maj])
    rng = np.random.default_rng(RANDOM_SEED)
    keep = []
    for c, n in dist.items():
        idx = np.where(y_tr == c)[0]
        if c == maj and n > target:
            idx = rng.choice(idx, size=target, replace=False)
        keep.append(idx)
    keep = np.concatenate(keep); rng.shuffle(keep)
    if logger:
        logger.info(f"  Undersample: {len(y_tr):,} → {len(keep):,} "
                    f"(majoritária {maj}: {dist[maj]:,} → {target:,})")
    return X_tr[keep], y_tr[keep]


def _k_smote(y_tr):
    dist = Counter(y_tr.astype(int))
    min_n = min(n for n in dist.values() if n > 1)
    return max(1, min(5, min_n - 1))


def smote(X_tr, y_tr, logger=None):
    from imblearn.over_sampling import SMOTE
    s = SMOTE(k_neighbors=_k_smote(y_tr), random_state=RANDOM_SEED)
    X_res, y_res = s.fit_resample(X_tr, y_tr)
    if logger:
        logger.info(f"  SMOTE: {len(y_tr):,} → {len(y_res):,}")
    return X_res, y_res


def borderline_smote(X_tr, y_tr, logger=None):
    from imblearn.over_sampling import BorderlineSMOTE
    s = BorderlineSMOTE(k_neighbors=_k_smote(y_tr), kind="borderline-2",
                        random_state=RANDOM_SEED)
    X_res, y_res = s.fit_resample(X_tr, y_tr)
    if logger:
        logger.info(f"  Borderline_SMOTE: {len(y_tr):,} → {len(y_res):,}")
    return X_res, y_res


def smote_enn(X_tr, y_tr, logger=None):
    from imblearn.combine import SMOTEENN
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import EditedNearestNeighbours
    s = SMOTEENN(
        smote=SMOTE(k_neighbors=_k_smote(y_tr), random_state=RANDOM_SEED),
        enn=EditedNearestNeighbours(n_neighbors=3, n_jobs=-1),
        random_state=RANDOM_SEED,
    )
    X_res, y_res = s.fit_resample(X_tr, y_tr)
    # ENN pode remover classes raras inteiras; reinjeta 1 amostra de cada.
    faltantes = set(np.unique(y_tr).tolist()) - set(np.unique(y_res).tolist())
    if faltantes:
        if logger:
            logger.info(f"  SMOTE_ENN: ENN removeu {faltantes}; reinjetando 1 de cada.")
        for c in faltantes:
            idx = np.where(y_tr == c)[0][:1]
            X_res = np.concatenate([X_res, X_tr[idx]])
            y_res = np.concatenate([y_res, y_tr[idx]])
    if logger:
        logger.info(f"  SMOTE_ENN: {len(y_tr):,} → {len(y_res):,}")
    return X_res, y_res


RESAMPLERS = {
    "Undersample_Benign": undersample_benign,
    "SMOTE":              smote,
    "Borderline_SMOTE":   borderline_smote,
    "SMOTE_ENN":          smote_enn,
}


# ─── Montagem das estratégias por família ──────────────────────────────────

def montar_estrategias(familia: str, y_tr, n_cls: int):
    """Devolve [(nome, resampler|None, kwargs_para_treina_avalia)].

    `resampler` (se não-None) é aplicado a (X_tr, y_tr) antes do treino.
    `kwargs` são repassados a _models.treina_avalia.
    """
    if familia == "arvore":
        return [
            ("Sem_Tratamento",     None,               dict(balanceamento_nativo=False)),
            ("ClassWeight_Nativo", None,               dict(balanceamento_nativo=True)),
            ("Undersample_Benign", undersample_benign, dict(balanceamento_nativo=False)),
            ("SMOTE",              smote,              dict(balanceamento_nativo=False)),
            ("Borderline_SMOTE",   borderline_smote,   dict(balanceamento_nativo=False)),
            ("SMOTE_ENN",          smote_enn,          dict(balanceamento_nativo=False)),
        ]

    # família = rede: inclui reponderação de perda
    from _models import focal_loss_pura, cb_focal_loss
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(y_tr)
    w = compute_class_weight("balanced", classes=classes, y=y_tr)
    cw = {int(c): float(wi) for c, wi in zip(classes, w)}
    cc = np.bincount(y_tr.astype(int), minlength=n_cls)
    cce = "sparse_categorical_crossentropy"
    return [
        ("Sem_Tratamento",       None,               dict(loss_fn=cce)),
        ("ClassWeight_Balanced", None,               dict(loss_fn=cce, class_weight=cw)),
        ("FocalLoss",            None,               dict(loss_fn=focal_loss_pura(2.0))),
        ("CB_FocalLoss",         None,               dict(loss_fn=cb_focal_loss(cc))),
        ("Undersample_Benign",   undersample_benign, dict(loss_fn=cce)),
        ("SMOTE",                smote,              dict(loss_fn=cce)),
        ("Borderline_SMOTE",     borderline_smote,   dict(loss_fn=cce)),
        ("SMOTE_ENN",            smote_enn,          dict(loss_fn=cce)),
    ]


def estrategia_por_nome(nome: str, familia: str, y_tr, n_cls: int):
    """Recupera (resampler, kwargs) de uma estratégia nomeada — usado pela Inv. 3
    para reproduzir o balanceamento vencedor da Inv. 2."""
    for n, resampler, kwargs in montar_estrategias(familia, y_tr, n_cls):
        if n == nome:
            return resampler, kwargs
    raise KeyError(f"Estratégia '{nome}' não existe para família '{familia}'.")
