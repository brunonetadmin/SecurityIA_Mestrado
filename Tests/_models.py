"""
_models.py — Fábrica uniforme de modelos do pipeline encadeado SecurityIA.
=========================================================================
Expõe `treina_avalia(modelo, X_tr, y_tr, X_te, y_te, n_cls, ...)`, uma API
única que constrói, treina e avalia tanto árvores (RandomForest, ExtraTrees,
XGBoost, CatBoost) quanto redes densas (MLP+BN, ResNet Tabular). Permite que
as Investigações 2-4 operem sobre o modelo vencedor da Investigação 1 sem
duplicar código de construção/treino.

Convenções de balanceamento:
  - Árvores aceitam `balanceamento_nativo=True` → pondera por classe na própria
    árvore (RF/ExtraTrees: class_weight='balanced_subsample'; XGBoost:
    sample_weight balanceado; CatBoost: auto_class_weights='Balanced').
  - Reamostragem (SMOTE/ENN/undersampling) é aplicada FORA daqui, sobre
    (X_tr, y_tr), e o modelo é treinado com `balanceamento_nativo=False`
    para não corrigir o desbalanceamento duas vezes.
  - Redes aceitam `loss_fn` e `class_weight` (reponderação de perda).

Hiperparâmetros default das árvores espelham a Investigação 1, para que o
modelo que percorre o pipeline seja o MESMO avaliado na comparação inicial.
`hparams` sobrescreve os defaults (usado pela otimização da Investigação 4).
"""
from __future__ import annotations

import time
from typing import Optional

import numpy as np

try:
    from config import RANDOM_SEED
except Exception:          # permite uso/teste fora do ambiente completo
    RANDOM_SEED = 42

from _test_logging import metricas_completas

ARVORES = ("RandomForest", "ExtraTrees", "XGBoost", "CatBoost")
REDES   = ("MLP_BN", "ResNet_Tabular", "SNN")

# Defaults espelhando a Investigação 1
DEFAULTS_ARVORE = {
    "RandomForest": dict(n_estimators=300, max_depth=None, min_samples_leaf=2),
    "ExtraTrees":   dict(n_estimators=400, max_depth=None, min_samples_leaf=2,
                         bootstrap=False),
    "XGBoost":      dict(n_estimators=400, max_depth=8, learning_rate=0.1),
    "CatBoost":     dict(iterations=500, depth=8, learning_rate=0.1),
}

# Early stopping das árvores boosting (CatBoost). 0 desliga. Aplica-se a TODOS
# os fits de CatBoost (trials do Optuna, reavaliação do vencedor e treino
# final), via fatia interna do treino — ver _avalia_arvore.
EARLY_STOPPING_ROUNDS = 50


# ═══════════════════════════════════════════════════════════════════════════
#   ESPAÇO DE BUSCA (Optuna/TPE) — usado pela Investigação 4
# ═══════════════════════════════════════════════════════════════════════════

def espaco_busca(modelo: str, trial) -> dict:
    """Sugere um conjunto de hiperparâmetros para `modelo` a partir de um
    `trial` do Optuna. O dict retornado é passado a treina_avalia(..., hparams=)
    e lá faz `DEFAULTS_ARVORE[modelo].update(hparams)` — portanto só precisa
    conter as chaves que o ramo correspondente de _avalia_arvore/_avalia_rede
    de fato lê. Parâmetros fixados internamente (random_seed, thread_count/
    n_jobs, verbose, loss_function, auto_class_weights/balanceamento) NÃO são
    sugeridos aqui de propósito: alterá-los não teria efeito ou duplicaria o
    tratamento de desbalanceamento.

    As faixas espelham e ampliam moderadamente os defaults da Investigação 1,
    mantendo o modelo dentro da mesma família avaliada na comparação inicial.
    """
    if modelo == "CatBoost":
        return {
            "iterations":    trial.suggest_int("iterations", 300, 600, step=100),
            "depth":         trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
        }

    if modelo == "XGBoost":
        return {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 1000, step=100),
            "max_depth":        trial.suggest_int("max_depth", 4, 12),
            "learning_rate":    trial.suggest_float("learning_rate", 1e-2, 3e-1, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }

    if modelo in ("RandomForest", "ExtraTrees"):
        hp = {
            "n_estimators":     trial.suggest_int("n_estimators", 200, 800, step=100),
            "max_depth":        trial.suggest_categorical("max_depth", [None, 10, 20, 30, 50]),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        }
        if modelo == "ExtraTrees":
            hp["bootstrap"] = trial.suggest_categorical("bootstrap", [False, True])
        return hp

    if modelo in ("MLP_BN", "SNN"):
        return {
            "hidden":   trial.suggest_categorical("hidden", [128, 256, 512]),
            "dropout":  trial.suggest_float("dropout", 0.05, 0.5),
            "n_layers": trial.suggest_int("n_layers", 2, 5),
        }

    if modelo == "ResNet_Tabular":
        return {
            "hidden":  trial.suggest_categorical("hidden", [128, 256, 512]),
            "dropout": trial.suggest_float("dropout", 0.05, 0.5),
        }

    raise ValueError(f"Sem espaço de busca definido para o modelo: {modelo}")


# ═══════════════════════════════════════════════════════════════════════════
#   ÁRVORES
# ═══════════════════════════════════════════════════════════════════════════

def _avalia_arvore(modelo, X_tr, y_tr, X_te, y_te, n_cls,
                   balanceamento_nativo, hparams, logger):
    hp = dict(DEFAULTS_ARVORE[modelo])
    if hparams:
        hp.update(hparams)
    t0 = time.time()

    if modelo == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(
            n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
            min_samples_leaf=hp["min_samples_leaf"], n_jobs=-1,
            random_state=RANDOM_SEED,
            class_weight="balanced_subsample" if balanceamento_nativo else None,
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

    elif modelo == "ExtraTrees":
        from sklearn.ensemble import ExtraTreesClassifier
        clf = ExtraTreesClassifier(
            n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
            min_samples_leaf=hp["min_samples_leaf"], bootstrap=hp["bootstrap"],
            n_jobs=-1, random_state=RANDOM_SEED,
            class_weight="balanced_subsample" if balanceamento_nativo else None,
        )
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)

    elif modelo == "XGBoost":
        import xgboost as xgb
        sample_weight = None
        if balanceamento_nativo:
            from sklearn.utils.class_weight import compute_sample_weight
            sample_weight = compute_sample_weight("balanced", y_tr)
        clf = xgb.XGBClassifier(
            n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
            learning_rate=hp["learning_rate"], objective="multi:softprob",
            num_class=n_cls, tree_method="hist", n_jobs=-1,
            random_state=RANDOM_SEED, eval_metric="mlogloss", verbosity=0,
            subsample=hp.get("subsample", 1.0),
            colsample_bytree=hp.get("colsample_bytree", 1.0),
        )
        clf.fit(X_tr, y_tr, sample_weight=sample_weight, verbose=False)
        y_pred = clf.predict(X_te)

    elif modelo == "CatBoost":
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(
            iterations=hp["iterations"], depth=hp["depth"],
            learning_rate=hp["learning_rate"], loss_function="MultiClass",
            thread_count=-1, random_seed=RANDOM_SEED, verbose=False,
            allow_writing_files=False,
            boosting_type="Plain",   # explícito: base grande — modo escalável em CPU
            auto_class_weights="Balanced" if balanceamento_nativo else None,
        )
        es = int(hp.get("early_stopping_rounds", EARLY_STOPPING_ROUNDS) or 0)
        # O split estratificado do early stopping exige >=2 amostras por classe.
        # Após SMOTE-ENN, classes raras podem ficar com 1 amostra (o ENN remove a
        # classe inteira e a reinjeção a repõe). Nesse caso, treina-se sem early
        # stopping — preserva a classe rara no ajuste e evita quebrar o split.
        _, _cnt = np.unique(y_tr, return_counts=True)
        if es > 0 and _cnt.min() >= 2:
            # Early stopping sobre uma fatia INTERNA do treino (10%), separada de
            # forma estratificada. NÃO usa X_te (conjunto de avaliação da dobra),
            # para a parada não ser sintonizada no mesmo dado em que a métrica é
            # reportada — evita otimismo. use_best_model restaura a melhor
            # iteração. Corta o custo dos trials cujo `iterations` já convergiu.
            from sklearn.model_selection import train_test_split
            X_fit, X_es, y_fit, y_es = train_test_split(
                X_tr, y_tr, test_size=0.10, random_state=RANDOM_SEED,
                stratify=y_tr,
            )
            clf.fit(X_fit, y_fit, eval_set=(X_es, y_es),
                    early_stopping_rounds=es, use_best_model=True)
        else:
            clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te).ravel().astype(int)

    else:
        raise ValueError(f"Árvore desconhecida: {modelo}")

    m = metricas_completas(y_te, y_pred, n_cls)
    m["tempo_s"] = time.time() - t0
    return m


# ═══════════════════════════════════════════════════════════════════════════
#   REDES DENSAS
# ═══════════════════════════════════════════════════════════════════════════

def build_mlp_bn(n_feat, n_cls, loss_fn="sparse_categorical_crossentropy",
                 hidden=256, n_layers=3, dropout=0.3, optimizer=None):
    import tensorflow as tf
    from tensorflow.keras.layers import (Input, Dense, Dropout,
                                          BatchNormalization, Activation)
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    inp = Input(shape=(n_feat,))
    x = inp
    unidades = hidden
    for i in range(n_layers):
        x = Dense(unidades)(x); x = BatchNormalization()(x)
        x = Activation("relu")(x); x = Dropout(dropout)(x)
        unidades = max(unidades // 2, 32)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="MLP_BN")
    m.compile(optimizer=optimizer or Adam(1e-3), loss=loss_fn,
              metrics=["accuracy"])
    return m


def build_resnet_tabular(n_feat, n_cls, loss_fn="sparse_categorical_crossentropy",
                         blocks=4, hidden=256, dropout=0.2, optimizer=None):
    import tensorflow as tf
    from tensorflow.keras.layers import (Input, Dense, Dropout,
                                          BatchNormalization, Activation, Add)
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    inp = Input(shape=(n_feat,))
    x = Dense(hidden)(inp); x = BatchNormalization()(x); x = Activation("relu")(x)
    for _ in range(blocks):
        s = x
        h = Dense(hidden)(x); h = BatchNormalization()(h); h = Activation("relu")(h)
        h = Dropout(dropout)(h)
        h = Dense(hidden)(h); h = BatchNormalization()(h)
        x = Add()([s, h]); x = Activation("relu")(x)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="ResNet_Tabular")
    m.compile(optimizer=optimizer or Adam(1e-3), loss=loss_fn,
              metrics=["accuracy"])
    return m


def build_snn(n_feat, n_cls, loss_fn="sparse_categorical_crossentropy",
              hidden=256, n_layers=3, dropout=0.10, optimizer=None):
    import tensorflow as tf
    from tensorflow.keras.layers import Input, Dense, AlphaDropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam
    inp = Input(shape=(n_feat,))
    x = inp
    unidades = hidden
    for _ in range(n_layers):
        x = Dense(unidades, activation="selu",
                  kernel_initializer="lecun_normal")(x)
        x = AlphaDropout(dropout)(x)
        unidades = max(unidades // 2, 32)
    out = Dense(n_cls, activation="softmax")(x)
    m = Model(inp, out, name="SNN")
    m.compile(optimizer=optimizer or Adam(1e-3), loss=loss_fn,
              metrics=["accuracy"])
    return m


def focal_loss_pura(gamma=2.0):
    import tensorflow as tf
    def fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p = tf.gather(y_pred, y_true, axis=1, batch_dims=1)
        return tf.reduce_mean(tf.pow(1.0 - p, gamma) * (-tf.math.log(p)))
    return fn


def cb_focal_loss(class_counts, gamma=2.0, beta=0.9999):
    import tensorflow as tf
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


def _avalia_rede(modelo, X_tr, y_tr, X_te, y_te, n_cls, X_val, y_val,
                 loss_fn, class_weight, hparams, epochs, batch_size,
                 patience, logger):
    import tensorflow as tf
    from tensorflow.keras import backend as K
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from _test_logging import EpochLogger
    hp = dict(hparams or {})
    K.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)
    t0 = time.time()
    builders = {"ResNet_Tabular": build_resnet_tabular, "SNN": build_snn}
    build = builders.get(modelo, build_mlp_bn)
    extra = {}
    if "n_layers" in hp and modelo in ("MLP_BN", "SNN"):
        extra["n_layers"] = hp["n_layers"]
    m = build(X_tr.shape[1], n_cls, loss_fn=loss_fn,
              hidden=hp.get("hidden", 256), dropout=hp.get("dropout", 0.3),
              **extra)
    cb = [
        EarlyStopping(patience=patience, restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7, monitor="val_loss"),
        EpochLogger(logger, prefix=f"{modelo} ") if logger else None,
    ]
    cb = [c for c in cb if c is not None]
    val = (X_val, y_val) if X_val is not None else None
    m.fit(X_tr, y_tr, validation_data=val, epochs=epochs,
          batch_size=batch_size, callbacks=cb, verbose=0,
          class_weight=class_weight)
    y_pred = np.argmax(m.predict(X_te, batch_size=batch_size, verbose=0), axis=1)
    out = metricas_completas(y_te, y_pred, n_cls)
    out["tempo_s"] = time.time() - t0
    return out


# ═══════════════════════════════════════════════════════════════════════════
#   DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════

def treina_avalia(modelo, X_tr, y_tr, X_te, y_te, n_cls, *,
                  X_val=None, y_val=None,
                  balanceamento_nativo=False,
                  loss_fn="sparse_categorical_crossentropy",
                  class_weight=None, hparams=None,
                  epochs=25, batch_size=4096, patience=6,
                  logger=None) -> dict:
    """Treina e avalia `modelo`, devolvendo o dict de métricas completas.

    Árvores ignoram loss_fn/class_weight/epochs e usam balanceamento_nativo.
    Redes ignoram balanceamento_nativo e usam loss_fn/class_weight.
    """
    if modelo in ARVORES:
        return _avalia_arvore(modelo, X_tr, y_tr, X_te, y_te, n_cls,
                              balanceamento_nativo, hparams, logger)
    if modelo in REDES:
        return _avalia_rede(modelo, X_tr, y_tr, X_te, y_te, n_cls,
                           X_val, y_val, loss_fn, class_weight, hparams,
                           epochs, batch_size, patience, logger)
    raise ValueError(f"Modelo desconhecido: {modelo}")
