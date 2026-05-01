"""
IDS/modules/custom_layers.py
============================
Componentes Keras customizados do SecurityIA, registrados para serialização
nativa do Keras 3.

Os decorators `@register_keras_serializable` GARANTEM que `keras.load_model()`
consiga reconstruir o modelo sem precisar receber `custom_objects=...`.

REGRA CRÍTICA: este módulo precisa ser IMPORTADO antes de qualquer
`load_model()` que envolva esses componentes. Os decorators só rodam no
import. Por isso, tanto `ids_learn.py` (treinamento) quanto `incident_engine.py`
(detecção) importam este módulo no topo.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.saving import register_keras_serializable


# ─── Atenção de Bahdanau ───────────────────────────────────────────────────

@register_keras_serializable(package="SecurityIA", name="BahdanauAttention")
class BahdanauAttention(tf.keras.layers.Layer):
    """
    Atenção aditiva (Bahdanau et al., 2015).

        e_t = v^T · tanh(W_h · h_t + b_a)
        α_t = softmax(e_t)
        c   = Σ α_t · h_t

    Implementação serializável via `@register_keras_serializable`. Modelos
    salvos com esta versão podem ser recarregados sem `custom_objects`.
    """

    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.W = Dense(self.units, use_bias=True,
                       kernel_initializer="glorot_uniform")
        self.V = Dense(1, use_bias=False,
                       kernel_initializer="glorot_uniform")

    def build(self, input_shape):
        # Constrói as sub-camadas explicitamente para evitar warning de
        # 'unbuilt state' durante serialização/desserialização.
        self.W.build(input_shape)
        # Saída de W tem última dim = units
        w_output_shape = tuple(input_shape[:-1]) + (self.units,)
        self.V.build(w_output_shape)
        super().build(input_shape)

    def call(self, hidden_states, training=False):
        score   = self.V(tf.nn.tanh(self.W(hidden_states)))         # (n, T, 1)
        weights = tf.nn.softmax(score, axis=1)                       # (n, T, 1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)     # (n, 2u)
        return context, weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


# ─── Focal Loss reponderada (Cui et al., 2019; Lin et al., 2017) ───────────

@register_keras_serializable(package="SecurityIA", name="focal_loss_cb")
def focal_loss_cb_placeholder(y_true, y_pred):
    """
    Placeholder REGISTRADO para a função de perda 'focal_loss_cb' que é
    salva embarcada no .keras. Necessário para satisfazer a deserialização
    nominal do Keras 3.

    NÃO É USADA em inferência — o `incident_engine` carrega o modelo com
    `compile=False`. Em fine-tuning, o `ids_learn` recompila com uma
    Focal Loss construída dinamicamente a partir do `class_counts` atual,
    via `make_focal_loss_class_balanced`.
    """
    y_true = tf.cast(y_true, tf.int32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    p = tf.gather(y_pred, y_true, axis=1, batch_dims=1)
    return tf.reduce_mean(tf.pow(1.0 - p, 2.0) * (-tf.math.log(p)))


def make_focal_loss_class_balanced(class_counts, gamma: float = 2.0,
                                    beta: float = 0.9999):
    """
    Constrói Focal Loss com pesos por classe baseados em number of effective
    samples (Cui et al., 2019). Usada no TREINO.

    Importante: a função interna é nomeada 'focal_loss_cb' para que o nome
    salvo no .keras corresponda ao placeholder registrado acima.
    """
    cc = np.asarray(class_counts, dtype=np.float64)
    n_eff = (1.0 - np.power(beta, cc)) / (1.0 - beta)
    w = (1.0 - beta) / np.maximum(n_eff, 1e-12)
    w = w / w.sum() * len(w)
    wt = tf.constant(w, dtype=tf.float32)

    def focal_loss_cb(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        p = tf.gather(y_pred, y_true, axis=1, batch_dims=1)
        cw = tf.gather(wt, y_true)
        return tf.reduce_mean(cw * tf.pow(1.0 - p, gamma) * (-tf.math.log(p)))

    return focal_loss_cb


__all__ = [
    "BahdanauAttention",
    "make_focal_loss_class_balanced",
    "focal_loss_cb_placeholder",
]
