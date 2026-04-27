#!/usr/bin/env python3
"""
IDS/ids_learn.py — Motor de Treinamento e Fine-Tuning do Modelo IDS

Versão atualizada com:
  - Particionamento ANTES do balanceamento (correção D1: leakage)
  - Focal Loss reponderada por número efetivo de amostras (correção D2)
  - Balanceamento adaptativo Borderline-SMOTE-2 (correção D3)
  - class_weight no fit
  - Registro automático no triplete M0/Mp/Mc
  - Geração automática de relatório completo após cada treino/fine-tuning

Uso:
    python3 IDS/ids_learn.py train --force      # treinamento integral
    python3 IDS/ids_learn.py finetune           # fine-tuning sobre staging
    python3 IDS/ids_learn.py status             # status do modelo atual
"""

import argparse
import json
import math
import sys
import time
import warnings
from collections import Counter
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config

Config.configure_tensorflow()

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Bidirectional, Dense, Dropout, Input, LSTM
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, matthews_corrcoef,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
from scipy.stats import entropy as scipy_entropy

from IDS.modules.utils import get_learn_logger, get_app_logger, timed, format_duration
from IDS.modules.versioning import versioned_path

log  = get_learn_logger()
alog = get_app_logger()
tf.random.set_seed(Config.TRAINING_CONFIG["random_state"])
np.random.seed(Config.TRAINING_CONFIG["random_state"])


# ─────────────────────────────────────────────────────────────────────────────
# Atenção de Bahdanau
# ─────────────────────────────────────────────────────────────────────────────

class BahdanauAttention(tf.keras.layers.Layer):
    """
    Atenção aditiva (Bahdanau et al., 2015).
        e_t = v^T · tanh(W_h · h_t + b_a)
        α_t = softmax(e_t)
        c   = Σ α_t · h_t
    """
    def __init__(self, units: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = Dense(units, use_bias=True, kernel_initializer="glorot_uniform")
        self.V = Dense(1, use_bias=False, kernel_initializer="glorot_uniform")

    def call(self, hidden_states, training=False):
        score   = self.V(tf.nn.tanh(self.W(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss reponderada (Cui et al., 2019; Lin et al., 2017)
# ─────────────────────────────────────────────────────────────────────────────

def make_focal_loss_class_balanced(class_counts, gamma: float = 2.0,
                                   beta: float = 0.9999):
    """
    Constrói Focal Loss com pesos por classe baseados em number of effective
    samples. Compatível com sparse_categorical labels.
    """
    class_counts = np.asarray(class_counts, dtype=np.float64)
    n_eff = (1.0 - np.power(beta, class_counts)) / (1.0 - beta)
    weights = (1.0 - beta) / np.maximum(n_eff, 1e-12)
    weights = weights / weights.sum() * len(weights)
    weights_tf = tf.constant(weights, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
        probs = tf.gather(y_pred, y_true, axis=1, batch_dims=1)
        w = tf.gather(weights_tf, y_true)
        focal_term = tf.pow(1.0 - probs, gamma)
        ce_term = -tf.math.log(probs)
        return tf.reduce_mean(w * focal_term * ce_term)

    loss_fn.__name__ = "focal_loss_cb"
    return loss_fn


def compute_class_weight_dict(y_train_original):
    """Calcula class_weight no formato {class_id: weight} para Keras fit."""
    classes = np.unique(y_train_original)
    weights = compute_class_weight(
        class_weight="balanced", classes=classes, y=y_train_original,
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}


# ─────────────────────────────────────────────────────────────────────────────
# DataHandler
# ─────────────────────────────────────────────────────────────────────────────

class DataHandler:
    """Pipeline de pré-processamento. NÃO balanceia mais o conjunto inteiro."""

    def __init__(self) -> None:
        self.scaler        = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.selected_features: list = []
        self.label_mapping: dict     = {}
        Config.ensure_dirs()

    # ── Carregamento ──────────────────────────────────────────────────────────

    def load_dataset(self) -> pd.DataFrame:
        log.info(f"Carregando dataset de '{Config.DATA_DIR}' …")
        csvs     = list(Config.DATA_DIR.glob("*.csv"))
        parquets = list(Config.DATA_DIR.glob("*.parquet"))
        if not csvs and not parquets:
            raise FileNotFoundError(f"Nenhum CSV/Parquet em {Config.DATA_DIR}")

        frames = []
        for f in csvs:
            log.info(f"  CSV: {f.name}")
            frames.append(pd.read_csv(f, low_memory=False))
        for f in parquets:
            log.info(f"  Parquet: {f.name}")
            frames.append(pd.read_parquet(f))

        df = pd.concat(frames, ignore_index=True)
        log.info(f"Dataset bruto: {df.shape[0]:,} linhas × {df.shape[1]} colunas")
        return df

    def load_with_cache(self) -> pd.DataFrame:
        cache = Config.TEMP_DIR / "01_cleaned_dataset.parquet"
        if cache.exists() and not Config.PREPROCESSING_CONFIG["force_reload"]:
            log.info(f"Cache de limpeza encontrado: '{cache.name}'")
            return pd.read_parquet(cache)
        df = self.load_dataset()
        df = self.clean(df)
        df.to_parquet(cache, compression="snappy", index=False)
        log.info(f"Cache de limpeza salvo: '{cache}'")
        return df

    # ── Limpeza ───────────────────────────────────────────────────────────────

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        log.info("Limpando dados …")
        n0 = len(df)

        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        thresh = int(Config.PREPROCESSING_CONFIG["missing_value_threshold"] * len(df))
        df.dropna(axis=1, thresh=thresh, inplace=True)
        df.fillna(0, inplace=True)

        for col in df.columns:
            if df[col].dtype == "object" and col.lower() != "label":
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df.fillna(0, inplace=True)

        if Config.PREPROCESSING_CONFIG["apply_variance_filter"]:
            vt   = Config.PREPROCESSING_CONFIG["variance_threshold"]
            fcols = [c for c in df.columns if c != "Label"]
            low   = [c for c in fcols if df[c].var() <= vt]
            if low:
                log.info(f"Filtro variância: removendo {len(low)} coluna(s): {low}")
                df.drop(columns=low, inplace=True)

        meta_to_drop = [c for c in Config.META_COLUMNS if c in df.columns]
        if meta_to_drop:
            df.drop(columns=meta_to_drop, inplace=True)

        log.info(f"Limpeza: {n0:,} → {len(df):,} linhas | {df.shape[1]} colunas")
        return df

    # ── Seleção de features ───────────────────────────────────────────────────

    def _information_gain(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        n_bins  = Config.FEATURE_SELECTION_CONFIG["ig_discretization_bins"]
        counts  = np.bincount(y.astype(int))
        probs   = counts[counts > 0] / counts.sum()
        h_y     = scipy_entropy(probs, base=2)
        ig      = np.zeros(X.shape[1])

        for i in range(X.shape[1]):
            col  = X[:, i]
            bins = np.linspace(col.min(), col.max(), n_bins + 1)
            digi = np.clip(np.digitize(col, bins[:-1]) - 1, 0, n_bins - 1)
            h_cond = 0.0
            for b in range(n_bins):
                mask = digi == b
                nb   = mask.sum()
                if nb == 0:
                    continue
                pb   = nb / len(y)
                yb   = y[mask].astype(int)
                pb_y = np.bincount(yb, minlength=len(counts)) / nb
                pb_y = pb_y[pb_y > 0]
                h_cond += pb * scipy_entropy(pb_y, base=2)
            ig[i] = h_y - h_cond
        return ig

    def select_features(self, X: np.ndarray, y: np.ndarray, names: list) -> tuple:
        cfg  = Config.FEATURE_SELECTION_CONFIG
        k    = cfg["k_best"]
        w_ig = cfg["ig_weight"]
        w_mi = cfg["mi_weight"]
        eps  = cfg["normalization_epsilon"]

        log.info(f"Calculando Information Gain (peso={w_ig}) …")
        ig_raw = self._information_gain(X, y)

        log.info(f"Calculando Mutual Information (peso={w_mi}) …")
        mi_raw = mutual_info_classif(X, y, random_state=Config.TRAINING_CONFIG["random_state"])

        def _norm(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + eps)

        ig_n   = _norm(ig_raw)
        mi_n   = _norm(mi_raw)
        scores = w_ig * ig_n + w_mi * mi_n

        ranked = sorted(zip(names, scores, ig_raw, mi_raw),
                        key=lambda x: x[1], reverse=True)
        selected = [n for n, *_ in ranked[:k]]

        log.info(f"Top-{k} features selecionadas (IG×{w_ig} + MI×{w_mi}):")
        feature_scores = {}
        for rank, (n, sc, ig, mi) in enumerate(ranked, 1):
            log.info(f"  [{rank:02d}] {n:<40s} score={sc:.4f}  IG={ig:.4f}  MI={mi:.4f}")
            i = names.index(n)
            feature_scores[n] = {
                "ig_raw": float(ig_raw[i]),  "mi_raw": float(mi_raw[i]),
                "ig_norm": float(ig_n[i]),   "mi_norm": float(mi_n[i]),
                "combined": float(scores[i]),
                "selected": n in selected,
            }

        self.selected_features = selected
        return selected, feature_scores

    # ── Balanceamento ADAPTATIVO ──────────────────────────────────────────────

    def balance(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """
        Balanceamento adaptativo (correção D3).

        Estratégia 'adaptive_borderline':
          1. n_c_alvo = min(r_c * n_c, beta * n_majoritaria) por classe minoritária
          2. k_c = min(k_max, ceil(alpha * sqrt(n_c)))
          3. BorderlineSMOTE-2
          4. RandomUnderSampler na majoritária
          5. ENN para limpeza

        IMPORTANTE: este método deve receber APENAS o conjunto de treino.
        """
        cfg = Config.BALANCING_CONFIG
        if cfg["strategy"] == "none":
            log.info("Balanceamento desativado (strategy='none')")
            return X, y

        if cfg["strategy"] == "classic_smote_enn":
            return self._balance_classic(X, y)

        # Default: 'adaptive_borderline'
        dist = Counter(y)
        n_majority = max(dist.values())
        majority_class = max(dist, key=dist.get)

        sampling_strategy = {}
        k_per_class = {}
        for c, n_c in dist.items():
            if c == majority_class:
                continue
            n_target = min(
                cfg["target_ratio_per_class"] * n_c,
                int(cfg["max_fraction_of_majority"] * n_majority),
            )
            n_target = max(n_target, n_c)
            sampling_strategy[c] = n_target
            k_per_class[c] = min(
                cfg["smote_k_neighbors_max"],
                max(1, math.ceil(cfg["smote_k_alpha"] * math.sqrt(n_c))),
            )

        log.info(f"Balanceamento adaptativo — distribuição original: {dict(dist)}")
        log.info(f"Alvos por classe minoritária: {sampling_strategy}")
        log.info(f"k adaptativo por classe: {k_per_class}")

        if not sampling_strategy:
            log.info("Sem classes minoritárias para oversampling — pulando.")
            return X, y

        k_med = max(1, int(np.median(list(k_per_class.values()))))
        log.info(f"k_neighbors usado pelo BorderlineSMOTE: {k_med}")

        smote = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=k_med,
            kind=cfg["borderline_kind"],
            random_state=Config.TRAINING_CONFIG["random_state"],
            n_jobs=-1,
        )
        Xr, yr = smote.fit_resample(X, y)
        log.info(f"Após BorderlineSMOTE: {Xr.shape[0]:,} amostras")

        target_majority = int(
            cfg["majority_undersample_factor"] * max(sampling_strategy.values())
        )
        target_majority = min(target_majority, dist[majority_class])
        rus = RandomUnderSampler(
            sampling_strategy={majority_class: target_majority},
            random_state=Config.TRAINING_CONFIG["random_state"],
        )
        Xr, yr = rus.fit_resample(Xr, yr)
        log.info(f"Após RandomUnderSampler: {Xr.shape[0]:,} amostras")

        enn = EditedNearestNeighbours(
            n_neighbors=cfg["enn_n_neighbors"],
            kind_sel=cfg["enn_kind_sel"],
            n_jobs=-1,
        )
        Xb, yb = enn.fit_resample(Xr, yr)
        log.info(f"Após ENN: {Xb.shape[0]:,} amostras | dist={dict(Counter(yb))}")
        return Xb, yb

    def _balance_classic(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Estratégia legada SMOTE → RUS → ENN. Mantida para compatibilidade."""
        cfg = Config.BALANCING_CONFIG
        dist = Counter(y)
        log.info(f"Balanceamento clássico — dist original: {dict(dist)}")

        try:
            benign_enc = self.label_encoder.transform(["Benign"])[0]
        except (ValueError, AttributeError):
            benign_enc = max(dist, key=dist.get)

        over_strat = {
            c: max(n, cfg["n_samples_minority"])
            for c, n in dist.items() if c != benign_enc
        }
        under_strat = {benign_enc: cfg["n_samples_majority"]}

        smote = SMOTE(
            sampling_strategy=over_strat,
            k_neighbors=cfg["smote_k_neighbors"],
            random_state=Config.TRAINING_CONFIG["random_state"],
        )
        Xr, yr = smote.fit_resample(X, y)
        rus = RandomUnderSampler(
            sampling_strategy=under_strat,
            random_state=Config.TRAINING_CONFIG["random_state"],
        )
        Xr, yr = rus.fit_resample(Xr, yr)
        enn = EditedNearestNeighbours(n_neighbors=cfg["enn_n_neighbors"])
        Xb, yb = enn.fit_resample(Xr, yr)
        log.info(f"Após balanceamento clássico: {Xb.shape[0]:,} amostras")
        return Xb, yb

    # ── Pré-processamento SEM balanceamento ───────────────────────────────────

    @timed("Pré-processamento")
    def preprocess_with_cache(self, df: pd.DataFrame) -> tuple:
        """
        Retorna (X_scaled, y, selected_features) — dados PROCESSADOS porém
        NÃO BALANCEADOS. O balanceamento é aplicado depois, somente ao
        conjunto de treino, via cmd_train(). Corrige leakage D1.
        """
        cx = Config.TEMP_DIR / "03_X_scaled_unbalanced.pkl"
        cy = Config.TEMP_DIR / "03_y_unbalanced.pkl"

        if (cx.exists() and cy.exists()
                and not Config.PREPROCESSING_CONFIG["force_preprocess"]):
            log.info("Cache de arrays escalados (sem balanceamento) — carregando …")
            X_s = joblib.load(cx)
            y = joblib.load(cy)
            self.scaler = joblib.load(Config.MODEL_DIR / Config.SCALER_FILENAME)
            self.label_encoder = joblib.load(Config.MODEL_DIR / Config.LABEL_ENCODER_FILENAME)
            with open(Config.MODEL_DIR / Config.MODEL_INFO_FILENAME, encoding="utf-8") as f:
                info = json.load(f)
            self.selected_features = info["selected_features"]
            self.label_mapping = {int(k): v for k, v in info["label_mapping"].items()}
            return X_s, y, self.selected_features

        log.info("Iniciando pré-processamento (SEM balanceamento)…")
        self.label_encoder.fit(df["Label"])
        self.label_mapping = {i: lbl for i, lbl in enumerate(self.label_encoder.classes_)}

        n_sample = min(Config.PREPROCESSING_CONFIG["sample_size_for_selection"], len(df))
        feat_names = [c for c in df.columns if c != "Label"]
        y_all = self.label_encoder.transform(df["Label"])

        if n_sample < len(df):
            _, df_s, _, y_s = train_test_split(
                df, y_all,
                test_size=n_sample / len(df),
                random_state=Config.TRAINING_CONFIG["random_state"],
                stratify=y_all,
            )
            X_s_for_sel = df_s[feat_names].values
        else:
            X_s_for_sel = df[feat_names].values
            y_s = y_all

        selected, feat_scores = self.select_features(X_s_for_sel, y_s, feat_names)

        X_full = df[selected].values
        log.info("Normalizando com StandardScaler…")
        X_scaled = self.scaler.fit_transform(X_full)
        y_full = y_all

        log.info("Salvando cache de arrays escalados (sem balanceamento)…")
        joblib.dump(X_scaled, cx)
        joblib.dump(y_full, cy)
        joblib.dump(self.scaler, Config.MODEL_DIR / Config.SCALER_FILENAME)
        joblib.dump(self.label_encoder, Config.MODEL_DIR / Config.LABEL_ENCODER_FILENAME)

        info = {
            "version": f"v{datetime.now().strftime('%Y%m%d%H%M')}",
            "trained_at": datetime.now().isoformat(),
            "selected_features": selected,
            "label_mapping": self.label_mapping,
            "feature_selection": {
                "k_best": Config.FEATURE_SELECTION_CONFIG["k_best"],
                "ig_weight": Config.FEATURE_SELECTION_CONFIG["ig_weight"],
                "mi_weight": Config.FEATURE_SELECTION_CONFIG["mi_weight"],
            },
            "feature_scores": feat_scores,
        }
        with open(Config.MODEL_DIR / Config.MODEL_INFO_FILENAME, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)

        return X_scaled, y_full, selected


# ─────────────────────────────────────────────────────────────────────────────
# ModelTrainer
# ─────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """Constrói, treina, avalia e salva Bi-LSTM + Atenção de Bahdanau."""

    def __init__(self) -> None:
        self.model   = None
        self.history = None
        Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def build(self, n_features: int, n_classes: int,
              class_counts=None) -> None:
        cfg = Config.MODEL_CONFIG
        inp = Input(shape=(n_features, 1), name="input")

        x = Bidirectional(
            LSTM(cfg["lstm_units_1"], return_sequences=True,
                 recurrent_dropout=cfg["recurrent_dropout_rate"]),
            name="bilstm_1",
        )(inp)
        x = Dropout(cfg["dropout_rate"], name="drop_1")(x)

        x = Bidirectional(
            LSTM(cfg["lstm_units_2"], return_sequences=True,
                 recurrent_dropout=cfg["recurrent_dropout_rate"]),
            name="bilstm_2",
        )(x)
        x = Dropout(cfg["dropout_rate"], name="drop_2")(x)

        ctx, _ = BahdanauAttention(cfg["attention_units"], name="attention")(x)

        x = Dense(cfg["dense_units"], activation="relu", name="dense_1")(ctx)
        x = Dropout(cfg["dropout_rate"] * 0.5, name="drop_3")(x)
        out = Dense(n_classes, activation="softmax", name="output")(x)

        self.model = Model(inp, out, name="SecurityIA_BiLSTM_Bahdanau")

        if cfg["loss_function"] == "focal_loss_cb" and class_counts is not None:
            log.info(f"Compilando com Focal Loss reponderada — "
                     f"gamma={cfg['focal_gamma']} beta={cfg['focal_class_balanced_beta']}")
            loss = make_focal_loss_class_balanced(
                class_counts=class_counts,
                gamma=cfg["focal_gamma"],
                beta=cfg["focal_class_balanced_beta"],
            )
        else:
            log.info("Compilando com sparse_categorical_crossentropy")
            loss = "sparse_categorical_crossentropy"

        spe = Config.TRAINING_CONFIG.get("steps_per_execution", 1)
        self.model.compile(
            optimizer=Adam(learning_rate=cfg["learning_rate"]),
            loss=loss,
            metrics=cfg["metrics"],
            steps_per_execution=spe,
        )
        log.info(f"steps_per_execution={spe}")
        self.model.summary(print_fn=log.info)

        try:
            diag = versioned_path(Config.MODEL_DIR, "model_architecture", "png")
            plot_model(self.model, to_file=str(diag), show_shapes=True,
                       show_layer_names=True)
            log.info(f"Diagrama salvo em '{diag}'")
        except Exception:
            pass

    def load_for_finetune(self, n_classes: int) -> None:
        mp = Config.MODEL_DIR / Config.MODEL_FILENAME
        if not mp.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {mp}")
        log.info(f"Carregando modelo para fine-tuning: {mp}")
        custom = {"BahdanauAttention": BahdanauAttention}
        # Em fine-tuning, a Focal Loss salva no .keras pode não ser
        # serializável; recarregamos sem compilar e recompilamos.
        self.model = load_model(str(mp), custom_objects=custom, compile=False)
        ft_lr = Config.FINE_TUNING_CONFIG["learning_rate"]
        self.model.compile(
            optimizer=Adam(learning_rate=ft_lr),
            loss="sparse_categorical_crossentropy",
            metrics=Config.MODEL_CONFIG["metrics"],
        )
        log.info(f"Modelo recompilado para fine-tuning com lr={ft_lr}")

    @timed("Treinamento")
    def train(
        self, X_tr: np.ndarray, y_tr: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        finetune: bool = False,
        class_weight: dict = None,
    ) -> None:
        cfg = Config.TRAINING_CONFIG
        ft_cfg = Config.FINE_TUNING_CONFIG

        X_tr_r  = X_tr.reshape(len(X_tr),   X_tr.shape[1],  1)
        X_val_r = X_val.reshape(len(X_val), X_val.shape[1], 1)

        epochs   = ft_cfg["epochs"]   if finetune else cfg["epochs"]
        patience = ft_cfg["patience"] if finetune else cfg["patience"]
        bs       = cfg["batch_size"]

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=patience,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=max(3, patience // 2),
                              min_lr=1e-7, verbose=1),
        ]

        if not (class_weight is not None and cfg.get("use_class_weight", False)):
            class_weight = None

        log.info(
            f"{'Fine-tuning' if finetune else 'Treinamento'} iniciado — "
            f"épocas={epochs} batch={bs} "
            f"class_weight={'sim' if class_weight else 'não'}"
        )
        t0 = time.time()

        self.history = self.model.fit(
            X_tr_r, y_tr,
            validation_data=(X_val_r, y_val),
            epochs=epochs,
            batch_size=bs,
            callbacks=callbacks,
            class_weight=class_weight,
            verbose=1,
        )
        elapsed = time.time() - t0
        log.info(f"Treinamento concluído em {format_duration(elapsed)}")

    def evaluate(self, X_te: np.ndarray, y_te: np.ndarray) -> np.ndarray:
        X_r = X_te.reshape(len(X_te), X_te.shape[1], 1)
        bs  = Config.EVALUATION_CONFIG.get("batch_size", 4096)
        return np.argmax(self.model.predict(X_r, batch_size=bs, verbose=0), axis=1)

    def save(self) -> None:
        mp = Config.MODEL_DIR / Config.MODEL_FILENAME
        self.model.save(str(mp))
        log.info(f"Modelo salvo: '{mp}'")

        if self.history:
            hp = versioned_path(Config.MODEL_DIR, "training_history", "json")
            hist = {k: [float(v) for v in vals]
                    for k, vals in self.history.history.items()}
            with open(hp, "w", encoding="utf-8") as f:
                json.dump(hist, f, indent=2)
            log.info(f"Histórico salvo: '{hp}'")


# ─────────────────────────────────────────────────────────────────────────────
# ReportGenerator (gráficos por execução, com nome versionado)
# ─────────────────────────────────────────────────────────────────────────────

class ReportGenerator:
    """Gráficos e relatório textual de classificação por execução."""

    def __init__(self, history, out_dir: Path) -> None:
        self.history = history.history if history else None
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use(Config.VIZ_CONFIG["style"])

    def plot_history(self) -> None:
        if not self.history:
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.plot(self.history.get("loss", []),     label="Treino")
        ax1.plot(self.history.get("val_loss", []), label="Validação")
        ax1.set(title="Loss", xlabel="Época", ylabel="Loss")
        ax1.legend()
        ax2.plot(self.history.get("accuracy", []),     label="Treino")
        ax2.plot(self.history.get("val_accuracy", []), label="Validação")
        ax2.set(title="Acurácia", xlabel="Época", ylabel="Acurácia")
        ax2.legend()
        fig.tight_layout()
        p = versioned_path(self.out_dir, "training_history", Config.VIZ_CONFIG["save_format"])
        fig.savefig(p, dpi=Config.VIZ_CONFIG["dpi"])
        plt.close(fig)
        log.info(f"Histórico de treinamento salvo: '{p}'")

    def plot_confusion_matrix(self, y_true, y_pred, label_map: dict) -> None:
        cm = confusion_matrix(y_true, y_pred)
        cm_n = cm.astype("float") / np.maximum(cm.sum(axis=1, keepdims=True), 1)
        labels = [label_map.get(i, str(i)) for i in sorted(label_map)]

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set(title="Matriz de Confusão Normalizada",
               xlabel="Classe Predita", ylabel="Classe Real")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        p = versioned_path(self.out_dir, "confusion_matrix", Config.VIZ_CONFIG["save_format"])
        fig.savefig(p, dpi=Config.VIZ_CONFIG["dpi"])
        plt.close(fig)
        log.info(f"Matriz de confusão salva: '{p}'")

    def save_report(self, y_true, y_pred, label_map: dict) -> None:
        labels = [label_map.get(i, str(i)) for i in sorted(label_map)]
        report = classification_report(y_true, y_pred, target_names=labels)
        p = versioned_path(self.out_dir, "classification_report", "txt")
        with open(p, "w", encoding="utf-8") as f:
            f.write("Relatório de Classificação\n")
            f.write("=" * 60 + "\n")
            f.write(report)
        log.info(f"Relatório de classificação salvo: '{p}'")
        print("\n" + report)


# ─────────────────────────────────────────────────────────────────────────────
# Funções auxiliares: métricas para registry
# ─────────────────────────────────────────────────────────────────────────────

def _metrics_for_registry(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    fpr_pc = []
    for c in range(cm.shape[0]):
        fp = cm[:, c].sum() - cm[c, c]
        tn = cm.sum() - cm[c, :].sum() - cm[:, c].sum() + cm[c, c]
        denom = fp + tn
        fpr_pc.append(fp / denom if denom else 0.0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "fpr_macro": float(np.mean(fpr_pc)),
        "n_test": int(len(y_true)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Comandos de alto nível
# ─────────────────────────────────────────────────────────────────────────────

@timed("Treinamento completo")
def cmd_train(force: bool = False) -> None:
    """
    Treinamento integral CORRIGIDO:
      1. Carrega/limpa dataset
      2. Pré-processa SEM balancear
      3. Split estratificado treino/val/teste sobre dados ORIGINAIS
      4. class_counts e class_weight a partir do TREINO
      5. Balanceia APENAS o treino
      6. Constrói modelo com Focal Loss reponderada
      7. Treina com class_weight no fit
      8. Avalia no teste ORIGINAL (não balanceado)
      9. Registra Mc no triplete; gera relatório completo M0/Mp/Mc
    """
    from IDS.modules.model_registry import register_framework_version
    from IDS.modules.full_report import generate as generate_full_report

    log.info("=" * 62)
    log.info("SecurityIA — TREINAMENTO COMPLETO (D1+D2+D3 corrigidos)")
    log.info("=" * 62)

    if force:
        Config.TRAINING_CONFIG["force_retrain"] = True
        Config.PREPROCESSING_CONFIG["force_reload"] = True
        Config.PREPROCESSING_CONFIG["force_preprocess"] = True

    dh = DataHandler()
    df = dh.load_with_cache()
    X, y, selected = dh.preprocess_with_cache(df)

    cfg = Config.TRAINING_CONFIG
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y,
        test_size=cfg["validation_split"] + cfg["test_split"],
        random_state=cfg["random_state"], stratify=y,
    )
    val_frac = cfg["validation_split"] / (cfg["validation_split"] + cfg["test_split"])
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp,
        test_size=1 - val_frac,
        random_state=cfg["random_state"], stratify=y_tmp,
    )
    log.info(
        f"Split ORIGINAL (sem balanceamento): "
        f"Treino={X_tr.shape[0]:,} | Val={X_val.shape[0]:,} | Teste={X_te.shape[0]:,}"
    )

    class_counts_train = np.bincount(y_tr.astype(int))
    class_weight = compute_class_weight_dict(y_tr) if cfg.get("use_class_weight") else None
    log.info(f"class_counts (pre-balanceamento): {class_counts_train.tolist()}")

    if cfg.get("balance_only_train", True):
        log.info("Aplicando balanceamento APENAS ao conjunto de treino…")
        X_tr, y_tr = dh.balance(X_tr, y_tr)
        log.info(f"Treino pós-balanceamento: {X_tr.shape[0]:,} amostras")

    trainer = ModelTrainer()
    mp = Config.MODEL_DIR / Config.MODEL_FILENAME

    if mp.exists() and Config.FINE_TUNING_CONFIG["enable"] and not force:
        log.warning(
            "Modelo existente detectado. Após mudanças estruturais (loss, "
            "balanceamento, split), TREINAMENTO DO ZERO é recomendado."
        )
        trainer.load_for_finetune(n_classes=len(dh.label_mapping))
        trainer.train(X_tr, y_tr, X_val, y_val, finetune=True,
                      class_weight=class_weight)
    else:
        trainer.build(
            n_features=X_tr.shape[1],
            n_classes=len(dh.label_mapping),
            class_counts=class_counts_train,
        )
        trainer.train(X_tr, y_tr, X_val, y_val, finetune=False,
                      class_weight=class_weight)

    log.info("Avaliando sobre conjunto de teste ORIGINAL (não balanceado)…")
    y_pred = trainer.evaluate(X_te, y_te)
    trainer.save()

    out_dir = Config.MODEL_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    rg = ReportGenerator(trainer.history, out_dir)
    rg.plot_history()
    rg.plot_confusion_matrix(y_te, y_pred, dh.label_mapping)
    rg.save_report(y_te, y_pred, dh.label_mapping)

    # Predições e dados de teste salvos com nome versionado
    np.save(versioned_path(out_dir, "y_test", "npy"), y_te)
    np.save(versioned_path(out_dir, "y_pred", "npy"), y_pred)
    np.save(versioned_path(out_dir, "X_test_scaled", "npy"), X_te)

    # Registro no triplete (Mc)
    register_framework_version(
        model_path=Config.MODEL_DIR / Config.MODEL_FILENAME,
        y_true=y_te,
        y_pred=y_pred,
        metrics=_metrics_for_registry(y_te, y_pred),
        source="train",
        extra={
            "loss": Config.MODEL_CONFIG["loss_function"],
            "balancing_strategy": Config.BALANCING_CONFIG["strategy"],
            "k_features": Config.FEATURE_SELECTION_CONFIG["k_best"],
            "epochs_run": (
                len(trainer.history.history["loss"]) if trainer.history else 0
            ),
        },
    )

    rep_dir = Config.IDS_REPORTS_DIR / f"full_train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    generate_full_report(rep_dir, label_map=dh.label_mapping)

    log.info("=" * 62)
    log.info("TREINAMENTO CONCLUÍDO — RELATÓRIO COMPLETO GERADO")
    log.info(f"Modelo: {Config.MODEL_DIR / Config.MODEL_FILENAME}")
    log.info(f"Relatório por execução: {out_dir}")
    log.info(f"Relatório completo M0/Mp/Mc: {rep_dir}")
    log.info("=" * 62)


def cmd_finetune() -> None:
    """
    Fine-tuning sobre dados de staging.
    Após o fine-tuning: registra Mc nova versão (Mc anterior vira Mp)
    e gera relatório completo comparando M0/Mp/Mc.
    """
    from IDS.modules.model_registry import register_framework_version
    from IDS.modules.full_report import generate as generate_full_report

    staging = Config.RETRAIN_CONFIG["staging_dir"]
    parquets = list(staging.glob("*.parquet"))
    if not parquets:
        print(f"  Nenhum arquivo de staging em '{staging}'")
        log.warning("Fine-tuning solicitado sem dados de staging.")
        return

    log.info(f"Fine-tuning sobre {len(parquets)} arquivo(s) de staging…")

    dh = DataHandler()
    Config.DATA_DIR = staging
    Config.PREPROCESSING_CONFIG["force_reload"] = True
    Config.PREPROCESSING_CONFIG["force_preprocess"] = True

    df = dh.load_with_cache()
    X, y, _ = dh.preprocess_with_cache(df)

    cfg = Config.TRAINING_CONFIG
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        X, y,
        test_size=cfg["validation_split"] + cfg["test_split"],
        random_state=cfg["random_state"], stratify=y,
    )
    val_frac = cfg["validation_split"] / (cfg["validation_split"] + cfg["test_split"])
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp,
        test_size=1 - val_frac,
        random_state=cfg["random_state"], stratify=y_tmp,
    )

    class_weight = compute_class_weight_dict(y_tr) if cfg.get("use_class_weight") else None

    if cfg.get("balance_only_train", True):
        X_tr, y_tr = dh.balance(X_tr, y_tr)

    trainer = ModelTrainer()
    trainer.load_for_finetune(n_classes=len(dh.label_mapping))
    trainer.train(X_tr, y_tr, X_val, y_val, finetune=True,
                  class_weight=class_weight)

    y_pred = trainer.evaluate(X_te, y_te)
    trainer.save()

    out_dir = Config.MODEL_DIR / f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    rg = ReportGenerator(trainer.history, out_dir)
    rg.plot_history()
    rg.plot_confusion_matrix(y_te, y_pred, dh.label_mapping)
    rg.save_report(y_te, y_pred, dh.label_mapping)

    np.save(versioned_path(out_dir, "y_test", "npy"), y_te)
    np.save(versioned_path(out_dir, "y_pred", "npy"), y_pred)

    register_framework_version(
        model_path=Config.MODEL_DIR / Config.MODEL_FILENAME,
        y_true=y_te,
        y_pred=y_pred,
        metrics=_metrics_for_registry(y_te, y_pred),
        source="finetune",
        extra={"staging_files": [p.name for p in parquets]},
    )

    rep_dir = Config.IDS_REPORTS_DIR / f"full_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    generate_full_report(rep_dir, label_map=dh.label_mapping)

    log.info("=" * 62)
    log.info("FINE-TUNING CONCLUÍDO — RELATÓRIO COMPLETO GERADO")
    log.info(f"Relatório completo M0/Mp/Mc: {rep_dir}")
    log.info("=" * 62)


def cmd_status() -> None:
    """Status do modelo atual e do triplete."""
    mp = Config.MODEL_DIR / Config.MODEL_FILENAME
    mi = Config.MODEL_DIR / Config.MODEL_INFO_FILENAME

    if not mp.exists():
        print("  Modelo não treinado ainda.")
        return

    stat = mp.stat()
    print(f"  Modelo   : {mp}")
    print(f"  Tamanho  : {stat.st_size / 1024**2:.1f} MiB")
    print(f"  Modificado: {datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")

    if mi.exists():
        with open(mi, encoding="utf-8") as f:
            info = json.load(f)
        print(f"  Versão   : {info.get('version', '?')}")
        print(f"  Treinado : {info.get('trained_at', '?')}")
        print(f"  Classes  : {len(info.get('label_mapping', {}))}")
        print(f"  Features : {len(info.get('selected_features', []))}")

    try:
        from IDS.modules.model_registry import _print_status
        _print_status()
    except ImportError:
        pass


def main() -> None:
    p = argparse.ArgumentParser(description="SecurityIA — Motor de Treinamento")
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train", help="Treinamento completo")
    tr.add_argument("--force", action="store_true", help="Ignora caches e re-treina do zero")

    sub.add_parser("finetune", help="Fine-tuning sobre dados de staging")
    sub.add_parser("status", help="Status do modelo atual")

    args = p.parse_args()

    if args.cmd == "train":
        cmd_train(force=getattr(args, "force", False))
    elif args.cmd == "finetune":
        cmd_finetune()
    elif args.cmd == "status":
        cmd_status()


if __name__ == "__main__":
    main()
