#!/usr/bin/env python3
"""
IDS/ids_learn.py — Motor de Treinamento e Fine-Tuning do Modelo IDS

Responsabilidades:
  - Carregamento e limpeza do dataset (CSV/Parquet) com cache em disco
  - Seleção de features via IG + MI ponderados (Teoria da Informação)
  - Balanceamento SMOTE → RandomUnderSampler → ENN
  - Construção da Bi-LSTM com Atenção de Bahdanau
  - Treinamento, avaliação e persistência de artefatos
  - Fine-tuning incremental sobre dados anotados do collector
  - Log detalhado em Logs/Learn.log

Uso:
    python3 IDS/ids_learn.py train             # treinamento completo
    python3 IDS/ids_learn.py finetune          # fine-tuning sobre staging
    python3 IDS/ids_learn.py status            # status do modelo atual
"""

import argparse
import json
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
from tensorflow.keras.layers import (
    Bidirectional, Dense, Dropout, Input, LSTM,
)
from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import mutual_info_classif
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
from scipy.stats import entropy as scipy_entropy

from IDS.modules.utils import get_learn_logger, get_app_logger, timed, format_duration

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
        self.W = Dense(units, use_bias=True,
                       kernel_initializer="glorot_uniform")
        self.V = Dense(1, use_bias=False,
                       kernel_initializer="glorot_uniform")

    def call(self, hidden_states, training=False):
        score   = self.V(tf.nn.tanh(self.W(hidden_states)))  # (n, T, 1)
        weights = tf.nn.softmax(score, axis=1)                 # (n, T, 1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)  # (n, 2u)
        return context, weights

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"units": self.units})
        return cfg


# ─────────────────────────────────────────────────────────────────────────────
# DataHandler
# ─────────────────────────────────────────────────────────────────────────────

class DataHandler:
    """Pipeline completo de pré-processamento com cache em Parquet."""

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

        # Filtro de variância (exclui colunas quase-constantes)
        if Config.PREPROCESSING_CONFIG["apply_variance_filter"]:
            vt   = Config.PREPROCESSING_CONFIG["variance_threshold"]
            fcols = [c for c in df.columns if c != "Label"]
            low   = [c for c in fcols if df[c].var() <= vt]
            if low:
                log.info(f"Filtro variância: removendo {len(low)} coluna(s): {low}")
                df.drop(columns=low, inplace=True)

        # Remove metadados de endereçamento (irrelevantes para treinamento)
        meta_to_drop = [c for c in Config.META_COLUMNS if c in df.columns]
        if meta_to_drop:
            df.drop(columns=meta_to_drop, inplace=True)

        log.info(f"Limpeza: {n0:,} → {len(df):,} linhas | {df.shape[1]} colunas")
        return df

    # ── Seleção de features (IG + MI ponderados) ──────────────────────────────

    def _information_gain(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """IG(X_i, Y) = H(Y) - H(Y|X_i) via discretização uniforme."""
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

    def select_features(
        self, X: np.ndarray, y: np.ndarray, names: list
    ) -> tuple:
        """
        Retorna (selected_names, feature_scores_dict).
        score = ig_weight * IG_norm + mi_weight * MI_norm
        """
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

    # ── Balanceamento ─────────────────────────────────────────────────────────

    def balance(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """SMOTE → RandomUnderSampler → ENN (pipeline otimizado para IDS)."""
        cfg = Config.BALANCING_CONFIG
        dist = Counter(y)
        log.info(f"Distribuição original: {dict(dist)}")

        benign_enc = self.label_encoder.transform(["Benign"])[0]
        over_strat = {
            c: max(n, cfg["n_samples_minority"])
            for c, n in dist.items()
            if c != benign_enc
        }
        under_strat = {benign_enc: cfg["n_samples_majority"]}

        log.info(f"SMOTE oversampling: {over_strat}")
        log.info(f"RandomUnder:        {under_strat}")

        pipe = ImbPipeline(steps=[
            ("smote", SMOTE(
                sampling_strategy=over_strat,
                k_neighbors=cfg["smote_k_neighbors"],
                random_state=Config.TRAINING_CONFIG["random_state"],
            )),
            ("rus", RandomUnderSampler(
                sampling_strategy=under_strat,
                random_state=Config.TRAINING_CONFIG["random_state"],
            )),
        ])
        Xr, yr = pipe.fit_resample(X, y)
        log.info("Aplicando ENN (limpeza de ruído) …")
        enn = EditedNearestNeighbours(n_neighbors=cfg["enn_n_neighbors"])
        Xb, yb = enn.fit_resample(Xr, yr)
        log.info(f"Após balanceamento: {Xb.shape[0]:,} amostras | dist={dict(Counter(yb))}")
        return Xb, yb

    # ── Pré-processamento com cache ───────────────────────────────────────────

    @timed("Pré-processamento")
    def preprocess_with_cache(self, df: pd.DataFrame) -> tuple:
        """
        Retorna (X_balanced, y_balanced, selected_features).
        Cache em Temp/ para evitar reprocessamento demorado.
        """
        cx = Config.TEMP_DIR / "03_X_balanced.pkl"
        cy = Config.TEMP_DIR / "03_y_balanced.pkl"

        if (cx.exists() and cy.exists()
                and not Config.PREPROCESSING_CONFIG["force_preprocess"]):
            log.info("Cache de arrays balanceados encontrado — carregando …")
            X_b = joblib.load(cx)
            y_b = joblib.load(cy)
            self.scaler        = joblib.load(Config.MODEL_DIR / Config.SCALER_FILENAME)
            self.label_encoder = joblib.load(Config.MODEL_DIR / Config.LABEL_ENCODER_FILENAME)
            with open(Config.MODEL_DIR / Config.MODEL_INFO_FILENAME, encoding="utf-8") as f:
                info = json.load(f)
            self.selected_features = info["selected_features"]
            self.label_mapping = {int(k): v for k, v in info["label_mapping"].items()}
            return X_b, y_b, self.selected_features

        log.info("Iniciando pré-processamento completo …")

        # Amostra estratificada para seleção de features (eficiência)
        n_sample = min(Config.PREPROCESSING_CONFIG["sample_size_for_selection"], len(df))
        df_s = df.sample(n=n_sample, random_state=Config.TRAINING_CONFIG["random_state"])
        X_s  = df_s.drop(columns=["Label"]).values
        y_s  = self.label_encoder.fit_transform(df_s["Label"])
        feat_names = df_s.drop(columns=["Label"]).columns.tolist()

        selected, feat_scores = self.select_features(X_s, y_s, feat_names)

        # Aplica ao dataset completo
        X_full = df[selected].values
        y_full = self.label_encoder.transform(df["Label"])
        self.label_mapping = {i: lbl for i, lbl in enumerate(self.label_encoder.classes_)}

        log.info("Normalizando com StandardScaler …")
        X_scaled = self.scaler.fit_transform(X_full)

        X_b, y_b = self.balance(X_scaled, y_full)

        # Persistência
        log.info("Salvando artefatos e cache …")
        joblib.dump(X_b, cx)
        joblib.dump(y_b, cy)
        joblib.dump(self.scaler,        Config.MODEL_DIR / Config.SCALER_FILENAME)
        joblib.dump(self.label_encoder, Config.MODEL_DIR / Config.LABEL_ENCODER_FILENAME)

        info = {
            "version":            f"v{datetime.now().strftime('%Y%m%d%H%M')}",
            "trained_at":         datetime.now().isoformat(),
            "selected_features":  selected,
            "label_mapping":      self.label_mapping,
            "feature_selection":  {
                "k_best":    Config.FEATURE_SELECTION_CONFIG["k_best"],
                "ig_weight": Config.FEATURE_SELECTION_CONFIG["ig_weight"],
                "mi_weight": Config.FEATURE_SELECTION_CONFIG["mi_weight"],
            },
            "feature_scores":     feat_scores,
        }
        with open(Config.MODEL_DIR / Config.MODEL_INFO_FILENAME, "w", encoding="utf-8") as f:
            json.dump(info, f, indent=4, ensure_ascii=False)

        log.info(f"Artefatos salvos em '{Config.MODEL_DIR}'")
        return X_b, y_b, selected


# ─────────────────────────────────────────────────────────────────────────────
# ModelTrainer
# ─────────────────────────────────────────────────────────────────────────────

class ModelTrainer:
    """Constrói, treina, avalia e salva o modelo Bi-LSTM + Bahdanau."""

    def __init__(self) -> None:
        self.model   = None
        self.history = None
        Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def build(self, n_features: int, n_classes: int) -> None:
        """Arquitetura: Input → Bi-LSTM(128) → Bi-LSTM(64) → Atenção → Dense → Softmax"""
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

        # Atenção de Bahdanau
        ctx, _ = BahdanauAttention(cfg["attention_units"], name="attention")(x)

        x = Dense(cfg["dense_units"], activation="relu", name="dense_1")(ctx)
        x = Dropout(cfg["dropout_rate"] * 0.5, name="drop_3")(x)
        out = Dense(n_classes, activation="softmax", name="output")(x)

        self.model = Model(inp, out, name="SecurityIA_BiLSTM_Bahdanau")
        self.model.compile(
            optimizer=Adam(learning_rate=cfg["learning_rate"]),
            loss=cfg["loss_function"],
            metrics=cfg["metrics"],
        )
        self.model.summary(print_fn=log.info)

        # Diagrama da arquitetura
        try:
            diag = Config.MODEL_DIR / "model_architecture.png"
            plot_model(self.model, to_file=str(diag), show_shapes=True,
                       show_layer_names=True)
            log.info(f"Diagrama salvo em '{diag}'")
        except Exception:
            pass

    def load_for_finetune(self, n_classes: int) -> None:
        """Carrega modelo existente e ajusta learning rate para fine-tuning."""
        mp = Config.MODEL_DIR / Config.MODEL_FILENAME
        if not mp.exists():
            raise FileNotFoundError(f"Modelo não encontrado: {mp}")
        log.info(f"Carregando modelo para fine-tuning: {mp}")
        self.model = load_model(str(mp), custom_objects={"BahdanauAttention": BahdanauAttention})
        ft_lr = Config.FINE_TUNING_CONFIG["learning_rate"]
        tf.keras.backend.set_value(self.model.optimizer.learning_rate, ft_lr)
        log.info(f"Learning rate ajustado para fine-tuning: {ft_lr}")

    @timed("Treinamento")
    def train(
        self, X_tr: np.ndarray, y_tr: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        finetune: bool = False,
    ) -> None:
        cfg    = Config.TRAINING_CONFIG
        ft_cfg = Config.FINE_TUNING_CONFIG

        X_tr_r  = X_tr.reshape(len(X_tr),   X_tr.shape[1],  1)
        X_val_r = X_val.reshape(len(X_val),  X_val.shape[1], 1)

        epochs  = ft_cfg["epochs"]  if finetune else cfg["epochs"]
        patience = ft_cfg["patience"] if finetune else cfg["patience"]

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=patience,
                          restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=max(3, patience // 2),
                              min_lr=1e-7, verbose=1),
        ]

        log.info(f"{'Fine-tuning' if finetune else 'Treinamento'} iniciado — "
                 f"épocas={epochs} batch={cfg['batch_size']}")
        t0 = time.time()

        self.history = self.model.fit(
            X_tr_r, y_tr,
            validation_data=(X_val_r, y_val),
            epochs=cfg["batch_size"],
            batch_size=cfg["batch_size"],
            callbacks=callbacks,
            verbose=1,
        )
        elapsed = time.time() - t0
        log.info(f"Treinamento concluído em {format_duration(elapsed)}")

    def evaluate(self, X_te: np.ndarray, y_te: np.ndarray) -> np.ndarray:
        X_r = X_te.reshape(len(X_te), X_te.shape[1], 1)
        return np.argmax(self.model.predict(X_r, verbose=0), axis=1)

    def save(self) -> None:
        mp = Config.MODEL_DIR / Config.MODEL_FILENAME
        self.model.save(str(mp))
        log.info(f"Modelo salvo: '{mp}'")

        if self.history:
            hp = Config.MODEL_DIR / f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            hist = {k: [float(v) for v in vals]
                    for k, vals in self.history.history.items()}
            with open(hp, "w", encoding="utf-8") as f:
                json.dump(hist, f, indent=2)
            log.info(f"Histórico salvo: '{hp}'")


# ─────────────────────────────────────────────────────────────────────────────
# ReportGenerator
# ─────────────────────────────────────────────────────────────────────────────

class ReportGenerator:
    """Gráficos e relatório textual de classificação."""

    def __init__(self, history, out_dir: Path) -> None:
        self.history  = history.history if history else None
        self.out_dir  = out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use(Config.VIZ_CONFIG["style"])

    def plot_history(self) -> None:
        if not self.history:
            return
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        ax1.plot(self.history["loss"],     label="Treino")
        ax1.plot(self.history["val_loss"], label="Validação")
        ax1.set(title="Loss", xlabel="Época", ylabel="Loss")
        ax1.legend()
        ax2.plot(self.history["accuracy"],     label="Treino")
        ax2.plot(self.history["val_accuracy"], label="Validação")
        ax2.set(title="Acurácia", xlabel="Época", ylabel="Acurácia")
        ax2.legend()
        fig.tight_layout()
        p = self.out_dir / f"training_history.{Config.VIZ_CONFIG['save_format']}"
        fig.savefig(p, dpi=Config.VIZ_CONFIG["dpi"])
        plt.close(fig)
        log.info(f"Histórico de treinamento salvo: '{p}'")

    def plot_confusion_matrix(self, y_true, y_pred, label_map: dict) -> None:
        cm     = confusion_matrix(y_true, y_pred)
        cm_n   = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        labels = [label_map.get(i, str(i)) for i in sorted(label_map)]

        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(cm_n, annot=True, fmt=".2f", cmap="Blues",
                    xticklabels=labels, yticklabels=labels, ax=ax)
        ax.set(title="Matriz de Confusão Normalizada",
               xlabel="Classe Predita", ylabel="Classe Real")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        p = self.out_dir / f"confusion_matrix.{Config.VIZ_CONFIG['save_format']}"
        fig.savefig(p, dpi=Config.VIZ_CONFIG["dpi"])
        plt.close(fig)
        log.info(f"Matriz de confusão salva: '{p}'")

    def save_report(self, y_true, y_pred, label_map: dict) -> None:
        labels = [label_map.get(i, str(i)) for i in sorted(label_map)]
        report = classification_report(y_true, y_pred, target_names=labels)
        p = self.out_dir / "classification_report.txt"
        with open(p, "w", encoding="utf-8") as f:
            f.write("Relatório de Classificação\n")
            f.write("=" * 60 + "\n")
            f.write(report)
        log.info(f"Relatório de classificação salvo: '{p}'")
        print("\n" + report)


# ─────────────────────────────────────────────────────────────────────────────
# Comandos de alto nível
# ─────────────────────────────────────────────────────────────────────────────

@timed("Treinamento completo")
def cmd_train(force: bool = False) -> None:
    """Treinamento completo do zero (ou com force_retrain)."""
    log.info("=" * 62)
    log.info("SecurityIA — TREINAMENTO COMPLETO INICIADO")
    log.info("=" * 62)

    if force:
        Config.TRAINING_CONFIG["force_retrain"]       = True
        Config.PREPROCESSING_CONFIG["force_reload"]   = True
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
    log.info(f"Treino={X_tr.shape[0]:,} | Val={X_val.shape[0]:,} | Teste={X_te.shape[0]:,}")

    trainer = ModelTrainer()
    mp = Config.MODEL_DIR / Config.MODEL_FILENAME

    if mp.exists() and Config.FINE_TUNING_CONFIG["enable"] and not force:
        trainer.load_for_finetune(n_classes=len(dh.label_mapping))
        trainer.train(X_tr, y_tr, X_val, y_val, finetune=True)
    else:
        trainer.build(n_features=X_tr.shape[1], n_classes=len(dh.label_mapping))
        trainer.train(X_tr, y_tr, X_val, y_val, finetune=False)

    y_pred = trainer.evaluate(X_te, y_te)
    trainer.save()

    out_dir = Config.MODEL_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    rg = ReportGenerator(trainer.history, out_dir)
    rg.plot_history()
    rg.plot_confusion_matrix(y_te, y_pred, dh.label_mapping)
    rg.save_report(y_te, y_pred, dh.label_mapping)

    log.info("=" * 62)
    log.info("TREINAMENTO CONCLUÍDO COM SUCESSO")
    log.info(f"Modelo: {Config.MODEL_DIR / Config.MODEL_FILENAME}")
    log.info(f"Resultados: {out_dir}")
    log.info("=" * 62)


def cmd_finetune() -> None:
    """Fine-tuning sobre dados anotados no diretório de staging."""
    staging = Config.RETRAIN_CONFIG["staging_dir"]
    parquets = list(staging.glob("*.parquet"))
    if not parquets:
        print(f"  Nenhum arquivo de staging encontrado em '{staging}'")
        log.warning("Fine-tuning solicitado sem dados de staging disponíveis.")
        return

    log.info(f"Fine-tuning sobre {len(parquets)} arquivo(s) de staging …")
    Config.DATA_DIR = staging
    Config.PREPROCESSING_CONFIG["force_reload"]    = True
    Config.PREPROCESSING_CONFIG["force_preprocess"] = True
    Config.FINE_TUNING_CONFIG["enable"]            = True
    cmd_train(force=False)


def cmd_status() -> None:
    """Exibe status do modelo atual."""
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


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="SecurityIA — Motor de Treinamento do Modelo IDS",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    tr = sub.add_parser("train",    help="Treinamento completo")
    tr.add_argument("--force", action="store_true", help="Ignora caches e re-treina do zero")

    sub.add_parser("finetune", help="Fine-tuning sobre dados de staging")
    sub.add_parser("status",   help="Status do modelo atual")

    args = p.parse_args()

    if args.cmd == "train":
        cmd_train(force=getattr(args, "force", False))
    elif args.cmd == "finetune":
        cmd_finetune()
    elif args.cmd == "status":
        cmd_status()


if __name__ == "__main__":
    main()
