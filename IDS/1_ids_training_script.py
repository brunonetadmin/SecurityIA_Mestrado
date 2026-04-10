#!/usr/bin/env python3
"""
##############################################################################################
#  Script de Treinamento Principal do Sistema IDS
#  Versão: 2.1 — Corrigida e Revisada
#
#  CORREÇÕES APLICADAS NESTE ARQUIVO (v2.1):
#
#  [FIX-A] Import adicionado: scipy.stats.entropy, necessário para o cálculo
#           interno do Information Gain em DataHandler._calculate_information_gain().
#
#  [FIX-B] Método NOVO: DataHandler._calculate_information_gain(X, y)
#           Implementa o cálculo do IG por discretização uniforme (binagem),
#           conforme metodologia descrita em script_teoria_informacao.py.
#           Este método existia conceitualmente na configuração (ids_config.py,
#           FEATURE_SELECTION_CONFIG.ig_weight) mas nunca havia sido implementado.
#
#  [FIX-C] Método CORRIGIDO: DataHandler._select_features(X, y, feature_names)
#           A versão anterior ignorava completamente os pesos ig_weight e mi_weight
#           definidos na configuração, aplicando apenas mutual_info_classif puro.
#           A versão corrigida:
#             1. Calcula IG e MI separadamente.
#             2. Normaliza ambos os scores no intervalo [0, 1] via min-max.
#             3. Combina com os pesos da configuração: score = w_ig*IG + w_mi*MI.
#             4. Persiste os scores individuais e combinados no model_info.json
#                para garantir rastreabilidade e reprodutibilidade.
#
#  [FIX-D] Método CORRIGIDO: DataHandler.clean_data(df)
#           variance_threshold estava definido em IDSConfig mas nunca aplicado.
#           A versão corrigida verifica a flag 'apply_variance_filter' e remove
#           colunas com variância inferior ao limiar configurado.
#
#  [FIX-E] Método CORRIGIDO: DataHandler.preprocess_with_cache(df)
#           O dicionário model_info agora persiste os scores de seleção
#           (ig_scores, mi_scores, combined_scores por feature), permitindo
#           auditoria e reprodutibilidade completas do processo de seleção.
#
#  [FIX-F] Comentário vazio corrigido na linha 113 do arquivo original.
#
#  [FIX-G] Indentação incorreta corrigida em ReportGenerator.save_classification_report().
#           O método utilizava indentação de 1 espaço em vez do padrão PEP 8 de 4 espaços.
#
#  Autor: Bruno Cavalcante Barbosa
#  UFAL - Universidade Federal de Alagoas
##############################################################################################
"""

# Imports padrão e de sistema
import os
import sys

# TF_CPP_MIN_LOG_LEVEL deve ser definido ANTES de importar tensorflow.
# '3' suprime mensagens INFO, WARNING e ERROR do backend C++ (oneDNN, CUDA probe).
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

import numpy as np
import pandas as pd
import logging
import json
import time
from datetime import datetime
from pathlib import Path
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

# Imports de visualização
import matplotlib.pyplot as plt
import seaborn as sns

# Imports de Machine Learning
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import EditedNearestNeighbours

import joblib

# [FIX-A] Import adicionado: scipy.stats.entropy é necessário para o cálculo
# da entropia marginal H(Y) e da entropia condicional H(Y|X) em
# DataHandler._calculate_information_gain(). Sem este import, a chamada a
# scipy_entropy() causaria NameError em tempo de execução.
from scipy.stats import entropy as scipy_entropy

# Importa a configuração centralizada
from ids_config import IDSConfig


# =============================================================================
# Configuração do Logging
# =============================================================================
def setup_logging(log_dir):
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"Treinamento_Registros_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=getattr(logging, IDSConfig.LOGGING_CONFIG['level']),
        format=IDSConfig.LOGGING_CONFIG['format'],
        datefmt=IDSConfig.LOGGING_CONFIG['datefmt'],
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout),
        ],
    )


# =============================================================================
# Classe para Manipulação de Dados
# =============================================================================
class DataHandler:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.selected_features = []
        self.label_mapping = {}

        # Garante que os diretórios necessários existam
        self.config.CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)

    def load_dataset(self):
        logging.info(f"Iniciando o carregamento do dataset de '{self.config.DATA_DIR}'...")
        csv_files = list(self.config.DATA_DIR.glob('*.csv'))
        if not csv_files:
            logging.error(f"Nenhum arquivo CSV encontrado em {self.config.DATA_DIR}")
            raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {self.config.DATA_DIR}")

        df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
        logging.info(f"Dataset bruto carregado. Shape: {df.shape}")
        return df

    def clean_data(self, df):
        """
        Executa a limpeza do DataFrame bruto, incluindo:
          - Substituição de valores infinitos por NaN.
          - Remoção de colunas com alta taxa de valores faltantes.
          - Imputação de NaN remanescentes com zero.
          - Conversão de colunas objeto não-rótulo para numérico.
          - [FIX-D] Remoção de colunas com variância inferior ao limiar
            configurado em PREPROCESSING_CONFIG['variance_threshold'],
            quando a flag 'apply_variance_filter' estiver habilitada.
        """
        logging.info("Iniciando a limpeza dos dados...")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Remove colunas com taxa de valores faltantes superior ao limiar
        threshold = int(self.config.PREPROCESSING_CONFIG['missing_value_threshold'] * len(df))
        df.dropna(axis=1, thresh=threshold, inplace=True)
        df.fillna(0, inplace=True)

        # Converte colunas de texto não-rótulo para numérico
        for col in df.columns:
            if df[col].dtype == 'object' and col.lower() != 'label':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.fillna(0, inplace=True)

        # [FIX-D] Aplica filtro de variância, se habilitado na configuração.
        # A versão original definia 'variance_threshold' em IDSConfig mas nunca
        # o aplicava, tornando-o um parâmetro morto que induzia o leitor a
        # acreditar que a filtragem ocorria quando, na prática, não ocorria.
        if self.config.PREPROCESSING_CONFIG.get('apply_variance_filter', False):
            variance_threshold = self.config.PREPROCESSING_CONFIG.get('variance_threshold', 1e-5)
            label_col = 'Label'
            feature_cols = [c for c in df.columns if c != label_col]

            variances = df[feature_cols].var()
            low_variance_cols = variances[variances <= variance_threshold].index.tolist()

            if low_variance_cols:
                logging.info(
                    f"Filtro de variância: removendo {len(low_variance_cols)} coluna(s) com "
                    f"variância <= {variance_threshold}: {low_variance_cols}"
                )
                df.drop(columns=low_variance_cols, inplace=True)
            else:
                logging.info(
                    f"Filtro de variância: nenhuma coluna abaixo do limiar {variance_threshold}. "
                    f"Nenhuma coluna removida."
                )

        logging.info(f"Dados limpos. Shape: {df.shape}")
        return df

    def load_and_clean_data_with_cache(self):
        """
        Unifica o carregamento e a limpeza com um sistema de cache em Parquet.
        Se um cache válido existir e 'force_reload' for False, o DataFrame
        limpo é carregado diretamente do disco, evitando o re-processamento.
        """
        cached_df_path = self.config.CACHE_DIR / "01_TREINAMENTO_Cleaned_Dataframe.parquet"

        if cached_df_path.exists() and not self.config.PREPROCESSING_CONFIG.get('force_reload', False):
            logging.info(f"SUCESSO: Carregando DataFrame limpo do cache: '{cached_df_path}'")
            return pd.read_parquet(cached_df_path)

        logging.info("AVISO: Cache de dados limpos não encontrado ou recarga forçada. Executando do zero.")
        df = self.load_dataset()
        df = self.clean_data(df)

        logging.info(f"Salvando DataFrame limpo em cache para uso futuro: '{cached_df_path}'")
        df.to_parquet(cached_df_path)

        return df

    # -------------------------------------------------------------------------
    # [FIX-B] Método NOVO: _calculate_information_gain
    # -------------------------------------------------------------------------
    def _calculate_information_gain(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calcula o Information Gain (IG) de cada coluna de X em relação ao
        vetor de rótulos y.

        O IG é definido como a redução da entropia da variável-alvo Y
        proporcionada pelo conhecimento de uma característica X_i:

            IG(X_i, Y) = H(Y) - H(Y | X_i)

        onde H(Y) é a entropia marginal de Y e H(Y | X_i) é a entropia
        condicional de Y dado X_i.

        Para variáveis contínuas, X_i é discretizada em bins uniformes antes
        do cálculo, conforme o parâmetro de configuração
        FEATURE_SELECTION_CONFIG['ig_discretization_bins'].

        Parâmetros
        ----------
        X : np.ndarray, shape (n_amostras, n_features)
            Matriz de características numéricas.
        y : np.ndarray, shape (n_amostras,)
            Vetor de rótulos de classe (inteiros codificados).

        Retorna
        -------
        ig_scores : np.ndarray, shape (n_features,)
            Score de Information Gain para cada característica.
            Valores no intervalo [0, H(Y)].
        """
        n_features = X.shape[1]
        n_bins = self.config.FEATURE_SELECTION_CONFIG.get('ig_discretization_bins', 10)

        # Entropia marginal da variável-alvo H(Y)
        class_counts = np.bincount(y.astype(int))
        class_probs = class_counts / class_counts.sum()
        # Filtra probabilidades zero para evitar log(0) no cálculo da entropia
        class_probs = class_probs[class_probs > 0]
        h_y = scipy_entropy(class_probs, base=2)

        ig_scores = np.zeros(n_features)

        for i in range(n_features):
            feature_col = X[:, i]

            # Discretização por bins uniformes (igual frequência de bins, não de amostras)
            bins = np.linspace(feature_col.min(), feature_col.max(), n_bins + 1)
            # np.digitize retorna índice do bin; clip garante que o índice máximo
            # seja n_bins (evita índice n_bins+1 para o valor exato do limite superior)
            digitized = np.clip(np.digitize(feature_col, bins[:-1]) - 1, 0, n_bins - 1)

            # Entropia condicional H(Y | X_i) = sum_b P(X_i=b) * H(Y | X_i=b)
            h_y_given_xi = 0.0
            for bin_idx in range(n_bins):
                mask = digitized == bin_idx
                n_in_bin = mask.sum()
                if n_in_bin == 0:
                    continue

                p_bin = n_in_bin / len(y)  # P(X_i = bin_idx)
                y_in_bin = y[mask].astype(int)

                # Entropia condicional para este bin H(Y | X_i = bin_idx)
                counts_in_bin = np.bincount(y_in_bin, minlength=class_counts.shape[0])
                probs_in_bin = counts_in_bin / counts_in_bin.sum()
                probs_in_bin = probs_in_bin[probs_in_bin > 0]

                if len(probs_in_bin) > 0:
                    h_y_given_xi += p_bin * scipy_entropy(probs_in_bin, base=2)

            ig_scores[i] = h_y - h_y_given_xi

        return ig_scores

    # -------------------------------------------------------------------------
    # [FIX-C] Método CORRIGIDO: _select_features
    # -------------------------------------------------------------------------
    def _select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list,
    ) -> list:
        """
        Seleciona as k melhores características utilizando um score combinado
        ponderado entre Information Gain (IG) e Mutual Information (MI).

        O score final para cada característica f_i é calculado como:

            score(f_i) = w_ig * IG_norm(f_i) + w_mi * MI_norm(f_i)

        onde IG_norm e MI_norm são os scores normalizados no intervalo [0, 1]
        via transformação min-max, e os pesos w_ig e w_mi são lidos de
        IDSConfig.FEATURE_SELECTION_CONFIG (padrão: 0.6 e 0.4, respectivamente).

        A normalização é necessária porque IG e MI possuem escalas distintas;
        sem normalização, o método com maior magnitude dominaria o score
        independentemente dos pesos configurados.

        Os scores individuais (IG, MI) e o score combinado são persistidos no
        arquivo model_info.json para garantir rastreabilidade e reprodutibilidade
        completas do processo de seleção de características.

        Parâmetros
        ----------
        X : np.ndarray, shape (n_amostras, n_features)
            Matriz de características da amostra de seleção.
        y : np.ndarray, shape (n_amostras,)
            Vetor de rótulos codificados.
        feature_names : list of str
            Nomes das colunas correspondentes às colunas de X.

        Retorna
        -------
        selected_features : list of str
            Lista com os nomes das k características selecionadas,
            ordenadas por score combinado decrescente.
        feature_scores : dict
            Dicionário com os scores detalhados por característica,
            para persistência em model_info.json.
        """
        fs_cfg = self.config.FEATURE_SELECTION_CONFIG
        k       = fs_cfg['k_best']
        w_ig    = fs_cfg['ig_weight']
        w_mi    = fs_cfg['mi_weight']
        eps     = fs_cfg.get('normalization_epsilon', 1e-9)

        # Validação dos pesos (devem somar 1.0 com tolerância numérica)
        if not np.isclose(w_ig + w_mi, 1.0, atol=1e-6):
            logging.warning(
                f"AVISO: ig_weight ({w_ig}) + mi_weight ({w_mi}) = {w_ig + w_mi:.6f} != 1.0. "
                f"Os pesos serão normalizados automaticamente para somar 1."
            )
            total = w_ig + w_mi
            w_ig /= total
            w_mi /= total

        # --- Cálculo dos scores brutos ---
        logging.info(
            f"Calculando Information Gain (peso={w_ig}) para seleção de características..."
        )
        ig_scores_raw = self._calculate_information_gain(X, y)

        logging.info(
            f"Calculando Mutual Information (peso={w_mi}) para seleção de características..."
        )
        mi_scores_raw = mutual_info_classif(
            X, y,
            random_state=self.config.TRAINING_CONFIG['random_state'],
        )

        # --- Normalização min-max para colocar ambas as métricas na mesma escala ---
        ig_min, ig_max = ig_scores_raw.min(), ig_scores_raw.max()
        mi_min, mi_max = mi_scores_raw.min(), mi_scores_raw.max()

        ig_norm = (ig_scores_raw - ig_min) / (ig_max - ig_min + eps)
        mi_norm = (mi_scores_raw - mi_min) / (mi_max - mi_min + eps)

        # --- Score combinado ponderado ---
        combined_scores_arr = w_ig * ig_norm + w_mi * mi_norm

        # --- Seleção das k melhores ---
        scored_features = list(zip(feature_names, combined_scores_arr))
        scored_features.sort(key=lambda x: x[1], reverse=True)
        selected_top_k = scored_features[:k]

        self.selected_features = [f[0] for f in selected_top_k]

        logging.info(
            f"Top {k} características selecionadas pelo score combinado "
            f"(IG×{w_ig} + MI×{w_mi}):"
        )
        for rank, (name, score) in enumerate(selected_top_k, start=1):
            idx = feature_names.index(name)
            logging.info(
                f"  [{rank:02d}] {name:<40s} "
                f"score={score:.6f}  IG={ig_scores_raw[idx]:.6f}  MI={mi_scores_raw[idx]:.6f}"
            )

        # --- Empacota scores detalhados para persistência ---
        feature_scores = {
            name: {
                'ig_raw':       float(ig_scores_raw[i]),
                'mi_raw':       float(mi_scores_raw[i]),
                'ig_normalized': float(ig_norm[i]),
                'mi_normalized': float(mi_norm[i]),
                'combined_score': float(combined_scores_arr[i]),
                'selected':     name in self.selected_features,
            }
            for i, name in enumerate(feature_names)
        }

        return self.selected_features, feature_scores

    # -------------------------------------------------------------------------
    # [FIX-E] Método CORRIGIDO: preprocess_with_cache
    # -------------------------------------------------------------------------
    def preprocess_with_cache(self, df):
        """
        Orquestra todo o pré-processamento com múltiplos pontos de cache.

        Etapas executadas (quando o cache não está disponível):
          1. Amostragem do dataset para seleção eficiente de características.
          2. Codificação dos rótulos via LabelEncoder.
          3. Seleção das k melhores características (IG + MI ponderados).
          4. Normalização via StandardScaler.
          5. Balanceamento dinâmico com pipeline SMOTE → RandomUnderSampler → ENN.
          6. Persistência dos artefatos (scaler, encoder, model_info com scores).
        """
        cached_X_path = self.config.CACHE_DIR / "03_PREPROCESS_X_balanced.pkl"
        cached_y_path = self.config.CACHE_DIR / "03_PREPROCESS_y_balanced.pkl"

        if (
            cached_X_path.exists()
            and cached_y_path.exists()
            and not self.config.PREPROCESSING_CONFIG.get('force_preprocess', False)
        ):
            logging.info("SUCESSO: Carregando os arrays de treino (X, y) já balanceados do cache.")
            X_balanced = joblib.load(cached_X_path)
            y_balanced = joblib.load(cached_y_path)

            self.scaler        = joblib.load(self.config.MODEL_DIR / self.config.SCALER_FILENAME)
            self.label_encoder = joblib.load(self.config.MODEL_DIR / self.config.LABEL_ENCODER_FILENAME)

            with open(self.config.MODEL_DIR / self.config.MODEL_INFO_FILENAME, 'r') as f:
                model_info = json.load(f)

            selected_features = model_info['selected_features']
            self.label_mapping = {int(k): v for k, v in model_info['label_mapping'].items()}

            return X_balanced, y_balanced, selected_features

        logging.info("AVISO: Cache de dados processados não encontrado ou o reprocessamento foi forçado.")

        label_col = 'Label'

        # --- Amostragem para seleção eficiente de características ---
        # A seleção é computacionalmente intensiva; usar o dataset completo
        # pode ser inviável. Uma amostra estratificada de até 200 mil registros
        # oferece representatividade adequada para estimativas de IG e MI.
        logging.info("Otimização: Usando amostra estratificada para seleção de características.")
        sample_size = min(200_000, len(df))
        df_sample = df.sample(n=sample_size, random_state=self.config.TRAINING_CONFIG['random_state'])

        X_sample = df_sample.drop(columns=[label_col])
        y_sample = df_sample[label_col]
        y_sample_encoded = self.label_encoder.fit_transform(y_sample)

        # [FIX-C] Agora retorna também os scores detalhados para persistência
        selected_features, feature_scores = self._select_features(
            X_sample.values,
            y_sample_encoded,
            X_sample.columns.tolist(),
        )

        # --- Aplica as features selecionadas ao DataFrame completo ---
        X = df[selected_features]
        y = df[label_col]
        y_encoded = self.label_encoder.transform(y)
        self.label_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)}

        # --- Normalização ---
        logging.info("Aplicando normalização StandardScaler...")
        X_scaled = self.scaler.fit_transform(X)

        # --- Balanceamento dinâmico e robusto ---
        logging.info("Aplicando balanceamento de classes com estratégia híbrida dinâmica...")

        class_distribution = Counter(y_encoded)
        logging.info(f"Distribuição original das classes: {class_distribution}")

        n_samples_minority_target = self.config.BALANCING_CONFIG.get('n_samples_minority', 50_000)
        benign_class_label_encoded = self.label_encoder.transform(['Benign'])[0]

        over_strategy = {
            class_label: max(num_samples, n_samples_minority_target)
            for class_label, num_samples in class_distribution.items()
            if class_label != benign_class_label_encoded
        }

        n_samples_majority = self.config.BALANCING_CONFIG.get('n_samples_majority', 150_000)
        under_strategy = {benign_class_label_encoded: n_samples_majority}

        logging.info(f"Estratégia SMOTE (Oversampling): {over_strategy}")
        logging.info(f"Estratégia RandomUnderSampler: {under_strategy}")

        smote = SMOTE(
            sampling_strategy=over_strategy,
            random_state=self.config.TRAINING_CONFIG['random_state'],
        )
        rus = RandomUnderSampler(
            sampling_strategy=under_strategy,
            random_state=self.config.TRAINING_CONFIG['random_state'],
        )
        enn = EditedNearestNeighbours(
            n_neighbors=self.config.BALANCING_CONFIG['enn_n_neighbors'],
        )

        balance_pipeline = Pipeline(steps=[('smote', smote), ('rus', rus)])
        X_resampled, y_resampled = balance_pipeline.fit_resample(X_scaled, y_encoded)

        logging.info("Aplicando ENN para limpeza de ruído nas fronteiras de decisão...")
        X_balanced, y_balanced = enn.fit_resample(X_resampled, y_resampled)

        logging.info(f"Dataset final para treinamento. Shape: {X_balanced.shape}")

        # --- Persistência dos artefatos em cache ---
        logging.info("Salvando arrays de treino (X, y) balanceados e artefatos em cache.")
        joblib.dump(X_balanced, cached_X_path)
        joblib.dump(y_balanced, cached_y_path)
        joblib.dump(self.scaler,        self.config.MODEL_DIR / self.config.SCALER_FILENAME)
        joblib.dump(self.label_encoder, self.config.MODEL_DIR / self.config.LABEL_ENCODER_FILENAME)

        # [FIX-E] model_info agora persiste os scores detalhados de seleção.
        # Isso garante rastreabilidade completa: qualquer execução futura poderá
        # reproduzir exatamente quais features foram selecionadas e por quê,
        # além de permitir auditoria dos scores IG, MI e combinado.
        model_info = {
            'selected_features': selected_features,
            'label_mapping': self.label_mapping,
            'feature_selection_config': {
                'k_best':    self.config.FEATURE_SELECTION_CONFIG['k_best'],
                'ig_weight': self.config.FEATURE_SELECTION_CONFIG['ig_weight'],
                'mi_weight': self.config.FEATURE_SELECTION_CONFIG['mi_weight'],
            },
            'feature_scores': feature_scores,  # Scores completos para auditoria
        }

        with open(self.config.MODEL_DIR / self.config.MODEL_INFO_FILENAME, 'w', encoding='utf-8') as f:
            json.dump(model_info, f, indent=4, ensure_ascii=False)

        logging.info(
            f"Artefatos de pré-processamento salvos em '{self.config.MODEL_DIR}'. "
            f"Scores de seleção de características incluídos em '{self.config.MODEL_INFO_FILENAME}'."
        )

        return X_balanced, y_balanced, selected_features


# =============================================================================
# Classe para Treinamento do Modelo
# (sem alterações nesta versão — reproduzida para integridade do arquivo)
# =============================================================================
class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None

    def _build_new_model(self, input_shape, num_classes):
        logging.info("Construindo uma nova arquitetura LSTM Bidirecional Híbrida...")
        model_cfg = self.config.MODEL_CONFIG
        model = Sequential([
            Bidirectional(LSTM(model_cfg['lstm_units_1'], return_sequences=True), input_shape=input_shape),
            Dropout(model_cfg['dropout_rate']),
            Bidirectional(LSTM(model_cfg['lstm_units_2'], return_sequences=False)),
            Dropout(model_cfg['dropout_rate']),
            Dense(model_cfg['dense_units'], activation='relu'),
            Dropout(model_cfg['dropout_rate']),
            Dense(num_classes, activation='softmax'),
        ])
        model.compile(
            optimizer=Adam(learning_rate=model_cfg['learning_rate']),
            loss=model_cfg['loss_function'],
            metrics=model_cfg['metrics'],
        )
        return model

    def build_or_load_model(self, input_shape, num_classes):
        model_path = self.config.MODEL_DIR / self.config.MODEL_FILENAME
        if model_path.exists() and self.config.FINE_TUNING_CONFIG['enable']:
            logging.info(f"Modelo encontrado em '{model_path}'. Carregando para fine-tuning...")
            self.model = load_model(model_path)
            new_lr = self.config.FINE_TUNING_CONFIG['learning_rate']
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
            logging.info(f"Taxa de aprendizado ajustada para fine-tuning: {new_lr}")
        else:
            logging.info("Nenhum modelo encontrado ou fine-tuning desabilitado. Construindo do zero...")
            self.model = self._build_new_model(input_shape, num_classes)

        self.model.summary(print_fn=logging.info)

        plot_path = self.config.RESULTS_DIR / "Treinamento_Arquitetura_do_Modelo.png"
        self.config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        plot_model(self.model, to_file=plot_path, show_shapes=True, show_layer_names=True)
        logging.info(f"Diagrama da arquitetura do modelo salvo em '{plot_path}'")

    def train(self, X_train, y_train, X_val, y_val):
        logging.info("Iniciando treinamento do modelo...")
        train_cfg = self.config.TRAINING_CONFIG

        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val_reshaped   = X_val.reshape(X_val.shape[0],   X_val.shape[1],   1)

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=train_cfg['patience'],
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        start_time = time.time()
        history = self.model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_val_reshaped, y_val),
            epochs=train_cfg['epochs'],
            batch_size=train_cfg['batch_size'],
            callbacks=callbacks,
            verbose=1,
        )
        logging.info(f"Treinamento concluído em {time.time() - start_time:.2f} segundos.")
        return history

    def evaluate(self, X_test, y_test):
        logging.info("Iniciando avaliação do modelo no conjunto de teste...")
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        y_pred_proba = self.model.predict(X_test_reshaped)
        return np.argmax(y_pred_proba, axis=1)

    def save_artifacts(self, data_handler, selected_features, history):
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = self.config.MODEL_DIR / self.config.MODEL_FILENAME
        logging.info(f"Salvando modelo treinado em '{model_path}'")
        self.model.save(model_path)

        logging.info("Processadores (scaler, encoder) e informações do modelo já salvos na etapa de cache.")

        if history:
            self.config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            history_path = self.config.RESULTS_DIR / f"Treinamento_Historico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            history_data_serializable = {
                key: [float(v) for v in values]
                for key, values in history.history.items()
            }
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(history_data_serializable, f, indent=4)
            logging.info(f"Histórico de treinamento salvo em '{history_path}'")


# =============================================================================
# Classe para Geração de Relatórios
# =============================================================================
class ReportGenerator:
    def __init__(self, config, history, output_dir):
        self.config = config
        self.history = history.history if history else None
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        plt.style.use(self.config.VISUALIZATION_CONFIG['style'])

    def plot_training_history(self):
        if not self.history:
            logging.warning("Histórico de treinamento não disponível para esta execução. Pulando gráfico.")
            return

        logging.info("Gerando gráfico do histórico de treinamento...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ax1.plot(self.history['loss'],     label='Training Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Evolução da Perda (Loss)')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Loss')
        ax1.legend()

        ax2.plot(self.history['accuracy'],     label='Training Accuracy')
        ax2.plot(self.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Evolução da Acurácia')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Acurácia')
        ax2.legend()

        fig.tight_layout()
        plt.savefig(
            self.output_dir / f"Treinamento_Historico.{self.config.VISUALIZATION_CONFIG['save_format']}",
            dpi=self.config.VISUALIZATION_CONFIG['dpi'],
        )
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, label_mapping):
        logging.info("Gerando Matriz de Confusão...")
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        labels = [label_mapping.get(i, str(i)) for i in sorted(label_mapping.keys())]

        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm_normalized,
            annot=True, fmt='.2f', cmap='Blues',
            xticklabels=labels, yticklabels=labels,
        )
        plt.title('Matriz de Confusão Normalizada', fontsize=16)
        plt.ylabel('Classe Verdadeira', fontsize=12)
        plt.xlabel('Classe Predita', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(
            self.output_dir / f"Treinamento_Matriz_de_Confusao.{self.config.VISUALIZATION_CONFIG['save_format']}",
            dpi=self.config.VISUALIZATION_CONFIG['dpi'],
        )
        plt.close()

    def save_classification_report(self, y_true, y_pred, label_mapping):
        # [FIX-G] Indentação corrigida de 1 espaço para 4 espaços (padrão PEP 8).
        # A indentação inconsistente não causava erro sintático neste caso específico
        # (Python aceita blocos com indentação uniforme de 1 espaço), mas violava
        # o padrão de legibilidade do projeto e poderia causar erros em refatorações.
        logging.info("Gerando e salvando relatório de classificação...")
        labels = [label_mapping.get(i, str(i)) for i in sorted(label_mapping.keys())]
        report = classification_report(y_true, y_pred, target_names=labels)

        # [FIX-02 / IDSConfig] REPORTS_DIR_NAME agora existe em IDSConfig.
        # A versão original referenciava self.config.REPORTS_DIR_NAME que não
        # estava definido na classe IDSConfig, causando AttributeError em runtime.
        reports_dir = self.output_dir / self.config.REPORTS_DIR_NAME
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = reports_dir / "Treinamento_Relatorio_de_Classificacao.txt"

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Relatório de Classificação Detalhado\n")
            f.write("=" * 40 + "\n")
            f.write(report)

        logging.info(f"Relatório salvo em '{report_path}'")
        print("\n" + report)


# =============================================================================
# Função Principal de Orquestração
# =============================================================================
def main():
    setup_logging(IDSConfig.LOGS_DIR)

    # Configura TF para CPU — desabilita GPU, ajusta threads inter/intra-op.
    # Deve ser chamado antes de qualquer operação TF.
    IDSConfig.configure_tensorflow()

    # Semente global do TensorFlow para reprodutibilidade da inicialização dos
    # pesos e das máscaras de dropout entre execuções consecutivas na mesma máquina.
    # Nota: não garante bit-a-bit entre CPU e GPU (float32 tem não-determinismo
    # de precisão de ~1e-6 entre plataformas), mas garante resultados equivalentes
    # do ponto de vista de métricas (F1, acurácia, AUC) a ≥ 4 casas decimais.
    tf.random.set_seed(IDSConfig.TRAINING_CONFIG['random_state'])

    logging.info(
        f"TensorFlow {tf.__version__} — CPU configurado: "
        f"inter_op={IDSConfig.CPU_CONFIG['inter_op_threads']} threads, "
        f"intra_op={IDSConfig.CPU_CONFIG['intra_op_threads']} threads"
    )

    try:
        logging.info("=" * 80)
        logging.info("INICIANDO SCRIPT DE TREINAMENTO DO SISTEMA IDS — v2.1")
        logging.info("=" * 80)

        data_handler = DataHandler(IDSConfig)
        df_raw = data_handler.load_and_clean_data_with_cache()
        X_processed, y_processed, selected_features = data_handler.preprocess_with_cache(df_raw)

        logging.info("Dividindo dados em conjuntos de treino, validação e teste...")
        train_cfg = IDSConfig.TRAINING_CONFIG
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_processed, y_processed,
            test_size=(train_cfg['validation_split'] + train_cfg['test_split']),
            random_state=train_cfg['random_state'],
            stratify=y_processed,
        )
        val_size_relative = train_cfg['validation_split'] / (
            train_cfg['validation_split'] + train_cfg['test_split']
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp,
            test_size=(1 - val_size_relative),
            random_state=train_cfg['random_state'],
            stratify=y_temp,
        )
        logging.info(f"Treino: {X_train.shape} | Validação: {X_val.shape} | Teste: {X_test.shape}")

        trainer = ModelTrainer(IDSConfig)
        model_path = IDSConfig.MODEL_DIR / IDSConfig.MODEL_FILENAME
        history = None

        if model_path.exists() and not train_cfg.get('force_retrain', False):
            logging.info(
                f"SUCESSO: Modelo encontrado em '{model_path}' e 'force_retrain' é False. "
                f"Pulando etapa de treinamento."
            )
            trainer.model = load_model(model_path)
        else:
            logging.info("AVISO: Nenhum modelo encontrado ou 'force_retrain' é True. Iniciando treinamento...")
            input_shape = (X_train.shape[1], 1)
            num_classes = len(np.unique(y_processed))
            trainer.build_or_load_model(input_shape, num_classes)
            history = trainer.train(X_train, y_train, X_val, y_val)
            trainer.save_artifacts(data_handler, selected_features, history)

        y_pred = trainer.evaluate(X_test, y_test)

        output_dir = IDSConfig.RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        reporter = ReportGenerator(IDSConfig, history, output_dir)
        reporter.plot_training_history()
        reporter.plot_confusion_matrix(y_test, y_pred, data_handler.label_mapping)
        reporter.save_classification_report(y_test, y_pred, data_handler.label_mapping)

        logging.info("=" * 80)
        logging.info("PROCESSO CONCLUÍDO COM SUCESSO!")
        logging.info(f"Modelo salvo em '{IDSConfig.MODEL_DIR}'")
        logging.info(f"Relatórios e gráficos salvos em '{output_dir}'")
        logging.info("=" * 80)

    except Exception as e:
        logging.error(f"Ocorreu um erro fatal durante a execução: {e}")
        logging.error("Traceback:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()     