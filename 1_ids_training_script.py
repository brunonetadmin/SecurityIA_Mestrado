#!/usr/bin/env python3
"""
##############################################################################################
#  Script de Treinamento Principal do Sistema IDS
#  Versão Refatorada e Aprimorada
#
#  Este script orquestra o pipeline completo de Machine Learning:
#  1.  Carregamento e validação de dados do CSE-CIC-IDS2018 (com cache).
#  2.  Pré-processamento, incluindo seleção otimizada de características e balanceamento
#      dinâmico (com cache).
#  3.  Construção, carregamento (fine-tuning) ou treinamento de um modelo LSTM Híbrido,
#      com a opção de pular o retreinamento.
#  4.  Avaliação rigorosa da performance.
#  5.  Geração de relatórios, gráficos e salvamento de todos os artefatos.
#
#  Autor: Bruno Cavalcante Barbosa
#  UFAL - Universidade Federal de Alagoas
##############################################################################################
"""

# Imports padrão e de sistema
import os
import sys
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

# Importa a configuração centralizada
from ids_config import IDSConfig

# --- Configuração do Logging ---
def setup_logging(log_dir):    
    log_dir.mkdir(parents=True, exist_ok=True)
    log_filename = log_dir / f"Treinamento_Registros_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=getattr(logging, IDSConfig.LOGGING_CONFIG['level']),
        format=IDSConfig.LOGGING_CONFIG['format'],
        datefmt=IDSConfig.LOGGING_CONFIG['datefmt'],
        handlers=[
            logging.FileHandler(log_filename, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

# --- Classe para Manipulação de Dados ---
class DataHandler:
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.selected_features = []
        self.label_mapping = {}
        
        # Garante que o diretório de cache exista
        self.config.CACHE_DIR.mkdir(parents=True, exist_ok=True)

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
        logging.info("Iniciando a limpeza dos dados...")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(axis=1, thresh=int(self.config.PREPROCESSING_CONFIG['missing_value_threshold'] * len(df)), inplace=True)
        df.fillna(0, inplace=True)
        
        for col in df.columns:
            if df[col].dtype == 'object' and col.lower() != 'label':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        df.fillna(0, inplace=True)
        
        logging.info(f"Dados limpos. Shape: {df.shape}")
        return df

    # ---  ---
    def load_and_clean_data_with_cache(self):
        """Unifica o carregamento e a limpeza com um sistema de cache em Parquet."""
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

    def _select_features(self, X, y, feature_names):
        logging.info("Calculando Mutual Information para seleção de características...")
        mi_scores = mutual_info_classif(X, y, random_state=self.config.TRAINING_CONFIG['random_state'])
        
        combined_scores = {name: score for name, score in zip(feature_names, mi_scores)}
        
        k = self.config.FEATURE_SELECTION_CONFIG['k_best']
        selected = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        
        self.selected_features = [f[0] for f in selected]
        logging.info(f"Top {k} características selecionadas: {self.selected_features}")
        return self.selected_features

    def preprocess_with_cache(self, df):
        """Orquestra todo o pré-processamento com múltiplos pontos de cache."""
        
        # --- Cache para os arrays finais (X e y balanceados) ---
        cached_X_path = self.config.CACHE_DIR / "03_PREPROCESS_X_balanced.pkl"
        cached_y_path = self.config.CACHE_DIR / "03_PREPROCESS_y_balanced.pkl"
        
        if cached_X_path.exists() and cached_y_path.exists() and not self.config.PREPROCESSING_CONFIG.get('force_preprocess', False):
            logging.info("SUCESSO: Carregando os arrays de treino (X, y) já balanceados do cache.")
            X_balanced = joblib.load(cached_X_path)
            y_balanced = joblib.load(cached_y_path)
            
            # Carrega os processadores salvos que correspondem a estes dados
            self.scaler = joblib.load(self.config.MODEL_DIR / self.config.SCALER_FILENAME)
            self.label_encoder = joblib.load(self.config.MODEL_DIR / self.config.LABEL_ENCODER_FILENAME)
            
            with open(self.config.MODEL_DIR / self.config.MODEL_INFO_FILENAME, 'r') as f:
                model_info = json.load(f)
                selected_features = model_info['selected_features']
            
            self.label_mapping = {int(k): v for k, v in model_info['label_mapping'].items()}
            return X_balanced, y_balanced, selected_features

        logging.info("AVISO: Cache de dados processados não encontrado ou o reprocessamento foi forçado.")
        
        label_col = 'Label'
        
        # --- Otimização da Seleção de Features com Amostragem ---
        logging.info("Otimização: Usando uma amostra do dataset para seleção de features.")
        sample_size = min(200000, len(df))
        df_sample = df.sample(n=sample_size, random_state=self.config.TRAINING_CONFIG['random_state'])
        
        X_sample = df_sample.drop(columns=[label_col])
        y_sample = df_sample[label_col]
        y_sample_encoded = self.label_encoder.fit_transform(y_sample)
        
        selected_features = self._select_features(X_sample.values, y_sample_encoded, X_sample.columns.tolist())
        
        # Aplica as features selecionadas ao DataFrame completo
        X = df[selected_features]
        y = df[label_col]
        y_encoded = self.label_encoder.transform(y) # Usa 'transform' pois já foi 'fit' na amostra
        self.label_mapping = {i: label for i, label in enumerate(self.label_encoder.classes_)}

        # Normalização
        logging.info("Aplicando normalização StandardScaler...")
        X_scaled = self.scaler.fit_transform(X)

        # --- Balanceamento dinâmico e robusto ---
        logging.info("Aplicando balanceamento de classes com estratégia híbrida dinâmica...")
        
        class_distribution = Counter(y_encoded)
        logging.info(f"Distribuição original das classes: {class_distribution}")

        n_samples_minority_target = self.config.BALANCING_CONFIG.get('n_samples_minority', 50000)
        benign_class_label_encoded = self.label_encoder.transform(['Benign'])[0]

        over_strategy = {}
        for class_label, num_samples in class_distribution.items():
            if class_label != benign_class_label_encoded:
                over_strategy[class_label] = max(num_samples, n_samples_minority_target)

        n_samples_majority = self.config.BALANCING_CONFIG.get('n_samples_majority', 150000)
        under_strategy = {benign_class_label_encoded: n_samples_majority}

        logging.info(f"Estratégia SMOTE (Oversampling): {over_strategy}")
        logging.info(f"Estratégia RandomUnderSampler: {under_strategy}")

        smote = SMOTE(sampling_strategy=over_strategy, random_state=self.config.TRAINING_CONFIG['random_state'])
        rus = RandomUnderSampler(sampling_strategy=under_strategy, random_state=self.config.TRAINING_CONFIG['random_state'])
        enn = EditedNearestNeighbours(n_neighbors=self.config.BALANCING_CONFIG['enn_n_neighbors'])

        pipeline = Pipeline(steps=[('smote', smote), ('rus', rus)])
        X_resampled, y_resampled = pipeline.fit_resample(X_scaled, y_encoded)
        
        logging.info("Aplicando ENN para limpeza de ruído...")
        X_balanced, y_balanced = enn.fit_resample(X_resampled, y_resampled)
        
        logging.info(f"Dataset final para treinamento. Shape: {X_balanced.shape}")
        
        # --- Salvando os resultados em cache ---
        logging.info("Salvando arrays de treino (X, y) balanceados e processadores em cache.")
        joblib.dump(X_balanced, cached_X_path)
        joblib.dump(y_balanced, cached_y_path)
        # Salva os processadores e informações que correspondem a estes dados
        joblib.dump(self.scaler, self.config.MODEL_DIR / self.config.SCALER_FILENAME)
        joblib.dump(self.label_encoder, self.config.MODEL_DIR / self.config.LABEL_ENCODER_FILENAME)
        model_info = {'selected_features': selected_features, 'label_mapping': self.label_mapping}
        with open(self.config.MODEL_DIR / self.config.MODEL_INFO_FILENAME, 'w') as f:
            json.dump(model_info, f, indent=4)

        return X_balanced, y_balanced, selected_features

# --- Classe para Treinamento do Modelo ---
class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.model = None

    def _build_new_model(self, input_shape, num_classes):
        logging.info("Construindo uma nova arquitetura LSTM Híbrida...")
        model_cfg = self.config.MODEL_CONFIG
        model = Sequential([
            Bidirectional(LSTM(model_cfg['lstm_units_1'], return_sequences=True), input_shape=input_shape),
            Dropout(model_cfg['dropout_rate']),
            Bidirectional(LSTM(model_cfg['lstm_units_2'], return_sequences=False)),
            Dropout(model_cfg['dropout_rate']),
            Dense(model_cfg['dense_units'], activation='relu'),
            Dropout(model_cfg['dropout_rate']),
            Dense(num_classes, activation='softmax')
        ])
        model.compile(optimizer=Adam(learning_rate=model_cfg['learning_rate']), loss=model_cfg['loss_function'], metrics=model_cfg['metrics'])
        return model

    def build_or_load_model(self, input_shape, num_classes):
        model_path = self.config.MODEL_DIR / self.config.MODEL_FILENAME
        if model_path.exists() and self.config.FINE_TUNING_CONFIG['enable']:
            logging.info(f"Modelo encontrado em '{model_path}'. Carregando para fine-tuning ...")
            self.model = load_model(model_path)
            new_lr = self.config.FINE_TUNING_CONFIG['learning_rate']
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
            logging.info(f"Taxa de aprendizado ajustada para (fine-tuning): {new_lr}")
        else:
            logging.info("Nenhum modelo encontrado ou o fine-tuning está desabilitado. Construindo um modelo do zero ...")
            self.model = self._build_new_model(input_shape, num_classes)
        
        self.model.summary(print_fn=logging.info)
        
        plot_path = self.config.RESULTS_DIR / "Treinamento_Arquitetura_do_Modelo.png"
        plot_model(self.model, to_file=plot_path, show_shapes=True, show_layer_names=True)
        logging.info(f"Arquitetura do modelo salva em '{plot_path}'")

    def train(self, X_train, y_train, X_val, y_val):
        logging.info("Iniciando treinamento do modelo...")
        train_cfg = self.config.TRAINING_CONFIG
        
        X_train_reshaped = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_val_reshaped = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=train_cfg['patience'], restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
        ]
        
        start_time = time.time()
        history = self.model.fit(
            X_train_reshaped, y_train,
            validation_data=(X_val_reshaped, y_val),
            epochs=train_cfg['epochs'],
            batch_size=train_cfg['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        logging.info(f"Treinamento concluído em {time.time() - start_time:.2f} segundos.")
        return history

    def evaluate(self, X_test, y_test):
        logging.info("Iniciando avaliação do modelo no conjunto de teste...")
        X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
        y_pred_proba = self.model.predict(X_test_reshaped)
        y_pred = np.argmax(y_pred_proba, axis=1)
        return y_pred

    def save_artifacts(self, data_handler, selected_features, history):
        self.config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model_path = self.config.MODEL_DIR / self.config.MODEL_FILENAME
        logging.info(f"Salvando modelo treinado em '{model_path}'")
        self.model.save(model_path)
        
        # Salva processadores e informações do modelo (movido para preprocess)
        logging.info("Processadores (scaler, encoder) e informações do modelo já salvos na etapa de cache.")

        # Salva o histórico de treinamento para análises futuras
        if history:
            history_path = self.config.RESULTS_DIR / f"Treinamento_Historico_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            # --- Converte os tipos do histórico para serem compatíveis com JSON ---
            history_data_serializable = {
                key: [float(value) for value in values] 
                for key, values in history.history.items()
            }

            with open(history_path, 'w', encoding='utf-8') as f:
                # Salva o dicionário convertido
                json.dump(history_data_serializable, f, indent=4)
                
            logging.info(f"Histórico de treinamento salvo em '{history_path}'")

# --- Classe para Geração de Relatórios ---
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
        
        ax1.plot(self.history['loss'], label='Training Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Evolução da Perda (Loss)')
        ax1.set_xlabel('Época'); ax1.set_ylabel('Loss'); ax1.legend()
        
        ax2.plot(self.history['accuracy'], label='Training Accuracy')
        ax2.plot(self.history['val_accuracy'], label='Validation Accuracy')
        ax2.set_title('Evolução da Acurácia')
        ax2.set_xlabel('Época'); ax2.set_ylabel('Acurácia'); ax2.legend()
        
        fig.tight_layout()
        plt.savefig(self.output_dir / f"Treinamento_Historico.{self.config.VISUALIZATION_CONFIG['save_format']}", dpi=self.config.VISUALIZATION_CONFIG['dpi'])
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred, label_mapping):
        logging.info("Gerando Matriz de Confusão...")
        cm = confusion_matrix(y_true, y_pred)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        labels = [label_mapping.get(i, str(i)) for i in sorted(label_mapping.keys())]

        plt.figure(figsize=(12, 10))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title('Matriz de Confusão Normalizada', fontsize=16)
        plt.ylabel('Classe Verdadeira', fontsize=12)
        plt.xlabel('Classe Predita', fontsize=12)
        plt.xticks(rotation=45, ha='right'); plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(self.output_dir / f"Treinamento_Matriz_de_Confusao.{self.config.VISUALIZATION_CONFIG['save_format']}", dpi=self.config.VISUALIZATION_CONFIG['dpi'])
        plt.close()

    def save_classification_report(self, y_true, y_pred, label_mapping):
     logging.info("Gerando e salvando relatório de classificação...")
     labels = [label_mapping.get(i, str(i)) for i in sorted(label_mapping.keys())]
     report = classification_report(y_true, y_pred, target_names=labels)
     
     # --- Cria o subdiretório e define o novo caminho ---
     # Usa a variável do arquivo de configuração para o nome do diretório
     reports_dir = self.output_dir / self.config.REPORTS_DIR_NAME
     reports_dir.mkdir(parents=True, exist_ok=True)
     report_path = reports_dir / "Treinamento_Relatorio_de_Classificacao.txt"
     # --- FIM DA ALTERAÇÃO ---

     with open(report_path, 'w', encoding='utf-8') as f:
         f.write("Relatório de Classificação Detalhado\n")
         f.write("="*40 + "\n")
         f.write(report)
     logging.info(f"Relatório salvo em '{report_path}'")
     print("\n" + report)


# --- Função Principal de Orquestração ---
def main():
    setup_logging(IDSConfig.LOGS_DIR)

    logging.info("Verificando disponibilidade de aceleração de GPU (CUDA)...")
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logging.info(f"==> SUCESSO: {len(gpus)} GPU(s) encontradas. Aceleração de hardware está HABILITADA.")
        except RuntimeError as e:
            logging.error(f"Erro ao configurar a GPU: {e}")
    else:
        logging.warning("==> AVISO: Nenhuma GPU compatível foi encontrada. O treinamento será executado na CPU.")
    
    try:
        logging.info("="*80)
        logging.info("INICIANDO SCRIPT DE TREINAMENTO DO SISTEMA IDS")
        logging.info("="*80)

        # Manipulação de Dados com Cache
        data_handler = DataHandler(IDSConfig)
        df_raw = data_handler.load_and_clean_data_with_cache()
        X_processed, y_processed, selected_features = data_handler.preprocess_with_cache(df_raw)

        # Divisão dos Dados
        logging.info("Dividindo dados em conjuntos de treino, validação e teste...")
        train_cfg = IDSConfig.TRAINING_CONFIG
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_processed, y_processed, 
            test_size=(train_cfg['validation_split'] + train_cfg['test_split']), 
            random_state=train_cfg['random_state'], stratify=y_processed
        )
        val_size_relative = train_cfg['validation_split'] / (train_cfg['validation_split'] + train_cfg['test_split'])
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=(1-val_size_relative), 
            random_state=train_cfg['random_state'], stratify=y_temp
        )
        logging.info(f"Treino: {X_train.shape}, Validação: {X_val.shape}, Teste: {X_test.shape}")

        # Treinamento do Modelo (com opção de pular)
        trainer = ModelTrainer(IDSConfig)
        model_path = IDSConfig.MODEL_DIR / IDSConfig.MODEL_FILENAME
        history = None
        
        # --- Lógica para pular o treinamento ---
        if model_path.exists() and not train_cfg.get('force_retrain', False):
            logging.info(f"SUCESSO: Foi encontrado um Modelo já treinado em '{model_path}' e a opção 'force_retrain' é False. Pulando a etapa de treinamento.")
            trainer.model = load_model(model_path)
        else:
            logging.info("AVISO: Nenhum Modelo encontrado ou 'force_retrain' é True. Iniciando processo de treinamento ...")
            input_shape = (X_train.shape[1], 1)
            num_classes = len(np.unique(y_processed))
            trainer.build_or_load_model(input_shape, num_classes)
            history = trainer.train(X_train, y_train, X_val, y_val)
            # Salvamento dos Artefatos
            trainer.save_artifacts(data_handler, selected_features, history)
        
        # Avaliação
        y_pred = trainer.evaluate(X_test, y_test)
        
        # Geração de Relatórios
        output_dir = IDSConfig.RESULTS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        reporter = ReportGenerator(IDSConfig, history, output_dir)
        reporter.plot_training_history()
        reporter.plot_confusion_matrix(y_test, y_pred, data_handler.label_mapping)
        reporter.save_classification_report(y_test, y_pred, data_handler.label_mapping)

        logging.info("="*80)
        logging.info("PROCESSO CONCLUÍDO COM SUCESSO!")
        logging.info(f"Modelo salvo em '{IDSConfig.MODEL_DIR}'")
        logging.info(f"Relatórios e gráficos salvos em '{output_dir}'")
        logging.info("="*80)

    except Exception as e:
        logging.error(f"Ocorreu um erro fatal durante a execução: {e}")
        logging.error("Traceback:", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()      