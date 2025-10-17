#!/usr/bin/env python3
"""
##############################################################################################
#  Arquivo de Configuração Centralizado para o Sistema IDS
#
#  Define todos os parâmetros, constantes e configurações do sistema, incluindo
#  arquitetura do modelo, caminhos de arquivo, parâmetros de treinamento e de
#  pós-processamento.
#
#  Autor: Bruno Cavalcante Barbosa
#  UFAL - Universidade Federal de Alagoas
##############################################################################################
"""

from pathlib import Path

class IDSConfig:
    """
    Classe de configuração que serve como um namespace para todos os
    parâmetros ajustáveis do projeto.
    """
    
    # --- Configurações de Diretórios e Arquivos ---
    # Usamos pathlib para garantir compatibilidade entre sistemas operacionais
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "Base/CSE-CIC-IDS2018/"
    MODEL_DIR = BASE_DIR / "Model/"
    RESULTS_DIR = BASE_DIR / "Results/"
    REPORTS_DIR = BASE_DIR / "Reports/"
    LOGS_DIR = BASE_DIR / "Logs/"
    CACHE_DIR = BASE_DIR / "Cache/"
    
    # Nomes de arquivos padronizados
    MODEL_FILENAME = "ids_lstm_model.keras"
    SCALER_FILENAME = "scaler.pkl"
    LABEL_ENCODER_FILENAME = "label_encoder.pkl"
    MODEL_INFO_FILENAME = "model_info.json"
    SELECTED_FEATURES_FILENAME = 'ids_selected_features.json'
    MODEL_INFO_FILENAME = 'ids_model_info.json'

    # --- Configurações de Pré-Processamento ---
    PREPROCESSING_CONFIG = {
        'sample_fraction': 1.0, # Use < 1.0 para testes rápidos com uma fração dos dados
        'missing_value_threshold': 0.5, # Remove colunas com mais de 50% de valores faltantes
        'variance_threshold': 0.0, # Remove features com variância zero
        'force_reload': False,  # Mude para True para ignorar o cache do dataframe
        'force_preprocess': False # Mude para True para ignorar o cache dos dados processados
    }

    # --- Configurações de Treinamento e Modelo ---
    MODEL_CONFIG = {
        'lstm_units_1': 128,
        'lstm_units_2': 64,
        'dense_units': 32,
        'dropout_rate': 0.5,
        'recurrent_dropout_rate': 0.2, # Lembre-se que pode impactar a performance da GPU
        'learning_rate': 1e-3,
        'loss_function': 'sparse_categorical_crossentropy',
        'metrics': ['accuracy']
    }
    
    TRAINING_CONFIG = {
        'random_state': 42,
        'validation_split': 0.15,
        'test_split': 0.15,
        'epochs': 50,
        'batch_size': 1024,
        'patience': 10,
        'force_retrain': False # Mude para True para forçar o retreinamento mesmo que um modelo salvo exista
    }
    
    # Configuração para Fine-Tuning (Treinamento Incremental / "Reforço")
    FINE_TUNING_CONFIG = {
        'enable': True,  # Habilita ou desabilita a funcionalidade
        'learning_rate': 1e-5  # Taxa de aprendizado bem menor para ajuste fino
    }
    
    FEATURE_SELECTION_CONFIG = {
        'k_best': 23, # Número de características a serem selecionadas
        'ig_weight': 0.6, # Peso do Information Gain no score combinado
        'mi_weight': 0.4  # Peso da Mutual Information no score combinado
    }
    
    BALANCING_CONFIG = {
        'enn_n_neighbors': 3,
        'n_samples_minority': 50000,
        'n_samples_majority': 150000
    }
    
    # --- Configurações de Logging e Visualização ---
    LOGGING_CONFIG = {
        'level': 'INFO',
        'format': '%(asctime)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }

    VISUALIZATION_CONFIG = {
        'style': 'ggplot',
        'save_format': 'png',
        'dpi': 300
    }

    # --- Definições do Dataset (Exemplo: CSE-CIC-IDS2018) ---
    # O mapeamento de labels é carregado dinamicamente, mas podemos ter um de referência
    REFERENCE_ATTACK_TYPES = {
        0: 'Benign',
        1: 'Bot',
        2: 'DDoS',
        3: 'DoS GoldenEye',
        4: 'DoS Hulk',
        5: 'DoS Slowhttptest',
        6: 'DoS slowloris',
        7: 'FTP-Patator',
        8: 'Heartbleed',
        9: 'Infiltration',
        10: 'PortScan',
        11: 'SSH-Patator',
        12: 'Web Attack – Brute Force',
        13: 'Web Attack – Sql Injection',
        14: 'Web Attack – XSS'
    }