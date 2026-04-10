#!/usr/bin/env python3
"""
################################################################################
#  SecurityIA — Arquivo de Configuração Centralizado (SSOT)
#  Versão: 3.0
#
#  ÚNICA fonte de verdade do sistema. Todos os scripts (IDS/, Tests/) importam
#  daqui. Edite as seções marcadas com ← para o seu ambiente.
#
#  Estrutura de diretórios:
#    SecurityIA/
#    ├── config.py          ← este arquivo
#    ├── install.sh
#    ├── Base/              ← dataset CSE-CIC-IDS2018 (.csv ou .parquet)
#    ├── Model/             ← artefatos do modelo treinado
#    ├── Logs/              ← App.log | Collector.log | Learn.log
#    ├── Temp/              ← cache e staging de re-treinamento
#    ├── Reports/           ← relatórios HTML do IDS
#    ├── Tests/             ← scripts de validação metodológica (não modificar)
#    └── IDS/               ← sistema de coleta, detecção e aprendizado
#
#  Autor: Bruno Cavalcante Barbosa — PPGI/IC/UFAL
#  Orient.: Prof. Dr. André Luiz Lins de Aquino
################################################################################
"""

import os
from pathlib import Path


# ──────────────────────────────────────────────────────────────────────────────
# Configuração antecipada de ambiente (deve preceder qualquer import TF)
# ──────────────────────────────────────────────────────────────────────────────
os.environ.setdefault("CUDA_VISIBLE_DEVICES",  "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL",  "3")
os.environ.setdefault("TF_DETERMINISTIC_OPS",  "1")
os.environ.setdefault("PYTHONHASHSEED",         "42")


class Config:
    """
    Namespace centralizado. Todos os atributos são variáveis de classe.
    Acesse via Config.ATTR — sem instanciação.
    """

    # =========================================================================
    # § 1 — DIRETÓRIOS RAIZ
    # =========================================================================
    BASE_DIR    = Path(__file__).resolve().parent        # raiz do projeto
    DATA_DIR    = BASE_DIR / "Base" / "CSE-CIC-IDS2018"  # ← dataset
    MODEL_DIR   = BASE_DIR / "Model"
    LOGS_DIR    = BASE_DIR / "Logs"
    TEMP_DIR    = BASE_DIR / "Temp"
    REPORTS_DIR = BASE_DIR / "Reports"                   # relatórios HTML do IDS
    TESTS_DIR   = BASE_DIR / "Tests"

    # =========================================================================
    # § 2 — ARQUIVOS DE LOG (3 logs independentes)
    # =========================================================================
    LOG_APP       = LOGS_DIR / "App.log"         # geral: erros, warnings, info
    LOG_COLLECTOR = LOGS_DIR / "Collector.log"   # coleta: início, erros, fim
    LOG_LEARN     = LOGS_DIR / "Learn.log"       # treinamento: épocas, métricas

    LOG_CONFIG = {
        "level":   "INFO",
        "format":  "%(asctime)s  [%(levelname)-8s]  %(name)-18s  %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
        "max_bytes":   50 * 1024 * 1024,  # 50 MiB por arquivo
        "backup_count": 5,
    }

    # =========================================================================
    # § 3 — PIPELINE DE COLETA (ids_collector.py)
    # =========================================================================
    CAPTURE_INTERFACE    = "eth1"               # ← porta mirror do roteador
    COLLECTOR_DIR        = TEMP_DIR / "capture" # saída dos parquets diários
    COLLECTOR_SAMPLE_RATE = 1.0                 # 1.0 = 100 % dos fluxos
    COLLECTOR_BUDGET_GB  = 7.0                  # budget máximo de armazenamento/dia
    COLLECTOR_ACTIVE_TIMEOUT  = 120             # s — expirar fluxo ativo
    COLLECTOR_IDLE_TIMEOUT    = 30              # s — expirar fluxo ocioso
    COLLECTOR_MAX_PKT_FLOW    = 10_000          # máx. pacotes por fluxo
    COLLECTOR_PKT_QUEUE_SIZE  = 200_000
    COLLECTOR_FLOW_QUEUE_SIZE = 50_000
    COLLECTOR_FLUSH_ROWS      = 50_000          # linhas por flush em disco
    COLLECTOR_FLUSH_SECS      = 60             # flush periódico (segundos)

    # =========================================================================
    # § 4 — ARTEFATOS DO MODELO
    # =========================================================================
    MODEL_FILENAME             = "ids_lstm_model.keras"
    SCALER_FILENAME            = "scaler.pkl"
    LABEL_ENCODER_FILENAME     = "label_encoder.pkl"
    MODEL_INFO_FILENAME        = "ids_model_info.json"
    SELECTED_FEATURES_FILENAME = "ids_selected_features.json"

    # =========================================================================
    # § 5 — FEATURES (23 — compatíveis com CSE-CIC-IDS2018)
    # Deve ser idêntico ao schema do collector e ao treinamento.
    # =========================================================================
    FEATURE_COLUMNS = [
        "Flow_Duration",         "Total_Fwd_Packets",    "Flow_Bytes_s",
        "Flow_Packets_s",        "Fwd_Packet_Length_Mean","Bwd_Packet_Length_Mean",
        "Flow_IAT_Mean",         "Fwd_IAT_Total",        "Bwd_IAT_Total",
        "Packet_Length_Variance","Flow_IAT_Std",          "Active_Mean",
        "Active_Std",            "Idle_Mean",             "Idle_Std",
        "TCP_Flag_Count",        "Protocol_Type",         "Service_Type",
        "Flag_Type",             "Source_Bytes",          "Destination_Bytes",
        "Count",                 "Same_Service_Rate",
    ]

    META_COLUMNS = [
        "meta_src_ip", "meta_dst_ip",
        "meta_src_port", "meta_dst_port",
        "meta_flow_start", "meta_flow_end",
    ]

    # =========================================================================
    # § 6 — PRÉ-PROCESSAMENTO
    # =========================================================================
    PREPROCESSING_CONFIG = {
        "missing_value_threshold": 0.5,   # remove col se > 50 % NaN
        "variance_threshold":      1e-5,
        "apply_variance_filter":   True,
        "force_reload":            False,  # True = ignora cache de CSV
        "force_preprocess":        False,  # True = ignora cache de arrays
        "sample_size_for_selection": 200_000,  # amostra para IG+MI
    }

    # =========================================================================
    # § 7 — SELEÇÃO DE CARACTERÍSTICAS (Teoria da Informação)
    # score = ig_weight * IG_norm + mi_weight * MI_norm
    # =========================================================================
    FEATURE_SELECTION_CONFIG = {
        "k_best":                 23,
        "ig_weight":              0.6,
        "mi_weight":              0.4,
        "normalization_epsilon":  1e-9,
        "ig_discretization_bins": 10,
    }

    # =========================================================================
    # § 8 — ARQUITETURA DO MODELO (Bi-LSTM + Atenção de Bahdanau)
    # =========================================================================
    MODEL_CONFIG = {
        "lstm_units_1":          128,
        "lstm_units_2":          64,
        "dense_units":           32,
        "attention_units":       64,   # Bahdanau attention dim
        "dropout_rate":          0.5,
        "recurrent_dropout_rate": 0.0, # 0.0 = caminho CuDNN-compatível (CPU-safe)
        "learning_rate":         1e-3,
        "loss_function":         "sparse_categorical_crossentropy",
        "metrics":               ["accuracy"],
        "sequence_length":       1,    # reshape (n, features, 1) — fluxos tabulares
    }

    # =========================================================================
    # § 9 — TREINAMENTO
    # =========================================================================
    TRAINING_CONFIG = {
        "random_state":     42,
        "validation_split": 0.15,
        "test_split":       0.15,
        "epochs":           50,
        "batch_size":       1024,
        "patience":         10,        # early stopping
        "force_retrain":    False,
    }

    FINE_TUNING_CONFIG = {
        "enable":        True,
        "learning_rate": 1e-5,
        "epochs":        20,
        "patience":      5,
    }

    # =========================================================================
    # § 10 — BALANCEAMENTO (SMOTE → RUS → ENN)
    # =========================================================================
    BALANCING_CONFIG = {
        "smote_k_neighbors":  5,
        "enn_n_neighbors":    3,
        "n_samples_minority": 50_000,
        "n_samples_majority": 150_000,
    }

    # =========================================================================
    # § 11 — DETECÇÃO (limiares de confiança)
    # =========================================================================
    CONF_MEDIUM = 0.60   # limiar mínimo para registrar incidente
    CONF_HIGH   = 0.85   # limiar para incluir no dataset de re-treinamento
    BENIGN_LABEL = "Benign"

    # Mapeamento de referência do dataset CIC-IDS2018
    REFERENCE_ATTACK_TYPES = {
        0:  "Benign",
        1:  "Bot",
        2:  "DDoS",
        3:  "DoS GoldenEye",
        4:  "DoS Hulk",
        5:  "DoS Slowhttptest",
        6:  "DoS slowloris",
        7:  "FTP-Patator",
        8:  "Heartbleed",
        9:  "Infiltration",
        10: "PortScan",
        11: "SSH-Patator",
        12: "Web Attack \u2013 Brute Force",
        13: "Web Attack \u2013 Sql Injection",
        14: "Web Attack \u2013 XSS",
    }

    # =========================================================================
    # § 12 — RE-TREINAMENTO AUTOMÁTICO
    # =========================================================================
    RETRAIN_CONFIG = {
        "staging_dir":   TEMP_DIR / "staging",
        "state_file":    TEMP_DIR / ".retrain_state.json",
        "min_days":      3,        # mínimo de dias acumulados antes de re-treinar
        "min_flows":     50_000,   # mínimo de fluxos anotados
        "method":        "direct", # "direct" | "flag"
        "flag_file":     TEMP_DIR / "RETRAIN_READY.flag",
    }

    # =========================================================================
    # § 13 — AVALIAÇÃO E BENCHMARK
    # =========================================================================
    EVALUATION_CONFIG = {
        "eval_dir":           TEMP_DIR / "evaluation",
        "benchmark_fraction": 0.10,  # 10 % do dataset para benchmark congelado
        "history_file":       TEMP_DIR / "evaluation" / "eval_history.json",
        "significance_alpha": 0.001,
        "batch_size":         4096,
    }

    # =========================================================================
    # § 14 — HARDWARE / TensorFlow (CPU-only)
    # Servidor alvo: 20 vCPUs, 64 GB RAM
    # =========================================================================
    CPU_CONFIG = {
        "inter_op_threads": 4,   # paralelismo entre ops TF independentes
        "intra_op_threads": 16,  # paralelismo interno (BLAS/MKL/oneDNN)
    }

    # =========================================================================
    # § 15 — VISUALIZAÇÃO
    # =========================================================================
    VIZ_CONFIG = {
        "style":       "ggplot",
        "save_format": "png",
        "dpi":         300,
    }

    # =========================================================================
    # § 16 — RELATÓRIOS (IDS)
    # =========================================================================
    REPORT_COUNTER_FILE = REPORTS_DIR / ".report_counter"
    MANAGER_STATE_FILE  = TEMP_DIR / ".manager_state.json"

    # =========================================================================
    # Métodos utilitários de classe
    # =========================================================================

    @classmethod
    def configure_tensorflow(cls) -> None:
        """
        Configura TF para CPU-only antes de qualquer operação TF.
        Deve ser chamado no topo de qualquer script que importe TF.
        Não altera pesos, loss ou métricas (diferença < 1e-6 float32).
        """
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_inter_op_parallelism_threads(
            cls.CPU_CONFIG["inter_op_threads"]
        )
        tf.config.threading.set_intra_op_parallelism_threads(
            cls.CPU_CONFIG["intra_op_threads"]
        )
        tf.get_logger().setLevel("ERROR")

    @classmethod
    def ensure_dirs(cls) -> None:
        """Cria todos os diretórios necessários (idempotente)."""
        dirs = [
            cls.DATA_DIR, cls.MODEL_DIR, cls.LOGS_DIR, cls.TEMP_DIR,
            cls.REPORTS_DIR, cls.TESTS_DIR, cls.COLLECTOR_DIR,
            cls.RETRAIN_CONFIG["staging_dir"],
            cls.RETRAIN_CONFIG["staging_dir"].parent,
            cls.EVALUATION_CONFIG["eval_dir"],
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def next_report_number(cls) -> int:
        """Contador global de relatórios — thread-safe via lock de arquivo."""
        import time
        cls.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        lock = cls.REPORT_COUNTER_FILE.with_suffix(".lock")
        for _ in range(30):
            try:
                lock.touch(exist_ok=False)
                break
            except FileExistsError:
                time.sleep(0.1)
        try:
            n = int(cls.REPORT_COUNTER_FILE.read_text().strip()) + 1 \
                if cls.REPORT_COUNTER_FILE.exists() else 1
            cls.REPORT_COUNTER_FILE.write_text(str(n))
        finally:
            lock.unlink(missing_ok=True)
        return n

    @classmethod
    def report_path(cls, label: str = "", ext: str = "html") -> Path:
        """
        Gera caminho padronizado para relatório:
        Reports/relatorio_NNN_LABEL_YYYYMMDD.html
        """
        from datetime import date
        n   = cls.next_report_number()
        lbl = label.replace(" ", "_") if label else "ids"
        dt  = date.today().strftime("%Y%m%d")
        return cls.REPORTS_DIR / f"relatorio_{n:03d}_{lbl}_{dt}.{ext}"

    @classmethod
    def summary(cls) -> str:
        """Resumo formatado das configurações principais."""
        sep = "═" * 62
        return "\n".join([
            sep,
            "  SecurityIA — Configuração Ativa",
            sep,
            f"  Interface de captura : {cls.CAPTURE_INTERFACE}",
            f"  Diretório de captura : {cls.COLLECTOR_DIR}",
            f"  Amostragem           : {cls.COLLECTOR_SAMPLE_RATE * 100:.0f} %",
            f"  Budget diário        : {cls.COLLECTOR_BUDGET_GB:.1f} GiB",
            f"  Dataset              : {cls.DATA_DIR}",
            f"  Modelo               : {cls.MODEL_DIR}",
            f"  Relatórios           : {cls.REPORTS_DIR}",
            f"  Logs                 : {cls.LOGS_DIR}",
            f"  Threads TF           : inter={cls.CPU_CONFIG['inter_op_threads']} "
            f"intra={cls.CPU_CONFIG['intra_op_threads']}",
            f"  k-best features      : {cls.FEATURE_SELECTION_CONFIG['k_best']}",
            f"  LSTM units           : {cls.MODEL_CONFIG['lstm_units_1']} / "
            f"{cls.MODEL_CONFIG['lstm_units_2']}",
            f"  Dropout              : {cls.MODEL_CONFIG['dropout_rate']}",
            f"  Atenção (Bahdanau)   : {cls.MODEL_CONFIG['attention_units']} unidades",
            sep,
        ])


# Alias para compatibilidade com scripts antigos que importam IDSConfig
IDSConfig = Config
