#!/usr/bin/env python3
"""
SecurityIA — Configuração central (atualizada)

Inclui:
  - Particionamento ANTES do balanceamento (correção D1: leakage)
  - Focal Loss reponderada (correção D2)
  - Balanceamento adaptativo Borderline-SMOTE-2 (correção D3)
  - Configuração para baseline RF
  - Métricas operacionais ampliadas (FPR, MCC, AUC-PR, alarmes/h)
  - Compatibilidade total com scripts legados (Tests/, app_menu.py)
"""

from __future__ import annotations

import os
import textwrap
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("PYTHONHASHSEED", "42")
warnings.filterwarnings("ignore")


class Config:
    # ------------------------------------------------------------------
    # Diretórios principais
    # ------------------------------------------------------------------
    BASE_DIR = Path(__file__).resolve().parent
    ROOT_DIR = BASE_DIR
    TESTS_DIR = BASE_DIR / "Tests"
    DATA_DIR = BASE_DIR / "Base" / "CSE-CIC-IDS2018"
    DATASET_DIR = DATA_DIR
    MODEL_DIR = BASE_DIR / "Model"
    IDS_DIR = BASE_DIR / "IDS"
    LOGS_DIR = BASE_DIR / "Logs"
    TEMP_DIR = BASE_DIR / "Temp"

    IDS_REPORTS_DIR = BASE_DIR / "Reports"
    REPORTS_DIR = IDS_REPORTS_DIR
    TEST_REPORTS_DIR = TESTS_DIR / "Reports"

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------
    LOG_APP = LOGS_DIR / "App.log"
    LOG_COLLECTOR = LOGS_DIR / "Collector.log"
    LOG_LEARN = LOGS_DIR / "Learn.log"

    LOG_CONFIG = {
        "level": "INFO",
        "format": "%(asctime)s  [%(levelname)-8s]  %(name)-18s  %(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
        "max_bytes": 50 * 1024 * 1024,
        "backup_count": 5,
    }

    # ------------------------------------------------------------------
    # IDS / operação
    # ------------------------------------------------------------------
    CAPTURE_INTERFACE = "eth1"
    COLLECTOR_DIR = TEMP_DIR / "capture"
    COLLECTOR_SAMPLE_RATE = 1.0
    COLLECTOR_BUDGET_GB = 7.0
    COLLECTOR_ACTIVE_TIMEOUT = 120
    COLLECTOR_IDLE_TIMEOUT = 30
    COLLECTOR_MAX_PKT_FLOW = 10_000
    COLLECTOR_PKT_QUEUE_SIZE = 200_000
    COLLECTOR_FLOW_QUEUE_SIZE = 50_000
    COLLECTOR_FLUSH_ROWS = 50_000
    COLLECTOR_FLUSH_SECS = 60

    MODEL_FILENAME = "ids_lstm_model.keras"
    SCALER_FILENAME = "scaler.pkl"
    LABEL_ENCODER_FILENAME = "label_encoder.pkl"
    MODEL_INFO_FILENAME = "ids_model_info.json"
    SELECTED_FEATURES_FILENAME = "ids_selected_features.json"

    FEATURE_COLUMNS = [
        "Flow_Duration", "Total_Fwd_Packets", "Flow_Bytes_s",
        "Flow_Packets_s", "Fwd_Packet_Length_Mean", "Bwd_Packet_Length_Mean",
        "Flow_IAT_Mean", "Fwd_IAT_Total", "Bwd_IAT_Total",
        "Packet_Length_Variance", "Flow_IAT_Std", "Active_Mean",
        "Active_Std", "Idle_Mean", "Idle_Std",
        "TCP_Flag_Count", "Protocol_Type", "Service_Type",
        "Flag_Type", "Source_Bytes", "Destination_Bytes",
        "Count", "Same_Service_Rate",
    ]

    META_COLUMNS = [
        "meta_src_ip", "meta_dst_ip",
        "meta_src_port", "meta_dst_port",
        "meta_flow_start", "meta_flow_end",
    ]

    PREPROCESSING_CONFIG = {
        "missing_value_threshold": 0.5,
        "variance_threshold": 1e-5,
        "apply_variance_filter": True,
        "force_reload": False,
        "force_preprocess": False,
        "sample_size_for_selection": 200_000,
    }

    FEATURE_SELECTION_CONFIG = {
        "k_best": 23,
        "ig_weight": 0.6,
        "mi_weight": 0.4,
        "normalization_epsilon": 1e-9,
        "ig_discretization_bins": 10,
    }

    # ------------------------------------------------------------------
    # MODEL_CONFIG — agora com Focal Loss reponderada
    # ------------------------------------------------------------------
    MODEL_CONFIG = {
        "lstm_units_1": 128,
        "lstm_units_2": 64,
        "dense_units": 32,
        "attention_units": 64,
        "dropout_rate": 0.5,
        "recurrent_dropout_rate": 0.0,
        "learning_rate": 1e-3,
        # Função de perda: 'focal_loss_cb' aciona reponderação por número
        # efetivo de amostras (Cui et al., 2019); 'sparse_ce' usa
        # sparse_categorical_crossentropy clássico.
        "loss_function": "focal_loss_cb",
        "focal_gamma": 2.0,
        "focal_class_balanced_beta": 0.9999,
        "metrics": ["accuracy"],
        "sequence_length": 100,
    }

    # ------------------------------------------------------------------
    # TRAINING_CONFIG — particionamento antes do balanceamento (D1)
    # ------------------------------------------------------------------
    TRAINING_CONFIG = {
        "random_state": 42,
        "validation_split": 0.15,
        "test_split": 0.15,
        "epochs": 50,
        "batch_size": 4096,
        "patience": 10,
        "force_retrain": False,
        "steps_per_execution": 8,
        # Correções de protocolo:
        "split_before_balancing": True,   # OBRIGATÓRIO — corrige leakage
        "use_class_weight": True,         # ativa class_weight no fit
        "balance_only_train": True,       # balanceia somente o treino
    }

    FINE_TUNING_CONFIG = {
        "enable": True,
        "learning_rate": 1e-5,
        "epochs": 20,
        "patience": 5,
    }

    # ------------------------------------------------------------------
    # BALANCING_CONFIG — Borderline-SMOTE-2 adaptativo (D3)
    # ------------------------------------------------------------------
    BALANCING_CONFIG = {
        # 'adaptive_borderline' | 'classic_smote_enn' | 'none'
        "strategy": "adaptive_borderline",
        # Borderline-SMOTE-2 (Han et al., 2005)
        "borderline_kind": "borderline-2",
        "smote_k_neighbors_max": 11,
        "smote_k_alpha": 0.25,
        # ENN
        "enn_n_neighbors": 3,
        "enn_kind_sel": "mode",
        # Proporção-alvo por classe minoritária
        "target_ratio_per_class": 5,
        "max_fraction_of_majority": 0.10,
        # Undersampling da majoritária
        "majority_undersample_factor": 1.5,
        # Compatibilidade legada (consumida pelo método balance() antigo)
        "smote_k_neighbors": 5,
        "n_samples_minority": 50_000,
        "n_samples_majority": 150_000,
    }

    CONF_MEDIUM = 0.60
    CONF_HIGH = 0.85
    BENIGN_LABEL = "Benign"

    REFERENCE_ATTACK_TYPES = {
        0: "Benign",
        1: "Bot",
        2: "DDoS",
        3: "DoS GoldenEye",
        4: "DoS Hulk",
        5: "DoS Slowhttptest",
        6: "DoS slowloris",
        7: "FTP-Patator",
        8: "Heartbleed",
        9: "Infiltration",
        10: "PortScan",
        11: "SSH-Patator",
        12: "Web Attack – Brute Force",
        13: "Web Attack – Sql Injection",
        14: "Web Attack – XSS",
    }

    RETRAIN_CONFIG = {
        "staging_dir": TEMP_DIR / "staging",
        "state_file": TEMP_DIR / ".retrain_state.json",
        "min_days": 3,
        "min_flows": 50_000,
        "method": "direct",
        "flag_file": TEMP_DIR / "RETRAIN_READY.flag",
    }

    # ------------------------------------------------------------------
    # EVALUATION_CONFIG — métricas operacionais ampliadas (L2)
    # ------------------------------------------------------------------
    EVALUATION_CONFIG = {
        "eval_dir": TEMP_DIR / "evaluation",
        "benchmark_fraction": 0.10,
        "history_file": TEMP_DIR / "evaluation" / "eval_history.json",
        "significance_alpha": 0.001,
        "batch_size": 4096,
        # Métricas estendidas
        "compute_fpr_per_class": True,
        "compute_mcc": True,
        "compute_auc_pr": True,
        "compute_alarm_rate": True,
        "lambda_benign_per_hour": 1_000_000,
        # Comparação contra baseline
        "compare_against_baseline": True,
    }

    # ------------------------------------------------------------------
    # BASELINE_CONFIG — Random Forest (M0)
    # ------------------------------------------------------------------
    BASELINE_CONFIG = {
        "model_filename": "baseline_rf.pkl",
        "n_estimators": 500,
        "max_depth": 20,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "class_weight": "balanced_subsample",
        "n_jobs": -1,
        "random_state": 42,
    }

    CPU_CONFIG = {
        "inter_op_threads": 2,
        "intra_op_threads": 20,
    }

    VIZ_CONFIG = {
        "style": "ggplot",
        "save_format": "png",
        "dpi": 300,
    }

    REPORT_COUNTER_FILE = IDS_REPORTS_DIR / ".report_counter"
    MANAGER_STATE_FILE = TEMP_DIR / ".manager_state.json"

    @classmethod
    def configure_tensorflow(cls) -> None:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        tf.config.threading.set_inter_op_parallelism_threads(cls.CPU_CONFIG["inter_op_threads"])
        tf.config.threading.set_intra_op_parallelism_threads(cls.CPU_CONFIG["intra_op_threads"])
        tf.get_logger().setLevel("ERROR")

    @classmethod
    def ensure_dirs(cls) -> None:
        dirs = [
            cls.DATA_DIR, cls.MODEL_DIR, cls.LOGS_DIR, cls.TEMP_DIR,
            cls.IDS_REPORTS_DIR, cls.TESTS_DIR, cls.TEST_REPORTS_DIR,
            cls.COLLECTOR_DIR,
            cls.RETRAIN_CONFIG["staging_dir"],
            cls.EVALUATION_CONFIG["eval_dir"],
            cls.MODEL_DIR / "registry",
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)

        for d in [
            cls.TEST_REPORTS_DIR / "Relatorio_1_Arquiteturas",
            cls.TEST_REPORTS_DIR / "Relatorio_2_Balanceamento",
            cls.TEST_REPORTS_DIR / "Relatorio_3_Teoria_Informacao",
            cls.TEST_REPORTS_DIR / "Relatorio_4_Otimizacao_Validacao",
        ]:
            (d / "figuras").mkdir(parents=True, exist_ok=True)
            (d / "tabelas").mkdir(parents=True, exist_ok=True)

    @classmethod
    def next_report_number(cls) -> int:
        import time
        cls.IDS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        lock = cls.REPORT_COUNTER_FILE.with_suffix(".lock")
        for _ in range(30):
            try:
                lock.touch(exist_ok=False)
                break
            except FileExistsError:
                time.sleep(0.1)
        try:
            n = int(cls.REPORT_COUNTER_FILE.read_text().strip()) + 1 if cls.REPORT_COUNTER_FILE.exists() else 1
            cls.REPORT_COUNTER_FILE.write_text(str(n))
        finally:
            lock.unlink(missing_ok=True)
        return n

    @classmethod
    def report_path(cls, label: str = "", ext: str = "html") -> Path:
        from datetime import date
        n = cls.next_report_number()
        lbl = label.replace(" ", "_") if label else "ids"
        dt = date.today().strftime("%Y%m%d")
        return cls.IDS_REPORTS_DIR / f"relatorio_{n:03d}_{lbl}_{dt}.{ext}"

    @classmethod
    def summary(cls) -> str:
        sep = "═" * 70
        return "\n".join([
            sep,
            "  SecurityIA — Configuração Ativa",
            sep,
            f"  Projeto              : {cls.BASE_DIR}",
            f"  Tests                : {cls.TESTS_DIR}",
            f"  Dataset              : {cls.DATA_DIR}",
            f"  Model                : {cls.MODEL_DIR}",
            f"  IDS Reports          : {cls.IDS_REPORTS_DIR}",
            f"  Tests Reports        : {cls.TEST_REPORTS_DIR}",
            f"  Logs                 : {cls.LOGS_DIR}",
            f"  Temp                 : {cls.TEMP_DIR}",
            f"  Interface de captura : {cls.CAPTURE_INTERFACE}",
            f"  Diretório de captura : {cls.COLLECTOR_DIR}",
            f"  Threads TF           : inter={cls.CPU_CONFIG['inter_op_threads']} intra={cls.CPU_CONFIG['intra_op_threads']}",
            f"  LSTM units           : {cls.MODEL_CONFIG['lstm_units_1']} / {cls.MODEL_CONFIG['lstm_units_2']}",
            f"  Dropout              : {cls.MODEL_CONFIG['dropout_rate']}",
            f"  Atenção              : {cls.MODEL_CONFIG['attention_units']}",
            f"  Função de perda      : {cls.MODEL_CONFIG['loss_function']}",
            f"  Estratégia balanc.   : {cls.BALANCING_CONFIG['strategy']}",
            f"  Split antes balanc.  : {cls.TRAINING_CONFIG['split_before_balancing']}",
            f"  class_weight no fit  : {cls.TRAINING_CONFIG['use_class_weight']}",
            sep,
        ])


IDSConfig = Config

# =============================================================================
# API LEGADA PARA OS TESTES ACADÊMICOS (Tests/analise_*.py)
# =============================================================================

ROOT_DIR = Config.ROOT_DIR
BASE_DIR = Config.BASE_DIR
TESTS_DIR = Config.TESTS_DIR
DATASET_DIR = Config.DATA_DIR
MODEL_DIR = Config.MODEL_DIR
IDS_DIR = Config.IDS_DIR
LOGS_DIR = Config.LOGS_DIR
TEMP_DIR = Config.TEMP_DIR
IDS_REPORTS_DIR = Config.IDS_REPORTS_DIR
TEST_REPORTS_DIR = Config.TEST_REPORTS_DIR
REPORTS_DIR = TEST_REPORTS_DIR

REPORT_NAMES = {
    1: "Análise Comparativa de Arquiteturas de Redes Neurais",
    2: "Análise de Estratégias de Balanceamento de Classes",
    3: "Análise da Aplicabilidade da Teoria da Informação",
    4: "Análise de Estratégias de Otimização e Validação",
}

REPORT_DIRS = {
    1: REPORTS_DIR / "Relatorio_1_Arquiteturas",
    2: REPORTS_DIR / "Relatorio_2_Balanceamento",
    3: REPORTS_DIR / "Relatorio_3_Teoria_Informacao",
    4: REPORTS_DIR / "Relatorio_4_Otimizacao_Validacao",
}

# -----------------------------------------------------------------------------
# Constantes acadêmicas / compatibilidade
# -----------------------------------------------------------------------------
RANDOM_SEED = 42
TF_INTER_OP_THREADS = Config.CPU_CONFIG["inter_op_threads"]
TF_INTRA_OP_THREADS = Config.CPU_CONFIG["intra_op_threads"]

LSTM_UNITS_L1 = Config.MODEL_CONFIG["lstm_units_1"]
LSTM_UNITS_L2 = Config.MODEL_CONFIG["lstm_units_2"]
LSTM_DENSE_UNITS = Config.MODEL_CONFIG["dense_units"]
LSTM_N_CLASSES = 15
DROPOUT_RATE = Config.MODEL_CONFIG["dropout_rate"]
RECURRENT_DROPOUT_RATE = Config.MODEL_CONFIG["recurrent_dropout_rate"]
LEARNING_RATE_INITIAL = Config.MODEL_CONFIG["learning_rate"]
LEARNING_RATE_FINETUNE = Config.FINE_TUNING_CONFIG["learning_rate"]
BATCH_SIZE = Config.TRAINING_CONFIG["batch_size"]
MAX_EPOCHS = Config.TRAINING_CONFIG["epochs"]
EARLY_STOPPING_PATIENCE = Config.TRAINING_CONFIG["patience"]
SEQUENCE_LENGTH = Config.MODEL_CONFIG["sequence_length"]
ATTENTION_UNITS = Config.MODEL_CONFIG["attention_units"]

N_FEATURES = 23
CLASS_NAMES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
CLASS_DIST = [0.80, 0.12, 0.05, 0.02, 0.01]
FEATURE_NAMES = [
    "Flow_Duration", "Dst_Port", "Total_Fwd_Packets", "Flow_IAT_Mean",
    "Pkt_Length_Mean", "Total_Bwd_Packets", "Flow_Bytes_s", "TCP_Flag_Count",
    "Protocol_Type", "Service_Type", "Active_Mean", "Active_Std",
    "Idle_Mean", "Idle_Std", "Count", "Same_Service_Rate",
    "Source_Bytes", "Destination_Bytes", "Flag_Type", "Fwd_Pkt_Length_Mean",
    "Pkt_Length_Variance", "Flow_IAT_Std", "Active_Std_2",
]

N_SAMPLES = {
    "arquiteturas": 8000,
    "balanceamento": 8000,
    "informacao": 6000,
    "otimizacao": 3000,
}

SMOTE_K = Config.BALANCING_CONFIG["smote_k_neighbors"]
ENN_K = Config.BALANCING_CONFIG["enn_n_neighbors"]
IG_WEIGHT = Config.FEATURE_SELECTION_CONFIG["ig_weight"]
MI_WEIGHT = Config.FEATURE_SELECTION_CONFIG["mi_weight"]
CV_FOLDS = 5
ALPHA_SIGNIFICANCE = Config.EVALUATION_CONFIG["significance_alpha"]

PLOT_STYLE = Config.VIZ_CONFIG["style"]
PLOT_DPI = Config.VIZ_CONFIG["dpi"]
FIG_TITLE_FS = 11
PLOT_PALETTE = "deep"

# Aliases novos
FOCAL_GAMMA = Config.MODEL_CONFIG["focal_gamma"]
FOCAL_CB_BETA = Config.MODEL_CONFIG["focal_class_balanced_beta"]
SMOTE_K_MAX = Config.BALANCING_CONFIG["smote_k_neighbors_max"]
SMOTE_K_ALPHA = Config.BALANCING_CONFIG["smote_k_alpha"]
TARGET_RATIO = Config.BALANCING_CONFIG["target_ratio_per_class"]
BASELINE_FILENAME = Config.BASELINE_CONFIG["model_filename"]

DATASET_FILES = [
    "bot.csv",
    "brute force -web.csv",
    "brute force -xss.csv",
    "ddos attack-hoic.csv",
    "ddos attack-loic-udp.csv",
    "ddos attacks-loic-http.csv",
    "dos attacks-goldeneye.csv",
    "dos attacks-hulk.csv",
    "dos attacks-slowhttptest.csv",
    "dos attacks-slowloris.csv",
    "ftp-bruteforce.csv",
    "infilteration.csv",
    "sql injection.csv",
    "ssh-bruteforce.csv",
]


def setup_environment() -> None:
    Config.ensure_dirs()
    for d in REPORT_DIRS.values():
        (d / "figuras").mkdir(parents=True, exist_ok=True)
        (d / "tabelas").mkdir(parents=True, exist_ok=True)

    try:
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(TF_INTER_OP_THREADS)
        tf.config.threading.set_intra_op_parallelism_threads(TF_INTRA_OP_THREADS)
        tf.random.set_seed(RANDOM_SEED)
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass


def apply_plot_style() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.style.use(PLOT_STYLE)
    sns.set_palette(PLOT_PALETTE)
    plt.rcParams.update({
        "figure.dpi": PLOT_DPI,
        "savefig.dpi": PLOT_DPI,
        "axes.titlesize": FIG_TITLE_FS,
        "axes.titleweight": "bold",
        "axes.grid": True,
    })


def fig_path(analise_id: int, nome: str) -> Path:
    """
    Caminho VERSIONADO para figuras dos testes acadêmicos.
    O nome é gerado com sufixo _YYYYMMDD-N pelo módulo
    IDS.modules.versioning, permitindo comparar execuções consecutivas.
    """
    base = REPORT_DIRS[analise_id] / "figuras"
    base.mkdir(parents=True, exist_ok=True)
    try:
        from IDS.modules.versioning import versioned_path
        return versioned_path(base, nome, "png")
    except ImportError:
        # Fallback caso o módulo de versionamento ainda não exista
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%Y%m%d-%H%M%S")
        return base / f"{nome}_{ts}.png"


def tab_path(analise_id: int, nome: str) -> Path:
    """Caminho VERSIONADO para tabelas dos testes acadêmicos."""
    base = REPORT_DIRS[analise_id] / "tabelas"
    base.mkdir(parents=True, exist_ok=True)
    try:
        from IDS.modules.versioning import versioned_path
        return versioned_path(base, nome, "csv")
    except ImportError:
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%Y%m%d-%H%M%S")
        return base / f"{nome}_{ts}.csv"


def print_config() -> None:
    print(Config.summary())


def _dataset_csvs() -> list[Path]:
    if not DATASET_DIR.exists():
        return []
    return sorted(p for p in DATASET_DIR.glob("*.csv") if p.is_file())


def _dataset_presente() -> tuple[bool, list[str]]:
    csvs = _dataset_csvs()
    if not csvs:
        return False, DATASET_FILES.copy()
    existentes = {p.name.strip().lower() for p in csvs}
    ausentes = [f for f in DATASET_FILES if f.lower() not in existentes]
    return len(ausentes) == 0, ausentes


def _instrucoes_manuais() -> None:
    print("\n  INSTRUÇÕES DE DATASET")
    print("  ─────────────────────")
    print(f"  Coloque os CSVs do CSE-CIC-IDS2018 cleaned em: {DATASET_DIR}")
    print("  Arquivos esperados:")
    for f in DATASET_FILES:
        print(f"    • {f}")
    print("  Observação: o diretório fixo esperado é Base/CSE-CIC-IDS2018/.")


def verificar_dataset(interativo: bool = True) -> bool:
    presente, ausentes = _dataset_presente()
    if presente:
        csvs = _dataset_csvs()
        print(f"  ✓ Dataset encontrado: {len(csvs)} arquivo(s) CSV em {DATASET_DIR}")
        return True
    print(f"\n  ⚠ Dataset real não encontrado em: {DATASET_DIR}")
    if ausentes:
        print(f"  Arquivos ausentes: {len(ausentes)} de {len(DATASET_FILES)}")
    if not interativo:
        print("  Modo não-interativo: prosseguindo com dados sintéticos.")
        return False
    _instrucoes_manuais()
    return False


def _resolve_label_column(df):
    """Resolve coluna de rótulo, normalizando espaços. Retorna nome real."""
    for col in df.columns:
        if col.strip().lower() == "label":
            return col
    return None


# Colunas tipicamente NÃO discriminantes em CSE-CIC-IDS2018 que causam leakage
# ou ruído (timestamps, IDs, portas como números absolutos).
_DROP_PATTERNS = (
    "timestamp", "flow id", "src ip", "dst ip", "source ip", "destination ip",
    "src port", "dst port", "source port", "destination port",
    "fwd header length.1",  # coluna duplicada conhecida
)


def _drop_meta_cols(df):
    drop = []
    for c in df.columns:
        cl = c.strip().lower()
        for p in _DROP_PATTERNS:
            if cl == p:
                drop.append(c); break
    return df.drop(columns=drop, errors="ignore")


def carregar_dataset_real(n_amostras_max: int = 500_000,
                           force_reload: bool = False):
    """
    Carrega CSE-CIC-IDS2018 com:
      - normalização de nomes de coluna (strip)
      - filtro de meta-colunas (timestamp, IPs, portas)
      - seleção das k=N_FEATURES MELHORES via Information Gain ranqueado
        sobre TODAS as features numéricas disponíveis
      - LabelEncoder consistente persistido em cache
      - sample estratificado por classe (preserva distribuição)
      - cache em Temp/cse_real_cache.npz para reuso entre análises

    Retorna (X, y, label_encoder) ou None.
    """
    try:
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import train_test_split
        from sklearn.feature_selection import mutual_info_classif
    except Exception as exc:
        print(f"  ⚠ Dependência ausente para dataset real: {exc}")
        return None

    cache_dir = Config.TEMP_DIR
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_npz = cache_dir / f"cse_real_cache_{n_amostras_max}.npz"
    cache_le  = cache_dir / f"cse_real_le_{n_amostras_max}.pkl"

    if cache_npz.exists() and cache_le.exists() and not force_reload:
        try:
            import joblib
            data = np.load(cache_npz, allow_pickle=False)
            X = data["X"].astype("float32")
            y = data["y"].astype("int64")
            le = joblib.load(cache_le)
            print(f"  ✓ Cache carregado: {cache_npz.name} "
                  f"X={X.shape}, classes={len(le.classes_)}")
            return X, y, le
        except Exception as exc:
            print(f"  ⚠ Cache inválido ({exc}); regenerando…")

    presente, _ = _dataset_presente()
    if not presente:
        return None

    csvs = _dataset_csvs()
    if not csvs:
        return None

    amostras_por_arquivo = max(1, n_amostras_max // max(1, len(csvs)))
    frames = []
    print(f"  Carregando {len(csvs)} CSV(s) (até {amostras_por_arquivo:,}/arquivo)…")

    for csv in csvs:
        try:
            df = pd.read_csv(csv, low_memory=False)
            if df.empty:
                continue
            df.columns = [c.strip() for c in df.columns]
            label_col = _resolve_label_column(df)
            if label_col is None:
                # Fallback determinístico: rótulo derivado do nome do arquivo
                stem = csv.stem.lower()
                if "benign" in stem or "normal" in stem:
                    df["Label"] = "Benign"
                else:
                    df["Label"] = csv.stem.replace("_", " ").title()
                label_col = "Label"
            elif label_col != "Label":
                df = df.rename(columns={label_col: "Label"})
            # Sample estratificado quando possível
            if len(df) > amostras_por_arquivo:
                try:
                    df, _ = train_test_split(
                        df, train_size=amostras_por_arquivo,
                        stratify=df["Label"], random_state=RANDOM_SEED)
                except Exception:
                    df = df.sample(amostras_por_arquivo, random_state=RANDOM_SEED)
            frames.append(df)
        except Exception as exc:
            print(f"  ⚠ Falha ao ler {csv.name}: {exc}")

    if not frames:
        return None

    df_all = pd.concat(frames, ignore_index=True)
    df_all = _drop_meta_cols(df_all)

    y_raw = df_all["Label"].astype(str).fillna("Unknown").str.strip()
    X_df = df_all.drop(columns=["Label"], errors="ignore")
    X_df = X_df.select_dtypes(include=["number"]).replace(
        [float("inf"), float("-inf")], 0).fillna(0)

    if X_df.empty:
        print("  ⚠ Dataset sem colunas numéricas utilizáveis.")
        return None

    print(f"  Universo de features numéricas: {X_df.shape[1]} colunas, "
          f"{len(X_df):,} amostras totais")

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    print(f"  Classes detectadas ({len(le.classes_)}): {list(le.classes_)}")

    X_full = X_df.to_numpy(dtype="float32")

    # Seleção REAL das k=N_FEATURES melhores via MI (rápido e determinístico)
    k_target = min(N_FEATURES, X_full.shape[1])
    if X_full.shape[1] > k_target:
        # Subamostra para acelerar MI
        n_mi = min(50_000, len(X_full))
        rng = np.random.default_rng(RANDOM_SEED)
        idx_mi = rng.choice(len(X_full), size=n_mi, replace=False)
        print(f"  Calculando MI sobre {n_mi:,} amostras para selecionar {k_target} features…")
        scores = mutual_info_classif(X_full[idx_mi], y[idx_mi],
                                      random_state=RANDOM_SEED)
        top_idx = np.argsort(scores)[::-1][:k_target]
        top_idx = np.sort(top_idx)
        X = X_full[:, top_idx]
        feature_names = X_df.columns[top_idx].tolist()
        print(f"  Features selecionadas: {feature_names}")
    else:
        X = X_full
        print(f"  Usando todas as {X.shape[1]} features disponíveis.")

    # Sample final estratificado
    if len(X) > n_amostras_max:
        try:
            from sklearn.model_selection import train_test_split as _tts
            _, X, _, y = _tts(X, y, test_size=n_amostras_max,
                              stratify=y, random_state=RANDOM_SEED)
        except Exception:
            idx = np.random.default_rng(RANDOM_SEED).choice(
                len(X), size=n_amostras_max, replace=False)
            X = X[idx]; y = y[idx]

    X = X.astype("float32")
    y = y.astype("int64")

    # Persiste cache
    try:
        import joblib
        np.savez(cache_npz, X=X, y=y)
        joblib.dump(le, cache_le)
        print(f"  ✓ Cache salvo: {cache_npz.name}")
    except Exception as exc:
        print(f"  ⚠ Falha ao salvar cache: {exc}")

    return X, y, le


@dataclass
class Relatorio:
    analise_id: int

    def __post_init__(self):
        setup_environment()
        self.id = self.analise_id
        self.titulo = REPORT_NAMES[self.id]
        self.dir = REPORT_DIRS[self.id]
        # Arquivo do relatório também versionado
        try:
            from IDS.modules.versioning import versioned_path
            self.arquivo = versioned_path(self.dir, f"Relatorio_{self.id}", "md")
        except ImportError:
            ts = datetime.now().strftime("%Y%m%d-%H%M%S")
            self.arquivo = self.dir / f"Relatorio_{self.id}_{ts}.md"
        self.linhas: list[str] = []
        self._cabecalho()

    def _cabecalho(self):
        self.linhas += [
            f"# Relatório {self.id}: {self.titulo}",
            "",
            "**Projeto**: SecurityIA  ",
            "**Trilha**: Testes e análises acadêmicas  ",
            f"**Gerado em**: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}  ",
            f"**Dataset**: {DATASET_DIR}  ",
            f"**Ambiente**: CPU-only | Threads inter={TF_INTER_OP_THREADS} intra={TF_INTRA_OP_THREADS}",
            "",
        ]

    def secao(self, titulo: str):
        self.linhas += [f"## {titulo}", ""]
        return self

    def subsecao(self, titulo: str):
        self.linhas += [f"### {titulo}", ""]
        return self

    def texto(self, texto: str):
        bloco = textwrap.dedent(str(texto)).strip("\n")
        if bloco:
            self.linhas += [bloco, ""]
        return self

    def metrica(self, nome: str, valor: str):
        self.linhas += [f"- **{nome}**: {valor}", ""]
        return self

    def tabela_df(self, df, legenda: str = ""):
        if legenda:
            self.linhas += [f"**{legenda}**", ""]
        try:
            table = df.to_markdown(index=False)
        except Exception:
            table = "```\n" + df.to_string(index=False) + "\n```"
        self.linhas += [table, ""]
        return self

    def figura(self, nome: str, legenda: str = ""):
        caption = legenda or nome
        # Para evitar referência inválida, mantém apenas o nome curto
        self.linhas += [f"![{caption}](figuras/{nome}.png)", ""]
        return self

    def salvar(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.arquivo.write_text("\n".join(self.linhas).rstrip() + "\n", encoding="utf-8")
        print(f"  ✓ Relatório salvo: {self.arquivo}")
        return self.arquivo


# Garante diretórios no import
setup_environment()
