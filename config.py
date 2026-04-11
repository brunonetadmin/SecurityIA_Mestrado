#!/usr/bin/env python3
"""
SecurityIA — Configuração central compatível com:
  - nova arquitetura do projeto (Base/CSE-CIC-IDS2018, Model, IDS, Reports, Temp)
  - scripts legados de testes em Tests/
  - app_menu.py

Este arquivo mantém uma API moderna via Config/IDSConfig e, ao mesmo tempo,
expõe aliases legados em nível de módulo para evitar quebrar os scripts de
análise acadêmica.
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

    # Relatórios do IDS (HTML / operação)
    IDS_REPORTS_DIR = BASE_DIR / "Reports"
    REPORTS_DIR = IDS_REPORTS_DIR

    # Relatórios dos testes acadêmicos
    TEST_REPORTS_DIR = TESTS_DIR / "Reports"

    # ------------------------------------------------------------------
    # Configuração de logs / operação IDS
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

    MODEL_CONFIG = {
        "lstm_units_1": 128,
        "lstm_units_2": 64,
        "dense_units": 32,
        "attention_units": 64,
        "dropout_rate": 0.5,
        "recurrent_dropout_rate": 0.0,
        "learning_rate": 1e-3,
        "loss_function": "sparse_categorical_crossentropy",
        "metrics": ["accuracy"],
        # Mantemos 100 para compatibilidade com os testes acadêmicos
        "sequence_length": 100,
    }

    TRAINING_CONFIG = {
        "random_state": 42,
        "validation_split": 0.15,
        "test_split": 0.15,
        "epochs": 50,
        "batch_size": 4096,
        "patience": 10,
        "force_retrain": False,
        # Agrupa N steps em uma única chamada ao backend — reduz overhead
        # Python entre batches. Seguro para CPU; valor conservador.
        "steps_per_execution": 8,
    }

    FINE_TUNING_CONFIG = {
        "enable": True,
        "learning_rate": 1e-5,
        "epochs": 20,
        "patience": 5,
    }

    BALANCING_CONFIG = {
        "smote_k_neighbors": 5,
        "enn_n_neighbors": 3,
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

    EVALUATION_CONFIG = {
        "eval_dir": TEMP_DIR / "evaluation",
        "benchmark_fraction": 0.10,
        "history_file": TEMP_DIR / "evaluation" / "eval_history.json",
        "significance_alpha": 0.001,
        "batch_size": 4096,
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
            cls.DATA_DIR,
            cls.MODEL_DIR,
            cls.LOGS_DIR,
            cls.TEMP_DIR,
            cls.IDS_REPORTS_DIR,
            cls.TESTS_DIR,
            cls.TEST_REPORTS_DIR,
            cls.COLLECTOR_DIR,
            cls.RETRAIN_CONFIG["staging_dir"],
            cls.EVALUATION_CONFIG["eval_dir"],
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
            sep,
        ])


IDSConfig = Config

# =============================================================================
# API LEGADA PARA OS TESTES ACADÊMICOS
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
# Para scripts legados, REPORTS_DIR deve apontar para os relatórios de testes.
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

# Dataset cleaned realmente esperado na pasta Base/CSE-CIC-IDS2018/
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
    path = REPORT_DIRS[analise_id] / "figuras" / f"{nome}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def tab_path(analise_id: int, nome: str) -> Path:
    path = REPORT_DIRS[analise_id] / "tabelas" / f"{nome}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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
    for col in ["Label", "label", " Label"]:
        if col in df.columns:
            return col
    return None


def carregar_dataset_real(n_amostras_max: int = 500_000):
    try:
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder
    except Exception as exc:
        print(f"  ⚠ Dependência ausente para dataset real: {exc}")
        return None

    presente, _ = _dataset_presente()
    if not presente:
        return None

    csvs = _dataset_csvs()
    if not csvs:
        return None

    amostras_por_arquivo = max(1, n_amostras_max // max(1, len(csvs)))
    frames = []

    for csv in csvs:
        try:
            df = pd.read_csv(csv, low_memory=False)
            if df.empty:
                continue

            label_col = _resolve_label_column(df)
            if label_col is None:
                # fallback: usa o nome do arquivo como rótulo
                df["Label"] = csv.stem
                label_col = "Label"

            if len(df) > amostras_por_arquivo:
                df = df.sample(amostras_por_arquivo, random_state=RANDOM_SEED)

            frames.append(df)
        except Exception as exc:
            print(f"  ⚠ Falha ao ler {csv.name}: {exc}")

    if not frames:
        return None

    df_all = pd.concat(frames, ignore_index=True)
    label_col = _resolve_label_column(df_all) or "Label"

    # Mantém apenas colunas numéricas e remove rótulo/meta textual.
    y_raw = df_all[label_col].astype(str).fillna("unknown")
    X_df = df_all.drop(columns=[c for c in [label_col, "Timestamp", "Flow ID", "Src IP", "Dst IP"] if c in df_all.columns], errors="ignore")
    X_df = X_df.select_dtypes(include=["number"]).replace([float("inf"), float("-inf")], 0).fillna(0)

    if X_df.empty:
        print("  ⚠ Dataset real lido, mas sem colunas numéricas utilizáveis.")
        return None

    # Ajusta número de features para o esperado pelos testes.
    if X_df.shape[1] >= N_FEATURES:
        X_df = X_df.iloc[:, :N_FEATURES]
    else:
        for i in range(N_FEATURES - X_df.shape[1]):
            X_df[f"pad_{i}"] = 0.0
        X_df = X_df.iloc[:, :N_FEATURES]

    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    X = X_df.to_numpy(dtype="float32")

    if len(X) > n_amostras_max:
        idx = np.random.default_rng(RANDOM_SEED).choice(len(X), size=n_amostras_max, replace=False)
        X = X[idx]
        y = y[idx]

    return X, y, le


@dataclass
class Relatorio:
    analise_id: int

    def __post_init__(self):
        setup_environment()
        self.id = self.analise_id
        self.titulo = REPORT_NAMES[self.id]
        self.dir = REPORT_DIRS[self.id]
        self.arquivo = self.dir / f"Relatorio_{self.id}.md"
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
        self.linhas += [f"![{caption}](figuras/{nome}.png)", ""]
        return self

    def salvar(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.arquivo.write_text("\n".join(self.linhas).rstrip() + "\n", encoding="utf-8")
        print(f"  ✓ Relatório salvo: {self.arquivo}")
        return self.arquivo


# Garante diretórios no import para evitar erro ao salvar figuras/tabelas.
setup_environment()