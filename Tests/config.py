"""
config.py — Configuração Central
=================================
Dissertação: Detecção de Anomalias em Redes de Computadores Utilizando AM
Autor  : Bruno Cavalcante Barbosa — bcb@ic.ufal.br
Orient.: Prof. Dr. André Luiz Lins de Aquino — PPGI/IC/UFAL

Centraliza: caminhos, hiperparâmetros, configuração de ambiente,
utilitários de relatório e verificação/download do dataset.
"""

# ── Ambiente CPU-only (deve preceder qualquer import de TF) ──────────────────
import os, sys, time, warnings
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["TF_DETERMINISTIC_OPS"]  = "1"
os.environ["PYTHONHASHSEED"]        = "42"
warnings.filterwarnings("ignore")

# ── Imports padrão ────────────────────────────────────────────────────────────
import shutil
import textwrap
import hashlib
from pathlib import Path
from datetime import datetime

import numpy as np
np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════════════════
# 1. CAMINHOS BASE
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR     = Path("/opt/Testes")
DATASET_DIR  = BASE_DIR / "Base"
REPORTS_DIR  = BASE_DIR / "Reports"

# Sub-diretórios de relatório (um por análise)
REPORT_DIRS = {
    1: REPORTS_DIR / "Relatorio_1_Arquiteturas",
    2: REPORTS_DIR / "Relatorio_2_Balanceamento",
    3: REPORTS_DIR / "Relatorio_3_Teoria_Informacao",
    4: REPORTS_DIR / "Relatorio_4_Otimizacao_Validacao",
}

REPORT_NAMES = {
    1: "Análise Comparativa de Arquiteturas de Redes Neurais",
    2: "Análise de Estratégias de Balanceamento de Classes",
    3: "Análise da Aplicabilidade da Teoria da Informação",
    4: "Análise de Estratégias de Otimização e Validação",
}

# Arquivos esperados do dataset CIC-IDS2018 (versão Mendeley — Rabelo et al., 2024)
DATASET_FILES = [
    "02-14-2018.csv", "02-15-2018.csv", "02-16-2018.csv",
    "02-20-2018.csv", "02-21-2018.csv", "02-22-2018.csv",
    "02-23-2018.csv", "02-28-2018.csv", "03-01-2018.csv",
    "03-02-2018.csv",
]

# URL da versão reduzida (Rabelo et al., 2024 — Mendeley DOI: 10.17632/29hdbdzx2r.1)
DATASET_MENDELEY_DOI  = "10.17632/29hdbdzx2r.1"
DATASET_MENDELEY_URL  = "https://data.mendeley.com/datasets/29hdbdzx2r/1"
DATASET_KAGGLE_SLUG   = "solarmainframe/ids-intrusion-csv"  # alternativa pública

# ═══════════════════════════════════════════════════════════════════════════════
# 2. RANDOM SEED E THREADS
# ═══════════════════════════════════════════════════════════════════════════════
RANDOM_SEED          = 42
TF_INTER_OP_THREADS  = 4
TF_INTRA_OP_THREADS  = 16   # calibrado para 20 vCPUs

# ═══════════════════════════════════════════════════════════════════════════════
# 3. HIPERPARÂMETROS DO MODELO
# ═══════════════════════════════════════════════════════════════════════════════
LSTM_UNITS_L1           = 128
LSTM_UNITS_L2           = 64
LSTM_DENSE_UNITS        = 32
LSTM_N_CLASSES          = 15
DROPOUT_RATE            = 0.5
RECURRENT_DROPOUT_RATE  = 0.0   # CuDNN-compatible (equivalente float32)
LEARNING_RATE_INITIAL   = 1e-3
LEARNING_RATE_FINETUNE  = 1e-5
BATCH_SIZE              = 1_024
MAX_EPOCHS              = 50
EARLY_STOPPING_PATIENCE = 10
SEQUENCE_LENGTH         = 100
ATTENTION_UNITS         = 64

# ═══════════════════════════════════════════════════════════════════════════════
# 4. DATASET SINTÉTICO
# ═══════════════════════════════════════════════════════════════════════════════
N_FEATURES    = 23
CLASS_NAMES   = ["Normal", "DoS", "Probe", "R2L", "U2R"]
CLASS_DIST    = [0.80, 0.12, 0.05, 0.02, 0.01]

FEATURE_NAMES = [
    "Flow_Duration", "Dst_Port", "Total_Fwd_Packets", "Flow_IAT_Mean",
    "Pkt_Length_Mean", "Total_Bwd_Packets", "Flow_Bytes_s", "TCP_Flag_Count",
    "Protocol_Type", "Service_Type", "Active_Mean", "Active_Std",
    "Idle_Mean", "Idle_Std", "Count", "Same_Service_Rate",
    "Source_Bytes", "Destination_Bytes", "Flag_Type", "Fwd_Pkt_Length_Mean",
    "Pkt_Length_Variance", "Flow_IAT_Std", "Active_Std_2",
]

# Amostras por script (reduzidas para viabilidade CPU)
N_SAMPLES = {
    "arquiteturas":  8_000,
    "balanceamento": 8_000,
    "informacao":    6_000,
    "otimizacao":    3_000,
}

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SMOTE-ENN
# ═══════════════════════════════════════════════════════════════════════════════
SMOTE_K       = 5
ENN_K         = 3
MAX_MINORITY  = 50_000
MAX_MAJORITY  = 150_000

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SELEÇÃO DE CARACTERÍSTICAS
# ═══════════════════════════════════════════════════════════════════════════════
IG_WEIGHT = 0.6
MI_WEIGHT = 0.4

# ═══════════════════════════════════════════════════════════════════════════════
# 7. VALIDAÇÃO CRUZADA
# ═══════════════════════════════════════════════════════════════════════════════
CV_FOLDS          = 5
ALPHA_SIGNIFICANCE = 0.001

# ═══════════════════════════════════════════════════════════════════════════════
# 8. ESTILO VISUAL
# ═══════════════════════════════════════════════════════════════════════════════
PLOT_STYLE   = "seaborn-v0_8"
PLOT_PALETTE = "husl"
PLOT_DPI     = 300
FIG_TITLE_FS = 14
FIG_LABEL_FS = 11


# ═══════════════════════════════════════════════════════════════════════════════
# 9. FUNÇÕES UTILITÁRIAS
# ═══════════════════════════════════════════════════════════════════════════════

def setup_environment() -> None:
    """Cria estrutura de diretórios e configura TF."""
    for d in [DATASET_DIR, REPORTS_DIR, *REPORT_DIRS.values()]:
        d.mkdir(parents=True, exist_ok=True)
        (d / "figuras").mkdir(exist_ok=True)
        (d / "tabelas").mkdir(exist_ok=True)

    try:
        import tensorflow as tf
        tf.config.threading.set_inter_op_parallelism_threads(TF_INTER_OP_THREADS)
        tf.config.threading.set_intra_op_parallelism_threads(TF_INTRA_OP_THREADS)
        tf.random.set_seed(RANDOM_SEED)
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            tf.config.set_visible_devices([], "GPU")
    except ImportError:
        pass


def apply_plot_style() -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.style.use(PLOT_STYLE)
    sns.set_palette(PLOT_PALETTE)


def fig_path(analise_id: int, nome: str) -> Path:
    """Retorna caminho padronizado para uma figura."""
    return REPORT_DIRS[analise_id] / "figuras" / f"{nome}.png"


def tab_path(analise_id: int, nome: str) -> Path:
    """Retorna caminho padronizado para uma tabela CSV."""
    return REPORT_DIRS[analise_id] / "tabelas" / f"{nome}.csv"


# ═══════════════════════════════════════════════════════════════════════════════
# 10. GERADOR DE RELATÓRIO MARKDOWN
# ═══════════════════════════════════════════════════════════════════════════════

class Relatorio:
    """
    Gera um relatório estruturado em Markdown para cada análise.

    Seções padrão
    -------------
    1. Cabeçalho (título, data, ambiente)
    2. Resumo executivo
    3. Metodologia
    4. Resultados (tabelas + referências às figuras)
    5. Conclusões
    """

    def __init__(self, analise_id: int) -> None:
        self.id      = analise_id
        self.titulo  = REPORT_NAMES[analise_id]
        self.dir     = REPORT_DIRS[analise_id]
        self.linhas: list[str] = []
        self._cabecalho()

    def _cabecalho(self) -> None:
        self.linhas += [
            f"# Relatório {self.id}: {self.titulo}",
            "",
            f"**Dissertação**: Detecção de Anomalias em Redes de Computadores Utilizando AM  ",
            f"**Autor**: Bruno Cavalcante Barbosa — bcb@ic.ufal.br  ",
            f"**Orientador**: Prof. Dr. André Luiz Lins de Aquino — PPGI/IC/UFAL  ",
            f"**Gerado em**: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}  ",
            f"**Ambiente**: CPU-only | Threads: inter={TF_INTER_OP_THREADS}, intra={TF_INTRA_OP_THREADS}",
            "",
            "---",
            "",
        ]

    def secao(self, titulo: str) -> "Relatorio":
        self.linhas += ["", f"## {titulo}", ""]
        return self

    def subsecao(self, titulo: str) -> "Relatorio":
        self.linhas += ["", f"### {titulo}", ""]
        return self

    def texto(self, txt: str) -> "Relatorio":
        self.linhas += [textwrap.dedent(txt).strip(), ""]
        return self

    def figura(self, nome_arquivo: str, legenda: str) -> "Relatorio":
        self.linhas += [
            f"![{legenda}](figuras/{nome_arquivo}.png)",
            f"*Figura: {legenda}*",
            "",
        ]
        return self

    def tabela_df(self, df, legenda: str = "") -> "Relatorio":
        import pandas as pd
        self.linhas += [df.to_markdown(index=False), ""]
        if legenda:
            self.linhas += [f"*Tabela: {legenda}*", ""]
        return self

    def metrica(self, nome: str, valor: str) -> "Relatorio":
        self.linhas += [f"- **{nome}**: {valor}"]
        return self

    def salvar(self) -> Path:
        caminho = self.dir / f"Relatorio_{self.id}.md"
        caminho.write_text("\n".join(self.linhas), encoding="utf-8")
        print(f"\n  📄 Relatório salvo: {caminho}")
        return caminho

    def separador(self) -> "Relatorio":
        self.linhas += ["", "---", ""]
        return self


# ═══════════════════════════════════════════════════════════════════════════════
# 11. VERIFICAÇÃO E DOWNLOAD DO DATASET
# ═══════════════════════════════════════════════════════════════════════════════

def _dataset_presente() -> tuple[bool, list[str]]:
    """Retorna (existe, lista_de_ausentes)."""
    ausentes = [f for f in DATASET_FILES if not (DATASET_DIR / f).exists()]
    # Aceita também se houver qualquer .csv no diretório
    csvs_existentes = list(DATASET_DIR.glob("*.csv"))
    if csvs_existentes and not ausentes:
        return True, []
    if csvs_existentes and len(csvs_existentes) >= 5:
        return True, []  # versão parcial — aceita
    return len(ausentes) == 0, ausentes


def _tamanho_formatado(nbytes: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if nbytes < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024
    return f"{nbytes:.1f} TB"


def _baixar_kaggle() -> bool:
    """Tenta baixar via kaggle CLI."""
    if shutil.which("kaggle") is None:
        return False
    print("  Iniciando download via Kaggle CLI...")
    ret = os.system(
        f"kaggle datasets download -d {DATASET_KAGGLE_SLUG} "
        f"--path {DATASET_DIR} --unzip"
    )
    return ret == 0


def _baixar_mendeley() -> bool:
    """Tenta baixar o arquivo ZIP do Mendeley via requests."""
    try:
        import requests
        # URL de download direto do dataset compactado (versão reduzida Rabelo 2024)
        url = (
            "https://prod-dcd-datasets-cache-zipfiles.s3.eu-west-1.amazonaws.com"
            "/29hdbdzx2r-1.zip"
        )
        print(f"  Conectando: {url}")
        resp = requests.get(url, stream=True, timeout=30)
        if resp.status_code != 200:
            print(f"  Erro HTTP {resp.status_code}.")
            return False

        total = int(resp.headers.get("content-length", 0))
        zip_path = DATASET_DIR / "cic_ids2018.zip"
        baixado = 0
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                baixado += len(chunk)
                if total:
                    pct = baixado / total * 100
                    print(f"\r  Download: {pct:.1f}% ({_tamanho_formatado(baixado)})", end="")
        print()

        print("  Descompactando...")
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(DATASET_DIR)
        zip_path.unlink()
        return True
    except Exception as e:
        print(f"  Falha no download: {e}")
        return False


def _instrucoes_manuais() -> None:
    print("\n" + "─" * 60)
    print("  DOWNLOAD MANUAL — OPÇÕES DISPONÍVEIS")
    print("─" * 60)
    print(f"\n  1. Mendeley (versão reduzida — Rabelo et al., 2024):")
    print(f"     {DATASET_MENDELEY_URL}")
    print(f"     DOI: {DATASET_MENDELEY_DOI}")
    print(f"\n  2. Kaggle (requer conta gratuita):")
    print(f"     https://www.kaggle.com/datasets/{DATASET_KAGGLE_SLUG}")
    print(f"     Comando: kaggle datasets download -d {DATASET_KAGGLE_SLUG}")
    print(f"\n  3. CIC/UNB (versão completa ~50GB):")
    print(f"     https://www.unb.ca/cic/datasets/ids-2018.html")
    print(f"\n  Destino: {DATASET_DIR}")
    print("─" * 60)


def verificar_dataset(interativo: bool = True) -> bool:
    """
    Verifica se o dataset CIC-IDS2018 existe no diretório configurado.

    Se ausente e interativo=True, pergunta ao usuário se deseja
    baixar automaticamente.

    Retorna
    -------
    bool : True se o dataset está disponível (ou download bem-sucedido).
    """
    presente, ausentes = _dataset_presente()

    if presente:
        csvs = list(DATASET_DIR.glob("*.csv"))
        print(f"  ✓ Dataset encontrado: {len(csvs)} arquivo(s) CSV em {DATASET_DIR}")
        return True

    print(f"\n  ⚠ Dataset CIC-IDS2018 não encontrado em: {DATASET_DIR}")
    if ausentes:
        print(f"  Arquivos ausentes: {len(ausentes)} de {len(DATASET_FILES)}")

    if not interativo:
        print("  Modo não-interativo: prosseguindo com dados sintéticos.")
        return False

    print("\n  Deseja baixar o dataset automaticamente?")
    print("  [1] Sim — tentar Mendeley (versão reduzida ~800MB)")
    print("  [2] Sim — tentar Kaggle CLI (requer 'pip install kaggle' + credenciais)")
    print("  [3] Não — exibir instruções de download manual")
    print("  [4] Não — prosseguir com dados sintéticos (experimentos de justificativa)")

    while True:
        opcao = input("\n  Opção [1-4]: ").strip()
        if opcao == "1":
            DATASET_DIR.mkdir(parents=True, exist_ok=True)
            if _baixar_mendeley():
                presente, _ = _dataset_presente()
                if presente:
                    print("  ✓ Download concluído com sucesso!")
                    return True
                else:
                    print("  Download aparentemente concluído, mas CSVs não detectados.")
                    print("  Verifique manualmente em:", DATASET_DIR)
                    return False
            else:
                print("  Download via Mendeley falhou. Tente a opção Kaggle ou manual.")
                return False
        elif opcao == "2":
            DATASET_DIR.mkdir(parents=True, exist_ok=True)
            if _baixar_kaggle():
                print("  ✓ Download via Kaggle concluído!")
                return True
            else:
                print("  Kaggle CLI não encontrado ou falhou.")
                print("  Instale: pip install kaggle")
                print("  Configure: https://github.com/Kaggle/kaggle-api#api-credentials")
                return False
        elif opcao == "3":
            _instrucoes_manuais()
            return False
        elif opcao == "4":
            print("  Prosseguindo com dados sintéticos.")
            return False
        else:
            print("  Opção inválida. Digite 1, 2, 3 ou 4.")


def carregar_dataset_real(n_amostras_max: int = 500_000):
    """
    Carrega os CSVs do CIC-IDS2018 e retorna X, y como arrays numpy.

    Aplicado quando o dataset está disponível. Usado pelos scripts que
    podem se beneficiar de dados reais (especialmente Script 2).

    Retorna
    -------
    (X, y, label_encoder) ou None se o dataset não estiver disponível.
    """
    try:
        import pandas as pd
        from sklearn.preprocessing import LabelEncoder

        csvs = sorted(DATASET_DIR.glob("*.csv"))
        if not csvs:
            return None

        print(f"  Carregando {len(csvs)} arquivo(s) CSV...")
        frames = []
        for csv in csvs:
            try:
                df = pd.read_csv(csv, low_memory=False)
                df.columns = df.columns.str.strip()
                frames.append(df)
            except Exception as e:
                print(f"  Aviso: erro ao ler {csv.name}: {e}")

        if not frames:
            return None

        df_all = pd.concat(frames, ignore_index=True)
        print(f"  Total carregado: {len(df_all):,} instâncias")

        # Identifica coluna de label
        label_col = next(
            (c for c in df_all.columns if "label" in c.lower()), None
        )
        if label_col is None:
            print("  Coluna de rótulo não encontrada.")
            return None

        # Limpeza básica
        df_all.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
        df_all.dropna(inplace=True)

        # Amostragem estratificada se necessário
        if len(df_all) > n_amostras_max:
            df_all = df_all.groupby(label_col, group_keys=False).apply(
                lambda x: x.sample(
                    min(len(x), int(n_amostras_max * len(x) / len(df_all))),
                    random_state=RANDOM_SEED,
                )
            )
            print(f"  Amostrado: {len(df_all):,} instâncias")

        # Features numéricas
        num_cols = df_all.select_dtypes(include="number").columns.tolist()
        if label_col in num_cols:
            num_cols.remove(label_col)

        X = df_all[num_cols].values.astype("float32")
        le = LabelEncoder()
        y = le.fit_transform(df_all[label_col].astype(str))

        print(f"  Features: {X.shape[1]} | Classes: {len(le.classes_)}")
        return X, y, le

    except Exception as e:
        print(f"  Erro ao carregar dataset: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# 12. SUMÁRIO DE CONFIGURAÇÃO
# ═══════════════════════════════════════════════════════════════════════════════

def print_config() -> None:
    sep = "═" * 62
    print(f"\n{sep}")
    print("  CONFIGURAÇÃO — IDS Bi-LSTM + Atenção | PPGI/UFAL")
    print(sep)
    print(f"  Base dir      : {BASE_DIR}")
    print(f"  Dataset dir   : {DATASET_DIR}")
    print(f"  Reports dir   : {REPORTS_DIR}")
    print(f"  Seed          : {RANDOM_SEED}")
    print(f"  TF threads    : inter={TF_INTER_OP_THREADS}, intra={TF_INTRA_OP_THREADS}")
    print(f"  Features       : {N_FEATURES} | Dropout: {DROPOUT_RATE}")
    print(f"  LSTM L1/L2    : {LSTM_UNITS_L1}/{LSTM_UNITS_L2} | Atenção: {ATTENTION_UNITS}u")
    print(f"  SMOTE k/ENN k : {SMOTE_K}/{ENN_K} | IG/MI: {IG_WEIGHT}/{MI_WEIGHT}")
    print(f"  CV folds / α  : {CV_FOLDS} / {ALPHA_SIGNIFICANCE}")
    print(sep + "\n")


# Inicializa diretórios ao importar
setup_environment()
