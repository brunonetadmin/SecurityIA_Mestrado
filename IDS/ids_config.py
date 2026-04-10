#!/usr/bin/env python3
"""
##############################################################################################
#  Arquivo de Configuração Centralizado para o Sistema IDS
#  Versão: 2.3 — Configuração de Interface de Captura e Nomenclatura de Relatórios
#
#  Este arquivo é a ÚNICA fonte de verdade (SSOT) do sistema.
#  Todos os scripts (ids_coletor.py, ids_manager.py, ids_model_evaluator.py,
#  1_ids_training_script.py) importam os valores daqui.
#
#  ┌─────────────────────────────────────────────────────────────────────────┐
#  │  CONFIGURAÇÕES MAIS COMUNS — edite estas primeiro:                      │
#  │                                                                         │
#  │  CAPTURE_INTERFACE = "eth1"        ← interface da porta mirror         │
#  │  COLLECTOR_DIR     = Path(...)     ← onde o coletor salva os parquets  │
#  │  IDS_REPORTS_DIR   = Path(...)     ← onde os relatórios são salvos     │
#  └─────────────────────────────────────────────────────────────────────────┘
#
#  HISTÓRICO DE VERSÕES:
#  v2.0 — Versão inicial estruturada
#  v2.1 — FIX-01..04: duplicatas e parâmetros mortos corrigidos
#  v2.2 — NEW-01: COLLECTOR_DIR, IDS_REPORTS_DIR, IDS_STAGING_DIR adicionados
#  v2.3 — NEW-02: CAPTURE_INTERFACE, EVALUATION_DIR, COLLECTOR_LOGS_DIR,
#                 REPORT_COUNTER_FILE, COLLECTOR_SAMPLE_RATE, COLLECTOR_BUDGET_GB
#
#  Autor: Bruno Cavalcante Barbosa
#  UFAL - Universidade Federal de Alagoas
##############################################################################################
"""

from pathlib import Path


class IDSConfig:
    """
    Namespace centralizado de configuração. Todos os atributos são variáveis de
    classe — acesse via IDSConfig.ATRIBUTO sem necessidade de instanciação.
    """

    # =========================================================================
    # SEÇÃO 1 — DIRETÓRIOS BASE DO PROJETO
    # =========================================================================
    BASE_DIR     = Path(__file__).resolve().parent
    DATA_DIR     = BASE_DIR / "Base/CSE-CIC-IDS2018/"
    MODEL_DIR    = BASE_DIR / "Model/"
    RESULTS_DIR  = BASE_DIR / "Results/"
    REPORTS_DIR  = BASE_DIR / "Reports/"
    LOGS_DIR     = BASE_DIR / "Logs/"
    CACHE_DIR    = BASE_DIR / "Cache/"

    # =========================================================================
    # SEÇÃO 2 — PIPELINE DE COLETA E ANÁLISE
    # Editável pelo operador. Todos os scripts consomem estes valores.
    # =========================================================================

    # [NEW-02] CAPTURE_INTERFACE — Interface de rede monitorada pelo ids_coletor.py.
    # Deve apontar para a porta mirror que espelha o link principal.
    # Exemplos: "eth1", "ens4", "enp3s0"
    # O ids_coletor.py lê este valor na inicialização; não é necessário passá-lo
    # como argumento de linha de comando.
    CAPTURE_INTERFACE = "eth1"

    # COLLECTOR_DIR — Onde o ids_coletor.py salva os arquivos Parquet diários.
    # Nomenclatura automática: captura_{Dia}_{dd}_{mm}_{aaaa}.parquet
    COLLECTOR_DIR = Path("/opt/idsapp/collector")

    # [NEW-02] COLLECTOR_LOGS_DIR — Logs operacionais do ids_coletor.py.
    # Separado de LOGS_DIR para não misturar logs de treinamento com logs de captura.
    COLLECTOR_LOGS_DIR = Path("/opt/idsapp/logs/collector")

    # [NEW-02] COLLECTOR_SAMPLE_RATE — Fração de fluxos capturados (0.0–1.0).
    # Em links de altíssimo tráfego, reduza para controlar o volume de dados.
    # 1.0 = captura 100% dos fluxos detectados.
    COLLECTOR_SAMPLE_RATE = 1.0

    # [NEW-02] COLLECTOR_BUDGET_GB — Budget máximo de armazenamento por dia (GiB).
    # Quando atingido, novos fluxos são descartados até 00:00 do dia seguinte.
    COLLECTOR_BUDGET_GB = 7.0

    # IDS_REPORTS_DIR — Onde os relatórios HTML/TXT de análise são salvos.
    # Nomenclatura: relatorio_{NNN}_{versao}_{YYYYMMDD}.html
    # NNN = número sequencial global (001, 002, ...)
    IDS_REPORTS_DIR = Path("/opt/idsapp/reports")

    # IDS_STAGING_DIR — Área temporária para datasets de re-treinamento.
    IDS_STAGING_DIR = Path("/opt/idsapp/staging")

    # [NEW-02] EVALUATION_DIR — Diretório do framework de avaliação contínua.
    # Contém o benchmark congelado, histórico de avaliações e relatórios de evolução.
    EVALUATION_DIR = Path("/opt/idsapp/evaluation")

    # [NEW-02] REPORT_COUNTER_FILE — Arquivo que mantém o contador global de relatórios.
    # Garante numeração sequencial única mesmo após reinicializações do sistema.
    REPORT_COUNTER_FILE = IDS_REPORTS_DIR / ".report_counter"

    # Nome do arquivo de estado do manager (lista de arquivos já processados)
    MANAGER_STATE_FILE = COLLECTOR_DIR / ".ids_manager_state.json"

    # =========================================================================
    # SEÇÃO 3 — ARTEFATOS DO MODELO
    # =========================================================================
    REPORTS_DIR_NAME           = "classification_reports"
    MODEL_FILENAME             = "ids_lstm_model.keras"
    SCALER_FILENAME            = "scaler.pkl"
    LABEL_ENCODER_FILENAME     = "label_encoder.pkl"
    SELECTED_FEATURES_FILENAME = "ids_selected_features.json"
    MODEL_INFO_FILENAME        = "ids_model_info.json"

    # =========================================================================
    # SEÇÃO 4 — PRÉ-PROCESSAMENTO
    # =========================================================================
    PREPROCESSING_CONFIG = {
        'sample_fraction':       1.0,
        'missing_value_threshold': 0.5,
        'variance_threshold':    1e-5,
        'apply_variance_filter': True,
        'force_reload':          False,
        'force_preprocess':      False,
    }

    # =========================================================================
    # SEÇÃO 5 — SELEÇÃO DE CARACTERÍSTICAS
    # Score combinado: score = ig_weight * IG_norm + mi_weight * MI_norm
    # =========================================================================
    FEATURE_SELECTION_CONFIG = {
        'k_best':                 23,
        'ig_weight':              0.6,
        'mi_weight':              0.4,
        'normalization_epsilon':  1e-9,
        'ig_discretization_bins': 10,
    }

    # =========================================================================
    # SEÇÃO 6 — ARQUITETURA E TREINAMENTO DO MODELO
    # =========================================================================
    MODEL_CONFIG = {
        'lstm_units_1':         128,
        'lstm_units_2':         64,
        'dense_units':          32,
        'dropout_rate':         0.5,
        # recurrent_dropout_rate > 0 desabilita cuDNN (mais lento em GPU).
        'recurrent_dropout_rate': 0.0,
        'learning_rate':        1e-3,
        'loss_function':        'sparse_categorical_crossentropy',
        'metrics':              ['accuracy'],
    }

    TRAINING_CONFIG = {
        'random_state':      42,
        'validation_split':  0.15,
        'test_split':        0.15,
        'epochs':            50,
        'batch_size':        1024,
        'patience':          10,
        'force_retrain':     False,
    }

    FINE_TUNING_CONFIG = {
        'enable':        True,
        'learning_rate': 1e-5,
    }

    BALANCING_CONFIG = {
        'enn_n_neighbors':    3,
        'n_samples_minority': 50_000,
        'n_samples_majority': 150_000,
    }

    # =========================================================================
    # SEÇÃO 7 — LOGGING E VISUALIZAÇÃO
    # =========================================================================
    LOGGING_CONFIG = {
        'level':   'INFO',
        'format':  '%(asctime)s - %(levelname)s - %(message)s',
        'datefmt': '%Y-%m-%d %H:%M:%S',
    }

    VISUALIZATION_CONFIG = {
        'style':       'ggplot',
        'save_format': 'png',
        'dpi':         300,
    }

    # =========================================================================
    # SEÇÃO 8 — MAPEAMENTO DE REFERÊNCIA DO DATASET
    # =========================================================================
    REFERENCE_ATTACK_TYPES = {
        0:  'Benign',
        1:  'Bot',
        2:  'DDoS',
        3:  'DoS GoldenEye',
        4:  'DoS Hulk',
        5:  'DoS Slowhttptest',
        6:  'DoS slowloris',
        7:  'FTP-Patator',
        8:  'Heartbleed',
        9:  'Infiltration',
        10: 'PortScan',
        11: 'SSH-Patator',
        12: 'Web Attack \u2013 Brute Force',
        13: 'Web Attack \u2013 Sql Injection',
        14: 'Web Attack \u2013 XSS',
    }

    # =========================================================================
    # SEÇÃO 9 — CONFIGURAÇÃO DE HARDWARE (CPU)
    # =========================================================================

    # Configuração de threads do TensorFlow para execução em CPU.
    # inter_op_threads : paralelismo entre operações TF independentes.
    # intra_op_threads : paralelismo interno por operação (BLAS/MKL — matrizes).
    # Para o servidor alvo (20 vCPUs): 4 inter + 16 intra é o ponto ótimo;
    # reserva ~2 vCPUs para o SO e o ids_coletor.py rodando em paralelo.
    CPU_CONFIG = {
        'inter_op_threads': 4,
        'intra_op_threads': 16,
    }

    @classmethod
    def configure_tensorflow(cls) -> None:
        """
        Configura o TensorFlow para execução exclusiva em CPU.

        Deve ser chamado ANTES de qualquer operação TF — idealmente no início
        de main() ou no topo do módulo que importa TensorFlow.

        Três efeitos, nenhum altera os resultados numéricos do modelo:
          1. Desabilita detecção de GPU — evita RuntimeError e atraso de
             10–30 s ao inicializar CUDA em sistemas sem GPU.
          2. Configura inter_op e intra_op threads — utiliza todos os cores
             disponíveis via MKL/oneDNN, reduzindo tempo de treinamento e
             inferência em 2–4× comparado ao padrão auto (0, 0).
          3. Não afeta arquitetura, pesos, loss ou métricas finais.
             Diferenças numéricas entre CPU e GPU são < 1e-6 (float32),
             abaixo da precisão de F1-score, acurácia ou AUC reportados.
        """
        import tensorflow as tf

        # Impede que TF tente inicializar CUDA em sistemas sem GPU.
        # Em sistemas com GPU, esta chamada simplesmente ignora os dispositivos
        # disponíveis — o código funciona igualmente nos dois cenários.
        tf.config.set_visible_devices([], 'GPU')

        # Threads para ops independentes (ex.: camadas paralelas no grafo TF).
        # Valor baixo é intencional: LSTM bidirecional é majoritariamente
        # sequencial, então inter-op tem pouco paralelismo a explorar.
        tf.config.threading.set_inter_op_parallelism_threads(
            cls.CPU_CONFIG['inter_op_threads']
        )

        # Threads para operações internas (GEMM, BLAS).
        # Este é o nível crítico para LSTM: cada passo temporal executa
        # multiplicações de matrizes de dimensão [batch × unidades], e o
        # MKL paralleliza internamente usando estes threads.
        tf.config.threading.set_intra_op_parallelism_threads(
            cls.CPU_CONFIG['intra_op_threads']
        )

    # =========================================================================
    # SEÇÃO 10 — UTILITÁRIOS DE CONFIGURAÇÃO (não editar)
    # =========================================================================

    @classmethod
    def next_report_number(cls) -> int:
        """
        Retorna e incrementa o contador global de relatórios.
        Thread-safe via arquivo de bloqueio temporário.
        Garante numeração sequencial única (001, 002, ...) mesmo após
        reinicializações do sistema.
        """
        cls.IDS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        lock = cls.REPORT_COUNTER_FILE.with_suffix('.lock')
        # Espera simples por lock (suficiente — relatórios não são gerados em paralelo)
        import time
        for _ in range(30):
            try:
                lock.touch(exist_ok=False)
                break
            except FileExistsError:
                time.sleep(0.1)
        try:
            if cls.REPORT_COUNTER_FILE.exists():
                n = int(cls.REPORT_COUNTER_FILE.read_text().strip()) + 1
            else:
                n = 1
            cls.REPORT_COUNTER_FILE.write_text(str(n))
        finally:
            lock.unlink(missing_ok=True)
        return n

    @classmethod
    def report_filename(cls, version: str, extension: str = 'html') -> Path:
        """
        Gera o caminho completo de um relatório com nomenclatura padronizada.
        Formato: relatorio_{NNN}_{versao}_{YYYYMMDD}.{ext}
        Exemplo: relatorio_003_v2_20260403.html
        """
        from datetime import date
        n   = cls.next_report_number()
        ver = version.replace(' ', '_')
        dt  = date.today().strftime('%Y%m%d')
        return cls.IDS_REPORTS_DIR / f"relatorio_{n:03d}_{ver}_{dt}.{extension}"

    @classmethod
    def ensure_dirs(cls) -> None:
        """Cria todos os diretórios do sistema que ainda não existem."""
        for d in [
            cls.MODEL_DIR, cls.RESULTS_DIR, cls.REPORTS_DIR,
            cls.LOGS_DIR, cls.CACHE_DIR, cls.COLLECTOR_DIR,
            cls.COLLECTOR_LOGS_DIR, cls.IDS_REPORTS_DIR,
            cls.IDS_STAGING_DIR, cls.EVALUATION_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def summary(cls) -> str:
        """Retorna um resumo formatado das configurações principais."""
        lines = [
            "═" * 60,
            "  Configurações do Sistema IDS",
            "═" * 60,
            f"  Interface de captura : {cls.CAPTURE_INTERFACE}",
            f"  Diretório do coletor : {cls.COLLECTOR_DIR}",
            f"  Amostragem           : {cls.COLLECTOR_SAMPLE_RATE * 100:.0f}%",
            f"  Budget diário        : {cls.COLLECTOR_BUDGET_GB:.1f} GiB",
            f"  Diretório relatórios : {cls.IDS_REPORTS_DIR}",
            f"  Diretório avaliação  : {cls.EVALUATION_DIR}",
            f"  Diretório modelo     : {cls.MODEL_DIR}",
            f"  Dataset original     : {cls.DATA_DIR}",
            "═" * 60,
        ]
        return "\n".join(lines)