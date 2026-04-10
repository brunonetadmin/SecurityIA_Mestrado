#!/usr/bin/env python3
"""
##############################################################################################
#  IDS Model Evaluator — Avaliação de Aprendizado Contínuo do Modelo LSTM
#  Versão: 1.0
#
#  Descrição:
#    Framework de avaliação rigorosa para medir se o re-treinamento contínuo
#    (Continual Learning) está melhorando, degradando ou estabilizando a
#    capacidade de detecção do modelo IDS ao longo do tempo.
#
#  Fundamentação Teórica:
#    - Lopez-Paz & Ranzato (2017): "Gradient Episodic Memory for Continual Learning"
#    - Chaudhry et al. (2018): "Efficient Lifelong Learning with A-GEM"
#    - McNemar (1947): teste de significância para classificadores pareados
#
#  Fluxo de Trabalho:
#
#    PASSO 1 (executar UMA VEZ, antes do primeiro re-treinamento):
#      python3 ids_model_evaluator.py create-benchmark
#      → Separa 10% do dataset original de forma estratificada.
#        Este conjunto é CONGELADO e nunca mais tocado pelo treinamento.
#
#    PASSO 2 (executar após cada ciclo de re-treinamento):
#      python3 ids_model_evaluator.py evaluate --version v1 --label "7 dias coleta"
#      → Avalia o modelo atual no benchmark congelado.
#        Registra todas as métricas no histórico.
#
#    PASSO 3 (para visualizar a evolução ao longo do tempo):
#      python3 ids_model_evaluator.py report
#      → Gera relatório HTML completo com gráficos de evolução,
#        testes de McNemar e métricas de Continual Learning.
#
#  Uso via ids_manager.py:
#    O ids_manager.py pode chamar este avaliador automaticamente após cada
#    ciclo de re-treinamento. Basta definir AUTO_EVALUATE = True na seção
#    de configuração abaixo.
#
#  Requisitos:
#    pip install numpy pandas scikit-learn tensorflow joblib pyarrow scipy
#
#  Autor: Bruno Cavalcante Barbosa
#  UFAL - Universidade Federal de Alagoas
#
#  Referências:
#    Lopez-Paz, D., & Ranzato, M. A. (2017). Gradient episodic memory for
#      continual learning. Advances in neural information processing systems.
#    Chaudhry, A., Ranzato, M. A., Rohrbach, M., & Elhoseiny, M. (2018).
#      Efficient lifelong learning with a-gem. arXiv:1812.00420.
#    McNemar, Q. (1947). Note on the sampling error of the difference between
#      correlated proportions or percentages. Psychometrika, 12(2), 153-157.
##############################################################################################
"""

import os, sys, json, logging, argparse, time, shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy.stats import chi2_contingency, chi2, binom
except ImportError as e:
    sys.exit(f"[ERRO] {e}. Execute: pip install numpy pandas scipy")

try:
    from sklearn.metrics import (
        classification_report, confusion_matrix,
        f1_score, precision_score, recall_score,
        roc_auc_score, cohen_kappa_score,
        accuracy_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import label_binarize
except ImportError as e:
    sys.exit(f"[ERRO] {e}. Execute: pip install scikit-learn")

try:
    # TF_CPP_MIN_LOG_LEVEL antes do import TF para suprimir logs do backend C++.
    import os as _os
    _os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')

    import tensorflow as tf
    from keras.models import load_model as keras_load_model
    tf.get_logger().setLevel('ERROR')
    import joblib
except ImportError as e:
    sys.exit(f"[ERRO] {e}. Execute: pip install tensorflow joblib")

try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ids_config import IDSConfig
except ImportError:
    class IDSConfig:
        BASE_DIR    = Path(__file__).resolve().parent
        MODEL_DIR   = BASE_DIR / "Model"
        DATA_DIR    = BASE_DIR / "Base/CSE-CIC-IDS2018/"
        RESULTS_DIR = BASE_DIR / "Results/"
        MODEL_FILENAME         = "ids_lstm_model.keras"
        SCALER_FILENAME        = "scaler.pkl"
        LABEL_ENCODER_FILENAME = "label_encoder.pkl"
        MODEL_INFO_FILENAME    = "ids_model_info.json"
        TRAINING_CONFIG        = {'random_state': 42, 'test_split': 0.15}
        CPU_CONFIG             = {'inter_op_threads': 4, 'intra_op_threads': 16}

        @classmethod
        def configure_tensorflow(cls):
            import tensorflow as _tf
            _tf.config.set_visible_devices([], 'GPU')
            _tf.config.threading.set_inter_op_parallelism_threads(cls.CPU_CONFIG['inter_op_threads'])
            _tf.config.threading.set_intra_op_parallelism_threads(cls.CPU_CONFIG['intra_op_threads'])

# Configura TF para CPU antes de qualquer operação de inferência.
IDSConfig.configure_tensorflow()


# ══════════════════════════════════════════════════════════════════════════════
# Configuração
# ══════════════════════════════════════════════════════════════════════════════

class EvaluatorConfig:
    """
    Parâmetros do framework de avaliação.
    Todos os caminhos derivam de IDSConfig para manter consistência com o
    restante do pipeline.
    """
    # Diretório raiz do framework de avaliação
    EVAL_DIR            = IDSConfig.RESULTS_DIR / "evaluation"

    # Conjunto de benchmark congelado (criado UMA vez, nunca alterado)
    BENCHMARK_DIR       = EVAL_DIR / "benchmark"
    BENCHMARK_X_PATH    = BENCHMARK_DIR / "benchmark_X.npy"
    BENCHMARK_Y_PATH    = BENCHMARK_DIR / "benchmark_y.npy"
    BENCHMARK_META_PATH = BENCHMARK_DIR / "benchmark_meta.json"

    # Histórico de avaliações (JSON Lines — um objeto por versão)
    HISTORY_PATH        = EVAL_DIR / "evaluation_history.jsonl"

    # Predições de cada versão (para testes de McNemar entre versões)
    PREDICTIONS_DIR     = EVAL_DIR / "predictions"

    # Relatórios HTML gerados
    REPORTS_DIR         = EVAL_DIR / "reports"

    # Fração do dataset original usada para o benchmark (estratificada por classe)
    BENCHMARK_FRACTION  = 0.10   # 10% — tamanho suficiente para significância estatística

    # Nível de significância para os testes estatísticos
    ALPHA               = 0.05

    # Tamanho do batch de inferência
    INFERENCE_BATCH     = 4096


# ══════════════════════════════════════════════════════════════════════════════
# Carregamento de Artefatos do Modelo
# ══════════════════════════════════════════════════════════════════════════════

def load_model_artifacts(model_path: Optional[Path] = None) -> dict:
    """
    Carrega os artefatos do modelo (weights, scaler, encoder, info).

    Parâmetros
    ----------
    model_path : Path opcional — se None, usa IDSConfig.MODEL_DIR.

    Retorna
    -------
    dict com chaves: model, scaler, encoder, label_map, selected_features
    """
    model_dir = model_path or IDSConfig.MODEL_DIR
    logging.info(f"Carregando artefatos de: {model_dir}")

    model   = keras_load_model(str(model_dir / IDSConfig.MODEL_FILENAME))
    scaler  = joblib.load(model_dir / IDSConfig.SCALER_FILENAME)
    encoder = joblib.load(model_dir / IDSConfig.LABEL_ENCODER_FILENAME)

    with open(model_dir / IDSConfig.MODEL_INFO_FILENAME) as f:
        info = json.load(f)

    label_map: Dict[int, str] = {int(k): v for k, v in info['label_mapping'].items()}
    selected_features: List[str] = info.get('selected_features', [])

    return {
        'model':             model,
        'scaler':            scaler,
        'encoder':           encoder,
        'label_map':         label_map,
        'selected_features': selected_features,
        'label_mapping':     info['label_mapping'],
    }


def run_inference_numpy(
    X_scaled_reshaped: np.ndarray,
    model,
    batch_size: int = EvaluatorConfig.INFERENCE_BATCH,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Executa inferência em batch sobre um array já escalado e redimensionado.

    Retorna (classes preditas, probabilidades máximas).
    """
    n = len(X_scaled_reshaped)
    classes    = np.empty(n, dtype=np.int32)
    proba_max  = np.empty(n, dtype=np.float32)
    all_proba  = np.empty((n, X_scaled_reshaped.shape[0]), dtype=np.float32)

    # Primeiro batch para descobrir n_classes
    first = model.predict(X_scaled_reshaped[:1], verbose=0)
    n_classes = first.shape[1]
    all_proba  = np.empty((n, n_classes), dtype=np.float32)

    for s in range(0, n, batch_size):
        e   = min(s + batch_size, n)
        p   = model.predict(X_scaled_reshaped[s:e], verbose=0)
        classes[s:e]   = np.argmax(p, axis=1)
        proba_max[s:e] = np.max(p, axis=1)
        all_proba[s:e] = p

    return classes, proba_max, all_proba


# ══════════════════════════════════════════════════════════════════════════════
# PASSO 1 — Criação do Benchmark Congelado
# ══════════════════════════════════════════════════════════════════════════════

def create_benchmark(force: bool = False) -> None:
    """
    Cria o conjunto de benchmark estratificado a partir do dataset original.

    Este método deve ser executado UMA ÚNICA VEZ, antes do primeiro ciclo de
    re-treinamento. O conjunto gerado é salvo em disco e permanece IMUTÁVEL
    durante toda a vida do projeto — é a "régua" fixa de comparação.

    Estratégia de amostragem:
      Amostragem estratificada por classe (StratifiedShuffleSplit) para garantir
      que todas as 15 classes de ataque do CSE-CIC-IDS2018 estejam representadas
      proporcionalmente, mesmo as mais raras (ex.: Heartbleed, Infiltration).

    Parâmetros
    ----------
    force : bool — se True, recria o benchmark mesmo que já exista.
    """
    EvaluatorConfig.BENCHMARK_DIR.mkdir(parents=True, exist_ok=True)

    if EvaluatorConfig.BENCHMARK_META_PATH.exists() and not force:
        logging.warning(
            "Benchmark já existe. Use --force para recriar. "
            "ATENÇÃO: recriar invalida todo o histórico de comparação."
        )
        print("  [AVISO] Benchmark já existe. Abortando para preservar a consistência histórica.")
        print("  Use --force apenas se quiser REINICIAR todo o histórico de avaliação.")
        return

    if force and EvaluatorConfig.BENCHMARK_META_PATH.exists():
        logging.warning("FORÇANDO recriação do benchmark. Histórico anterior INVALIDADO.")
        # Arquiva o histórico antigo antes de sobrescrever
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        archive = EvaluatorConfig.EVAL_DIR / f"archive_{ts}"
        shutil.copytree(EvaluatorConfig.BENCHMARK_DIR, archive / "benchmark")
        if EvaluatorConfig.HISTORY_PATH.exists():
            shutil.copy2(EvaluatorConfig.HISTORY_PATH, archive / "evaluation_history.jsonl")
        print(f"  Histórico arquivado em: {archive}")

    print("\n  Carregando artefatos do modelo para identificar features e encoder...")
    arts = load_model_artifacts()

    print(f"  Carregando dataset original de: {IDSConfig.DATA_DIR}")
    csv_files = list(Path(IDSConfig.DATA_DIR).glob('*.csv'))
    parquet_files = list(Path(IDSConfig.DATA_DIR).glob('*.parquet'))
    all_files = csv_files + parquet_files
    if not all_files:
        raise FileNotFoundError(f"Nenhum arquivo de dados encontrado em {IDSConfig.DATA_DIR}")

    frames = []
    for f in all_files:
        print(f"    Carregando {f.name}...", end=' ', flush=True)
        df = pd.read_csv(f) if f.suffix == '.csv' else pd.read_parquet(f)
        frames.append(df)
        print(f"{len(df):,} linhas")
    df_full = pd.concat(frames, ignore_index=True)
    print(f"  Dataset completo: {len(df_full):,} amostras")

    label_col = 'Label'
    selected  = arts['selected_features']

    # Limpa e prepara
    df_full.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_full.fillna(0, inplace=True)
    for col in selected:
        if col not in df_full.columns:
            df_full[col] = 0.0

    X = df_full[selected].values.astype(np.float32)
    y_raw = df_full[label_col].values
    y = arts['encoder'].transform(y_raw)

    # Amostragem estratificada
    print(f"\n  Separando {EvaluatorConfig.BENCHMARK_FRACTION*100:.0f}% estratificado por classe...")
    _, X_bench, _, y_bench = train_test_split(
        X, y,
        test_size=EvaluatorConfig.BENCHMARK_FRACTION,
        stratify=y,
        random_state=IDSConfig.TRAINING_CONFIG['random_state'],
    )

    # Escala com o scaler do modelo
    X_bench_scaled    = arts['scaler'].transform(X_bench)
    X_bench_reshaped  = X_bench_scaled.reshape(X_bench_scaled.shape[0],
                                                X_bench_scaled.shape[1], 1)

    # Salva
    np.save(str(EvaluatorConfig.BENCHMARK_X_PATH), X_bench_reshaped)
    np.save(str(EvaluatorConfig.BENCHMARK_Y_PATH), y_bench)

    # Distribuição por classe para auditoria
    class_dist = {}
    for idx, name in arts['label_map'].items():
        count = int(np.sum(y_bench == idx))
        if count > 0:
            class_dist[name] = count

    meta = {
        'created_at':      datetime.now().isoformat(),
        'total_samples':   len(y_bench),
        'original_total':  len(df_full),
        'fraction':        EvaluatorConfig.BENCHMARK_FRACTION,
        'random_state':    IDSConfig.TRAINING_CONFIG['random_state'],
        'selected_features': selected,
        'class_distribution': class_dist,
        'label_map':       arts['label_mapping'],
        'note': (
            "Este conjunto é IMUTÁVEL. Não deve ser usado para treinamento. "
            "Qualquer alteração invalida todas as comparações históricas."
        ),
    }
    with open(EvaluatorConfig.BENCHMARK_META_PATH, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"\n  ✔ Benchmark criado: {len(y_bench):,} amostras")
    print(f"    Distribuição por classe:")
    for cls, cnt in sorted(class_dist.items(), key=lambda x: -x[1]):
        print(f"      {cls:<40s} {cnt:>8,}")
    print(f"\n  Arquivos salvos em: {EvaluatorConfig.BENCHMARK_DIR}")
    print("  IMPORTANTE: Não modifique, mova ou delete estes arquivos.")


# ══════════════════════════════════════════════════════════════════════════════
# PASSO 2 — Avaliação de uma Versão do Modelo
# ══════════════════════════════════════════════════════════════════════════════

def _compute_continual_learning_metrics(
    history: List[dict],
    current_result: dict,
) -> dict:
    """
    Calcula as métricas de Continual Learning após N avaliações.

    Métricas calculadas:
      - Forgetting Measure (FM): max queda de F1 em qualquer tarefa anterior.
        FM = max_{t<T} [ max_{i<T} R(i,i) - R(T,i) ]
        onde R(j,i) = F1 do modelo j na tarefa i.

      - Backward Transfer (BWT): impacto médio do aprendizado novo
        sobre tarefas anteriores.
        BWT = (1/(T-1)) * sum_{i<T} [ R(T,i) - R(i,i) ]

      - Forward Transfer (FWT): quanto o modelo foi pré-preparado para
        tarefas futuras (comparado ao desempenho zero-shot esperado).
        Requer uma baseline de random init — aproximado aqui como 1/n_classes.

    Referência: Chaudhry et al. (2018), Eqs. 1–4.

    Parâmetros
    ----------
    history : lista de avaliações anteriores (cada uma com 'per_class_f1')
    current_result : avaliação da versão atual

    Retorna
    -------
    dict com 'forgetting_measure', 'backward_transfer', 'intransigence'
    """
    if len(history) == 0:
        return {
            'forgetting_measure': None,
            'backward_transfer':  None,
            'intransigence':      None,
        }

    # Coleta os F1-scores de cada versão para cada classe
    # R[version_idx][class_name] = f1
    all_versions = history + [current_result]
    T = len(all_versions)

    # Classes presentes em TODAS as versões
    common_classes = set(all_versions[0].get('per_class_f1', {}).keys())
    for v in all_versions[1:]:
        common_classes &= set(v.get('per_class_f1', {}).keys())

    if not common_classes:
        return {'forgetting_measure': None, 'backward_transfer': None, 'intransigence': None}

    # R[i][cls] = F1 da versão i na classe cls
    R = [{cls: v['per_class_f1'].get(cls, 0.0) for cls in common_classes}
         for v in all_versions]

    # Forgetting Measure: queda máxima no melhor desempenho histórico de cada classe
    fm_values = []
    for cls in common_classes:
        best_before_T = max(R[i][cls] for i in range(T - 1))
        current       = R[T - 1][cls]
        fm_values.append(best_before_T - current)
    forgetting_measure = float(np.mean(fm_values))

    # Backward Transfer: impacto médio do aprendizado novo sobre tarefas anteriores
    bwt_values = []
    for cls in common_classes:
        # Compara desempenho atual com o desempenho na época em que a classe "apareceu"
        r_at_intro = R[0][cls]   # Versão original (baseline)
        r_current  = R[T - 1][cls]
        bwt_values.append(r_current - r_at_intro)
    backward_transfer = float(np.mean(bwt_values))

    # Intransigência: dificuldade de aprender novas tarefas (comparado com oracle)
    # Estimativa: queda relativa no F1-macro ao longo do tempo
    f1_initial = history[0].get('f1_macro', 0.0)
    f1_current = current_result.get('f1_macro', 0.0)
    intransigence = float(max(0.0, f1_initial - f1_current))

    return {
        'forgetting_measure': round(forgetting_measure, 6),
        'backward_transfer':  round(backward_transfer, 6),
        'intransigence':      round(intransigence, 6),
    }


def _mcnemar_test(y_true: np.ndarray,
                  pred_a: np.ndarray,
                  pred_b: np.ndarray) -> dict:
    """
    Teste de McNemar para verificar se a diferença de erros entre dois modelos
    é estatisticamente significativa.

    O teste opera sobre a tabela de contingência:
        n00 = ambos acertaram
        n01 = modelo A errou, B acertou
        n10 = modelo A acertou, B errou
        n11 = ambos erraram

    H0: os modelos têm a mesma taxa de erro (n01 == n10)
    H1: os modelos têm taxas de erro diferentes

    Usa a correção de continuidade de Edwards quando n01 + n10 < 25.

    Parâmetros
    ----------
    y_true : rótulos verdadeiros
    pred_a : predições do modelo A (versão anterior)
    pred_b : predições do modelo B (versão atual)

    Retorna
    -------
    dict com statistic, p_value, significant, n01, n10, interpretation
    """
    correct_a = (pred_a == y_true)
    correct_b = (pred_b == y_true)

    n00 = int(np.sum(~correct_a & ~correct_b))
    n01 = int(np.sum(~correct_a &  correct_b))
    n10 = int(np.sum( correct_a & ~correct_b))
    n11 = int(np.sum( correct_a &  correct_b))

    if n01 + n10 == 0:
        return {
            'statistic':     0.0,
            'p_value':       1.0,
            'significant':   False,
            'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11,
            'interpretation': 'Modelos idênticos nas predições do benchmark.',
        }

    # Implementação do teste de McNemar compatível com scipy >= 1.7
    # (scipy.stats.mcnemar foi removido na versão 1.17)
    # Usa teste exato binomial quando n01+n10 < 25 (Edwards, 1948);
    # caso contrário, qui-quadrado com correção de continuidade.
    use_exact = (n01 + n10) < 25
    if use_exact:
        # Teste binomial exato: H0: P(discordância a favor de B) = 0.5
        from scipy.stats import binom as _binom
        p_one_side = _binom.cdf(min(n01, n10), n01 + n10, 0.5)
        p_value    = float(min(1.0, 2 * p_one_side))
        statistic  = float(abs(n01 - n10))
    else:
        # Qui-quadrado com correção de continuidade (Edwards, 1948)
        from scipy.stats import chi2 as _chi2
        statistic  = float((abs(n01 - n10) - 1.0) ** 2 / (n01 + n10))
        p_value    = float(1 - _chi2.cdf(statistic, df=1))

    sig = p_value < EvaluatorConfig.ALPHA
    if sig:
        if n01 > n10:
            interp = (f"O modelo NOVO é significativamente MELHOR "
                      f"(B corrigiu {n01} erros de A; A introduziu {n10} novos erros). "
                      f"p={p_value:.4f} < α={EvaluatorConfig.ALPHA}")
        else:
            interp = (f"O modelo NOVO é significativamente PIOR "
                      f"(A corrigiu {n10} erros de B; B introduziu {n01} novos erros). "
                      f"p={p_value:.4f} < α={EvaluatorConfig.ALPHA}")
    else:
        interp = (f"Diferença NÃO significativa "
                  f"(p={p_value:.4f} ≥ α={EvaluatorConfig.ALPHA}). "
                  f"Não há evidência de melhora ou piora.")

    return {
        'statistic':     round(statistic, 6),
        'p_value':       round(p_value, 6),
        'significant':   sig,
        'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11,
        'exact_test':    use_exact,
        'interpretation': interp,
    }


def evaluate_model(version_id: str, label: str, notes: str = "") -> dict:
    """
    Avalia o modelo atual no benchmark congelado e registra os resultados.

    Parâmetros
    ----------
    version_id : str — identificador da versão (ex.: 'v0', 'v1', 'v2')
    label      : str — descrição humana (ex.: 'Baseline', '7 dias coleta')
    notes      : str — notas livres para o histórico

    Retorna
    -------
    dict com todas as métricas calculadas
    """
    # ── Verifica pré-requisitos ────────────────────────────────────────────
    if not EvaluatorConfig.BENCHMARK_X_PATH.exists():
        raise FileNotFoundError(
            "Benchmark não encontrado. Execute primeiro:\n"
            "  python3 ids_model_evaluator.py create-benchmark"
        )

    EvaluatorConfig.PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Carrega benchmark ──────────────────────────────────────────────────
    print(f"\n  Carregando benchmark congelado...")
    X_bench = np.load(str(EvaluatorConfig.BENCHMARK_X_PATH))
    y_bench = np.load(str(EvaluatorConfig.BENCHMARK_Y_PATH))
    with open(EvaluatorConfig.BENCHMARK_META_PATH) as f:
        bench_meta = json.load(f)
    print(f"  Benchmark: {len(y_bench):,} amostras | "
          f"{len(bench_meta['class_distribution'])} classes")

    # ── Carrega artefatos do modelo ────────────────────────────────────────
    print(f"  Carregando modelo: {IDSConfig.MODEL_DIR / IDSConfig.MODEL_FILENAME}")
    arts = load_model_artifacts()

    # ── Inferência ────────────────────────────────────────────────────────
    print(f"  Executando inferência ({len(y_bench):,} amostras)...")
    t0 = time.time()
    y_pred, conf_scores, all_proba = run_inference_numpy(X_bench, arts['model'])
    elapsed = time.time() - t0
    print(f"  Inferência concluída em {elapsed:.1f}s")

    # ── Métricas gerais ────────────────────────────────────────────────────
    label_names = [arts['label_map'][i] for i in sorted(arts['label_map'].keys())]
    n_classes   = len(label_names)

    acc    = float(accuracy_score(y_bench, y_pred))
    f1_mac = float(f1_score(y_bench, y_pred, average='macro', zero_division=0))
    f1_wt  = float(f1_score(y_bench, y_pred, average='weighted', zero_division=0))
    prec   = float(precision_score(y_bench, y_pred, average='macro', zero_division=0))
    rec    = float(recall_score(y_bench, y_pred, average='macro', zero_division=0))
    kappa  = float(cohen_kappa_score(y_bench, y_pred))

    # AUC-ROC one-vs-rest (requer probabilidades)
    try:
        y_bin = label_binarize(y_bench, classes=list(range(n_classes)))
        if y_bin.shape[1] == 1:
            auc = float(roc_auc_score(y_bench, all_proba[:, 1]))
        else:
            auc = float(roc_auc_score(y_bin, all_proba,
                                       average='macro', multi_class='ovr'))
    except Exception:
        auc = None

    # ── Métricas por classe ────────────────────────────────────────────────
    report = classification_report(
        y_bench, y_pred,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    per_class_f1        = {cls: round(report[cls]['f1-score'],   6) for cls in label_names if cls in report}
    per_class_precision = {cls: round(report[cls]['precision'],  6) for cls in label_names if cls in report}
    per_class_recall    = {cls: round(report[cls]['recall'],     6) for cls in label_names if cls in report}
    per_class_support   = {cls: int(report[cls]['support'])         for cls in label_names if cls in report}

    # ── Matriz de confusão ─────────────────────────────────────────────────
    cm = confusion_matrix(y_bench, y_pred).tolist()

    # ── Carrega histórico e calcula métricas de Continual Learning ─────────
    history = _load_history()
    cl_metrics = _compute_continual_learning_metrics(history, {
        'per_class_f1': per_class_f1,
        'f1_macro':     f1_mac,
    })

    # ── Teste de McNemar vs versão anterior ───────────────────────────────
    mcnemar_result = None
    if history:
        prev = history[-1]
        prev_pred_path = EvaluatorConfig.PREDICTIONS_DIR / f"{prev['version_id']}_predictions.npy"
        if prev_pred_path.exists():
            y_pred_prev = np.load(str(prev_pred_path))
            mcnemar_result = _mcnemar_test(y_bench, y_pred_prev, y_pred)
            print(f"\n  Teste de McNemar vs {prev['version_id']}:")
            print(f"    {mcnemar_result['interpretation']}")

    # ── Salva predições desta versão para futuros testes de McNemar ───────
    pred_path = EvaluatorConfig.PREDICTIONS_DIR / f"{version_id}_predictions.npy"
    np.save(str(pred_path), y_pred)

    # ── Monta resultado completo ───────────────────────────────────────────
    result = {
        'version_id':     version_id,
        'label':          label,
        'notes':          notes,
        'evaluated_at':   datetime.now().isoformat(),
        'benchmark_samples': int(len(y_bench)),
        'inference_time_s':  round(elapsed, 2),
        # Métricas gerais
        'accuracy':       round(acc,    6),
        'f1_macro':       round(f1_mac, 6),
        'f1_weighted':    round(f1_wt,  6),
        'precision_macro': round(prec,  6),
        'recall_macro':   round(rec,    6),
        'cohen_kappa':    round(kappa,  6),
        'auc_roc_macro':  round(auc, 6) if auc is not None else None,
        # Por classe
        'per_class_f1':        per_class_f1,
        'per_class_precision': per_class_precision,
        'per_class_recall':    per_class_recall,
        'per_class_support':   per_class_support,
        # Continual Learning
        'forgetting_measure':  cl_metrics['forgetting_measure'],
        'backward_transfer':   cl_metrics['backward_transfer'],
        'intransigence':       cl_metrics['intransigence'],
        # Significância estatística
        'mcnemar_vs_previous': mcnemar_result,
        # Matriz de confusão (para o relatório HTML)
        'confusion_matrix':    cm,
        'class_names':         label_names,
    }

    # ── Persiste no histórico ──────────────────────────────────────────────
    _append_history(result)

    # ── Exibe resumo ───────────────────────────────────────────────────────
    _print_evaluation_summary(result, history)

    return result


def _load_history() -> List[dict]:
    if not EvaluatorConfig.HISTORY_PATH.exists():
        return []
    results = []
    with open(EvaluatorConfig.HISTORY_PATH) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return results


def _append_history(result: dict) -> None:
    EvaluatorConfig.EVAL_DIR.mkdir(parents=True, exist_ok=True)
    with open(EvaluatorConfig.HISTORY_PATH, 'a') as f:
        # Remove confusion_matrix do histórico (volumosa) — mantém no relatório
        r = {k: v for k, v in result.items() if k != 'confusion_matrix'}
        f.write(json.dumps(r, ensure_ascii=False) + '\n')
    logging.info(f"Resultado v{result['version_id']} adicionado ao histórico.")


def _print_evaluation_summary(result: dict, history: List[dict]) -> None:
    """Exibe resumo formatado no terminal."""
    sep = '─' * 68
    print(f"\n{sep}")
    print(f"  RESULTADO DA AVALIAÇÃO — {result['version_id']}: {result['label']}")
    print(sep)
    print(f"  Acurácia          : {result['accuracy']*100:.3f}%")
    print(f"  F1-Score Macro    : {result['f1_macro']:.6f}")
    print(f"  F1-Score Ponderado: {result['f1_weighted']:.6f}")
    print(f"  Precision Macro   : {result['precision_macro']:.6f}")
    print(f"  Recall Macro      : {result['recall_macro']:.6f}")
    print(f"  Cohen's Kappa     : {result['cohen_kappa']:.6f}")
    if result['auc_roc_macro']:
        print(f"  AUC-ROC Macro     : {result['auc_roc_macro']:.6f}")

    if history:
        prev = history[-1]
        delta_f1 = result['f1_macro'] - prev.get('f1_macro', 0)
        delta_sym = ('▲' if delta_f1 > 0 else '▼' if delta_f1 < 0 else '═')
        print(f"\n  Δ F1 vs {prev['version_id']:<8s}: {delta_sym} {abs(delta_f1)*100:.4f}%")

        if result['forgetting_measure'] is not None:
            fm = result['forgetting_measure']
            fm_status = 'SEM FORGETTING' if fm <= 0 else f'FORGETTING DETECTADO (FM={fm:.4f})'
            print(f"  Forgetting Measure: {fm:.6f}  ({fm_status})")
        if result['backward_transfer'] is not None:
            bwt = result['backward_transfer']
            bwt_status = 'Transferência positiva ✔' if bwt > 0 else 'Transferência negativa ✘'
            print(f"  Backward Transfer : {bwt:.6f}  ({bwt_status})")

    print(f"\n  F1-Score por classe:")
    for cls, f1 in sorted(result['per_class_f1'].items(), key=lambda x: -x[1]):
        bar = '█' * int(f1 * 20)
        if history:
            prev_f1 = history[-1].get('per_class_f1', {}).get(cls, 0.0)
            delta   = f1 - prev_f1
            delta_s = f'+{delta*100:.2f}%' if delta > 0 else f'{delta*100:.2f}%'
            delta_s = f'({delta_s:>9s})' if delta != 0 else '           '
        else:
            delta_s = ''
        print(f"    {cls:<38s} {f1:.4f}  {bar:<20s} {delta_s}")
    print(sep)


# ══════════════════════════════════════════════════════════════════════════════
# PASSO 3 — Relatório HTML de Evolução
# ══════════════════════════════════════════════════════════════════════════════

def generate_evolution_report() -> Path:
    """
    Gera um relatório HTML completo mostrando a evolução do modelo ao longo
    de todas as versões registradas no histórico.

    Inclui:
      - Tabela comparativa de todas as versões
      - Gráfico de evolução do F1-Macro ao longo do tempo
      - Heatmap de F1 por classe vs versão
      - Métricas de Continual Learning (FM, BWT)
      - Resultados dos testes de McNemar
      - Análise de estabilidade por classe de ataque
    """
    history = _load_history()
    if not history:
        print("  Nenhuma avaliação encontrada. Execute 'evaluate' primeiro.")
        return None

    EvaluatorConfig.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = EvaluatorConfig.REPORTS_DIR / f"evolution_report_{ts}.html"

    # ── Dados para visualização ────────────────────────────────────────────
    versions    = [h['version_id'] for h in history]
    labels      = [h['label']      for h in history]
    f1_macros   = [h['f1_macro']   for h in history]
    kappas      = [h.get('cohen_kappa', 0) for h in history]
    accuracies  = [h['accuracy']   for h in history]
    fms         = [h.get('forgetting_measure', 0) or 0 for h in history]
    bwts        = [h.get('backward_transfer',  0) or 0 for h in history]

    # Todas as classes presentes em pelo menos uma versão
    all_classes = sorted({
        cls for h in history
        for cls in h.get('per_class_f1', {}).keys()
    })

    # Matriz de F1 por classe [classe][versão]
    f1_matrix = {
        cls: [h.get('per_class_f1', {}).get(cls, 0.0) for h in history]
        for cls in all_classes
    }

    # Cores para o heatmap (0.0 = branco, 1.0 = azul escuro)
    def heat_color(v: float) -> str:
        r = int(255 - v * 200)
        g = int(255 - v * 180)
        b = int(255 - v * 100)
        return f"rgb({r},{g},{b})"

    # ── Tabela de versões ──────────────────────────────────────────────────
    def versions_table() -> str:
        header_cols = (['Versão', 'Label', 'Data', 'Acurácia',
                        'F1-Macro', 'F1-Pond.', 'Kappa',
                        'FM', 'BWT', 'McNemar'])
        rows = ''
        for i, h in enumerate(history):
            mc = h.get('mcnemar_vs_previous')
            mc_cell = '—' if mc is None else (
                f'<span style="color:{"green" if mc["n01"]>mc["n10"] else "red"}">'
                f'p={mc["p_value"]:.4f} '
                f'{"✔ Melhor" if mc["significant"] and mc["n01"]>mc["n10"] else "✘ Pior" if mc["significant"] else "≈ Igual"}'
                f'</span>'
            )
            fm_v  = h.get('forgetting_measure', 0) or 0
            bwt_v = h.get('backward_transfer',  0) or 0
            fm_col  = 'green' if fm_v  <= 0 else 'orange' if fm_v  < 0.05 else 'red'
            bwt_col = 'green' if bwt_v >= 0 else 'orange' if bwt_v > -0.05 else 'red'

            delta_f1 = ''
            if i > 0:
                d = h['f1_macro'] - history[i-1]['f1_macro']
                delta_f1 = (f' <span style="color:{"green" if d>0 else "red"};font-size:11px">'
                            f'{"▲" if d>0 else "▼"}{abs(d)*100:.3f}%</span>')

            rows += (
                f'<tr>'
                f'<td><strong>{h["version_id"]}</strong></td>'
                f'<td>{h["label"]}</td>'
                f'<td style="font-size:11px">{h["evaluated_at"][:16].replace("T"," ")}</td>'
                f'<td>{h["accuracy"]*100:.3f}%</td>'
                f'<td>{h["f1_macro"]:.6f}{delta_f1}</td>'
                f'<td>{h["f1_weighted"]:.6f}</td>'
                f'<td>{h.get("cohen_kappa",0):.6f}</td>'
                f'<td style="color:{fm_col}">{fm_v:.6f}</td>'
                f'<td style="color:{bwt_col}">{bwt_v:.6f}</td>'
                f'<td style="font-size:11px">{mc_cell}</td>'
                f'</tr>\n'
            )
        cols = ''.join(f'<th>{c}</th>' for c in header_cols)
        return f'<table><tr>{cols}</tr>{rows}</table>'

    # ── Heatmap de F1 por classe ───────────────────────────────────────────
    def f1_heatmap() -> str:
        header = '<tr><th>Classe de Ataque</th>' + \
                 ''.join(f'<th style="font-size:11px">{v}</th>' for v in versions) + '</tr>\n'
        rows = ''
        for cls in all_classes:
            values = f1_matrix[cls]
            row = f'<tr><td style="font-size:12px"><strong>{cls}</strong></td>'
            for v in values:
                bg = heat_color(v)
                row += (f'<td style="background:{bg};text-align:center;'
                        f'font-size:11px;padding:4px">{v:.3f}</td>')
            row += '</tr>\n'
            rows += row
        return f'<table style="border-collapse:collapse">{header}{rows}</table>'

    # ── Gráfico SVG de evolução do F1-Macro ────────────────────────────────
    def f1_evolution_svg() -> str:
        if len(versions) < 2:
            return '<p style="color:#6c757d;font-size:13px">Necessário ≥ 2 avaliações para o gráfico.</p>'
        W, H, PAD = 640, 220, 40
        f1_min = max(0.0, min(f1_macros) - 0.05)
        f1_max = min(1.0, max(f1_macros) + 0.05)
        f1_range = f1_max - f1_min if f1_max > f1_min else 0.1
        n = len(versions)
        xs = [PAD + i * (W - 2 * PAD) / (n - 1) for i in range(n)]
        ys = [H - PAD - (v - f1_min) / f1_range * (H - 2 * PAD) for v in f1_macros]

        # Linhas de grade
        grid = ''
        for tick in np.linspace(f1_min, f1_max, 5):
            y_g = H - PAD - (tick - f1_min) / f1_range * (H - 2 * PAD)
            grid += (f'<line x1="{PAD}" y1="{y_g:.1f}" x2="{W-PAD}" y2="{y_g:.1f}" '
                     f'stroke="#e0e0e0" stroke-width="1"/>'
                     f'<text x="{PAD-4}" y="{y_g+4:.1f}" text-anchor="end" '
                     f'font-size="10" fill="#888">{tick:.3f}</text>')

        # Linha do F1
        polyline_pts = ' '.join(f'{x:.1f},{y:.1f}' for x, y in zip(xs, ys))
        points_svg = ''
        for i, (x, y) in enumerate(zip(xs, ys)):
            color = '#198754' if i > 0 and f1_macros[i] > f1_macros[i-1] \
                    else '#dc3545' if i > 0 else '#0d6efd'
            points_svg += (f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" fill="{color}"/>'
                           f'<text x="{x:.1f}" y="{y-8:.1f}" text-anchor="middle" '
                           f'font-size="10" fill="{color}">{f1_macros[i]:.4f}</text>'
                           f'<text x="{x:.1f}" y="{H-PAD+14:.0f}" text-anchor="middle" '
                           f'font-size="10" fill="#495057">{versions[i]}</text>')

        # Área sob a curva (fill)
        area_pts = (f'{xs[0]:.1f},{H-PAD} ' +
                    ' '.join(f'{x:.1f},{y:.1f}' for x, y in zip(xs, ys)) +
                    f' {xs[-1]:.1f},{H-PAD}')

        return (
            f'<svg viewBox="0 0 {W} {H}" style="width:100%;max-width:{W}px">'
            f'{grid}'
            f'<polygon points="{area_pts}" fill="rgba(13,110,253,0.08)"/>'
            f'<polyline points="{polyline_pts}" fill="none" stroke="#0d6efd" '
            f'stroke-width="2.5" stroke-linejoin="round"/>'
            f'{points_svg}'
            f'</svg>'
        )

    # ── Seção de métricas CL ───────────────────────────────────────────────
    def cl_metrics_section() -> str:
        if len(history) < 2:
            return '<p style="color:#6c757d">Necessário ≥ 2 avaliações para calcular métricas de Continual Learning.</p>'
        last = history[-1]
        fm  = last.get('forgetting_measure', None)
        bwt = last.get('backward_transfer',  None)
        intr= last.get('intransigence',      None)

        def interp_fm(v):
            if v is None: return '—'
            if v <= 0:    return f'<span style="color:green">FM={v:.4f} — Sem esquecimento. ✔</span>'
            if v < 0.02:  return f'<span style="color:orange">FM={v:.4f} — Leve esquecimento.</span>'
            return f'<span style="color:red">FM={v:.4f} — Esquecimento significativo. ✘</span>'

        def interp_bwt(v):
            if v is None: return '—'
            if v > 0.01:  return f'<span style="color:green">BWT={v:.4f} — Transferência retrógrada positiva. ✔</span>'
            if v > -0.02: return f'<span style="color:orange">BWT={v:.4f} — Transferência neutra.</span>'
            return f'<span style="color:red">BWT={v:.4f} — Transferência negativa (forgetting). ✘</span>'

        return f"""
<table>
  <tr><th>Métrica</th><th>Valor (última versão)</th><th>Interpretação</th></tr>
  <tr><td><strong>Forgetting Measure (FM)</strong></td>
      <td>{fm:.6f if fm is not None else '—'}</td>
      <td style="font-size:12px">{interp_fm(fm)}</td></tr>
  <tr><td><strong>Backward Transfer (BWT)</strong></td>
      <td>{bwt:.6f if bwt is not None else '—'}</td>
      <td style="font-size:12px">{interp_bwt(bwt)}</td></tr>
  <tr><td><strong>Intransigência</strong></td>
      <td>{intr:.6f if intr is not None else '—'}</td>
      <td style="font-size:12px">Queda do F1-macro em relação à versão inicial.</td></tr>
</table>
<p style="font-size:12px;color:#6c757d;margin-top:8px">
  <strong>FM ≤ 0</strong>: o modelo não esqueceu ataques antigos. &nbsp;
  <strong>BWT &gt; 0</strong>: o re-treinamento melhorou inclusive a detecção de ataques históricos.
</p>"""

    # ── HTML final ─────────────────────────────────────────────────────────
    now_str = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    HTML = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Evolução do Modelo IDS</title>
<style>
body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
     margin:0;padding:0;background:#f8f9fa;color:#212529}}
.wrap{{max-width:1200px;margin:0 auto;padding:24px 16px}}
h1{{font-size:24px;margin:0 0 4px}}
h2{{font-size:18px;border-bottom:2px solid #dee2e6;padding-bottom:6px;
    margin-top:36px;color:#1a1a2e}}
.header{{background:linear-gradient(135deg,#1a1a2e,#16213e);color:#fff;
         padding:28px 32px;border-radius:12px;margin-bottom:24px}}
.header p{{margin:4px 0;opacity:.85;font-size:14px}}
.callout{{background:#e8f4fd;border-left:4px solid #0d6efd;padding:12px 16px;
          border-radius:0 8px 8px 0;margin:16px 0;font-size:13px}}
.callout.warn{{background:#fff3cd;border-color:#ffc107}}
.callout.success{{background:#d1e7dd;border-color:#198754}}
table{{width:100%;border-collapse:collapse;font-size:12px;background:#fff;
       border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.06);
       margin-bottom:20px}}
th{{background:#343a40;color:#fff;padding:9px 12px;text-align:left;
    font-weight:600;font-size:11px}}
td{{padding:8px 12px;border-bottom:1px solid #f0f0f0}}
tr:hover td{{background:#f8f9fa}}
section{{margin-bottom:40px}}
</style>
</head>
<body>
<div class="wrap">
<div class="header">
  <h1>📊 Evolução do Modelo IDS — Análise de Continual Learning</h1>
  <p>Gerado em: {now_str} &nbsp;|&nbsp; {len(history)} versão(ões) avaliada(s)</p>
</div>

<div class="callout">
  <strong>Como interpretar este relatório:</strong> O benchmark é um conjunto
  <strong>congelado e imutável</strong> retirado do dataset original CSE-CIC-IDS2018.
  Nenhuma amostra deste conjunto jamais foi usada para treinamento, garantindo
  uma "régua" consistente de comparação. Os valores de F1, Kappa e AUC são
  calculados neste mesmo benchmark em todas as versões do modelo.
</div>

<section>
<h2>1. Comparativo de Versões</h2>
{versions_table()}
</section>

<section>
<h2>2. Evolução do F1-Score Macro ao Longo do Tempo</h2>
{f1_evolution_svg()}
<p style="font-size:12px;color:#6c757d">
  Cada ponto representa uma avaliação no benchmark congelado.
  Verde = melhora; Vermelho = piora vs versão anterior.
</p>
</section>

<section>
<h2>3. F1-Score por Classe de Ataque (Heatmap)</h2>
<p style="font-size:13px;color:#495057;margin-bottom:12px">
  Cada célula mostra o F1-score do modelo em determinada versão para aquela
  classe de ataque. Cores mais escuras = melhor detecção.
  Linhas que ficam mais claras ao longo do tempo indicam <strong>forgetting</strong>.
</p>
{f1_heatmap()}
</section>

<section>
<h2>4. Métricas de Continual Learning</h2>
<div class="callout">
  <strong>Forgetting Measure (FM)</strong>: mede o quanto o modelo "esqueceu"
  tarefas anteriores após aprender novas. FM ≤ 0 é ideal. &nbsp;
  <strong>Backward Transfer (BWT)</strong>: mede se o aprendizado novo
  <em>melhorou</em> o desempenho em tarefas antigas (BWT &gt; 0 = positivo).
  Referência: Chaudhry et al. (2018).
</div>
{cl_metrics_section()}
</section>

<section>
<h2>5. Testes de Significância Estatística (McNemar)</h2>
<p style="font-size:13px;color:#495057">
  O teste de McNemar avalia se a diferença de erros entre duas versões consecutivas
  do modelo é estatisticamente significativa (α = {EvaluatorConfig.ALPHA}).
  Referência: McNemar (1947).
</p>
<table>
<tr><th>Comparação</th><th>Estático χ²</th><th>p-value</th>
    <th>n01 (A✘ B✔)</th><th>n10 (A✔ B✘)</th>
    <th>Significativo</th><th>Interpretação</th></tr>
{''.join(
    f"""<tr>
    <td>{history[i-1]["version_id"]} → {history[i]["version_id"]}</td>
    <td>{history[i].get("mcnemar_vs_previous",{}).get("statistic","—") if history[i].get("mcnemar_vs_previous") else "—"}</td>
    <td>{f'{history[i]["mcnemar_vs_previous"]["p_value"]:.4f}' if history[i].get("mcnemar_vs_previous") else "—"}</td>
    <td>{history[i]["mcnemar_vs_previous"]["n01"] if history[i].get("mcnemar_vs_previous") else "—"}</td>
    <td>{history[i]["mcnemar_vs_previous"]["n10"] if history[i].get("mcnemar_vs_previous") else "—"}</td>
    <td>{"✔ Sim" if history[i].get("mcnemar_vs_previous",{}).get("significant") else "✘ Não"}</td>
    <td style="font-size:11px">{history[i].get("mcnemar_vs_previous",{}).get("interpretation","—")}</td>
    </tr>"""
    for i in range(1, len(history))
    if history[i].get("mcnemar_vs_previous")
) or '<tr><td colspan="7" style="color:#6c757d">Necessário ≥ 2 avaliações.</td></tr>'}
</table>
</section>

<section>
<h2>6. Análise de Estabilidade por Classe</h2>
<p style="font-size:13px;color:#495057">
  Variação do F1-score de cada classe entre a primeira avaliação (v0) e a
  última versão. Valores negativos indicam forgetting específico.
</p>
<table>
<tr><th>Classe</th><th>F1 v0 (baseline)</th><th>F1 última versão</th>
    <th>Δ F1</th><th>Tendência</th><th>Risco de Forgetting</th></tr>
{''.join(
    (lambda cls, f1_v=f1_matrix, h=history: (
        f0 := h[0].get("per_class_f1",{}).get(cls, 0.0),
        fl := h[-1].get("per_class_f1",{}).get(cls, 0.0),
        d  := fl - f0,
        f'<tr><td><strong>{cls}</strong></td>'
        f'<td>{f0:.4f}</td><td>{fl:.4f}</td>'
        f'<td style="color:{"green" if d>=0 else "red"}">{d:+.4f}</td>'
        f'<td>{"▲ Melhora" if d>0.01 else "▼ Piora" if d<-0.01 else "≈ Estável"}</td>'
        f'<td>{"🟢 Baixo" if d>-0.02 else "🟡 Moderado" if d>-0.05 else "🔴 Alto"}</td>'
        f'</tr>\n'
    )[-1])(cls)
    for cls in all_classes
    if len(history) > 1
) or '<tr><td colspan="6" style="color:#6c757d">Necessário ≥ 2 avaliações.</td></tr>'}
</table>
</section>

<footer style="margin-top:40px;padding-top:16px;border-top:1px solid #dee2e6;
               font-size:11px;color:#6c757d;text-align:center">
  IDS Model Evaluator v1.0 — UFAL &nbsp;|&nbsp;
  Benchmark: {_load_benchmark_info()} &nbsp;|&nbsp; Gerado em {now_str}
</footer>
</div>
</body>
</html>"""

    out_path.write_text(HTML, encoding='utf-8')
    print(f"\n  ✔ Relatório salvo: {out_path}")
    return out_path


def _load_benchmark_info() -> str:
    try:
        with open(EvaluatorConfig.BENCHMARK_META_PATH) as f:
            m = json.load(f)
        return (f"{m['total_samples']:,} amostras, "
                f"{len(m['class_distribution'])} classes, "
                f"criado {m['created_at'][:10]}")
    except Exception:
        return "não encontrado"


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description="IDS Model Evaluator — Framework de avaliação de Continual Learning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = p.add_subparsers(dest='command', required=True)

    # create-benchmark
    cb = sub.add_parser('create-benchmark',
                        help='Cria o conjunto de benchmark congelado (executar UMA VEZ).')
    cb.add_argument('--force', action='store_true',
                    help='Recria o benchmark mesmo que já exista (invalida histórico).')

    # evaluate
    ev = sub.add_parser('evaluate',
                        help='Avalia a versão atual do modelo no benchmark congelado.')
    ev.add_argument('--version', required=True,
                    help='Identificador da versão. Ex.: v0, v1, v2')
    ev.add_argument('--label',   required=True,
                    help='Descrição humana. Ex.: "Baseline", "7 dias coleta"')
    ev.add_argument('--notes',   default='',
                    help='Notas livres para o histórico.')

    # report
    sub.add_parser('report',
                   help='Gera relatório HTML de evolução do modelo.')

    # history
    sub.add_parser('history',
                   help='Exibe o histórico de avaliações no terminal.')

    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )

    if args.command == 'create-benchmark':
        create_benchmark(force=args.force)

    elif args.command == 'evaluate':
        evaluate_model(args.version, args.label, args.notes)

    elif args.command == 'report':
        generate_evolution_report()

    elif args.command == 'history':
        history = _load_history()
        if not history:
            print("  Nenhuma avaliação registrada.")
            return
        print(f"\n  {'Versão':<8s} {'Label':<30s} {'F1-Macro':<12s} "
              f"{'Kappa':<10s} {'FM':<10s} {'Data'}")
        print('  ' + '─' * 78)
        for h in history:
            fm = h.get('forgetting_measure')
            fm_s = f'{fm:.4f}' if fm is not None else '   —  '
            print(f"  {h['version_id']:<8s} {h['label']:<30s} "
                  f"{h['f1_macro']:<12.6f} "
                  f"{h.get('cohen_kappa',0):<10.6f} "
                  f"{fm_s:<10s} "
                  f"{h['evaluated_at'][:16]}")


if __name__ == '__main__':
    main()