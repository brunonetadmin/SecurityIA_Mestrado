#!/usr/bin/env python3
"""
ids_learn.py — Aprendizado e Re-Treinamento do Modelo IDS

Responsabilidades:
  - Preparação do dataset anotado a partir dos resultados de análise.
  - Execução do fine-tuning do modelo LSTM via 1_ids_training_script.py.
  - Carregamento dinâmico do módulo ids_model_evaluator.py.

Importado por: ids_manager.py

Autor: Bruno Cavalcante Barbosa — UFAL
"""

import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import warnings
warnings.filterwarnings('ignore')

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError(f"Dependência ausente: {exc}. Execute: pip install pandas") from exc

from ids_config import IDSConfig
from ids_system import CONF_HIGH


# ──────────────────────────────────────────────────────────────────────────────
# Preparação do dataset de re-treinamento
# ──────────────────────────────────────────────────────────────────────────────

def _prepare_staging_dataset(results: List[Dict]) -> Optional[Path]:
    """
    Extrai fluxos com confiança >= CONF_HIGH dos resultados de análise,
    usa o rótulo predito como Label e salva um Parquet no staging.

    Apenas predições de alta confiança são usadas para evitar ruído
    no processo de fine-tuning (pseudo-label approach).

    Retorna o caminho do Parquet gerado, ou None se não houver dados válidos.
    """
    frames = []
    for result in results:
        df = result.get('df', pd.DataFrame())
        if df.empty:
            continue
        high_conf = df[df['_conf'] >= CONF_HIGH].copy()
        if high_conf.empty:
            continue
        high_conf['Label'] = high_conf['_label']
        drop_cols = [c for c in ('_label', '_conf') if c in high_conf.columns]
        high_conf.drop(columns=drop_cols, inplace=True)
        frames.append(high_conf)

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined = combined[combined['Label'].notna() & (combined['Label'] != 'Unknown')]

    if combined.empty:
        return None

    staging = IDSConfig.IDS_STAGING_DIR
    staging.mkdir(parents=True, exist_ok=True)
    out_path = staging / f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    combined.to_parquet(out_path, compression='snappy', index=False)
    return out_path


# ──────────────────────────────────────────────────────────────────────────────
# Fine-tuning
# ──────────────────────────────────────────────────────────────────────────────

def run_retraining(
    results: List[Dict],
    on_step: Optional[Callable[[int, str], None]] = None,
) -> bool:
    """
    Executa o ciclo completo de fine-tuning em três etapas:
      1. Prepara o dataset anotado com pseudo-labels de alta confiança.
      2. Configura IDSConfig para apontar ao staging e ativa force_retrain.
      3. Importa e executa 1_ids_training_script.py dinamicamente.

    Parâmetros
    ----------
    results : Resultados de analyze_file() — deve conter o campo 'df'.
    on_step : Callback(step: int, message: str) para feedback de progresso.

    Retorna True se o fine-tuning for concluído sem exceção.
    """
    def _step(n: int, msg: str) -> None:
        if on_step:
            on_step(n, msg)

    _step(1, "Preparando dataset anotado...")
    dataset_path = _prepare_staging_dataset(results)

    if dataset_path is None:
        _step(1, "Nenhum dado de alta confiança disponível. Re-treinamento cancelado.")
        return False

    n_rows = len(pd.read_parquet(dataset_path))
    _step(2, f"{n_rows:,} fluxos anotados — {dataset_path.name}")

    training_script = Path(__file__).resolve().parent / '1_ids_training_script.py'
    if not training_script.exists():
        _step(3, f"Script de treinamento não encontrado: {training_script}")
        _step(3, f"Dataset salvo. Execute manualmente: python3 1_ids_training_script.py")
        return False

    _step(3, "Iniciando fine-tuning...")

    # Aponta DATA_DIR para o staging com os dados novos anotados
    IDSConfig.DATA_DIR = IDSConfig.IDS_STAGING_DIR
    IDSConfig.TRAINING_CONFIG['force_retrain'] = True
    IDSConfig.FINE_TUNING_CONFIG['enable']     = True

    try:
        spec   = importlib.util.spec_from_file_location('training_script', training_script)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        module.main()
        _step(3, "Fine-tuning concluído com sucesso.")
        return True
    except Exception as exc:
        _step(3, f"Erro no re-treinamento: {exc}")
        return False


# ──────────────────────────────────────────────────────────────────────────────
# Integração com o avaliador
# ──────────────────────────────────────────────────────────────────────────────

def load_evaluator() -> Optional[Any]:
    """
    Importa ids_model_evaluator.py dinamicamente do mesmo diretório.

    Retorna o módulo em caso de sucesso, ou None com mensagem de erro.
    O carregamento dinâmico mantém o avaliador como módulo independente,
    sem acoplamento estático que criaria dependência circular com TensorFlow.
    """
    ev_path = Path(__file__).resolve().parent / 'ids_model_evaluator.py'
    if not ev_path.exists():
        print(f"  [ERRO] ids_model_evaluator.py não encontrado em {ev_path}")
        return None

    spec   = importlib.util.spec_from_file_location('ids_model_evaluator', ev_path)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        print(f"  [ERRO] Falha ao carregar avaliador: {exc}")
        return None