#!/usr/bin/env python3
"""
IDS/modules/versioning.py — Versionamento de nomes de artefatos

Utilitário central para gerar nomes versionados de tabelas, figuras e
relatórios, permitindo comparação direta entre execuções consecutivas.

Convenção:
    <stem>_YYYYMMDD-N.<ext>
onde N é um contador inteiro autoincrementado por dia, persistido em
um arquivo .counter no próprio diretório.

Exemplos:
    tabela_metricas_20260426-1.csv
    matriz_confusao_20260426-1.png
    matriz_confusao_20260426-2.png    # segunda execução do mesmo dia
    relatorio_completo_20260427-1.md  # próximo dia volta a 1

API:
    versioned_path(directory, stem, ext) -> Path
    current_run_tag() -> str           # "YYYYMMDD-N" para a execução corrente
    set_run_tag(tag)                   # força tag (testes, scripts em lote)
"""

from __future__ import annotations

import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


_LOCK = threading.Lock()
_RUN_TAG_OVERRIDE: Optional[str] = None
_RUN_TAG_CACHE: Optional[str] = None
_COUNTER_FILE_NAME = ".version_counter"


def set_run_tag(tag: Optional[str]) -> None:
    """
    Força o uso de uma tag específica para todos os artefatos da execução.
    Passe None para voltar ao comportamento automático (incremento por dia).
    """
    global _RUN_TAG_OVERRIDE, _RUN_TAG_CACHE
    with _LOCK:
        _RUN_TAG_OVERRIDE = tag
        _RUN_TAG_CACHE = None


def _read_counter(counter_file: Path, today: str) -> int:
    """Lê o contador do dia. Retorna 0 se ausente ou de dia anterior."""
    if not counter_file.exists():
        return 0
    try:
        raw = counter_file.read_text(encoding="utf-8").strip()
        date_part, num_part = raw.split(":")
        if date_part != today:
            return 0
        return int(num_part)
    except (ValueError, OSError):
        return 0


def _write_counter(counter_file: Path, today: str, n: int) -> None:
    """Escrita atômica via tmp + replace."""
    counter_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = counter_file.with_suffix(".tmp")
    tmp.write_text(f"{today}:{n}", encoding="utf-8")
    tmp.replace(counter_file)


def _resolve_run_tag(base_dir: Path) -> str:
    """
    Resolve a tag de execução. Se houver override, usa-o. Caso contrário,
    incrementa o contador armazenado em base_dir/.version_counter.

    A tag é resolvida UMA VEZ por execução (cache em _RUN_TAG_CACHE),
    garantindo que todos os artefatos de uma mesma rodada compartilhem
    o mesmo sufixo numérico.
    """
    global _RUN_TAG_CACHE
    with _LOCK:
        if _RUN_TAG_OVERRIDE is not None:
            return _RUN_TAG_OVERRIDE
        if _RUN_TAG_CACHE is not None:
            return _RUN_TAG_CACHE

        today = datetime.now().strftime("%Y%m%d")
        counter_file = base_dir / _COUNTER_FILE_NAME
        n = _read_counter(counter_file, today) + 1
        _write_counter(counter_file, today, n)
        _RUN_TAG_CACHE = f"{today}-{n}"
        return _RUN_TAG_CACHE


def current_run_tag(base_dir: Optional[Path] = None) -> str:
    """
    Retorna a tag da execução corrente. Se base_dir for None, usa um
    contador global em /tmp/securityia_version.counter (válido para
    chamadas isoladas que não tenham um diretório de saída específico).
    """
    if base_dir is None:
        base_dir = Path(os.environ.get("TMPDIR", "/tmp")) / "securityia_version"
    return _resolve_run_tag(Path(base_dir))


def versioned_path(directory: Path, stem: str, ext: str) -> Path:
    """
    Gera caminho versionado dentro de 'directory' no formato:
        directory/<stem>_YYYYMMDD-N.<ext>

    O contador é compartilhado entre TODOS os artefatos de uma mesma
    execução, ou seja: todos os arquivos de uma rodada terminam com o
    mesmo sufixo, facilitando o pareamento visual.

    Exemplo:
        versioned_path(Path("Reports"), "matriz_confusao", "png")
        -> Reports/matriz_confusao_20260426-1.png
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tag = _resolve_run_tag(directory)
    ext = ext.lstrip(".")
    return directory / f"{stem}_{tag}.{ext}"
