"""
IDS/modules — Pacote de módulos internos do SecurityIA

Módulos:
  flow_features    : FlowTracker, WelfordAccumulator, extração de 23 features
  incident_engine  : ModelArtifacts, run_inference, analyze_file, ManagerState
  utils            : logging, CLI helpers, progress_bar, decorators
  evaluator        : benchmark congelado, métricas estendidas, McNemar
  versioning       : nomes versionados <stem>_YYYYMMDD-N.<ext>
  model_registry   : triplete M0 / Mp / Mc
  full_report      : geração consolidada de relatório M0/Mp/Mc
"""

from IDS.modules.versioning import versioned_path, current_run_tag, set_run_tag
from IDS.modules.model_registry import (
    register_baseline, register_framework_version,
    load_triplet, list_versions, cleanup_old_versions,
)
from IDS.modules.full_report import generate as generate_full_report

__all__ = [
    "versioned_path", "current_run_tag", "set_run_tag",
    "register_baseline", "register_framework_version",
    "load_triplet", "list_versions", "cleanup_old_versions",
    "generate_full_report",
]
