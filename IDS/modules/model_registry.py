#!/usr/bin/env python3
"""
IDS/modules/model_registry.py — Registro Versionado de Modelos SecurityIA

Mantém triplete canônico de modelos para comparação direta:
    M0  : baseline imutável — Random Forest treinado sobre dataset puro
    Mp  : modelo anterior   — versão imediatamente precedente do framework
    Mc  : modelo atual      — versão recém-treinada do framework

Persistência em $MODEL_DIR/registry/:
    baseline/                    M0 (não rotaciona)
    v0001_<timestamp>/           versões do framework
    v0002_<timestamp>/
    ...
    registry.json                índice (lock-free com tmp+rename)

API:
    register_baseline(model_path, y_true, y_pred, metrics)
    register_framework_version(model_path, y_true, y_pred, metrics, source)
    load_triplet() -> dict('M0','Mp','Mc')   cada um é dict ou None
    list_versions() -> List[dict]
    cleanup_old_versions(keep=10)
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import Config


REGISTRY_DIR = Config.MODEL_DIR / "registry"
BASELINE_DIR = REGISTRY_DIR / "baseline"
INDEX_FILE = REGISTRY_DIR / "registry.json"


def _ensure_layout() -> None:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    if not INDEX_FILE.exists():
        _write_index({"baseline": None, "framework_versions": []})


def _read_index() -> dict:
    _ensure_layout()
    try:
        with open(INDEX_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {"baseline": None, "framework_versions": []}


def _write_index(idx: dict) -> None:
    _ensure_layout()
    tmp = INDEX_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(idx, f, indent=2, ensure_ascii=False)
    tmp.replace(INDEX_FILE)


def register_baseline(
    model_pickle_path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict,
    overwrite: bool = False,
) -> dict:
    """Registra M0. Por padrão imutável; use overwrite=True para substituir."""
    _ensure_layout()
    idx = _read_index()
    if idx.get("baseline") and not overwrite:
        print(
            f"  [registry] Baseline já registrado em "
            f"{idx['baseline']['registered_at']}.\n"
            f"             Use overwrite=True para substituir."
        )
        return idx["baseline"]

    if not Path(model_pickle_path).exists():
        raise FileNotFoundError(
            f"Modelo baseline não encontrado: {model_pickle_path}"
        )

    shutil.copy2(model_pickle_path, BASELINE_DIR / "model.pkl")
    np.save(BASELINE_DIR / "y_test.npy", np.asarray(y_true))
    np.save(BASELINE_DIR / "predictions.npy", np.asarray(y_pred))
    with open(BASELINE_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    entry = {
        "id": "baseline",
        "kind": "RandomForest",
        "registered_at": datetime.now().isoformat(),
        "path": str(BASELINE_DIR),
        "n_test": int(len(y_true)),
        "f1_macro": float(metrics.get("f1_macro", 0.0)),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "fpr_macro": float(metrics.get("fpr_macro", 0.0)),
        "mcc": float(metrics.get("mcc", 0.0)),
    }
    idx["baseline"] = entry
    _write_index(idx)
    print(f"  [registry] Baseline M0 registrado: {BASELINE_DIR}")
    return entry


def register_framework_version(
    model_path: Path,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metrics: dict,
    source: str = "train",
    extra: Optional[dict] = None,
) -> dict:
    """Registra nova versão do framework. Mc anterior vira Mp automaticamente."""
    _ensure_layout()
    idx = _read_index()

    n_existing = len(idx["framework_versions"])
    version_id = f"v{n_existing + 1:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    version_dir = REGISTRY_DIR / version_id
    version_dir.mkdir(parents=True, exist_ok=False)

    if Path(model_path).exists():
        try:
            shutil.copy2(model_path, version_dir / "model.keras")
        except (shutil.SameFileError, OSError):
            pass
    np.save(version_dir / "y_test.npy", np.asarray(y_true))
    np.save(version_dir / "predictions.npy", np.asarray(y_pred))
    with open(version_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    entry = {
        "id": version_id,
        "kind": "BiLSTM_Bahdanau",
        "registered_at": datetime.now().isoformat(),
        "path": str(version_dir),
        "source": source,
        "n_test": int(len(y_true)),
        "f1_macro": float(metrics.get("f1_macro", 0.0)),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "fpr_macro": float(metrics.get("fpr_macro", 0.0)),
        "mcc": float(metrics.get("mcc", 0.0)),
        "extra": extra or {},
    }
    idx["framework_versions"].append(entry)
    _write_index(idx)
    print(f"  [registry] Versão registrada: {version_id} (source={source})")
    return entry


def _load_artifacts(entry: Optional[dict]) -> Optional[dict]:
    if not entry:
        return None
    base = Path(entry["path"])
    out = dict(entry)
    try:
        with open(base / "metrics.json", encoding="utf-8") as f:
            out["metrics"] = json.load(f)
        out["y_true"] = np.load(base / "y_test.npy")
        out["y_pred"] = np.load(base / "predictions.npy")
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [registry] Falha ao carregar {entry['id']}: {exc}")
        return None
    return out


def load_triplet() -> dict:
    """Retorna dict com 'M0', 'Mp', 'Mc'; cada um é dict completo ou None."""
    idx = _read_index()
    versions = idx["framework_versions"]
    return {
        "M0": _load_artifacts(idx.get("baseline")),
        "Mp": _load_artifacts(versions[-2]) if len(versions) >= 2 else None,
        "Mc": _load_artifacts(versions[-1]) if len(versions) >= 1 else None,
    }


def list_versions() -> list:
    """Lista todas as versões registradas (mais recente primeiro)."""
    idx = _read_index()
    out = []
    if idx.get("baseline"):
        out.append(idx["baseline"])
    out.extend(reversed(idx["framework_versions"]))
    return out


def cleanup_old_versions(keep: int = 10) -> int:
    """Remove versões antigas, preservando 'keep' mais recentes. M0 nunca é removido."""
    idx = _read_index()
    versions = idx["framework_versions"]
    if len(versions) <= keep:
        return 0

    to_remove = versions[:-keep]
    for v in to_remove:
        path = Path(v["path"])
        try:
            if path.exists() and path.resolve().is_relative_to(REGISTRY_DIR.resolve()):
                shutil.rmtree(path, ignore_errors=True)
        except (OSError, ValueError):
            pass
    idx["framework_versions"] = versions[-keep:]
    _write_index(idx)
    print(f"  [registry] {len(to_remove)} versão(ões) antiga(s) removida(s).")
    return len(to_remove)


def _print_status() -> None:
    idx = _read_index()
    print("\n  REGISTRO DE MODELOS — SecurityIA")
    print("  " + "─" * 60)
    if idx.get("baseline"):
        b = idx["baseline"]
        print(f"  M0  Baseline       : {b['kind']:<20s}  F1={b['f1_macro']:.4f}  "
              f"({b['registered_at'][:19]})")
    else:
        print("  M0  Baseline       : NÃO REGISTRADO")

    versions = idx["framework_versions"]
    if not versions:
        print("  Mc  Atual          : NÃO REGISTRADO")
    else:
        c = versions[-1]
        print(f"  Mc  Atual          : {c['id']}  F1={c['f1_macro']:.4f}  "
              f"src={c.get('source','?')}  ({c['registered_at'][:19]})")
        if len(versions) >= 2:
            p = versions[-2]
            print(f"  Mp  Anterior       : {p['id']}  F1={p['f1_macro']:.4f}  "
                  f"src={p.get('source','?')}  ({p['registered_at'][:19]})")
        else:
            print("  Mp  Anterior       : (sem histórico — apenas Mc)")
    print(f"\n  Total de versões registradas: {len(versions)}")
    print("  " + "─" * 60)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SecurityIA — Registro de Modelos")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status", help="exibe status do triplete M0/Mp/Mc")
    sub.add_parser("list", help="lista todas as versões")
    cl = sub.add_parser("cleanup", help="remove versões antigas")
    cl.add_argument("--keep", type=int, default=10)

    args = p.parse_args()
    if args.cmd == "status":
        _print_status()
    elif args.cmd == "list":
        for v in list_versions():
            print(f"  {v['id']:<35s}  F1={v['f1_macro']:.4f}  "
                  f"acc={v['accuracy']:.4f}  ({v['registered_at'][:19]})")
    elif args.cmd == "cleanup":
        cleanup_old_versions(keep=args.keep)
