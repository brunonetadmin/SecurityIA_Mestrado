#!/usr/bin/env python3
"""
IDS/modules/incident_engine.py — Motor de Detecção de Incidentes

Conteúdo:
  - Constantes: SEVERITY, MITRE, ACTION (mapeamentos por classe de ataque)
  - ModelArtifacts  : singleton para carregamento dos artefatos do modelo
  - run_inference() : inferência em lote sobre DataFrame
  - analyze_file()  : processa arquivo Parquet → lista de incidentes
  - ManagerState    : persiste arquivos já analisados entre sessões
  - scan_new_files(): varredura do diretório do collector

Importado por: IDS/ids_detector.py, IDS/ids_learn.py, IDS/ids_manager.py
"""

import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import tensorflow as tf
from keras.models import load_model as keras_load_model
tf.get_logger().setLevel("ERROR")

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config import Config

Config.configure_tensorflow()


# ─────────────────────────────────────────────────────────────────────────────
# Constantes de detecção (MITRE ATT&CK Enterprise — CIC-IDS2018)
# ─────────────────────────────────────────────────────────────────────────────

BENIGN_LABEL = Config.BENIGN_LABEL
CONF_MEDIUM  = Config.CONF_MEDIUM
CONF_HIGH    = Config.CONF_HIGH

# (severidade_int, label_pt, cor_hex) por classe
SEVERITY: Dict[str, Tuple[int, str, str]] = {
    "Benign":                            (0, "NORMAL",   "#6c757d"),
    "PortScan":                          (1, "BAIXA",    "#0d6efd"),
    "FTP-Patator":                       (2, "MÉDIA",    "#fd7e14"),
    "SSH-Patator":                       (2, "MÉDIA",    "#fd7e14"),
    "Web Attack \u2013 Brute Force":     (3, "ALTA",     "#dc3545"),
    "Web Attack \u2013 XSS":             (3, "ALTA",     "#dc3545"),
    "Web Attack \u2013 Sql Injection":   (3, "ALTA",     "#dc3545"),
    "Bot":                               (4, "ALTA",     "#dc3545"),
    "DoS GoldenEye":                     (4, "ALTA",     "#dc3545"),
    "DoS Hulk":                          (4, "ALTA",     "#dc3545"),
    "DoS Slowhttptest":                  (4, "ALTA",     "#dc3545"),
    "DoS slowloris":                     (4, "ALTA",     "#dc3545"),
    "DDoS":                              (5, "CRÍTICA",  "#6f0000"),
    "Infiltration":                      (5, "CRÍTICA",  "#6f0000"),
    "Heartbleed":                        (5, "CRÍTICA",  "#6f0000"),
}

# (tática MITRE, técnica MITRE) por classe
MITRE: Dict[str, Tuple[str, str]] = {
    "PortScan":                          ("Reconhecimento",         "T1046"),
    "FTP-Patator":                       ("Acesso a Credenciais",   "T1110.001"),
    "SSH-Patator":                       ("Acesso a Credenciais",   "T1110.001"),
    "Bot":                               ("Comando e Controle",     "T1071"),
    "DoS GoldenEye":                     ("Impacto",                "T1499"),
    "DoS Hulk":                          ("Impacto",                "T1499"),
    "DoS Slowhttptest":                  ("Impacto",                "T1499.002"),
    "DoS slowloris":                     ("Impacto",                "T1499.002"),
    "DDoS":                              ("Impacto",                "T1498"),
    "Infiltration":                      ("Acesso Inicial",         "T1190"),
    "Heartbleed":                        ("Acesso a Credenciais",   "T1557"),
    "Web Attack \u2013 Brute Force":     ("Acesso a Credenciais",   "T1110.001"),
    "Web Attack \u2013 XSS":             ("Acesso Inicial",         "T1059.007"),
    "Web Attack \u2013 Sql Injection":   ("Acesso Inicial",         "T1190"),
}

# Ação operacional recomendada por classe
ACTION: Dict[str, str] = {
    "Benign":                            "Permitir",
    "PortScan":                          "Registrar e Monitorar",
    "FTP-Patator":                       "Bloquear IP Temporariamente",
    "SSH-Patator":                       "Bloquear IP Temporariamente",
    "Bot":                               "Quarentena do Host",
    "DoS GoldenEye":                     "Limitar Taxa da Origem",
    "DoS Hulk":                          "Limitar Taxa da Origem",
    "DoS Slowhttptest":                  "Limitar Taxa da Origem",
    "DoS slowloris":                     "Limitar Taxa da Origem",
    "DDoS":                              "Bloqueio de Emergência",
    "Infiltration":                      "Quarentena do Host",
    "Heartbleed":                        "Bloqueio de Emergência",
    "Web Attack \u2013 Brute Force":     "Bloquear Requisição",
    "Web Attack \u2013 XSS":             "Bloquear Requisição",
    "Web Attack \u2013 Sql Injection":   "Bloquear Requisição",
}

_INFERENCE_BATCH = 4096 * 8  # ~32 768 fluxos por iteração


# ─────────────────────────────────────────────────────────────────────────────
# ModelArtifacts — singleton do modelo treinado
# ─────────────────────────────────────────────────────────────────────────────

class ModelArtifacts:
    """
    Carrega e mantém em memória os artefatos do modelo (Keras + sklearn).
    Singleton por processo — evita recarga desnecessária no loop interativo.
    Use ModelArtifacts.reset() para forçar recarga após re-treinamento.
    """
    _instance: Optional["ModelArtifacts"] = None

    def __new__(cls) -> "ModelArtifacts":
        if cls._instance is None:
            obj = super().__new__(cls)
            obj._loaded = False
            cls._instance = obj
        return cls._instance

    def load(self) -> None:
        if self._loaded:
            return
        model_path = Config.MODEL_DIR / Config.MODEL_FILENAME
        if not model_path.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado: {model_path}\n"
                f"Execute o treinamento primeiro: python3 IDS/ids_learn.py train"
            )
        print(f"  ↳ Carregando {model_path.name} …", flush=True)
        self.model   = keras_load_model(str(model_path))
        self.scaler  = joblib.load(Config.MODEL_DIR / Config.SCALER_FILENAME)
        self.encoder = joblib.load(Config.MODEL_DIR / Config.LABEL_ENCODER_FILENAME)

        with open(Config.MODEL_DIR / Config.MODEL_INFO_FILENAME, encoding="utf-8") as f:
            info = json.load(f)

        self.selected_features: List[str]   = info.get("selected_features", Config.FEATURE_COLUMNS)
        self.label_map: Dict[int, str]      = {int(k): v for k, v in info.get("label_mapping", {}).items()}
        self.version:   str                 = info.get("version", "v?")
        self.trained_at: str                = info.get("trained_at", "—")
        self.feature_scores: dict           = info.get("feature_scores", {})
        self._loaded = True
        print(f"  ✓ Modelo v{self.version} | {len(self.label_map)} classes | {len(self.selected_features)} features")

    def decode(self, class_idx: int) -> str:
        return self.label_map.get(class_idx, f"Classe_{class_idx}")

    @classmethod
    def reset(cls) -> None:
        """Descarta o singleton — necessário após re-treinamento."""
        cls._instance = None


# ─────────────────────────────────────────────────────────────────────────────
# Inferência em lote
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    df: pd.DataFrame,
    arts: ModelArtifacts,
    batch_size: int = 4096,
) -> pd.DataFrame:
    """
    Executa inferência sobre um DataFrame de features.
    Adiciona colunas _label (str) e _conf (float) ao resultado.
    Features ausentes são preenchidas com 0.0.
    """
    for feat in arts.selected_features:
        if feat not in df.columns:
            df[feat] = 0.0

    X_raw      = df[arts.selected_features].values.astype(np.float32)
    X_scaled   = arts.scaler.transform(X_raw)
    X_reshaped = X_scaled.reshape(len(X_scaled), X_scaled.shape[1], 1)

    pred_cls  = np.empty(len(df), dtype=np.int32)
    pred_conf = np.empty(len(df), dtype=np.float32)

    for s in range(0, len(df), batch_size):
        e     = min(s + batch_size, len(df))
        proba = arts.model.predict(X_reshaped[s:e], verbose=0)
        pred_cls[s:e]  = np.argmax(proba, axis=1)
        pred_conf[s:e] = np.max(proba, axis=1)

    out = df.copy()
    out["_label"] = [arts.decode(c) for c in pred_cls]
    out["_conf"]  = pred_conf
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Construção de incidente
# ─────────────────────────────────────────────────────────────────────────────

_META_SET = set(Config.META_COLUMNS)

def _build_incident(row: pd.Series, has_meta: bool) -> dict:
    label    = str(row["_label"])
    conf     = float(row["_conf"])
    sev_int, sev_lbl, sev_col = SEVERITY.get(label, (3, "ALTA", "#dc3545"))
    mitre_t, mitre_tech       = MITRE.get(label, ("—", "—"))

    src_ip   = str(row.get("meta_src_ip",    "—")) if has_meta else "—"
    dst_ip   = str(row.get("meta_dst_ip",    "—")) if has_meta else "—"
    src_port = int(row.get("meta_src_port",   0))  if has_meta else 0
    dst_port = int(row.get("meta_dst_port",   0))  if has_meta else 0
    t_start  = float(row.get("meta_flow_start", 0.0)) if has_meta else 0.0

    proto_num = int(row.get("Protocol_Type", 0))
    protocol  = {6: "TCP", 17: "UDP", 1: "ICMP"}.get(proto_num, f"PROTO_{proto_num}")

    ts_str = (
        datetime.fromtimestamp(t_start, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        if t_start > 0 else "—"
    )

    return {
        "attack":           label,
        "confidence":       round(conf, 4),
        "conf_pct":         f"{conf * 100:.1f}%",
        "conf_level":       "ALTA" if conf >= CONF_HIGH else "MÉDIA",
        "severity":         sev_lbl,
        "sev_int":          sev_int,
        "sev_color":        sev_col,
        "src_ip":           src_ip,
        "dst_ip":           dst_ip,
        "src_port":         src_port,
        "dst_port":         dst_port,
        "protocol":         protocol,
        "flow_start":       ts_str,
        "flow_start_epoch": t_start,
        "action":           ACTION.get(label, "Investigar"),
        "mitre_tactic":     mitre_t,
        "mitre_tech":       mitre_tech,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Análise de arquivo Parquet
# ─────────────────────────────────────────────────────────────────────────────

def analyze_file(
    path: Path,
    arts: ModelArtifacts,
    on_incident: Optional[Callable[[dict], None]] = None,
    on_progress: Optional[Callable[[int, int, int], None]] = None,
) -> Dict:
    """
    Processa um arquivo Parquet do collector em batches.
    Retorna dict com: filename, path, rows, normal, incidents, elapsed_s, df.
    """
    pf       = pq.ParquetFile(str(path))
    total    = pf.metadata.num_rows
    has_meta = all(c in pf.schema_arrow.names for c in Config.META_COLUMNS)

    incidents: List[dict]        = []
    frames:    List[pd.DataFrame] = []
    done = 0
    t0   = time.time()

    for batch in pf.iter_batches(batch_size=_INFERENCE_BATCH):
        df_b = run_inference(batch.to_pandas(), arts)
        frames.append(df_b)
        done += len(df_b)

        mask = (df_b["_label"] != BENIGN_LABEL) & (df_b["_conf"] >= CONF_MEDIUM)
        for _, row in df_b[mask].iterrows():
            inc = _build_incident(row, has_meta)
            incidents.append(inc)
            if on_incident:
                on_incident(inc)

        if on_progress:
            on_progress(done, total, len(incidents))

    df_all  = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    elapsed = time.time() - t0

    n_attack = int((df_all["_label"] != BENIGN_LABEL).sum()) if not df_all.empty else 0
    normal   = total - n_attack

    # Estatísticas por tipo de ataque
    atk_counts: Dict[str, int] = {}
    for inc in incidents:
        atk_counts[inc["attack"]] = atk_counts.get(inc["attack"], 0) + 1

    fps_estimate = len(df_all) / max(elapsed, 0.01)  # flows/s

    return {
        "filename":   path.name,
        "path":       path,
        "rows":       total,
        "normal":     normal,
        "n_attack":   n_attack,
        "incidents":  incidents,
        "atk_counts": atk_counts,
        "elapsed_s":  round(elapsed, 2),
        "flows_per_s": round(fps_estimate, 1),
        "df":         df_all,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ManagerState — persistência de arquivos processados
# ─────────────────────────────────────────────────────────────────────────────

class ManagerState:
    """
    Rastreia arquivos analisados e treinados entre sessões.
    Escrita atômica via tempfile + rename (evita corrupção em crash).
    """

    def __init__(self) -> None:
        self._path = Config.MANAGER_STATE_FILE
        self._data = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            try:
                with open(self._path, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return {"analyzed": [], "trained": []}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)
        tmp.replace(self._path)

    def was_analyzed(self, name: str) -> bool:
        return name in self._data["analyzed"]

    def was_trained(self, name: str) -> bool:
        return name in self._data["trained"]

    def mark_analyzed(self, names: List[str]) -> None:
        for n in names:
            if n not in self._data["analyzed"]:
                self._data["analyzed"].append(n)
        self._save()

    def mark_trained(self, names: List[str]) -> None:
        for n in names:
            if n not in self._data["trained"]:
                self._data["trained"].append(n)
        self._save()

    def stats(self) -> dict:
        return {
            "analyzed": len(self._data["analyzed"]),
            "trained":  len(self._data["trained"]),
        }


def scan_new_files(state: ManagerState) -> List[Path]:
    """
    Retorna captura_*.parquet do COLLECTOR_DIR não analisados, em ordem cronológica.
    """
    d = Config.COLLECTOR_DIR
    if not d.exists():
        return []
    return [
        f for f in sorted(d.glob("captura_*.parquet"))
        if not state.was_analyzed(f.name)
    ]
