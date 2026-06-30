#!/usr/bin/env python3
"""
dump_ig_mi_23.py — Imprime o top-23 do IG_MI60/40 SEM treinar.

Reusa EXATAMENTE o método da dissertação (DataHandler.select_features de
IDS/ids_learn.py), de modo que o ranking aqui é idêntico ao reportado na
Investigação 3. Apenas calcula IG + MI sobre a amostra de seleção e ordena;
nenhum modelo é treinado.

Saídas:
    - imprime o ranking completo das 70 features (score IG×0,6 + MI×0,4)
    - marca as 23 selecionadas
    - grava Model/ids_selected_features_k23.json

Pré-requisito:
    Cache de limpeza Temp/01_cleaned_dataset.parquet (gerado por uma execução
    anterior do pré-processamento) OU o dataset acessível em Config.DATA_DIR.

Uso (no servidor, a partir da raiz do projeto):
    cd /opt/SecurityIA && python3 dump_ig_mi_23.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import Config
from IDS.ids_learn import DataHandler

K = 23


def main() -> None:
    # Força k=23 apenas neste processo (não persiste no config.py).
    Config.FEATURE_SELECTION_CONFIG["k_best"] = K

    dh = DataHandler()

    # Mesma fonte do pré-processamento: cache de limpeza (ou dataset bruto).
    df = dh.load_with_cache()
    print(f"  Dataset carregado: {df.shape[0]:,} linhas x {df.shape[1]} colunas")

    feat_names = [c for c in df.columns if c != "Label"]
    if len(feat_names) <= K:
        print(f"  [ERRO] dataset tem apenas {len(feat_names)} features (<= {K}); "
              "nada a selecionar.")
        sys.exit(1)

    dh.label_encoder.fit(df["Label"])
    y_all = dh.label_encoder.transform(df["Label"])

    # Reproduz a amostragem estratificada de preprocess_with_cache().
    n_sample = min(Config.PREPROCESSING_CONFIG["sample_size_for_selection"], len(df))
    if n_sample < len(df):
        _, df_s, _, y_s = train_test_split(
            df, y_all,
            test_size=n_sample / len(df),
            random_state=Config.TRAINING_CONFIG["random_state"],
            stratify=y_all,
        )
        X_s = df_s[feat_names].to_numpy(dtype=np.float64)
    else:
        X_s = df[feat_names].to_numpy(dtype=np.float64)
        y_s = y_all
    print(f"  Amostra de seleção: {X_s.shape[0]:,} linhas")

    # Reusa o método EXATO da dissertação.
    selected, scores = dh.select_features(X_s, y_s, feat_names)

    # Ranking completo ordenado por score combinado.
    ranked = sorted(
        scores.items(),
        key=lambda kv: kv[1].get("combined", 0.0),
        reverse=True,
    )

    print("\n  RANKING IG_MI60/40 (IG x0,6 + MI x0,4) — 70 features")
    print("  " + "-" * 70)
    print(f"  {'#':>3}  {'feature':<34}{'score':>9}{'IG':>9}{'MI':>9}  sel")
    for i, (name, sc) in enumerate(ranked, 1):
        mark = "  *" if name in selected else ""
        print(f"  {i:>3}  {name:<34}"
              f"{sc.get('combined', 0.0):>9.4f}"
              f"{sc.get('ig_raw', 0.0):>9.4f}"
              f"{sc.get('mi_raw', 0.0):>9.4f}{mark}")
        if i == K:
            print("  " + "." * 70 + "  <= corte k=23")

    out = Config.MODEL_DIR / "ids_selected_features_k23.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"k": K, "method": "IG_MI60/40",
             "ig_weight": Config.FEATURE_SELECTION_CONFIG["ig_weight"],
             "mi_weight": Config.FEATURE_SELECTION_CONFIG["mi_weight"],
             "selected_features": selected},
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\n  23 features gravadas em: {out}")
    print("  Envie esse JSON para o cruzamento com o coletor.")


if __name__ == "__main__":
    main()
