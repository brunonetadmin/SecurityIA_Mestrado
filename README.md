# SecurityIA

**Framework de baixo custo para detecção multiclasse de intrusões em redes, baseado em Aprendizado de Máquina e operável apenas em CPU.** Desenvolvido como parte da dissertação de mestrado no Programa de Pós-Graduação em Informática (PPGI) do Instituto de Computação da UFAL.

O SecurityIA classifica fluxos de rede em 15 categorias (tráfego benigno e 14 tipos de ataque) sobre o dataset **CSE-CIC-IDS2018**. A metodologia segue um estudo experimental comparativo, organizado em quatro investigações encadeadas e sob protocolo livre de *data leakage*, do qual emergiu o núcleo de detecção: **CatBoost + SMOTE-ENN**, sobre um subconjunto de **23 atributos** selecionados por Teoria da Informação e com hiperparâmetros ajustados por otimização bayesiana. Todo o pipeline foi projetado para rodar em hardware de propósito geral, sem aceleradores gráficos.

---

## Resumo em números

| Item | Valor |
|------|-------|
| **Núcleo de detecção** | CatBoost (Investigação 1) + SMOTE-ENN no treino (Investigação 2) |
| **Seleção de atributos** | Índice IG_MI₆₀/₄₀ (0,6·IG + 0,4·MI), `k = 23` (Investigação 3) |
| **Otimização** | Optuna / TPE sob validação cruzada estratificada de 5 *folds* (Investigação 4) |
| **Configuração final** | CatBoost `iterations=400`, `depth=10`, `learning_rate≈0,1206` |
| **Métricas (teste reservado)** | *recall*-macro **0,8773** · MCC **0,8567** · F1-macro **0,8509** · FPR-macro **0,0080** |
| **Critério de decisão** | MCC ≥ 0,80 → FPR-macro ≤ 0,010 → maximização do *recall*-macro |
| **Dataset** | CSE-CIC-IDS2018 Mendeley (≈ 9,6 M registros); subamostra estratificada de **366.017** para os experimentos e o treino do modelo |
| **Classes** | 15 (1 *Benign* + 14 ataques) |
| **Execução** | CPU-only (sem GPU); semente fixa `RANDOM_SEED = 42` |

> Os modelos recorrentes (BiLSTM com atenção) foram avaliados como hipótese secundária e **excluídos** da configuração final: vetores de estatísticas de fluxo agregado não têm ordenação temporal que justifique arquiteturas sequenciais. Os *ensembles* de árvores venceram a comparação a um custo computacional muito inferior.

---

## Sumário

- [Instalação rápida](#instalação-rápida)
- [Dataset](#dataset)
- [Uso](#uso)
- [Arquitetura do sistema](#arquitetura-do-sistema)
- [As quatro investigações](#as-quatro-investigações)
- [Progressão de modelos (Baseline · M1 · M2 · M3)](#progressão-de-modelos-baseline--m1--m2--m3)
- [Bibliotecas e requisitos](#bibliotecas-e-requisitos)
- [Estrutura de diretórios](#estrutura-de-diretórios)
- [Conformidade com diretrizes de IA generativa](#conformidade-com-diretrizes-de-ia-generativa)
- [Autoria](#autoria)
- [Licença](#licença)

---

## Instalação rápida

**Ambiente de referência:** Ubuntu Server 24.04 LTS, Python 3.10+ (testado em 3.11), CPU-only.

### Opção A — script de instalação (recomendado)

Instala dependências do sistema e Python, cria o `.venv`, aplica *tuning* de kernel para captura, configura `cap_net_raw` (captura sem root), *templates* systemd e rotação de logs.

```bash
git clone https://github.com/brunonetadmin/SecurityIA_Mestrado.git SecurityIA
cd SecurityIA
chmod +x install.sh
sudo ./install.sh                 # use --no-tuning --no-systemd para instalação mínima
```

### Opção B — manual (apenas Python)

```bash
git clone https://github.com/brunonetadmin/SecurityIA_Mestrado.git SecurityIA
cd SecurityIA
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install catboost xgboost scikit-learn imbalanced-learn optuna \
            tensorflow pandas numpy scipy pyarrow scapy matplotlib seaborn joblib
```

### Verificação

```bash
python3 -c "
import catboost, xgboost, sklearn, imblearn, optuna, pandas, numpy
print('CatBoost', catboost.__version__, '| imbalanced-learn', imblearn.__version__)
print('Dependências OK.')
"
python3 -c "from config import Config; Config.ensure_dirs(); print('Diretórios criados.')"
```

> Captura de pacotes em tempo real (Collector) exige `cap_net_raw`; o `install.sh` já configura. Manualmente:
> `sudo setcap cap_net_raw,cap_net_admin=eip $(readlink -f .venv/bin/python3)`

---

## Dataset

Base: **CSE-CIC-IDS2018** (Canadian Institute for Cybersecurity), versão *cleaned* disponível no Mendeley Data, com cerca de **9,6 milhões de registros** distribuídos em 14 arquivos rotulados por categoria de ataque.

Para viabilizar a execução em CPU, o projeto consome uma **subamostra estratificada de 366.017 registros** — proporção escolhida para os experimentos e para o treino do modelo de detecção. A amostragem é feita **por arquivo**, com semente fixa (`RANDOM_SEED = 42`), o que preserva a distribuição original entre classes (desbalanceamento característico da base, com *Benign* majoritária). O mesmo subconjunto alimenta as quatro investigações e o modelo implantado, garantindo correspondência entre o sistema avaliado e o operado.

Extraia os 14 CSVs para `Base/CSE-CIC-IDS2018/`:

```
Base/CSE-CIC-IDS2018/
├── bot.csv                       ├── dos attacks-slowhttptest.csv
├── brute force -web.csv          ├── dos attacks-slowloris.csv
├── brute force -xss.csv          ├── ftp-bruteforce.csv
├── ddos attack-hoic.csv          ├── infilteration.csv
├── ddos attack-loic-udp.csv      ├── sql injection.csv
├── ddos attacks-loic-http.csv    └── ssh-bruteforce.csv
├── dos attacks-goldeneye.csv
├── dos attacks-hulk.csv
```

Verificação: `python3 -c "from config import verificar_dataset; verificar_dataset()"`

---

## Uso

### Menu interativo (ponto de entrada)

```bash
python3 app_menu.py
```

Unifica em um terminal: execução das análises acadêmicas, dataset e relatórios; e operação do IDS (captura, detecção, treinamento, diagnósticos, logs, configurações).

### Treinamento do modelo

```bash
python3 IDS/ids_learn.py train            # treino completo
python3 IDS/ids_learn.py train --force    # ignora caches
python3 IDS/ids_learn.py finetune         # ajuste sobre dados anotados pelo Detector
python3 IDS/ids_learn.py status           # estado do modelo + progressão
```

O pipeline carrega a subamostra de 366k → seleção IG_MI₆₀/₄₀ (`k=23`) → *split* estratificado 70/15/15 **antes** de qualquer balanceamento → **SMOTE-ENN apenas no treino** → CatBoost → avaliação no teste original → registro na progressão de modelos. Para um re-treino limpo:

```bash
rm -f temp/01_cleaned_dataset.parquet temp/03_*.pkl
python3 IDS/ids_learn.py train --force
```

### Baseline e ranking de atributos

```bash
python3 baseline_rf.py                    # Baseline RandomForest (piso de referência)
python3 dump_ig_mi_23.py                  # ranking IG_MI60/40 e os 23 atributos selecionados
```

### Captura e detecção

```bash
python3 IDS/ids_collector.py --background --interface ens18   # captura contínua → Parquet/dia
python3 IDS/ids_detector.py watch --interval 60               # monitor contínuo
python3 IDS/ids_detector.py batch                             # lote de Parquets pendentes
python3 IDS/ids_detector.py file temp/capture/<arquivo>.parquet
```

O Detector carrega o modelo CatBoost (`.cbm`), classifica cada fluxo, mapeia para severidade e técnicas MITRE ATT&CK, exporta incidentes em JSONL (para SIEM) e dados anotados em *staging* (para *fine-tuning*), e gera relatório HTML.

### Execução das investigações em segundo plano

```bash
# via menu (item Tests → executar todas) ou diretamente:
nohup setsid ./run_all_tests.sh > Tests/run_all.out 2>&1 &
tail -f Tests/Logs/run_all.log        # acompanhar
pkill -f analise_                     # cancelar
```

O processo mestre é desacoplado da sessão (`nohup setsid`) e **imune ao encerramento do SSH**; cada análise roda em subprocess independente, e a falha de uma não interrompe as demais.

---

## Arquitetura do sistema

Quatro módulos independentes, orquestráveis pelo menu ou por CLI:

```
   Rede (TCP/UDP/ICMP)
        │
        ▼
   ┌──────────────┐   Módulo Coletor (ids_collector.py)
   │  COLETOR     │   Scapy → flow_features → 23 atributos CIC-IDS2018
   └──────┬───────┘   captura contínua, Parquet (Snappy) por dia
          │ vetores de 23 atributos
          ▼
   ┌──────────────┐   Módulo de Pré-processamento
   │ PRÉ-PROCESS. │   projeção nos 23 atributos do modelo
   └──────┬───────┘   (CatBoost é invariante à escala: sem normalização na inferência)
          │
          ▼
   ┌──────────────┐   Módulo de Detecção (ids_detector.py)
   │  DETECÇÃO    │   CatBoost (.cbm) → classe + probabilidade sobre 15 classes
   └──────┬───────┘
          │ classe + probabilidade
          ▼
   ┌──────────────┐   Sistema de Alertas (ids_reports.py)
   │   ALERTAS    │   limiar de confiança → JSONL (SIEM) + HTML + MITRE ATT&CK
   └──────────────┘
```

O **coletor emite os 23 atributos** no mesmo formato e unidades do CSE-CIC-IDS2018, eliminando divergência treino/produção (*train/serve skew*). O **sistema de alertas opera sobre a decisão do detector** (classe e probabilidade), não sobre os atributos — a redução de dimensionalidade incide apenas na entrada do detector, cujo desempenho é preservado pelo subconjunto de 23.

---

## As quatro investigações

Cada investigação encadeia a decisão da anterior, sob protocolo único e livre de vazamento de informação (seleção e balanceamento aplicados **apenas ao treino**; teste reservado isolado).

| # | Investigação | Decisão | Script |
|---|--------------|---------|--------|
| 1 | Comparação de arquiteturas | **CatBoost** vence (ensembles de árvores > redes profundas; recorrentes excluídos) | `Tests/analise_1_arquiteturas.py` |
| 2 | Tratamento de desbalanceamento | **SMOTE-ENN** (recall alto dentro das restrições de MCC e FPR) | `Tests/analise_2_balanceamento.py` |
| 3 | Seleção de atributos | **IG_MI₆₀/₄₀, k = 23** (preserva o desempenho com 70% menos atributos) | `Tests/analise_3_teoria_informacao.py` |
| 4 | Otimização e validação | **Optuna / TPE** + validação cruzada de 5 *folds* + teste reservado | `Tests/analise_4_otimizacao_validacao.py` |

**Critério de avaliação.** Métrica primária: *recall*-macro. Critério hierárquico de seleção: MCC ≥ 0,80 → FPR-macro ≤ 0,010 → maximização do *recall*-macro, com o tempo de treino em CPU como desempate. Acurácia e F1-*weighted* são reportados apenas para comparação com a literatura — o desbalanceamento severo (*Benign* dominante) os torna enganosos.

Cada análise gera relatórios em Markdown, figuras PNG (tons de cinza, sem título embutido) e tabelas CSV em `Tests/Reports/`.

---

## Progressão de modelos (Baseline · M1 · M2 · M3)

Cada treinamento é registrado em uma progressão de maturidade crescente, em `Model/registry/`:

- **Baseline** — RandomForest sobre o dataset bruto, sem as técnicas das investigações. Piso de referência (`baseline_rf.py`).
- **M1 — Modelo inicial** — CatBoost + SMOTE-ENN + IG_MI₆₀/₄₀ (k=23) + hiperparâmetros otimizados. Produto direto da metodologia.
- **M2 — Modelo anterior** — resultado de um ciclo de retreinamento (aprendizado contínuo, trabalho futuro).
- **M3 — Modelo atual** — versão em operação após o ciclo mais recente.

Após cada treino/*fine-tuning*, o sistema gera um relatório comparativo (métricas lado a lado, matrizes de confusão e decisão de promoção). M2 e M3 pertencem ao laço de aprendizado contínuo previsto como trabalho futuro.

---

## Bibliotecas e requisitos

**Hardware de referência:** 8+ vCPUs (recomendado 20+), 32 GB RAM (recomendado 64 GB), sem GPU. Interface dedicada para captura (opcional).

| Biblioteca | Papel |
|------------|-------|
| **CatBoost** | Núcleo de detecção (modelo final) |
| **imbalanced-learn** | SMOTE-ENN (balanceamento no treino) |
| **scikit-learn** | Seleção de atributos, métricas, *split* estratificado, Baseline RF |
| **Optuna** | Otimização bayesiana de hiperparâmetros (TPE) |
| **XGBoost** | Arquitetura comparada na Investigação 1 |
| **TensorFlow / Keras** | Redes densas/recorrentes comparadas na Investigação 1 (CPU-only) |
| **pandas · NumPy · SciPy** | Manipulação e cálculo numérico |
| **PyArrow** | Persistência em Parquet (Snappy) |
| **Scapy** | Captura de pacotes (Collector) |
| **Matplotlib · Seaborn** | Figuras dos relatórios |
| **joblib** | Serialização de *scaler* e *label encoder* |

---

## Estrutura de diretórios

```
SecurityIA/
├── app_menu.py                 # Menu interativo unificado (Tests + IDS)
├── config.py                   # Configuração central (caminhos, hiperparâmetros, amostragem)
├── baseline_rf.py              # Baseline RandomForest (piso de referência)
├── dump_ig_mi_23.py            # Ranking IG_MI60/40 e os 23 atributos selecionados
├── reconcile_registry.py       # Reconciliação do registro de modelos
├── run_all_tests.sh            # Orquestra as 4 investigações (nohup setsid)
├── install.sh                  # Instalação do ambiente (Ubuntu 24.04)
│
├── IDS/
│   ├── ids_collector.py        # Captura (Scapy) e agregação em fluxos
│   ├── ids_detector.py         # Detecção (CatBoost) → incidentes
│   ├── ids_learn.py            # Treinamento, fine-tuning e progressão de modelos
│   ├── ids_reports.py          # Relatórios HTML/JSONL + MITRE ATT&CK
│   └── modules/
│       ├── flow_features.py    # Extrator de fluxos (23 atributos CIC-IDS2018)
│       ├── incident_engine.py  # Classificação, severidade, MITRE ATT&CK
│       ├── model_registry.py   # Progressão Baseline · M1 · M2 · M3
│       ├── full_report.py      # Relatório comparativo entre versões
│       ├── versioning.py       # Versionamento de artefatos por execução
│       └── utils.py            # Logging, helpers, execução em background
│
├── Tests/                      # As 4 investigações + relatórios (CSV/PNG/MD)
│   ├── analise_1_arquiteturas.py
│   ├── analise_2_balanceamento.py
│   ├── analise_3_teoria_informacao.py
│   └── analise_4_otimizacao_validacao.py
│
├── Base/CSE-CIC-IDS2018/       # CSVs do dataset (não versionados)
├── Model/                      # cerebro_catboost.cbm, scaler.pkl, label_encoder.pkl,
│   └── registry/               #   ids_model_info.json, registry/ (Baseline·M1·M2·M3)
├── Reports/                    # Relatórios HTML do IDS (operação)
├── Logs/                       # App.log, Collector.log, Learn.log
└── temp/                       # Caches, capturas (capture/) e staging/
```

---

## Conformidade com diretrizes de IA generativa

Em conformidade com o Art. 9.º da Portaria CNPq n.º 2.664/2026, declara-se que ferramentas de IA generativa foram utilizadas como apoio à revisão textual, à depuração de *scripts* e à triagem bibliográfica preliminar. Todo o conteúdo apoiado por essas ferramentas foi revisado e validado pelo autor, que assume responsabilidade integral pela precisão, pela metodologia e pela integridade científica do trabalho.

---

## Autoria

| | |
|---|---|
| **Autor** | Bruno Cavalcante Barbosa — bcb@ic.ufal.br |
| **Orientador** | Prof. Dr. André Luiz Lins de Aquino |
| **Programa** | PPGI — Instituto de Computação — UFAL |
| **Dissertação** | SecurityIA: Um Framework de Baixo Custo para Detecção Multiclasse de Intrusões em Redes Utilizando Aprendizado de Máquina |

---

## Licença

Projeto desenvolvido no âmbito de uma dissertação de mestrado acadêmica. Consulte o autor/orientador para informações sobre uso e redistribuição.
