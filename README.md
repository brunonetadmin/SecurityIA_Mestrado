# SecurityIA

Sistema de Detecção de Intrusão em Redes de Computadores baseado em Aprendizado de Máquina profundo, desenvolvido como parte da dissertação de mestrado no Programa de Pós-Graduação em Informática (PPGI) do Instituto de Computação da UFAL.

O projeto combina rede neural Bidirecional LSTM com mecanismo de Atenção de Bahdanau e tratamento de desbalanceamento por Focal Loss reponderada (Cui et al., 2019), seleção de features fundamentada em Teoria da Informação (Information Gain + Mutual Information) e protocolo experimental livre de *data leakage*. Opera em ambiente CPU-only e é validado sobre o dataset CSE-CIC-IDS2018.

---

## Sumário

- [Arquitetura do Sistema](#arquitetura-do-sistema)
- [Decisões Metodológicas](#decisões-metodológicas)
- [Estrutura de Diretórios](#estrutura-de-diretórios)
- [Requisitos](#requisitos)
- [Instalação](#instalação)
- [Configuração](#configuração)
- [Dataset](#dataset)
- [Uso](#uso)
  - [Menu Interativo](#menu-interativo)
  - [Execução em Background](#execução-em-background)
  - [Testes Acadêmicos](#testes-acadêmicos)
  - [Treinamento do Modelo](#treinamento-do-modelo)
  - [Captura de Tráfego](#captura-de-tráfego)
  - [Detecção de Incidentes](#detecção-de-incidentes)
  - [Relatórios](#relatórios)
- [Pipeline de Dados](#pipeline-de-dados)
- [Modelo](#modelo)
- [Versionamento de Modelos (Triplete M0/Mp/Mc)](#versionamento-de-modelos-triplete-m0mpmc)
- [Conformidade com Diretrizes de IA Generativa](#conformidade-com-diretrizes-de-ia-generativa)
- [Referências Bibliográficas](#referências-bibliográficas)
- [Autoria](#autoria)
- [Licença](#licença)

---

## Arquitetura do Sistema

O SecurityIA é composto por quatro módulos que operam de forma independente e podem ser orquestrados pelo menu interativo (`app_menu.py`) ou por linha de comando direta:

```
                ┌─────────────┐
                │  Rede Local │
                └──────┬──────┘
                       │ pacotes IP (TCP/UDP/ICMP)
                       ▼
             ┌─────────────────┐
             │   Collector     │   ids_collector.py
             │  (3 threads)    │   Scapy → FlowTracker → Parquet (Snappy)
             └────────┬────────┘
                      │ .parquet por dia
                      ▼
             ┌─────────────────┐
             │   Detector      │   ids_detector.py
             │  (Bi-LSTM +     │   watch | batch | file
             │   Bahdanau)     │   load_model(compile=False)
             └───┬────────┬────┘
                 │        │
      JSONL      │        │  Parquet anotado
   (incidentes)  │        │  (staging para fine-tuning)
                 ▼        ▼
             ┌─────────────────┐
             │   Reports       │   ids_reports.py
             │  HTML + TXT     │   métricas, heatmap, MITRE ATT&CK
             └─────────────────┘

             ┌─────────────────┐
             │   Learn         │   ids_learn.py
             │  train | fine-  │   DataHandler → ModelTrainer → ReportGenerator
             │  tune | status  │   + Focal Loss reponderada + registry M0/Mp/Mc
             └─────────────────┘
```

O Collector captura pacotes em tempo real, agrega em fluxos bidirecionais com 23 features compatíveis com o CIC-IDS2018 e persiste em Parquet com compressão Snappy. O Detector carrega o modelo treinado, classifica cada fluxo e exporta incidentes em JSONL (para ingestão por SIEM) e datasets anotados em staging (para fine-tuning incremental). O módulo Learn gerencia todo o ciclo de treinamento sob protocolo livre de *data leakage*.

---

## Decisões Metodológicas

O projeto incorpora correções metodológicas identificadas durante a revisão da literatura e validadas empiricamente pelos quatro testes acadêmicos:

### D1 — Particionamento antes do balanceamento

O split treino/validação/teste ocorre **antes** de qualquer operação de balanceamento, evitando que amostras sintéticas vazem para o conjunto de teste. O balanceamento, quando aplicado, opera **apenas sobre o conjunto de treino**.

### D2 — Focal Loss reponderada (sem dupla penalização)

A função de perda é a Focal Loss balanceada por número efetivo de amostras (Cui et al., 2019), com γ=2.0 e β=0.9999. O parâmetro `use_class_weight` no `model.fit()` foi mantido **desligado** para evitar dupla penalização da mesma classe (uma vez pela loss, outra vez pelo peso de classe).

### D3 — Balanceamento sintético abandonado

A Análise 2 demonstrou empiricamente que estratégias de oversampling agressivo (SMOTE, Borderline-SMOTE-2, ADASYN) prejudicam o desempenho no CSE-CIC-IDS2018 quando comparadas a soluções baseadas em perda. O tratamento de desbalanceamento é delegado integralmente à Focal Loss reponderada (`BALANCING_CONFIG.strategy = "none"`).

### Critério de avaliação

A métrica primária para seleção de modelos é o **Recall macro** (sensibilidade média entre classes), com o **Coeficiente de Correlação de Matthews (MCC)** como métrica secundária. A acurácia e o F1-score weighted são reportados para compatibilidade com a literatura mas **não** são usados para decisão, dado que o severo desbalanceamento do dataset (Benign > 80%) os torna enganosos.

---

## Estrutura de Diretórios

```
SecurityIA/
├── app_menu.py                 # Menu interativo unificado (testes + IDS)
├── config.py                   # Configuração central (caminhos, hiperparâmetros)
├── baseline_rf.py              # Baseline RandomForest M0 (modelo de referência)
│
├── IDS/                        # Módulos do sistema IDS
│   ├── ids_collector.py        # Daemon de captura (3 threads)
│   ├── ids_detector.py         # Motor de detecção (watch/batch/file)
│   ├── ids_learn.py            # Treinamento e fine-tuning
│   ├── ids_reports.py          # Geração de relatórios HTML/TXT
│   ├── ids_manager.py          # CLI alternativa de gerenciamento
│   └── modules/
│       ├── utils.py            # Logging, cores ANSI, @timed, run_background
│       ├── custom_layers.py    # BahdanauAttention e Focal Loss serializáveis
│       ├── incident_engine.py  # Classificação, MITRE ATT&CK, severidade
│       ├── flow_features.py    # FlowTracker, PacketMinimal, extração
│       ├── versioning.py       # Versionamento de artefatos por execução
│       ├── model_registry.py   # Triplete M0/Mp/Mc (baseline/anterior/atual)
│       ├── full_report.py      # Relatório comparativo M0/Mp/Mc
│       └── evaluator.py        # Avaliação genérica de modelos
│
├── Tests/                      # Scripts de análise acadêmica
│   ├── _test_logging.py        # Logger compartilhado + safe_run + helpers
│   ├── analise_1_arquiteturas.py
│   ├── analise_2_balanceamento.py
│   ├── analise_3_teoria_informacao.py
│   ├── analise_4_otimizacao_validacao.py
│   ├── Logs/                   # Logs individuais por execução
│   └── Reports/                # Relatórios em Markdown + CSV + PNG
│       ├── Relatorio_1_Arquiteturas/
│       ├── Relatorio_2_Balanceamento/
│       ├── Relatorio_3_Teoria_Informacao/
│       └── Relatorio_4_Otimizacao_Validacao/
│
├── Base/
│   └── CSE-CIC-IDS2018/        # CSVs do dataset (não versionados)
│
├── Model/                      # Artefatos do modelo treinado
│   ├── ids_lstm_model.keras    # Modelo Keras serializado
│   ├── scaler.pkl              # StandardScaler (joblib)
│   ├── label_encoder.pkl       # LabelEncoder (joblib)
│   ├── ids_model_info.json     # Metadados: versão, features, label_mapping
│   └── registry/               # Triplete M0/Mp/Mc
│       ├── registry.json
│       └── baseline/
│
├── Reports/                    # Relatórios HTML do IDS (operação)
├── Logs/                       # App.log, Collector.log, Learn.log
├── temp/                       # Caches, staging, PID files (minúsculo)
│   ├── capture/                # Parquets capturados pelo Collector
│   └── staging/                # Dados anotados para fine-tuning
│
├── install.sh                  # Script de instalação do ambiente
└── requirements.txt            # Dependências Python
```

---

## Requisitos

**Hardware mínimo (ambiente de referência):**

| Recurso | Especificação |
|---------|---------------|
| CPU | 8+ vCPUs (recomendado: 20) |
| RAM | 32 GB (recomendado: 64 GB) |
| Disco | 200 GB livres (SAS 10K ou SSD) |
| GPU | Não requerida (otimizado para CPU) |
| Rede | Interface dedicada para captura (eth1, ens18, etc.) |
| SO | Ubuntu 22.04+ / Debian 12+ |

**Software:**

| Dependência | Versão testada |
|-------------|----------------|
| Python | 3.10+ |
| TensorFlow | 2.15+ (validado em 2.21) |
| scikit-learn | 1.3+ |
| imbalanced-learn | 0.11+ |
| pandas | 2.0+ |
| NumPy | 1.24+ |
| SciPy | 1.11+ |
| Scapy | 2.5+ (apenas para captura) |
| PyArrow | 14.0+ |
| Matplotlib | 3.7+ |
| Seaborn | 0.12+ |
| joblib | 1.3+ |
| xgboost | 2.0+ (apenas para Análise 1) |

---

## Instalação

### 1. Clonar o repositório

```bash
git clone https://github.com/<seu-usuario>/SecurityIA.git
cd SecurityIA
```

### 2. Criar o ambiente virtual

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Se o arquivo `requirements.txt` não estiver disponível, instale manualmente:

```bash
pip install tensorflow pandas numpy scipy scikit-learn imbalanced-learn \
    joblib matplotlib seaborn pyarrow scapy xgboost
```

### 4. Verificar a instalação

```bash
python3 -c "
import tensorflow as tf
import sklearn, imblearn, pandas, numpy, scipy
print(f'TensorFlow {tf.__version__}')
print(f'scikit-learn {sklearn.__version__}')
print(f'imbalanced-learn {imblearn.__version__}')
print('Todas as dependências OK.')
"
```

### 5. Criar diretórios do projeto

```bash
python3 -c "from config import Config; Config.ensure_dirs(); print('Diretórios criados.')"
```

### 6. Permissões para captura de pacotes (opcional)

Necessário apenas se for usar o Collector para captura em tempo real:

```bash
sudo setcap cap_net_raw,cap_net_admin=eip $(readlink -f .venv/bin/python3)
```

---

## Configuração

Toda a configuração é centralizada no arquivo `config.py`. Os parâmetros mais relevantes:

### Caminhos

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `DATA_DIR` | `Base/CSE-CIC-IDS2018/` | Diretório dos CSVs do dataset |
| `MODEL_DIR` | `Model/` | Artefatos do modelo treinado |
| `COLLECTOR_DIR` | `temp/capture/` | Saída do Collector (Parquets) |
| `REPORTS_DIR` | `Reports/` | Relatórios HTML do IDS |
| `TEMP_DIR` | `temp/` | Caches e staging (minúsculo) |

### Captura

| Parâmetro | Padrão | Descrição |
|-----------|--------|-----------|
| `CAPTURE_INTERFACE` | `eth1` | Interface de rede para captura |
| `COLLECTOR_BUDGET_GB` | `7.0` | Budget diário em GiB |
| `COLLECTOR_SAMPLE_RATE` | `1.0` | Fração de fluxos capturados (1.0 = todos) |
| `COLLECTOR_FLUSH_ROWS` | `50000` | Fluxos acumulados antes de flush em disco |

### Modelo e Treinamento

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `lstm_units_1` | `128` | Unidades da primeira camada Bi-LSTM |
| `lstm_units_2` | `64` | Unidades da segunda camada Bi-LSTM |
| `attention_units` | `64` | Unidades da camada de Atenção de Bahdanau |
| `dense_units` | `32` | Unidades da camada densa final |
| `dropout_rate` | `0.5` | Taxa de dropout entre camadas |
| `recurrent_dropout_rate` | `0.0` | Mantido em zero para preservar caminho oneDNN |
| `loss_function` | `focal_loss_cb` | Focal Loss reponderada (Cui et al., 2019) |
| `focal_gamma` | `2.0` | Parâmetro de focalização |
| `focal_class_balanced_beta` | `0.9999` | Parâmetro de reponderação por classe |
| `learning_rate` | `1e-3` | Taxa de aprendizado inicial (Adam) |
| `batch_size` | `4096` | Tamanho do batch (otimizado para CPU) |
| `epochs` | `30` | Épocas máximas (teto de segurança) |
| `patience` | `4` | Épocas sem melhoria antes de parar (EarlyStopping) |
| `early_stopping_min_delta` | `1e-3` | Ganho mínimo em val_loss para considerar melhoria |
| `lr_reduce_factor` | `0.3` | Fator de redução do LR no ReduceLROnPlateau |
| `lr_reduce_patience` | `2` | Épocas sem melhoria antes de reduzir LR |
| `lr_min` | `1e-6` | LR mínimo permitido |
| `steps_per_execution` | `64` | Steps agrupados por chamada Python (reduz overhead) |
| `use_class_weight` | `False` | Desligado (evita dupla penalização com Focal Loss) |
| `split_before_balancing` | `True` | Particionamento antes do balanceamento (D1) |
| `balance_only_train` | `True` | Balanceamento apenas no treino |

### Seleção de Features (Teoria da Informação)

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `k_best` | `23` | Número de features selecionadas |
| `ig_weight` | `0.6` | Peso do Information Gain no score combinado |
| `mi_weight` | `0.4` | Peso da Mutual Information no score combinado |
| `ig_discretization_bins` | `10` | Bins para discretização do IG |

### Balanceamento

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `strategy` | `"none"` | Sem balanceamento sintético (decisão D3) |

Os parâmetros legados (`smote_k_neighbors`, `enn_n_neighbors`, etc.) permanecem no arquivo apenas para reprodução das comparações feitas na Análise 2.

### CPU Threading

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `inter_op_threads` | `4` | Threads para operações independentes |
| `intra_op_threads` | `20` | Threads dentro de cada operação (álgebra linear) |

---

## Dataset

O projeto utiliza o **CSE-CIC-IDS2018** (Canadian Institute for Cybersecurity), versão cleaned disponível no Mendeley Data (< 800 MB). O dataset contém tráfego de rede rotulado com 15 classes:

- Benign
- Bot
- DDoS (HOIC, LOIC-UDP, LOIC-HTTP)
- DoS (GoldenEye, Hulk, Slowhttptest, Slowloris)
- FTP-Patator, SSH-Patator
- Heartbleed
- Infiltration
- Web Attack (Brute Force, SQL Injection, XSS)

### Preparação

Extraia os 14 CSVs para o diretório `Base/CSE-CIC-IDS2018/`:

```
Base/CSE-CIC-IDS2018/
├── bot.csv
├── brute force -web.csv
├── brute force -xss.csv
├── ddos attack-hoic.csv
├── ddos attack-loic-udp.csv
├── ddos attacks-loic-http.csv
├── dos attacks-goldeneye.csv
├── dos attacks-hulk.csv
├── dos attacks-slowhttptest.csv
├── dos attacks-slowloris.csv
├── ftp-bruteforce.csv
├── infilteration.csv
├── sql injection.csv
└── ssh-bruteforce.csv
```

### Verificação

```bash
python3 -c "from config import verificar_dataset; verificar_dataset()"
```

---

## Uso

### Menu Interativo

O ponto de entrada principal do projeto. Unifica testes acadêmicos e operação do IDS em uma interface de terminal.

```bash
python3 app_menu.py
```

```
══════════════════════════════════════════════════════════════
  SecurityIA  v3.0.0
  Unified CLI for Tests and Intelligent IDS Operations
──────────────────────────────────────────────────────────────
  [1] Realizar Testes
      Acesso ao conjunto de análises experimentais, dataset e relatórios.

  [2] Sistema IDS
      Operação do motor de IDS, treinamento, diagnósticos, logs e gestão.

  [0] Sair
```

O menu do IDS dá acesso direto a:

- **Captura de Pacotes** — iniciar/parar Collector, listar arquivos, importar PCAP
- **Detecção de Incidentes** — análise em lote, monitor contínuo, arquivo específico
- **Treinamento** — completo, forçado, fine-tuning incremental, avaliação
- **Relatórios** — listar HTML/JSONL, abrir no browser, exportar
- **Status do Sistema** — processos ativos, modelo, dados capturados, pendências
- **Diagnósticos** — benchmark de CPU/GPU, disco, rede, integridade do dataset
- **Logs** — visualização por módulo, exportação, limpeza por idade
- **Configurações** — diretórios, dependências, recriação de estrutura

### Execução em Background

As análises acadêmicas (item [1] do menu) podem ser executadas individualmente em foreground/background ou todas em sequência no background. Em ambos os casos, a execução é **imune ao encerramento da sessão SSH**:

```
Submenu Tests/
  [1] Análise 1 — Arquiteturas
      → escolha F (foreground) ou B (background)
  ...
  [5] Executar todas as análises (background)
```

Quando o modo background é escolhido, o processo mestre é desacoplado da sessão via `nohup setsid` e roda em novo grupo de sessão. O terminal é devolvido imediatamente. Cada análise roda em subprocess Python **independente** — sem leak de TensorFlow acumulado, sem `importlib.reload`. **Falha de uma análise não interrompe as demais**.

**Acompanhar progresso:**

```bash
tail -f Tests/Logs/run_all.log              # log mestre da fila
ls -lt Tests/Logs/analise_*.log             # logs individuais
pgrep -af analise_                           # verificar se ainda está rodando
```

**Cancelar execução em andamento:**

```bash
pkill -f analise_
```

### Testes Acadêmicos

Os quatro scripts validam empiricamente cada decisão metodológica da dissertação:

```bash
cd Tests/

# Análise 1 — Arquiteturas
# Compara RandomForest, XGBoost, MLP+BatchNorm, ResNet Tabular e Bi-LSTM com
# Atenção. A Bi-LSTM é avaliada como hipótese secundária — vetores de fluxo
# agregado não têm estrutura sequencial natural; o resultado serve como
# referência para validação metodológica.
python3 analise_1_arquiteturas.py

# Análise 2 — Tratamento de Desbalanceamento via Perda
# Compara cinco estratégias: Sem_Tratamento, ClassWeight_Balanced, FocalLoss
# puro, CB_FocalLoss (Cui et al., 2019) e Undersample_Benign. SMOTE foi
# removido da comparação após validação empírica de degradação (decisão D3).
python3 analise_2_balanceamento.py

# Análise 3 — Seleção de Features
# Avalia cinco métodos (Information_Gain, Mutual_Information, ANOVA_F,
# RF_Feature_Importance, IG_MI_60_40) sobre TODAS as features brutas
# (~77 colunas após limpeza), em grade k = [10, 15, 23, 32, 48, all].
python3 analise_3_teoria_informacao.py

# Análise 4 — Otimização e Validação
# Compara 10 combinações otimizador × scheduler (Adam, AdamW, RMSprop, SGD
# Momentum × none, ReduceLROnPlateau, CosineDecay) sobre dados reais com
# mesmo split estratificado.
python3 analise_4_otimizacao_validacao.py
```

Cada análise gera relatório em Markdown, figuras em PNG e tabelas em CSV no diretório `Tests/Reports/`. **Critério primário de seleção: Recall macro + MCC**. F1-macro é reportado como métrica complementar.

### Treinamento do Modelo

```bash
# Treinamento completo (do zero)
python3 IDS/ids_learn.py train

# Treinamento forçado (ignora todos os caches)
python3 IDS/ids_learn.py train --force

# Fine-tuning sobre dados anotados pelo Detector
python3 IDS/ids_learn.py finetune

# Status do modelo atual + triplete M0/Mp/Mc
python3 IDS/ids_learn.py status
```

O pipeline de treinamento executa:

1. Carregamento e limpeza (cache em Parquet em `temp/`)
2. Seleção de features (IG + MI ponderados, k=23)
3. **Split estratificado 70/15/15 sobre dados ORIGINAIS** (antes de qualquer balanceamento — correção D1)
4. Construção do modelo Bi-LSTM + Bahdanau com **Focal Loss reponderada** (correção D2)
5. Treinamento com EarlyStopping (`min_delta=1e-3`, `patience=4`) e ReduceLROnPlateau (`factor=0.3`, `patience=2`)
6. Avaliação sobre o teste **ORIGINAL** (não balanceado)
7. Registro automático no triplete M0/Mp/Mc
8. Geração de relatório comparativo (matriz de confusão, métricas por classe, evolução do treinamento)

Caches intermediários são salvos em `temp/` para evitar reprocessamento. Para um re-treino limpo:

```bash
rm -f temp/01_cleaned_dataset.parquet
rm -f temp/03_X_scaled_unbalanced.pkl temp/03_y_unbalanced.pkl
python3 IDS/ids_learn.py train --force
```

### Captura de Tráfego

O Collector opera como daemon com três threads coordenadas:

```bash
# Foreground (para depuração)
python3 IDS/ids_collector.py

# Background (produção)
python3 IDS/ids_collector.py --background

# Opções de override
python3 IDS/ids_collector.py --interface ens18 --budget-gb 15.0

# Via systemd (recomendado em produção)
sudo systemctl start ids-collector
```

A captura gera um Parquet por dia em `temp/capture/`, com rotação automática à meia-noite e controle de budget diário. Ao atingir o limite configurado, a captura é suspensa até o próximo dia sem encerrar o processo.

### Detecção de Incidentes

```bash
# Monitor contínuo (verifica novos Parquets a cada 60s)
python3 IDS/ids_detector.py watch --interval 60

# Monitor em background
python3 IDS/ids_detector.py watch --background

# Análise de todos os Parquets pendentes
python3 IDS/ids_detector.py batch

# Análise de arquivo específico
python3 IDS/ids_detector.py file temp/capture/captura_Seg_07_04_2026.parquet
```

Para cada arquivo processado, o Detector:

1. Carrega o modelo com `keras.load_model(..., compile=False)` — inferência não requer otimizador
2. Normaliza os fluxos com o mesmo `StandardScaler` do treinamento
3. Classifica cada fluxo via Bi-LSTM + Atenção de Bahdanau
4. Mapeia classes preditas para níveis de severidade e técnicas MITRE ATT&CK
5. Exporta incidentes em JSONL (para SIEM) e dados anotados em staging (para fine-tuning)
6. Gera relatório HTML com métricas, heatmap, top IPs e recomendações

### Relatórios

Os relatórios HTML do IDS são gerados automaticamente pelo Detector e salvos em `Reports/`. Incluem:

- Cards de métricas: fluxos totais, incidentes, taxa de ataque, críticos
- Tabela de ataques por tipo com severidade e mapeamento MITRE ATT&CK
- Top 10 IPs de origem e destino
- Heatmap de atividade por hora (UTC)
- Tabela detalhada de incidentes (até 200, ordenados por severidade)
- Recomendações operacionais contextualizadas por tipo de ameaça
- Informações do modelo (versão, data, classes, features)

```bash
# Listar relatórios
ls -lh Reports/*.html

# Abrir no navegador
xdg-open Reports/relatorio_001_v202604_20260411.html
```

---

## Pipeline de Dados

```
Rede (pacotes IP)
  │
  ▼
CaptureThread ─── Scapy AsyncSniffer + filtro BPF
  │                extrai PacketMinimal (80 bytes), descarta objeto Scapy
  │
  ▼
ProcessorThread ── FlowTracker online
  │                agrega pacotes em fluxos bidirecionais
  │                sweep de timeouts (idle=30s, active=120s)
  │                23 features CIC-IDS2018 + 6 metadados
  │
  ▼
WriterThread ───── Parquet (Snappy)
  │                flush a cada 50.000 fluxos ou 60s
  │                rotação diária, budget por GiB/dia
  │
  ▼
Detector ──────── Normalização (StandardScaler salvo)
  │               Classificação (Bi-LSTM + Bahdanau, compile=False)
  │               Mapeamento de severidade + MITRE ATT&CK
  │
  ├─── JSONL de incidentes (para SIEM)
  ├─── Parquet anotado em staging/ (para fine-tuning)
  └─── Relatório HTML
```

---

## Modelo

### Arquitetura

```
Input (23, 1)
  │
  ▼
Bidirectional LSTM (128 unidades, return_sequences=True)
  │
Dropout (0.5)
  │
  ▼
Bidirectional LSTM (64 unidades, return_sequences=True)
  │
Dropout (0.5)
  │
  ▼
Atenção de Bahdanau (64 unidades)
  │  e_t = V^T · tanh(W_h · h_t + b)
  │  α_t = softmax(e_t)
  │  c   = Σ α_t · h_t
  │
  ▼
Dense (32, ReLU)
  │
Dropout (0.25)
  │
  ▼
Dense (15, Softmax)
```

A `BahdanauAttention` está implementada em `IDS/modules/custom_layers.py` com o decorator `@keras.saving.register_keras_serializable(package="SecurityIA")`, garantindo que `keras.load_model()` reconstrua o modelo sem necessidade de passar `custom_objects` explicitamente.

### Função de Perda

```
Focal Loss reponderada por número efetivo de amostras (Cui et al., 2019):

    n_eff_c = (1 - β^n_c) / (1 - β)
    w_c     = (1 - β) / n_eff_c          (com β = 0.9999)
    L       = -Σ w_c · (1 - p_t)^γ · log(p_t)     (com γ = 2.0)
```

A reponderação é baseada na contagem de amostras **do conjunto de treino**, calculada após o split estratificado.

### Seleção de Features

As 23 features são selecionadas a partir do espaço original do CIC-IDS2018 usando um score combinado:

```
score(f) = 0.6 × IG_normalizado(f) + 0.4 × MI_normalizado(f)
```

Onde IG é calculado via discretização uniforme (10 bins) e MI via estimador baseado em k-vizinhos (scikit-learn).

### Métricas Reportadas

Para cada modelo treinado, o sistema reporta:

| Métrica | Uso |
|---------|-----|
| **Recall macro** | **Métrica primária** — sensibilidade média entre classes |
| **MCC** | **Métrica secundária** — correlação multiclasse, robusta ao desbalanceamento |
| F1-macro | Métrica complementar |
| F1-weighted | Compatibilidade com a literatura (deflacionado pelo desbalanceamento) |
| Acurácia | Compatibilidade com a literatura |
| FPR-macro | Taxa de falsos positivos média (interesse operacional) |
| Balanced accuracy | Acurácia média por classe |

---

## Versionamento de Modelos (Triplete M0/Mp/Mc)

Cada treinamento ou fine-tuning é registrado no triplete `M0/Mp/Mc`:

- **M0** — Modelo de referência (RandomForest baseline, gerado por `baseline_rf.py`). Imutável.
- **Mp** — Versão anterior do framework (modelo Bi-LSTM + Bahdanau previamente treinado).
- **Mc** — Versão corrente. Substitui Mp quando um novo treinamento é executado.

O registro é feito automaticamente em `Model/registry/registry.json` por `IDS/modules/model_registry.py`. Após cada treino ou fine-tuning, o sistema gera um **relatório comparativo** em `Reports/IDS/full_<tipo>_<timestamp>/` que apresenta:

- Métricas lado a lado de M0, Mp e Mc
- Matrizes de confusão das três versões
- Decisão de promoção (Mc supera Mp?) baseada nas métricas primárias

---

## Conformidade com Diretrizes de IA Generativa

Em conformidade com o Art. 9.º da Portaria CNPq n.º 2.664/2026, declara-se que ferramentas de IA generativa foram utilizadas como apoio à revisão textual, à depuração de scripts e à triagem bibliográfica preliminar. Todo o conteúdo apoiado por essas ferramentas foi revisado e validado pelo autor, que assume responsabilidade integral pela precisão, pela metodologia e pela integridade científica do trabalho.

---

## Referências Bibliográficas

- Sharafaldin, I., Lashkari, A. H., & Ghorbani, A. A. (2018). Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization. *ICISSP*.
- Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural Machine Translation by Jointly Learning to Align and Translate. *ICLR*.
- Lin, T.-Y., Goyal, P., Girshick, R., He, K., & Dollár, P. (2017). Focal Loss for Dense Object Detection. *ICCV*.
- Cui, Y., Jia, M., Lin, T.-Y., Song, Y., & Belongie, S. (2019). Class-Balanced Loss Based on Effective Number of Samples. *CVPR*.
- Peng, H., Long, F., & Ding, C. (2005). Feature Selection Based on Mutual Information: Criteria of Max-Dependency, Max-Relevance, and Min-Redundancy. *IEEE TPAMI*, 27(8), 1226–1238.
- Hochreiter, S., & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation*, 9(8), 1735–1780.
- Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic Minority Over-sampling Technique. *JAIR*, 16, 321–357.
- Wilson, D. L. (1972). Asymptotic Properties of Nearest Neighbor Rules Using Edited Data. *IEEE Transactions on Systems, Man, and Cybernetics*, 2(3), 408–421.

---

## Autoria

| | |
|---|---|
| **Autor** | Bruno Cavalcante Barbosa — bcb@ic.ufal.br |
| **Orientador** | Prof. Dr. André Luiz Lins de Aquino |
| **Programa** | PPGI — Instituto de Computação — UFAL |
| **Dissertação** | Detecção de Anomalias em Redes de Computadores Utilizando Aprendizado de Máquina |

---

## Licença

Este projeto é parte de uma dissertação de mestrado acadêmica. Consulte o orientador para informações sobre uso e redistribuição.
