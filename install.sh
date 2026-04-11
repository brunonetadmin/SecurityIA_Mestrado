#!/usr/bin/env bash
# ==============================================================================
#  SecurityIA — Script de Instalação e Configuração de Ambiente
#  Versão: 3.0
#
#  Compatível com: Ubuntu Server 24.04 LTS
#
#  O que este script faz:
#    1. Instala dependências do sistema (Python 3.11, libpcap, etc.)
#    2. Cria e configura o ambiente virtual Python
#    3. Instala todas as dependências Python do projeto
#    4. Aplica tuning de kernel para captura de alta performance
#    5. Configura capacidades de captura (cap_net_raw sem root)
#    6. Cria templates de serviços systemd
#    7. Configura rotação de logs
#
#  Uso:
#    chmod +x install.sh
#    sudo ./install.sh [--no-venv] [--no-tuning] [--no-systemd]
#
#  Autor: Bruno Cavalcante Barbosa — PPGI/IC/UFAL
# ==============================================================================

set -euo pipefail
IFS=$'\n\t'

# ── Cores ─────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

log()  { echo -e "${CYAN}[INFO ]${RESET} $*"; }
ok()   { echo -e "${GREEN}[OK   ]${RESET} $*"; }
warn() { echo -e "${YELLOW}[WARN ]${RESET} $*"; }
err()  { echo -e "${RED}[ERROR]${RESET} $*"; }
sep()  { echo -e "${BOLD}══════════════════════════════════════════════════════${RESET}"; }

# ── Configuração ──────────────────────────────────────────────────────────────
INSTALL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$INSTALL_DIR/.venv"
PYTHON_MIN="3.10"
DO_VENV=true
DO_TUNING=true
DO_SYSTEMD=true

for arg in "$@"; do
  case $arg in
    --no-venv)    DO_VENV=false ;;
    --no-tuning)  DO_TUNING=false ;;
    --no-systemd) DO_SYSTEMD=false ;;
  esac
done

# ── Verificação de root ───────────────────────────────────────────────────────
if [[ $EUID -ne 0 ]]; then
  err "Este script deve ser executado como root: sudo $0"
  exit 1
fi

REAL_USER="${SUDO_USER:-$(logname 2>/dev/null || echo 'ubuntu')}"
REAL_HOME=$(getent passwd "$REAL_USER" | cut -d: -f6)

sep
echo -e "${BOLD}  SecurityIA — Instalação e Configuração${RESET}"
echo "  Diretório : $INSTALL_DIR"
echo "  Usuário   : $REAL_USER"
echo "  Python    : $PYTHON_MIN+"
sep

# ==============================================================================
# § 1 — Dependências do sistema
# ==============================================================================
log "Atualizando índice APT …"
apt-get update

log "Instalando dependências do sistema …"
apt-get install -y --no-install-recommends \
  python3 python3-dev python3-venv python3-pip \
  build-essential gcc g++ \
  libpcap-dev libpcap0.8 \
  tcpdump net-tools ethtool \
  libhdf5-dev libhdf5-serial-dev \
  pkg-config \
  git curl wget \
  logrotate \
  procps lsof \
  numactl

ok "Dependências do sistema instaladas."

# ==============================================================================
# § 2 — Ambiente virtual Python
# ==============================================================================
if $DO_VENV; then
  log "Criando ambiente virtual em '$VENV_DIR' …"
  python3 -m venv "$VENV_DIR"
  PYTHON="$VENV_DIR/bin/python"
  PIP="$VENV_DIR/bin/pip"
  chown -R "$REAL_USER:$REAL_USER" "$VENV_DIR"
  ok "Ambiente virtual criado."
else
  PYTHON="$(which python3 || which python3)"
  PIP="$(which pip3)"
  warn "--no-venv: usando Python do sistema: $PYTHON"
fi

"$PIP" install --upgrade pip setuptools wheel -q

# ==============================================================================
# § 3 — Dependências Python
# ==============================================================================
log "Instalando dependências Python …"

PY_DEPS=(
  "numpy>=1.24.0,<2.0.0"
  "pandas>=2.0.0"
  "pyarrow>=14.0.0"
  "scipy>=1.11.0"
  "scikit-learn>=1.4.0"
  "imbalanced-learn>=0.12.0"
  "joblib>=1.3.0"
  "tensorflow-cpu>=2.15.0"
  "keras>=3.0.0"
  "scapy>=2.5.0"
  "matplotlib>=3.7.0"
  "seaborn>=0.13.0"
  "optuna>=3.5.0"
  "shap>=0.44.0"
  "psutil>=5.9.0"
  "tqdm>=4.66.0"
)

"$PIP" install -q "${PY_DEPS[@]}"

ok "Dependências Python instaladas."

# Verificação rápida
log "Verificando imports críticos …"
"$PYTHON" -c "
import numpy, pandas, pyarrow, sklearn, tensorflow, scapy, scipy
print(f'  numpy={numpy.__version__}')
print(f'  pandas={pandas.__version__}')
print(f'  tensorflow={tensorflow.__version__}')
print(f'  sklearn={sklearn.__version__}')
print(f'  scapy={scapy.__version__}')
"
ok "Imports verificados."

# ==============================================================================
# § 4 — Capacidades de captura (sem precisar de root em produção)
# ==============================================================================
log "Configurando cap_net_raw para Python …"
PYTHON_BIN="$(realpath "$PYTHON")"
setcap cap_net_raw,cap_net_admin=eip "$PYTHON_BIN" 2>/dev/null && \
  ok "cap_net_raw configurado em '$PYTHON_BIN'." || \
  warn "Não foi possível configurar cap_net_raw. Execute o collector como root."

# ==============================================================================
# § 5 — Tuning de kernel (Ubuntu 24.04 — captura de alta performance)
# ==============================================================================
if $DO_TUNING; then
  log "Aplicando tuning de kernel para captura de rede …"

  SYSCTL_FILE="/etc/sysctl.d/99-security-ia.conf"
  cat > "$SYSCTL_FILE" << 'EOF'
# ==============================================================
#  SecurityIA — Tuning de Kernel (Ubuntu 24.04)
#  Otimizado para: captura Scapy em 10 Gbps + TF training/inference
# ==============================================================

# ── Buffers de socket de rede (captura Scapy / pcap) ─────────
# Aumenta buffer de recepção para reduzir drops em alta taxa de pacotes.
# 256 MiB por socket — Scapy AsyncSniffer usa SO_RCVBUF implicitamente.
net.core.rmem_max          = 268435456
net.core.rmem_default      = 268435456
net.core.wmem_max          = 268435456
net.core.wmem_default      = 268435456
net.core.netdev_max_backlog = 300000
net.core.netdev_budget     = 600
net.core.netdev_budget_usecs = 8000

# ── Backlog de conexões TCP ───────────────────────────────────
net.core.somaxconn         = 65536
net.ipv4.tcp_max_syn_backlog = 65536

# ── Timeouts TCP ─────────────────────────────────────────────
net.ipv4.tcp_fin_timeout   = 15
net.ipv4.tcp_keepalive_time = 300
net.ipv4.tcp_keepalive_probes = 3
net.ipv4.tcp_keepalive_intvl = 15

# ── Memória do kernel para buffers de pacote ─────────────────
net.ipv4.tcp_rmem          = 4096 131072 268435456
net.ipv4.tcp_wmem          = 4096 65536  268435456
net.ipv4.udp_rmem_min      = 16384
net.ipv4.udp_wmem_min      = 16384

# ── Desabilita reverse path filtering (porta mirror) ─────────
# CRÍTICO para captura em porta mirror: pacotes vindos de IPs
# que não são rota da interface seriam descartados sem este ajuste.
net.ipv4.conf.all.rp_filter    = 0
net.ipv4.conf.default.rp_filter = 0

# ── Descomentar se usar eth1 especificamente ──────────────────
# net.ipv4.conf.eth1.rp_filter = 0

# ── Memória virtual (TensorFlow / NumPy) ──────────────────────
# vm.swappiness baixo mantém dados de modelo em RAM.
vm.swappiness             = 10
vm.dirty_ratio            = 15
vm.dirty_background_ratio = 5

# ── Hugepages para TensorFlow MKL (opcional, ajuda com matrizes grandes) ─────
# vm.nr_hugepages = 512  # descomente se tiver 64+ GB RAM

# ── Limite de arquivos abertos ────────────────────────────────
fs.file-max               = 2097152

# ── Parâmetros de IRQ / NAPI ─────────────────────────────────
kernel.pid_max            = 4194304
EOF

  sysctl -p "$SYSCTL_FILE" >/dev/null 2>&1
  ok "Parâmetros de kernel aplicados em '$SYSCTL_FILE'."

  # ── Limites de recursos (ulimits) ────────────────────────────────────────
  LIMITS_FILE="/etc/security/limits.d/99-security-ia.conf"
  cat > "$LIMITS_FILE" << EOF
# SecurityIA — limites de recursos do usuário $REAL_USER
$REAL_USER soft nofile 1048576
$REAL_USER hard nofile 1048576
$REAL_USER soft nproc  65536
$REAL_USER hard nproc  65536
$REAL_USER soft memlock unlimited
$REAL_USER hard memlock unlimited
EOF
  ok "Limites de recursos configurados em '$LIMITS_FILE'."

  # ── IRQ affinity e offloads da NIC ───────────────────────────────────────
  IFACE="eth1"  # mesma que Config.CAPTURE_INTERFACE
  if ip link show "$IFACE" &>/dev/null; then
    # Desabilita GRO/LRO na interface de captura (reduz latência de processamento)
    ethtool -K "$IFACE" gro off lro off 2>/dev/null && \
      ok "GRO/LRO desabilitado em $IFACE." || \
      warn "Não foi possível desabilitar GRO/LRO em $IFACE."

    # Aumenta ring buffer da NIC
    ethtool -G "$IFACE" rx 4096 tx 4096 2>/dev/null && \
      ok "Ring buffer aumentado em $IFACE." || \
      warn "Ring buffer em $IFACE não alterado (driver pode não suportar)."
  else
    warn "Interface '$IFACE' não encontrada. Configure CAPTURE_INTERFACE em config.py."
  fi

else
  warn "--no-tuning: otimizações de kernel ignoradas."
fi

# ==============================================================================
# § 6 — Criação de diretórios do projeto
# ==============================================================================
log "Criando estrutura de diretórios …"
for d in Base Model Logs Temp Temp/capture Temp/staging Temp/evaluation Reports Tests; do
  mkdir -p "$INSTALL_DIR/$d"
done
chown -R "$REAL_USER:$REAL_USER" "$INSTALL_DIR"
ok "Diretórios criados."

# ==============================================================================
# § 7 — Rotação de logs (logrotate)
# ==============================================================================
log "Configurando rotação de logs …"
cat > /etc/logrotate.d/security-ia << EOF
$INSTALL_DIR/Logs/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    copytruncate
    dateext
    dateformat -%Y%m%d
    create 0640 $REAL_USER $REAL_USER
}
EOF
ok "Logrotate configurado."

# ==============================================================================
# § 8 — Templates systemd
# ==============================================================================
if $DO_SYSTEMD; then
  log "Criando templates systemd …"

  VENV_PYTHON="${PYTHON}"

  # ── ids-collector.service ─────────────────────────────────────────────────
  cat > /etc/systemd/system/ids-collector.service << EOF
[Unit]
Description=SecurityIA — Collector Daemon de Captura de Rede
Documentation=file://$INSTALL_DIR/README.md
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$REAL_USER
Group=$REAL_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$VENV_PYTHON $INSTALL_DIR/IDS/ids_collector.py
Restart=on-failure
RestartSec=15
StandardOutput=append:$INSTALL_DIR/Logs/Collector.log
StandardError=append:$INSTALL_DIR/Logs/Collector.log
LimitNOFILE=1048576

# Capacidade de captura de pacotes (dispensa execução como root)
AmbientCapabilities=CAP_NET_RAW CAP_NET_ADMIN
CapabilityBoundingSet=CAP_NET_RAW CAP_NET_ADMIN

# Hardening
PrivateTmp=true
NoNewPrivileges=false
ProtectSystem=strict
ReadWritePaths=$INSTALL_DIR/Temp $INSTALL_DIR/Logs

[Install]
WantedBy=multi-user.target
EOF

  # ── ids-learn.service (oneshot — disparado por timer) ────────────────────
  cat > /etc/systemd/system/ids-learn.service << EOF
[Unit]
Description=SecurityIA — Treinamento/Fine-tuning do Modelo LSTM
After=network.target

[Service]
Type=oneshot
User=$REAL_USER
Group=$REAL_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$VENV_PYTHON $INSTALL_DIR/IDS/ids_learn.py train
TimeoutStartSec=7200
StandardOutput=append:$INSTALL_DIR/Logs/Learn.log
StandardError=append:$INSTALL_DIR/Logs/Learn.log
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
EOF

  # ── ids-learn.timer (semanal, domingo às 02:00) ───────────────────────────
  cat > /etc/systemd/system/ids-learn.timer << EOF
[Unit]
Description=SecurityIA — Timer de Re-treinamento Semanal

[Timer]
OnCalendar=Sun 02:00:00
Persistent=true
Unit=ids-learn.service

[Install]
WantedBy=timers.target
EOF

  # ── ids-detector.service ──────────────────────────────────────────────────
  cat > /etc/systemd/system/ids-detector.service << EOF
[Unit]
Description=SecurityIA — Detector de Incidentes (modo watch)
After=ids-collector.service
Wants=ids-collector.service

[Service]
Type=simple
User=$REAL_USER
Group=$REAL_USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$VENV_PYTHON $INSTALL_DIR/IDS/ids_detector.py watch --interval 60
Restart=on-failure
RestartSec=30
StandardOutput=append:$INSTALL_DIR/Logs/App.log
StandardError=append:$INSTALL_DIR/Logs/App.log
LimitNOFILE=1048576

[Install]
WantedBy=multi-user.target
EOF

  systemctl daemon-reload
  ok "Serviços systemd criados e daemon recarregado."

  echo
  log "Para ativar os serviços:"
  echo "    sudo systemctl enable --now ids-collector"
  echo "    sudo systemctl enable --now ids-detector"
  echo "    sudo systemctl enable --now ids-learn.timer  # re-treino semanal"
else
  warn "--no-systemd: serviços systemd não criados."
fi

# ==============================================================================
# § 9 — Verificação final
# ==============================================================================
sep
echo -e "${BOLD}  Verificação Final${RESET}"
sep

CHECKS=(
  "Scapy disponível:$("$PYTHON" -c 'import scapy; print("OK")' 2>/dev/null || echo 'FALHOU')"
  "TensorFlow (CPU):$("$PYTHON" -c 'import tensorflow as tf; tf.config.set_visible_devices([],"GPU"); print("OK")' 2>/dev/null || echo 'FALHOU')"
  "imbalanced-learn:$("$PYTHON" -c 'import imblearn; print("OK")' 2>/dev/null || echo 'FALHOU')"
  "pyarrow         :$("$PYTHON" -c 'import pyarrow; print("OK")' 2>/dev/null || echo 'FALHOU')"
  "scipy           :$("$PYTHON" -c 'import scipy; print("OK")' 2>/dev/null || echo 'FALHOU')"
)
for c in "${CHECKS[@]}"; do
  label="${c%%:*}"
  status="${c##*:}"
  if [[ "$status" == "OK" ]]; then
    ok "  $label"
  else
    err "  $label — FALHOU"
  fi
done

sep
echo -e "${BOLD}  Instalação concluída!${RESET}"
echo
echo "  Próximos passos:"
echo "  1. Edite config.py → CAPTURE_INTERFACE (interface da porta mirror)"
echo "  2. Coloque os CSVs do CIC-IDS2018 em: $INSTALL_DIR/Base/CSE-CIC-IDS2018/"
echo "  3. Execute o treinamento:"
echo "     source $VENV_DIR/bin/activate"
echo "     python3 IDS/ids_learn.py train"
echo "  4. Inicie o sistema:"
echo "     python3 IDS/ids_manager.py"
echo
sep
