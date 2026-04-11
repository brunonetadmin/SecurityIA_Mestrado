#!/usr/bin/env python3
"""
IDS/ids_collector.py — Daemon de Captura Contínua de Tráfego de Rede
Versão: 2.0

Arquitetura de 3 threads:
  CaptureThread  → extrai PacketMinimal (80 B) e descarta o objeto Scapy
  ProcessorThread→ FlowTracker online, sweep por timeout, amostragem
  WriterThread   → flush periódico para Parquet (Snappy), rotação diária

Features: 23 (CIC-IDS2018) + 6 metadados de endereçamento
Budget diário configurável. Shutdown gracioso via SIGTERM/SIGINT.

Uso:
    python3 IDS/ids_collector.py [--interface eth1] [--background]
    systemctl start ids-collector   # via systemd (recomendado em produção)
"""

import argparse
import os
import queue
import signal
import sys
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config

# ── Dependências de captura ───────────────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    sys.exit(f"[ERRO] {e}. Execute: pip install numpy pandas pyarrow")

try:
    from scapy.all import AsyncSniffer, IP, TCP, UDP, conf as scapy_conf
    scapy_conf.verb = 0
except ImportError:
    sys.exit("[ERRO] Scapy ausente. Execute: pip install scapy")

# ── Módulos internos ──────────────────────────────────────────────────────────
from IDS.modules.flow_features import (
    FlowTracker, PacketMinimal,
    PROTO_TCP, PROTO_UDP, PROTO_ICMP,
    FLAG_FIN, FLAG_RST,
)
from IDS.modules.utils import get_collector_logger, format_bytes, format_duration

log = get_collector_logger()

# ── Schema Parquet de saída ───────────────────────────────────────────────────
_FEAT_COLS = Config.FEATURE_COLUMNS
_META_COLS = Config.META_COLUMNS

_SCHEMA = pa.schema(
    [(c, pa.float64()) for c in _FEAT_COLS]
    + [("Label",         pa.string())]
    + [("meta_src_ip",   pa.string()),
       ("meta_dst_ip",   pa.string()),
       ("meta_src_port", pa.int32()),
       ("meta_dst_port", pa.int32()),
       ("meta_flow_start", pa.float64()),
       ("meta_flow_end",   pa.float64())]
)

_DIAS_PT = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui",
            4: "Sex", 5: "Sab", 6: "Dom"}

def _filename(d: date) -> str:
    return f"captura_{_DIAS_PT[d.weekday()]}_{d.strftime('%d_%m_%Y')}.parquet"


# ─────────────────────────────────────────────────────────────────────────────
# Thread 1 — CaptureThread
# ─────────────────────────────────────────────────────────────────────────────

class CaptureThread(threading.Thread):
    """
    Captura pacotes IP (TCP/UDP/ICMP) via Scapy AsyncSniffer com filtro BPF
    no kernel — reduz drasticamente o tráfego de dados entre kernel e userspace.
    Extrai apenas os campos necessários e descarta o objeto Scapy imediatamente.
    """
    BPF = "ip and (tcp or udp or icmp)"

    def __init__(self, iface: str, pkt_q: queue.Queue, stop: threading.Event) -> None:
        super().__init__(name="Capture", daemon=True)
        self._iface  = iface
        self._pkt_q  = pkt_q
        self._stop   = stop
        self._ok     = 0
        self._drop   = 0

    def run(self) -> None:
        log.info(f"[Capture] Iniciado — interface='{self._iface}' filtro='{self.BPF}'")
        try:
            sn = AsyncSniffer(
                iface=self._iface, filter=self.BPF,
                prn=self._on_pkt, store=False,
            )
            sn.start()
            while not self._stop.is_set():
                time.sleep(0.5)
            if sn.running:
                sn.stop()
        except Exception as e:
            log.error(f"[Capture] Erro fatal: {e}", exc_info=True)
        log.info(f"[Capture] Encerrado — capturados={self._ok:,} descartados={self._drop:,}")

    def _on_pkt(self, pkt) -> None:
        if IP not in pkt:
            return
        ip    = pkt[IP]
        proto = ip.proto
        ts    = float(pkt.time)
        src   = sys.intern(ip.src)
        dst   = sys.intern(ip.dst)
        length = len(ip)
        sp = dp = flags = 0

        if proto == PROTO_TCP and TCP in pkt:
            sp    = pkt[TCP].sport
            dp    = pkt[TCP].dport
            flags = int(pkt[TCP].flags)
        elif proto == PROTO_UDP and UDP in pkt:
            sp = pkt[UDP].sport
            dp = pkt[UDP].dport

        m = PacketMinimal(ts, src, dst, sp, dp, proto, length, flags)
        try:
            self._pkt_q.put_nowait(m)
            self._ok += 1
        except queue.Full:
            self._drop += 1


# ─────────────────────────────────────────────────────────────────────────────
# Thread 2 — ProcessorThread
# ─────────────────────────────────────────────────────────────────────────────

class ProcessorThread(threading.Thread):
    """
    Consome PacketMinimal da fila, mantém FlowTracker em memória.
    Faz sweep de timeouts a cada SWEEP_INTERVAL segundos.
    """
    SWEEP_INTERVAL = 10  # segundos

    def __init__(
        self, pkt_q: queue.Queue, flow_q: queue.Queue,
        stop: threading.Event, budget_hit: threading.Event,
        tracker: FlowTracker,
    ) -> None:
        super().__init__(name="Processor", daemon=True)
        self._pkt_q      = pkt_q
        self._flow_q     = flow_q
        self._stop       = stop
        self._budget_hit = budget_hit
        self._tracker    = tracker
        self._done       = 0
        self._drop       = 0

    def run(self) -> None:
        log.info("[Processor] Iniciado")
        last_sweep = time.time()

        while True:
            try:
                pkt = self._pkt_q.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():
                    break
                now = time.time()
                if now - last_sweep >= self.SWEEP_INTERVAL:
                    self._drain(self._tracker.sweep(now))
                    last_sweep = now
                continue

            if not self._budget_hit.is_set():
                feat = self._tracker.process(pkt)
                if feat:
                    self._enq(feat)

        # Drena o restante após sinal de stop
        while not self._pkt_q.empty():
            try:
                feat = self._tracker.process(self._pkt_q.get_nowait())
                if feat:
                    self._enq(feat)
            except queue.Empty:
                break
        self._drain(self._tracker.flush_all())
        self._flow_q.put(None)  # sentinel
        log.info(f"[Processor] Encerrado — fluxos={self._done:,} descartados={self._drop:,}")

    def _drain(self, feats: list) -> None:
        for f in feats:
            self._enq(f)

    def _enq(self, feat: dict) -> None:
        try:
            self._flow_q.put_nowait(feat)
            self._done += 1
        except queue.Full:
            self._drop += 1


# ─────────────────────────────────────────────────────────────────────────────
# Thread 3 — WriterThread
# ─────────────────────────────────────────────────────────────────────────────

class WriterThread(threading.Thread):
    """
    Consome features da fila e as persiste em Parquet (Snappy).
    Rotação automática à meia-noite. Controle de budget diário.
    """

    def __init__(
        self,
        output_dir: Path, budget_gb: float,
        flow_q: queue.Queue, stop: threading.Event,
        budget_hit: threading.Event,
        flush_rows: int = Config.COLLECTOR_FLUSH_ROWS,
        flush_secs: int = Config.COLLECTOR_FLUSH_SECS,
    ) -> None:
        super().__init__(name="Writer", daemon=True)
        self._dir          = output_dir
        self._budget_bytes = int(budget_gb * 1024 ** 3)
        self._flow_q       = flow_q
        self._stop         = stop
        self._budget_hit   = budget_hit
        self._flush_rows   = flush_rows
        self._flush_secs   = flush_secs
        self._buf: list    = []
        self._cur_date: Optional[date] = None
        self._writer       = None
        self._path: Optional[Path] = None
        self._bytes_today  = 0
        self._rows_today   = 0
        self._total_rows   = 0
        self._last_flush   = time.time()
        self._t_start      = time.time()

    def run(self) -> None:
        log.info(f"[Writer] Iniciado — saída='{self._dir}' budget={self._budget_bytes/1024**3:.1f} GiB")
        while True:
            try:
                item = self._flow_q.get(timeout=0.5)
            except queue.Empty:
                self._maybe_flush()
                continue
            if item is None:   # sentinel
                break
            self._buf.append(item)
            if len(self._buf) >= self._flush_rows:
                self._flush()
            else:
                self._maybe_flush()

        self._flush(force=True)
        self._close()
        elapsed = time.time() - self._t_start
        log.info(
            f"[Writer] Encerrado — total={self._total_rows:,} linhas | "
            f"tempo={format_duration(elapsed)} | "
            f"tamanho≈{format_bytes(self._bytes_today)}"
        )

    def _maybe_flush(self) -> None:
        if time.time() - self._last_flush >= self._flush_secs:
            self._flush()

    def _flush(self, force: bool = False) -> None:
        if not self._buf:
            return

        today = date.today()
        if self._cur_date != today:
            self._close()
            self._cur_date    = today
            self._bytes_today = 0
            self._rows_today  = 0
            self._budget_hit.clear()
            self._open(today)

        if self._writer is None:
            self._buf.clear()
            return

        if self._budget_hit.is_set():
            log.warning(
                f"[Writer] Budget atingido ({self._budget_bytes/1024**3:.1f} GiB). "
                f"Descartando {len(self._buf):,} fluxos."
            )
            self._buf.clear()
            return

        try:
            df = pd.DataFrame(self._buf)
            df["Label"] = "Unknown"

            # Garante todas as colunas do schema
            for col in _FEAT_COLS:
                if col not in df.columns:
                    df[col] = 0.0
            for col in ("meta_src_ip", "meta_dst_ip"):
                if col not in df.columns:
                    df[col] = ""
            for col in ("meta_src_port", "meta_dst_port"):
                if col not in df.columns:
                    df[col] = 0

            col_order = _FEAT_COLS + ["Label"] + _META_COLS
            df = df[[c for c in col_order if c in df.columns]]

            tbl = pa.Table.from_pandas(df, schema=_SCHEMA, preserve_index=False)
            self._writer.write_table(tbl)

            est = int(df.memory_usage(deep=True).sum() * 0.28)  # compressão Snappy ~72%
            self._bytes_today += est
            self._rows_today  += len(df)
            self._total_rows  += len(df)
            self._last_flush   = time.time()

            log.info(
                f"[Writer] {len(df):,} fluxos → '{self._path.name}' | "
                f"hoje={format_bytes(self._bytes_today)} / {self._budget_bytes/1024**3:.1f} GiB"
            )

            if self._bytes_today >= self._budget_bytes:
                log.warning("[Writer] Budget diário atingido. Captura suspensa até meia-noite.")
                self._budget_hit.set()

        except Exception as e:
            log.error(f"[Writer] Erro no flush: {e}", exc_info=True)
        finally:
            self._buf.clear()

    def _open(self, d: date) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path   = self._dir / _filename(d)
        self._writer = pq.ParquetWriter(str(self._path), schema=_SCHEMA, compression="snappy")
        log.info(f"[Writer] Arquivo aberto: '{self._path.name}'")

    def _close(self) -> None:
        if self._writer:
            try:
                self._writer.close()
                log.info(
                    f"[Writer] Arquivo fechado: '{self._path.name}' — "
                    f"{self._rows_today:,} fluxos | {format_bytes(self._bytes_today)}"
                )
            except Exception as e:
                log.error(f"[Writer] Erro ao fechar arquivo: {e}")
            finally:
                self._writer = None
                self._path   = None


# ─────────────────────────────────────────────────────────────────────────────
# Daemon principal
# ─────────────────────────────────────────────────────────────────────────────

class CollectorDaemon:
    """Orquestra as 3 threads e gerencia o ciclo de vida do daemon."""

    def __init__(
        self,
        interface:   str   = Config.CAPTURE_INTERFACE,
        output_dir:  Path  = Config.COLLECTOR_DIR,
        sample_rate: float = Config.COLLECTOR_SAMPLE_RATE,
        budget_gb:   float = Config.COLLECTOR_BUDGET_GB,
    ) -> None:
        self._interface   = interface
        self._output_dir  = output_dir
        self._sample_rate = sample_rate
        self._budget_gb   = budget_gb
        self._pkt_q       = queue.Queue(maxsize=Config.COLLECTOR_PKT_QUEUE_SIZE)
        self._flow_q      = queue.Queue(maxsize=Config.COLLECTOR_FLOW_QUEUE_SIZE)
        self._stop        = threading.Event()
        self._budget_hit  = threading.Event()
        self._tracker     = FlowTracker(
            sample_rate    = sample_rate,
            active_timeout = Config.COLLECTOR_ACTIVE_TIMEOUT,
            idle_timeout   = Config.COLLECTOR_IDLE_TIMEOUT,
            max_packets    = Config.COLLECTOR_MAX_PKT_FLOW,
        )

    def _setup_signals(self) -> None:
        def _handler(sig, _):
            log.info(f"[Daemon] {signal.Signals(sig).name} recebido — encerrando…")
            self._stop.set()
        signal.signal(signal.SIGTERM, _handler)
        signal.signal(signal.SIGINT,  _handler)

    def run(self) -> None:
        self._setup_signals()
        Config.ensure_dirs()

        log.info("=" * 62)
        log.info("SecurityIA Collector v2.0 — Daemon de Captura de Tráfego")
        log.info("=" * 62)
        log.info(f"Interface   : {self._interface}")
        log.info(f"Saída       : {self._output_dir}")
        log.info(f"Amostragem  : {self._sample_rate * 100:.0f} %")
        log.info(f"Budget/dia  : {self._budget_gb:.1f} GiB")
        log.info(f"Arquivo hoje: {_filename(date.today())}")
        log.info("=" * 62)

        threads = [
            WriterThread(self._output_dir, self._budget_gb,
                         self._flow_q, self._stop, self._budget_hit),
            ProcessorThread(self._pkt_q, self._flow_q, self._stop,
                            self._budget_hit, self._tracker),
            CaptureThread(self._interface, self._pkt_q, self._stop),
        ]
        for t in threads:
            t.start()

        MONITOR = 30
        while not self._stop.is_set():
            self._stop.wait(timeout=MONITOR)
            if not self._stop.is_set():
                log.info(
                    f"[Monitor] pkt_q={self._pkt_q.qsize():,} "
                    f"flow_q={self._flow_q.qsize():,} "
                    f"fluxos_ativos={self._tracker.active_count:,}"
                )

        for t in threads:
            t.join(timeout=60)
            if t.is_alive():
                log.warning(f"[Daemon] Thread {t.name} não encerrou no prazo.")
        log.info("[Daemon] Encerrado com sucesso.")


# ─────────────────────────────────────────────────────────────────────────────
# Ponto de entrada
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="SecurityIA — Collector Daemon de captura contínua de tráfego.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Toda a configuração é lida de config.py. Os argumentos abaixo são\n"
            "opcionais e sobrescrevem o config apenas para esta execução.\n\n"
            "Exemplo (systemd):\n"
            "  ExecStart=/usr/bin/python3 /opt/SecurityIA/IDS/ids_collector.py\n\n"
            "Requer cap_net_raw:\n"
            "  sudo setcap cap_net_raw,cap_net_admin=eip $(which python3)"
        ),
    )
    p.add_argument("--interface",   "-i", default=None,      help=f"Interface de rede (padrão: {Config.CAPTURE_INTERFACE})")
    p.add_argument("--output",      "-o", type=Path, default=None, help=f"Diretório de saída (padrão: {Config.COLLECTOR_DIR})")
    p.add_argument("--sample-rate", type=float, default=None, help=f"Fração de fluxos 0.0–1.0 (padrão: {Config.COLLECTOR_SAMPLE_RATE})")
    p.add_argument("--budget-gb",   type=float, default=None, help=f"Budget diário em GiB (padrão: {Config.COLLECTOR_BUDGET_GB})")
    p.add_argument("--background",  action="store_true",      help="Executa em segundo plano (nohup)")
    args = p.parse_args()

    if args.background:
        import subprocess
        cmd = [sys.executable] + [a for a in sys.argv if a != "--background"]
        log_file = Config.LOG_COLLECTOR
        log_fh = open(log_file, "a")
        proc = subprocess.Popen(
            cmd, stdout=log_fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        log_fh.close()
        print(f"  Collector iniciado em background — PID {proc.pid}")
        print(f"  Log: {log_file}")
        return

    CollectorDaemon(
        interface   = args.interface   or Config.CAPTURE_INTERFACE,
        output_dir  = args.output      or Config.COLLECTOR_DIR,
        sample_rate = args.sample_rate if args.sample_rate is not None else Config.COLLECTOR_SAMPLE_RATE,
        budget_gb   = args.budget_gb   if args.budget_gb   is not None else Config.COLLECTOR_BUDGET_GB,
    ).run()


if __name__ == "__main__":
    main()
