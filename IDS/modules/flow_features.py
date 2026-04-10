#!/usr/bin/env python3
"""
IDS/modules/flow_features.py — Extração de Features de Fluxo de Rede

Implementa o pipeline de extração das 23 features estatísticas compatíveis
com o dataset CSE-CIC-IDS2018, operando em O(1) de memória por fluxo.

Componentes:
  - WelfordAccumulator : média/variância online (Welford, 1962)
  - ActiveIdleTracker  : períodos ativo/ocioso de cada fluxo
  - FlowState          : estado completo de um fluxo bidirecional
  - SlidingWindowTracker: Count e Same_Service_Rate em janela temporal
  - FlowTracker        : tabela de fluxos ativos + finalização
  - extract_features() : calcula as 23 features de um FlowState finalizado

Importado por: IDS/ids_collector.py
"""

import math
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

# ── Constantes de protocolo e flags TCP ──────────────────────────────────────
PROTO_TCP  = 6
PROTO_UDP  = 17
PROTO_ICMP = 1

FLAG_FIN = 0x01
FLAG_SYN = 0x02
FLAG_RST = 0x04
FLAG_PSH = 0x08
FLAG_ACK = 0x10
FLAG_URG = 0x20

IDLE_THRESHOLD = 1.0   # segundos de silêncio para abrir período ocioso


# ─────────────────────────────────────────────────────────────────────────────
# Acumuladores estatísticos O(1)
# ─────────────────────────────────────────────────────────────────────────────

class WelfordAccumulator:
    """
    Computa média e variância amostral online (Welford, 1962).
    Uso de memória: O(1) — armazena apenas (n, mean, M2, total).
    """
    __slots__ = ("_n", "_mean", "_M2", "_total")

    def __init__(self) -> None:
        self._n = 0; self._mean = 0.0; self._M2 = 0.0; self._total = 0.0

    def update(self, v: float) -> None:
        self._n += 1; self._total += v
        delta = v - self._mean
        self._mean += delta / self._n
        self._M2   += delta * (v - self._mean)

    @property
    def count(self) -> int:   return self._n
    @property
    def mean(self) -> float:  return self._mean if self._n > 0 else 0.0
    @property
    def variance(self) -> float:
        return (self._M2 / (self._n - 1)) if self._n > 1 else 0.0
    @property
    def std(self) -> float:
        v = self.variance; return math.sqrt(v) if v > 0 else 0.0
    @property
    def total(self) -> float: return self._total


class ActiveIdleTracker:
    """
    Rastreia períodos ativo/ocioso de um fluxo em tempo linear.
    Usa listas internas para calcular média e std de cada período.
    """
    __slots__ = ("_act_start", "_last", "_thresh", "_acts", "_idles")

    def __init__(self, t0: float, thresh: float = IDLE_THRESHOLD) -> None:
        self._thresh = thresh; self._act_start = t0; self._last = t0
        self._acts: List[float] = []; self._idles: List[float] = []

    def update(self, t: float) -> None:
        gap = t - self._last
        if gap >= self._thresh:
            dur = self._last - self._act_start
            if dur > 0:
                self._acts.append(dur)
            self._idles.append(gap)
            self._act_start = t
        self._last = t

    def finalize(self, t: float) -> None:
        dur = t - self._act_start
        if dur > 0:
            self._acts.append(dur)

    @staticmethod
    def _stat(lst: List[float]) -> Tuple[float, float]:
        if not lst:
            return 0.0, 0.0
        n = len(lst); m = sum(lst) / n
        s = math.sqrt(sum((x - m) ** 2 for x in lst) / (n - 1)) if n > 1 else 0.0
        return m, s

    @property
    def active_mean(self) -> float: return self._stat(self._acts)[0]
    @property
    def active_std(self) -> float:  return self._stat(self._acts)[1]
    @property
    def idle_mean(self) -> float:   return self._stat(self._idles)[0]
    @property
    def idle_std(self) -> float:    return self._stat(self._idles)[1]


# ─────────────────────────────────────────────────────────────────────────────
# PacketMinimal — estrutura mínima de pacote (evita manter obj Scapy em memória)
# ─────────────────────────────────────────────────────────────────────────────

class PacketMinimal:
    """
    Representação compacta de um pacote IP. ~80 bytes vs ~1500 bytes do pacote
    Scapy original. Descartamos o objeto Scapy logo após a extração.
    """
    __slots__ = ("timestamp", "src_ip", "dst_ip", "src_port",
                 "dst_port", "protocol", "ip_length", "tcp_flags")

    def __init__(
        self,
        timestamp: float, src_ip: str, dst_ip: str,
        src_port: int, dst_port: int, protocol: int,
        ip_length: int, tcp_flags: int,
    ) -> None:
        self.timestamp  = timestamp
        self.src_ip     = src_ip
        self.dst_ip     = dst_ip
        self.src_port   = src_port
        self.dst_port   = dst_port
        self.protocol   = protocol
        self.ip_length  = ip_length
        self.tcp_flags  = tcp_flags


# ─────────────────────────────────────────────────────────────────────────────
# FlowState — estado de um fluxo bidirecional
# ─────────────────────────────────────────────────────────────────────────────

class FlowState:
    """Estado completo de um fluxo bidirecional com acumuladores online."""
    __slots__ = (
        "flow_key", "start_time", "last_time", "packet_count", "max_packets",
        "iat_acc", "fwd_len_acc", "bwd_len_acc", "all_len_acc",
        "fwd_count", "fwd_bytes", "fwd_first", "fwd_last",
        "bwd_count", "bwd_bytes", "bwd_first", "bwd_last",
        "syn_count", "psh_urg_count", "protocol", "dst_port", "active_idle",
    )

    def __init__(self, flow_key: tuple, pkt: PacketMinimal, max_packets: int) -> None:
        self.flow_key    = flow_key
        self.start_time  = pkt.timestamp
        self.last_time   = pkt.timestamp
        self.packet_count = 1
        self.max_packets  = max_packets

        self.iat_acc     = WelfordAccumulator()
        self.fwd_len_acc = WelfordAccumulator()
        self.bwd_len_acc = WelfordAccumulator()
        self.all_len_acc = WelfordAccumulator()

        self.fwd_count   = 1
        self.fwd_bytes   = pkt.ip_length
        self.fwd_first   = pkt.timestamp
        self.fwd_last    = pkt.timestamp
        self.fwd_len_acc.update(float(pkt.ip_length))

        self.bwd_count   = 0
        self.bwd_bytes   = 0
        self.bwd_first   = 0.0
        self.bwd_last    = 0.0

        self.all_len_acc.update(float(pkt.ip_length))

        f = pkt.tcp_flags
        self.syn_count     = 1 if (f & FLAG_SYN) else 0
        self.psh_urg_count = 1 if (f & (FLAG_PSH | FLAG_URG)) else 0
        self.protocol      = pkt.protocol
        self.dst_port      = pkt.dst_port
        self.active_idle   = ActiveIdleTracker(pkt.timestamp)

    def update(self, pkt: PacketMinimal, is_fwd: bool) -> bool:
        """
        Atualiza o estado do fluxo com um novo pacote.
        Retorna False se o fluxo atingiu max_packets (deve ser finalizado).
        """
        self.packet_count += 1
        if self.packet_count > self.max_packets:
            return False

        iat = pkt.timestamp - self.last_time
        self.iat_acc.update(iat)
        self.active_idle.update(pkt.timestamp)
        self.all_len_acc.update(float(pkt.ip_length))

        f = pkt.tcp_flags
        if f & FLAG_SYN:           self.syn_count     += 1
        if f & (FLAG_PSH | FLAG_URG): self.psh_urg_count += 1

        if is_fwd:
            self.fwd_count += 1
            self.fwd_bytes += pkt.ip_length
            self.fwd_last   = pkt.timestamp
            self.fwd_len_acc.update(float(pkt.ip_length))
        else:
            self.bwd_count += 1
            self.bwd_bytes += pkt.ip_length
            if self.bwd_count == 1:
                self.bwd_first = pkt.timestamp
            self.bwd_last = pkt.timestamp
            self.bwd_len_acc.update(float(pkt.ip_length))

        self.last_time = pkt.timestamp
        return True


# ─────────────────────────────────────────────────────────────────────────────
# SlidingWindowTracker — Count e Same_Service_Rate
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowTracker:
    """
    Rastreia fluxos recentes em janela deslizante para calcular:
      - Count         : nº de fluxos para o mesmo dst_ip na janela
      - Same_Service  : fração com o mesmo dst_port
    """
    __slots__ = ("_window", "_secs")

    def __init__(self, secs: float = 2.0) -> None:
        self._secs   = secs
        self._window: deque = deque()

    def record(self, ts: float, dst_ip: str, dst_port: int) -> None:
        self._window.append((ts, dst_ip, dst_port))

    def query(self, ts: float, dst_ip: str, dst_port: int) -> Tuple[float, float]:
        cut = ts - self._secs
        while self._window and self._window[0][0] < cut:
            self._window.popleft()
        cnt = same = 0
        for _, d, p in self._window:
            if d == dst_ip:
                cnt += 1
                same += (1 if p == dst_port else 0)
        return float(cnt), (same / cnt if cnt > 0 else 0.0)


# ─────────────────────────────────────────────────────────────────────────────
# Extração das 23 features
# ─────────────────────────────────────────────────────────────────────────────

def extract_features(
    state: FlowState,
    sw: SlidingWindowTracker,
) -> Optional[dict]:
    """
    Calcula as 23 features + 6 metadados de um FlowState finalizado.
    Retorna None se o fluxo for muito curto (duração ≤ 0).
    """
    dur = state.last_time - state.start_time
    if dur <= 0.0:
        return None

    sd  = max(dur, 1e-9)
    tb  = state.fwd_bytes + state.bwd_bytes
    tp  = state.fwd_count + state.bwd_count
    fi  = (state.fwd_last - state.fwd_first) if state.fwd_count > 1 else 0.0
    bi  = (state.bwd_last - state.bwd_first) if state.bwd_count > 1 else 0.0

    state.active_idle.finalize(state.last_time)

    sw.record(state.last_time, state.flow_key[1], state.dst_port)
    cnt, rate = sw.query(state.last_time, state.flow_key[1], state.dst_port)

    return {
        # ── 23 features CIC-IDS2018 ──────────────────────────────────────
        "Flow_Duration":          dur,
        "Total_Fwd_Packets":      float(state.fwd_count),
        "Flow_Bytes_s":           tb / sd,
        "Flow_Packets_s":         tp / sd,
        "Fwd_Packet_Length_Mean": state.fwd_len_acc.mean,
        "Bwd_Packet_Length_Mean": state.bwd_len_acc.mean,
        "Flow_IAT_Mean":          state.iat_acc.mean,
        "Fwd_IAT_Total":          fi,
        "Bwd_IAT_Total":          bi,
        "Packet_Length_Variance": state.all_len_acc.variance,
        "Flow_IAT_Std":           state.iat_acc.std,
        "Active_Mean":            state.active_idle.active_mean,
        "Active_Std":             state.active_idle.active_std,
        "Idle_Mean":              state.active_idle.idle_mean,
        "Idle_Std":               state.active_idle.idle_std,
        "TCP_Flag_Count":         float(state.syn_count),
        "Protocol_Type":          float(state.protocol),
        "Service_Type":           float(state.dst_port),
        "Flag_Type":              float(state.psh_urg_count),
        "Source_Bytes":           float(state.fwd_bytes),
        "Destination_Bytes":      float(state.bwd_bytes),
        "Count":                  cnt,
        "Same_Service_Rate":      rate,
        # ── 6 metadados de endereçamento (descartados no treinamento) ────
        "meta_src_ip":            state.flow_key[0],
        "meta_dst_ip":            state.flow_key[1],
        "meta_src_port":          int(state.flow_key[2]),
        "meta_dst_port":          int(state.flow_key[3]),
        "meta_flow_start":        state.start_time,
        "meta_flow_end":          state.last_time,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FlowTracker — tabela de fluxos ativos
# ─────────────────────────────────────────────────────────────────────────────

class FlowTracker:
    """
    Mantém um dicionário de fluxos ativos, cria/atualiza/finaliza FlowStates.
    Thread-safe apenas para uso por um único ProcessorThread.
    """

    def __init__(
        self,
        sample_rate: float,
        active_timeout: int,
        idle_timeout: int,
        max_packets: int,
        sliding_secs: float = 2.0,
    ) -> None:
        self._rate         = sample_rate
        self._active_to    = active_timeout
        self._idle_to      = idle_timeout
        self._max_packets  = max_packets
        self._active: Dict[tuple, FlowState] = {}
        self._sw           = SlidingWindowTracker(sliding_secs)

    @staticmethod
    def _flow_key(pkt: PacketMinimal) -> Tuple[tuple, bool]:
        """
        Chave bidirecional canônica: menor 4-tupla lexicográfica é a chave.
        Retorna (chave, is_forward).
        """
        a = (pkt.src_ip, pkt.dst_ip, pkt.src_port, pkt.dst_port, pkt.protocol)
        b = (pkt.dst_ip, pkt.src_ip, pkt.dst_port, pkt.src_port, pkt.protocol)
        if a <= b:
            return a, True
        return b, False

    def process(self, pkt: PacketMinimal) -> Optional[dict]:
        """
        Processa um pacote. Retorna dict de features se o fluxo for finalizado,
        None caso contrário.
        """
        key, fwd = self._flow_key(pkt)

        if key not in self._active:
            # Aplica amostragem apenas em novos fluxos
            if random.random() >= self._rate:
                return None
            self._active[key] = FlowState(key, pkt, self._max_packets)
            return None

        state = self._active[key]
        ok    = state.update(pkt, fwd)

        # Finaliza por: max_packets atingido ou FIN/RST TCP
        close = (not ok) or (
            pkt.protocol == PROTO_TCP and bool(pkt.tcp_flags & (FLAG_FIN | FLAG_RST))
        )
        if close:
            del self._active[key]
            return extract_features(state, self._sw)
        return None

    def sweep(self, now: float) -> List[dict]:
        """Expira fluxos por timeout ativo ou ocioso. Retorna features finalizados."""
        expired = [
            k for k, st in self._active.items()
            if (now - st.last_time  >= self._idle_to)
            or (now - st.start_time >= self._active_to)
        ]
        results = []
        for k in expired:
            f = extract_features(self._active.pop(k), self._sw)
            if f:
                results.append(f)
        return results

    def flush_all(self) -> List[dict]:
        """Finaliza todos os fluxos ativos (shutdown do daemon)."""
        results = []
        for k in list(self._active):
            f = extract_features(self._active.pop(k), self._sw)
            if f:
                results.append(f)
        return results

    @property
    def active_count(self) -> int:
        return len(self._active)
