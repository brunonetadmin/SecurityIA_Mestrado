#!/usr/bin/env python3
"""
##############################################################################################
#  ids_coletor.py — Daemon de Captura Contínua de Tráfego de Rede
#  Versão: 1.0
#
#  Totalmente autônomo — lê TODA a configuração de ids_config.py.
#  Após a primeira execução, opera indefinidamente em segundo plano sem
#  necessidade de intervenção do operador.
#
#  Comportamento automático:
#    • Interface de captura lida de IDSConfig.CAPTURE_INTERFACE
#    • Diretório de saída lido de IDSConfig.COLLECTOR_DIR
#    • Fração de amostragem lida de IDSConfig.COLLECTOR_SAMPLE_RATE
#    • Budget diário lido de IDSConfig.COLLECTOR_BUDGET_GB
#    • 1 arquivo por dia: captura_{Dia}_{dd}_{mm}_{aaaa}.parquet
#    • Rotação automática à meia-noite
#    • Shutdown gracioso via SIGTERM / SIGINT / Ctrl+C
#
#  Instalação:
#    pip install scapy pyarrow pandas numpy
#    sudo setcap cap_net_raw,cap_net_admin=eip $(which python3)
#
#  Uso:
#    # Primeira vez (e toda vez que quiser iniciar):
#    python3 ids_coletor.py
#
#    # Ou como serviço systemd (recomendado para produção):
#    # Ver template em /etc/systemd/system/ids-coletor.service abaixo
#
#    # Opcional — sobrescreve apenas a interface sem alterar o config:
#    python3 ids_coletor.py --interface eth2
#
#  Arquivo systemd (/etc/systemd/system/ids-coletor.service):
#    [Unit]
#    Description=IDS Coletor Daemon
#    After=network-online.target
#    Wants=network-online.target
#
#    [Service]
#    User=root
#    ExecStart=/usr/bin/python3 /opt/idsapp/ids_coletor.py
#    Restart=on-failure
#    RestartSec=15
#    StandardOutput=journal
#    StandardError=journal
#
#    [Install]
#    WantedBy=multi-user.target
#
#  Autor: Bruno Cavalcante Barbosa
#  UFAL - Universidade Federal de Alagoas
##############################################################################################
"""

import os, sys, signal, logging, argparse, time, math, random
import threading, queue
from datetime import datetime, date
from pathlib import Path
from collections import deque
from typing import Dict, List, Optional, Tuple, NamedTuple

# ── Dependências de dados ──────────────────────────────────────────────────
try:
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as e:
    sys.exit(f"[ERRO] Dependência ausente: {e}\nExecute: pip install numpy pandas pyarrow")

# ── Scapy ──────────────────────────────────────────────────────────────────
try:
    from scapy.all import AsyncSniffer, IP, TCP, UDP, conf as scapy_conf
    scapy_conf.verb = 0
except ImportError:
    sys.exit("[ERRO] Scapy não encontrado.\nExecute: pip install scapy")

# ── Configuração central ───────────────────────────────────────────────────
try:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from ids_config import IDSConfig
except ImportError:
    # Fallback mínimo para execução sem ids_config.py
    class IDSConfig:
        CAPTURE_INTERFACE     = "eth1"
        COLLECTOR_DIR         = Path("/opt/idsapp/collector")
        COLLECTOR_LOGS_DIR    = Path("/opt/idsapp/logs/collector")
        COLLECTOR_SAMPLE_RATE = 1.0
        COLLECTOR_BUDGET_GB   = 7.0
        LOGGING_CONFIG        = {'level':'INFO',
                                 'format':'%(asctime)s [%(threadName)s] %(levelname)s — %(message)s',
                                 'datefmt':'%Y-%m-%d %H:%M:%S'}


# ══════════════════════════════════════════════════════════════════════════════
# Constantes e Schema
# ══════════════════════════════════════════════════════════════════════════════

DIAS_PT = {0:'Seg',1:'Ter',2:'Qua',3:'Qui',4:'Sex',5:'Sab',6:'Dom'}

FEATURE_COLUMNS: List[str] = [
    'Flow_Duration','Total_Fwd_Packets','Flow_Bytes_s','Flow_Packets_s',
    'Fwd_Packet_Length_Mean','Bwd_Packet_Length_Mean','Flow_IAT_Mean',
    'Fwd_IAT_Total','Bwd_IAT_Total','Packet_Length_Variance',
    'Flow_IAT_Std','Active_Mean','Active_Std','Idle_Mean','Idle_Std',
    'TCP_Flag_Count','Protocol_Type','Service_Type','Flag_Type',
    'Source_Bytes','Destination_Bytes','Count','Same_Service_Rate',
]
META_COLUMNS: List[str] = [
    'meta_src_ip','meta_dst_ip','meta_src_port','meta_dst_port',
    'meta_flow_start','meta_flow_end',
]
LABEL_COLUMN  = 'Label'
DEFAULT_LABEL = 'Unknown'

OUTPUT_SCHEMA = pa.schema(
    [(c, pa.float64()) for c in FEATURE_COLUMNS]
    + [(LABEL_COLUMN, pa.string())]
    + [('meta_src_ip',pa.string()),('meta_dst_ip',pa.string()),
       ('meta_src_port',pa.int32()),('meta_dst_port',pa.int32()),
       ('meta_flow_start',pa.float64()),('meta_flow_end',pa.float64())]
)

PROTO_TCP=6; PROTO_UDP=17; PROTO_ICMP=1
FLAG_FIN=0x01; FLAG_SYN=0x02; FLAG_RST=0x04
FLAG_PSH=0x08; FLAG_ACK=0x10; FLAG_URG=0x20
IDLE_THRESHOLD = 1.0


def make_filename(d: date) -> str:
    return f"captura_{DIAS_PT[d.weekday()]}_{d.strftime('%d_%m_%Y')}.parquet"


# ══════════════════════════════════════════════════════════════════════════════
# Estruturas de dados O(1) memória
# ══════════════════════════════════════════════════════════════════════════════

class PacketMinimal(NamedTuple):
    timestamp:float; src_ip:str; dst_ip:str
    src_port:int;    dst_port:int; protocol:int
    ip_length:int;   tcp_flags:int


class WelfordAccumulator:
    __slots__=('_n','_mean','_M2','_total')
    def __init__(self): self._n=0;self._mean=0.0;self._M2=0.0;self._total=0.0
    def update(self,v:float):
        self._n+=1;self._total+=v;d=v-self._mean
        self._mean+=d/self._n;self._M2+=d*(v-self._mean)
    @property
    def count(self)->int:   return self._n
    @property
    def mean(self)->float:  return self._mean if self._n>0 else 0.0
    @property
    def variance(self)->float: return(self._M2/(self._n-1))if self._n>1 else 0.0
    @property
    def std(self)->float:   v=self.variance;return math.sqrt(v)if v>0 else 0.0
    @property
    def total(self)->float: return self._total


class ActiveIdleTracker:
    __slots__=('_act_start','_last','_thresh','_acts','_idles')
    def __init__(self,t0:float,thresh:float=IDLE_THRESHOLD):
        self._thresh=thresh;self._act_start=t0;self._last=t0
        self._acts:List[float]=[];self._idles:List[float]=[]
    def update(self,t:float):
        gap=t-self._last
        if gap>=self._thresh:
            dur=self._last-self._act_start
            if dur>0:self._acts.append(dur)
            self._idles.append(gap);self._act_start=t
        self._last=t
    def finalize(self,t:float):
        dur=t-self._act_start
        if dur>0:self._acts.append(dur)
    def _stat(self,lst)->Tuple[float,float]:
        if not lst:return 0.0,0.0
        n=len(lst);m=sum(lst)/n
        s=math.sqrt(sum((x-m)**2 for x in lst)/(n-1))if n>1 else 0.0
        return m,s
    @property
    def active_mean(self)->float:return self._stat(self._acts)[0]
    @property
    def active_std(self)->float: return self._stat(self._acts)[1]
    @property
    def idle_mean(self)->float:  return self._stat(self._idles)[0]
    @property
    def idle_std(self)->float:   return self._stat(self._idles)[1]


class FlowState:
    __slots__=(
        'flow_key','start_time','last_time','packet_count','max_packets',
        'iat_acc','fwd_len_acc','bwd_len_acc','all_len_acc',
        'fwd_packet_count','fwd_bytes','fwd_first_time','fwd_last_time',
        'bwd_packet_count','bwd_bytes','bwd_first_time','bwd_last_time',
        'syn_count','psh_urg_count','protocol','dst_port','active_idle',
    )
    def __init__(self,flow_key,first_pkt:PacketMinimal,max_packets:int):
        self.flow_key=flow_key;self.start_time=first_pkt.timestamp
        self.last_time=first_pkt.timestamp;self.packet_count=1
        self.max_packets=max_packets
        self.iat_acc=WelfordAccumulator();self.fwd_len_acc=WelfordAccumulator()
        self.bwd_len_acc=WelfordAccumulator();self.all_len_acc=WelfordAccumulator()
        self.fwd_packet_count=1;self.fwd_bytes=first_pkt.ip_length
        self.fwd_first_time=first_pkt.timestamp;self.fwd_last_time=first_pkt.timestamp
        self.fwd_len_acc.update(float(first_pkt.ip_length))
        self.bwd_packet_count=0;self.bwd_bytes=0
        self.bwd_first_time=0.0;self.bwd_last_time=0.0
        self.all_len_acc.update(float(first_pkt.ip_length))
        f=first_pkt.tcp_flags
        self.syn_count=1 if(f&FLAG_SYN) else 0
        self.psh_urg_count=1 if(f&(FLAG_PSH|FLAG_URG)) else 0
        self.protocol=first_pkt.protocol;self.dst_port=first_pkt.dst_port
        self.active_idle=ActiveIdleTracker(first_pkt.timestamp)

    def update(self,pkt:PacketMinimal,is_fwd:bool)->bool:
        self.packet_count+=1
        if self.packet_count>self.max_packets:return False
        self.iat_acc.update(pkt.timestamp-self.last_time)
        self.active_idle.update(pkt.timestamp)
        self.all_len_acc.update(float(pkt.ip_length))
        f=pkt.tcp_flags
        if f&FLAG_SYN:self.syn_count+=1
        if f&(FLAG_PSH|FLAG_URG):self.psh_urg_count+=1
        if is_fwd:
            self.fwd_packet_count+=1;self.fwd_bytes+=pkt.ip_length
            self.fwd_last_time=pkt.timestamp;self.fwd_len_acc.update(float(pkt.ip_length))
        else:
            self.bwd_packet_count+=1;self.bwd_bytes+=pkt.ip_length
            if self.bwd_packet_count==1:self.bwd_first_time=pkt.timestamp
            self.bwd_last_time=pkt.timestamp;self.bwd_len_acc.update(float(pkt.ip_length))
        self.last_time=pkt.timestamp;return True


class SlidingWindowTracker:
    __slots__=('_window','_secs')
    def __init__(self,secs:float=2.0):
        self._secs=secs;self._window:deque=deque()
    def record(self,ts:float,dst_ip:str,dst_port:int):
        self._window.append((ts,dst_ip,dst_port))
    def query(self,ts:float,dst_ip:str,dst_port:int)->Tuple[float,float]:
        cut=ts-self._secs
        while self._window and self._window[0][0]<cut:self._window.popleft()
        cnt=0;same=0
        for _,d,p in self._window:
            if d==dst_ip:cnt+=1;same+=(1 if p==dst_port else 0)
        return float(cnt),(same/cnt if cnt>0 else 0.0)


def extract_features(state:FlowState,sw:SlidingWindowTracker)->Optional[dict]:
    dur=state.last_time-state.start_time
    if dur<=0.0:return None
    sd=max(dur,1e-9)
    tb=state.fwd_bytes+state.bwd_bytes
    tp=state.fwd_packet_count+state.bwd_packet_count
    fi=state.fwd_last_time-state.fwd_first_time if state.fwd_packet_count>1 else 0.0
    bi=state.bwd_last_time-state.bwd_first_time if state.bwd_packet_count>1 else 0.0
    state.active_idle.finalize(state.last_time)
    sw.record(state.last_time,state.flow_key[1],state.dst_port)
    cnt,rate=sw.query(state.last_time,state.flow_key[1],state.dst_port)
    return {
        'Flow_Duration':          dur,            'Total_Fwd_Packets':       float(state.fwd_packet_count),
        'Flow_Bytes_s':           tb/sd,          'Flow_Packets_s':          tp/sd,
        'Fwd_Packet_Length_Mean': state.fwd_len_acc.mean,
        'Bwd_Packet_Length_Mean': state.bwd_len_acc.mean,
        'Flow_IAT_Mean':          state.iat_acc.mean,
        'Fwd_IAT_Total':          fi,             'Bwd_IAT_Total':           bi,
        'Packet_Length_Variance': state.all_len_acc.variance,
        'Flow_IAT_Std':           state.iat_acc.std,
        'Active_Mean':            state.active_idle.active_mean,
        'Active_Std':             state.active_idle.active_std,
        'Idle_Mean':              state.active_idle.idle_mean,
        'Idle_Std':               state.active_idle.idle_std,
        'TCP_Flag_Count':         float(state.syn_count),
        'Protocol_Type':          float(state.protocol),
        'Service_Type':           float(state.dst_port),
        'Flag_Type':              float(state.psh_urg_count),
        'Source_Bytes':           float(state.fwd_bytes),
        'Destination_Bytes':      float(state.bwd_bytes),
        'Count':                  cnt,            'Same_Service_Rate':       rate,
        'meta_src_ip':            state.flow_key[0],
        'meta_dst_ip':            state.flow_key[1],
        'meta_src_port':          int(state.flow_key[2]),
        'meta_dst_port':          int(state.flow_key[3]),
        'meta_flow_start':        state.start_time,
        'meta_flow_end':          state.last_time,
    }


# ══════════════════════════════════════════════════════════════════════════════
# FlowTracker
# ══════════════════════════════════════════════════════════════════════════════

class FlowTracker:
    def __init__(self,sample_rate:float,active_timeout:int,idle_timeout:int,
                 max_packets:int,sliding_secs:float):
        self._sample_rate=sample_rate
        self._active_timeout=active_timeout
        self._idle_timeout=idle_timeout
        self._max_packets=max_packets
        self._active:Dict[tuple,FlowState]={}
        self._sw=SlidingWindowTracker(sliding_secs)

    @staticmethod
    def _key(p:PacketMinimal)->Tuple[tuple,bool]:
        a=(p.src_ip,p.dst_ip,p.src_port,p.dst_port,p.protocol)
        b=(p.dst_ip,p.src_ip,p.dst_port,p.src_port,p.protocol)
        return(a,True)if a<=b else(b,False)

    def process(self,pkt:PacketMinimal)->Optional[dict]:
        key,fwd=self._key(pkt)
        if key not in self._active:
            if random.random()>=self._sample_rate:return None
            self._active[key]=FlowState(key,pkt,self._max_packets)
            return None
        st=self._active[key]
        ok=st.update(pkt,fwd)
        close=(not ok)or(pkt.protocol==PROTO_TCP and bool(pkt.tcp_flags&(FLAG_FIN|FLAG_RST)))
        if close:
            self._active.pop(key,None)
            return extract_features(st,self._sw)
        return None

    def sweep(self,now:float)->List[dict]:
        expired=[k for k,st in self._active.items()
                 if now-st.last_time>=self._idle_timeout
                 or now-st.start_time>=self._active_timeout]
        done=[]
        for k in expired:
            f=extract_features(self._active.pop(k),self._sw)
            if f:done.append(f)
        return done

    def flush_all(self)->List[dict]:
        done=[]
        for k in list(self._active):
            f=extract_features(self._active.pop(k),self._sw)
            if f:done.append(f)
        return done

    @property
    def active_count(self)->int:return len(self._active)


# ══════════════════════════════════════════════════════════════════════════════
# Threads
# ══════════════════════════════════════════════════════════════════════════════

class CaptureThread(threading.Thread):
    BPF="ip and (tcp or udp or icmp)"
    def __init__(self,iface:str,pq:queue.Queue,stop:threading.Event):
        super().__init__(name="Capture",daemon=True)
        self._iface=iface;self._pq=pq;self._stop=stop
        self._ok=0;self._drop=0

    def run(self):
        logging.info(f"[Capture] interface='{self._iface}' filtro='{self.BPF}'")
        try:
            sn=AsyncSniffer(iface=self._iface,filter=self.BPF,prn=self._on_pkt,store=False)
            sn.start()
            while not self._stop.is_set():time.sleep(0.5)
            if sn.running:sn.stop()
        except Exception as e:
            logging.error(f"[Capture] {e}",exc_info=True)
        logging.info(f"[Capture] capturados={self._ok:,} descartados={self._drop:,}")

    def _on_pkt(self,pkt):
        if IP not in pkt:return
        ip=pkt[IP];proto=ip.proto;ts=float(pkt.time)
        src=sys.intern(ip.src);dst=sys.intern(ip.dst)
        ip_len=len(ip);sp=dp=flags=0
        if proto==PROTO_TCP and TCP in pkt:
            sp=pkt[TCP].sport;dp=pkt[TCP].dport;flags=int(pkt[TCP].flags)
        elif proto==PROTO_UDP and UDP in pkt:
            sp=pkt[UDP].sport;dp=pkt[UDP].dport
        m=PacketMinimal(ts,src,dst,sp,dp,proto,ip_len,flags)
        try:    self._pq.put_nowait(m);self._ok+=1
        except queue.Full:self._drop+=1


class ProcessorThread(threading.Thread):
    SWEEP_INTERVAL=10
    def __init__(self,pq:queue.Queue,fq:queue.Queue,stop:threading.Event,
                 budget_hit:threading.Event,tracker:FlowTracker):
        super().__init__(name="Processor",daemon=True)
        self._pq=pq;self._fq=fq;self._stop=stop
        self._budget_hit=budget_hit;self._tracker=tracker
        self._done=0;self._drop=0

    def run(self):
        last_sweep=time.time()
        while True:
            try:pkt=self._pq.get(timeout=0.1)
            except queue.Empty:
                if self._stop.is_set():break
                now=time.time()
                if now-last_sweep>=self.SWEEP_INTERVAL:
                    for f in self._tracker.sweep(now):self._enq(f)
                    last_sweep=now
                continue
            if not self._budget_hit.is_set():
                f=self._tracker.process(pkt)
                if f:self._enq(f)
        while not self._pq.empty():
            try:
                f=self._tracker.process(self._pq.get_nowait())
                if f:self._enq(f)
            except queue.Empty:break
        for f in self._tracker.flush_all():self._enq(f)
        self._fq.put(None)
        logging.info(f"[Processor] fluxos={self._done:,} descartados={self._drop:,}")

    def _enq(self,feat:dict):
        try:    self._fq.put_nowait(feat);self._done+=1
        except queue.Full:self._drop+=1


class WriterThread(threading.Thread):
    def __init__(self,output_dir:Path,budget_gb:float,
                 fq:queue.Queue,stop:threading.Event,budget_hit:threading.Event,
                 flush_rows:int=100_000,flush_secs:int=60):
        super().__init__(name="Writer",daemon=True)
        self._dir=output_dir;self._budget_bytes=int(budget_gb*1024**3)
        self._fq=fq;self._stop=stop;self._budget_hit=budget_hit
        self._flush_rows=flush_rows;self._flush_secs=flush_secs
        self._buf:List[dict]=[];self._cur_date:Optional[date]=None
        self._writer=None;self._path=None
        self._bytes_today=0;self._rows_today=0;self._total_rows=0
        self._last_flush=time.time()

    def run(self):
        while True:
            try:item=self._fq.get(timeout=0.5)
            except queue.Empty:
                self._maybe_flush();continue
            if item is None:break
            self._buf.append(item)
            if len(self._buf)>=self._flush_rows:self._flush()
            self._maybe_flush()
        self._flush(force=True);self._close()
        logging.info(f"[Writer] total_linhas={self._total_rows:,}")

    def _maybe_flush(self):
        if time.time()-self._last_flush>=self._flush_secs:self._flush()

    def _flush(self,force:bool=False):
        if not self._buf:return
        today=date.today()
        if self._cur_date!=today:
            self._close();self._cur_date=today
            self._bytes_today=0;self._rows_today=0
            self._budget_hit.clear();self._open(today)
        if self._writer is None:self._buf.clear();return
        if self._budget_hit.is_set():
            logging.warning(f"[Writer] Budget {self._budget_bytes/1024**3:.1f}GiB atingido. Descartando {len(self._buf):,} registros.")
            self._buf.clear();return
        try:
            df=pd.DataFrame(self._buf)
            df[LABEL_COLUMN]=DEFAULT_LABEL
            df=df[FEATURE_COLUMNS+[LABEL_COLUMN]+META_COLUMNS]
            tbl=pa.Table.from_pandas(df,schema=OUTPUT_SCHEMA,preserve_index=False)
            self._writer.write_table(tbl)
            est=int(df.memory_usage(deep=True).sum()*0.28)
            self._bytes_today+=est;self._rows_today+=len(df);self._total_rows+=len(df)
            self._last_flush=time.time()
            logging.info(f"[Writer] {len(df):,} fluxos → '{self._path.name}' | "
                         f"hoje={self._bytes_today/1024**2:.0f}MiB/{self._budget_bytes/1024**3:.1f}GiB")
            if self._bytes_today>=self._budget_bytes:
                logging.warning("[Writer] Budget diário atingido.");self._budget_hit.set()
        except Exception as e:
            logging.error(f"[Writer] {e}",exc_info=True)
        finally:
            self._buf.clear()

    def _open(self,d:date):
        self._dir.mkdir(parents=True,exist_ok=True)
        self._path=self._dir/make_filename(d)
        self._writer=pq.ParquetWriter(str(self._path),schema=OUTPUT_SCHEMA,compression='snappy')
        logging.info(f"[Writer] Arquivo aberto: '{self._path.name}'")

    def _close(self):
        if self._writer:
            try:
                self._writer.close()
                logging.info(f"[Writer] Fechado: '{self._path.name}' | "
                             f"{self._rows_today:,} fluxos | {self._bytes_today/1024**2:.1f}MiB")
            except Exception as e:logging.error(f"[Writer] {e}")
            finally:self._writer=None;self._path=None


# ══════════════════════════════════════════════════════════════════════════════
# Daemon Principal
# ══════════════════════════════════════════════════════════════════════════════

class ColetorDaemon:
    # Parâmetros internos (não expostos no config para simplificar)
    _PKT_QUEUE_SIZE    = 200_000
    _FLOW_QUEUE_SIZE   = 50_000
    _ACTIVE_TIMEOUT    = 120
    _IDLE_TIMEOUT      = 30
    _MAX_PACKETS_FLOW  = 10_000
    _SLIDING_WIN_SECS  = 2.0

    def __init__(self, interface: str, output_dir: Path,
                 sample_rate: float, budget_gb: float):
        self._interface   = interface
        self._output_dir  = output_dir
        self._sample_rate = sample_rate
        self._budget_gb   = budget_gb
        self._pq          = queue.Queue(maxsize=self._PKT_QUEUE_SIZE)
        self._fq          = queue.Queue(maxsize=self._FLOW_QUEUE_SIZE)
        self._stop        = threading.Event()
        self._budget_hit  = threading.Event()
        self._tracker     = FlowTracker(
            sample_rate   = sample_rate,
            active_timeout= self._ACTIVE_TIMEOUT,
            idle_timeout  = self._IDLE_TIMEOUT,
            max_packets   = self._MAX_PACKETS_FLOW,
            sliding_secs  = self._SLIDING_WIN_SECS,
        )

    def _setup_logging(self):
        log_dir = IDSConfig.COLLECTOR_LOGS_DIR
        log_dir.mkdir(parents=True, exist_ok=True)
        lf = log_dir / f"coletor_{datetime.now().strftime('%Y%m%d')}.log"
        fmt = logging.Formatter(
            IDSConfig.LOGGING_CONFIG['format'],
            datefmt=IDSConfig.LOGGING_CONFIG['datefmt'])
        root = logging.getLogger()
        root.setLevel(getattr(logging, IDSConfig.LOGGING_CONFIG['level'], logging.INFO))
        root.handlers.clear()
        for h in [logging.FileHandler(lf, encoding='utf-8'),
                  logging.StreamHandler(sys.stdout)]:
            h.setFormatter(fmt); root.addHandler(h)

    def _signals(self):
        def _h(sig, _):
            logging.info(f"[Daemon] {signal.Signals(sig).name} — encerrando...")
            self._stop.set()
        signal.signal(signal.SIGTERM, _h)
        signal.signal(signal.SIGINT,  _h)

    def run(self):
        self._setup_logging()
        self._signals()
        logging.info("═"*60)
        logging.info("IDS COLETOR v1.0 — Daemon de Captura de Tráfego")
        logging.info("═"*60)
        logging.info(f"Interface   : {self._interface}")
        logging.info(f"Saída       : {self._output_dir}")
        logging.info(f"Amostragem  : {self._sample_rate*100:.0f}%")
        logging.info(f"Budget/dia  : {self._budget_gb:.1f} GiB")
        logging.info(f"Arquivo hoje: {make_filename(date.today())}")
        logging.info("═"*60)

        threads = [
            WriterThread(self._output_dir, self._budget_gb,
                         self._fq, self._stop, self._budget_hit),
            ProcessorThread(self._pq, self._fq, self._stop,
                            self._budget_hit, self._tracker),
            CaptureThread(self._interface, self._pq, self._stop),
        ]
        for t in threads: t.start()

        MONITOR = 30
        while not self._stop.is_set():
            self._stop.wait(timeout=MONITOR)
            if not self._stop.is_set():
                logging.info(f"[Monitor] pkt_queue={self._pq.qsize():,} "
                             f"flow_queue={self._fq.qsize():,} "
                             f"fluxos_ativos={self._tracker.active_count:,}")

        for t in threads:
            t.join(timeout=60)
            if t.is_alive():
                logging.warning(f"[Daemon] Thread {t.name} não encerrou no prazo.")
        logging.info("[Daemon] Encerrado.")


# ══════════════════════════════════════════════════════════════════════════════
# Ponto de Entrada
# ══════════════════════════════════════════════════════════════════════════════

def main():
    p = argparse.ArgumentParser(
        description=(
            "IDS Coletor — Daemon de captura contínua de tráfego.\n"
            "Toda a configuração é lida de ids_config.py.\n"
            "Os argumentos abaixo são OPCIONAIS e sobrescrevem o config apenas\n"
            "para esta execução, sem alterar o arquivo de configuração."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        '--interface', '-i',
        default=None,
        help=(f"Interface de rede (padrão: IDSConfig.CAPTURE_INTERFACE = "
              f"'{IDSConfig.CAPTURE_INTERFACE}')"),
    )
    p.add_argument(
        '--output', '-o',
        type=Path, default=None,
        help=(f"Diretório de saída (padrão: IDSConfig.COLLECTOR_DIR = "
              f"'{IDSConfig.COLLECTOR_DIR}')"),
    )
    p.add_argument(
        '--sample-rate', type=float, default=None,
        help=(f"Fração de fluxos capturados 0.0–1.0 "
              f"(padrão: {IDSConfig.COLLECTOR_SAMPLE_RATE})"),
    )
    p.add_argument(
        '--budget-gb', type=float, default=None,
        help=(f"Budget diário em GiB "
              f"(padrão: {IDSConfig.COLLECTOR_BUDGET_GB})"),
    )
    args = p.parse_args()

    # Argumentos opcionais sobrescrevem o config apenas se fornecidos
    interface   = args.interface   or IDSConfig.CAPTURE_INTERFACE
    output_dir  = args.output      or IDSConfig.COLLECTOR_DIR
    sample_rate = args.sample_rate if args.sample_rate is not None else IDSConfig.COLLECTOR_SAMPLE_RATE
    budget_gb   = args.budget_gb   if args.budget_gb   is not None else IDSConfig.COLLECTOR_BUDGET_GB

    ColetorDaemon(interface, output_dir, sample_rate, budget_gb).run()


if __name__ == '__main__':
    main()