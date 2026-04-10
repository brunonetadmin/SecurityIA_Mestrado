#!/usr/bin/env python3
"""
IDS/ids_detector.py — Motor de Detecção de Incidentes de Segurança

Modos de operação:
  watch  : Monitora COLLECTOR_DIR continuamente (daemon em background)
  batch  : Processa todos os Parquets não analisados de uma vez
  file   : Processa um único arquivo Parquet especificado

Saídas geradas:
  - Logs estruturados em Logs/App.log
  - Relatório HTML em Reports/ (via ids_reports.py)
  - Arquivo JSONL de incidentes (para ingestão por SIEM)
  - Dataset anotado em Temp/staging/ (para fine-tuning)

Uso:
    python3 IDS/ids_detector.py watch [--interval 60]
    python3 IDS/ids_detector.py batch
    python3 IDS/ids_detector.py file caminho/captura.parquet
    python3 IDS/ids_detector.py watch --background
"""

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config
from IDS.modules.incident_engine import (
    ModelArtifacts, ManagerState, scan_new_files, analyze_file,
    SEVERITY, CONF_HIGH,
)
from IDS.modules.utils import (
    get_app_logger, progress_bar, format_duration,
    _c, _C, _sep, _section,
)

log = get_app_logger()

# ─────────────────────────────────────────────────────────────────────────────
# Exibição em tempo real
# ─────────────────────────────────────────────────────────────────────────────

def _sev_color(sev: str) -> str:
    return {
        "NORMAL":  _C.GRAY,
        "BAIXA":   _C.BLUE,
        "MÉDIA":   _C.YELLOW,
        "ALTA":    _C.ORANGE,
        "CRÍTICA": _C.RED,
    }.get(sev, _C.WHITE)


def _on_incident(inc: dict) -> None:
    sev   = inc["severity"]
    color = _sev_color(sev)
    ts    = inc["flow_start"]
    print(
        f"  {_c(f'[{sev:^7s}]', color, _C.BOLD)}  "
        f"{_c(inc['attack'], _C.WHITE)}  "
        f"{_c(inc['src_ip'], _C.DIM)}:{inc['src_port']} → "
        f"{_c(inc['dst_ip'], _C.DIM)}:{inc['dst_port']}  "
        f"conf={_c(inc['conf_pct'], _C.CYAN)}  "
        f"MITRE={_c(inc['mitre_tech'], _C.GRAY)}  "
        f"{_c(ts, _C.DIM)}"
    )


def _on_progress(done: int, total: int, n_inc: int) -> None:
    bar = progress_bar(done, total, width=28,
                       suffix=f"{_c(f'{n_inc} incidentes', _C.YELLOW)}")
    print(f"\r  {bar}", end="", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# Exportação de incidentes (JSONL + staging para fine-tuning)
# ─────────────────────────────────────────────────────────────────────────────

def _export_incidents(results: List[dict]) -> None:
    """Salva incidentes em JSONL estruturado para ingestão por SIEM."""
    all_inc = []
    for r in results:
        all_inc.extend(r["incidents"])
    if not all_inc:
        return

    Config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Config.REPORTS_DIR / f"incidents_{ts}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for inc in all_inc:
            f.write(json.dumps(inc, ensure_ascii=False) + "\n")
    log.info(f"JSONL de incidentes: '{out}' ({len(all_inc):,} registros)")


def _export_staging(results: List[dict]) -> None:
    """Salva fluxos de alta confiança para re-treinamento."""
    import pandas as pd
    frames = []
    for r in results:
        df = r.get("df", pd.DataFrame())
        if df.empty:
            continue
        mask = df["_conf"] >= CONF_HIGH
        hi   = df[mask].copy()
        if not hi.empty:
            hi["Label"] = hi["_label"]
            hi.drop(columns=["_label", "_conf"], errors="ignore", inplace=True)
            frames.append(hi)

    if not frames:
        return

    combined = pd.concat(frames, ignore_index=True)
    combined  = combined[combined["Label"].notna() & (combined["Label"] != "Unknown")]
    if combined.empty:
        return

    staging = Config.RETRAIN_CONFIG["staging_dir"]
    staging.mkdir(parents=True, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = staging / f"annotated_{ts}.parquet"
    combined.to_parquet(out, compression="snappy", index=False)
    log.info(f"Dataset de re-treinamento: '{out}' ({len(combined):,} fluxos)")


# ─────────────────────────────────────────────────────────────────────────────
# Sumário de sessão
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(results: List[dict]) -> None:
    _section("SUMÁRIO DA SESSÃO DE DETECÇÃO")

    total_flows = sum(r["rows"] for r in results)
    total_inc   = sum(len(r["incidents"]) for r in results)
    total_normal = sum(r["normal"] for r in results)
    atk_rate    = (total_inc / total_flows * 100) if total_flows else 0

    print(f"  Arquivos analisados : {len(results):,}")
    print(f"  Fluxos totais       : {total_flows:,}")
    print(f"  Fluxos normais      : {total_normal:,}")
    print(f"  Incidentes          : {_c(str(total_inc), _C.YELLOW, _C.BOLD)}")
    print(f"  Taxa de ataque      : {_c(f'{atk_rate:.2f}%', _C.RED if atk_rate > 1 else _C.GREEN)}")
    print()

    # Breakdown por tipo de ataque
    atk_total: dict = {}
    for r in results:
        for a, n in r["atk_counts"].items():
            atk_total[a] = atk_total.get(a, 0) + n

    if atk_total:
        print(f"  {_c('Tipo de Ataque', _C.BOLD, _C.CYAN):<40s}  {_c('Contagem', _C.BOLD, _C.CYAN)}")
        print("  " + "─" * 52)
        for atk, cnt in sorted(atk_total.items(), key=lambda x: -x[1]):
            sev_int, sev_lbl, _ = SEVERITY.get(atk, (3, "ALTA", ""))
            sc = _sev_color(sev_lbl)
            print(f"  {_c(f'[{sev_lbl}]', sc):<20s}  {atk:<30s}  {cnt:>6,}")

    print()
    elapsed = sum(r["elapsed_s"] for r in results)
    fps_avg = total_flows / max(elapsed, 0.01)
    print(f"  Tempo total     : {format_duration(elapsed)}")
    print(f"  Throughput médio: {fps_avg:,.0f} fluxos/s")
    print(_sep("─"))


# ─────────────────────────────────────────────────────────────────────────────
# Geração de relatório HTML
# ─────────────────────────────────────────────────────────────────────────────

def _generate_report(results: List[dict], arts: ModelArtifacts) -> None:
    try:
        from IDS.ids_reports import generate_report
        path_html, path_txt = generate_report(results, arts)
        print(f"\n  {_c('Relatório HTML:', _C.GREEN)}  {path_html}")
        print(f"  {_c('Relatório TXT: ', _C.GREEN)}  {path_txt}")
        log.info(f"Relatório gerado: '{path_html}'")
    except Exception as e:
        log.error(f"Erro ao gerar relatório: {e}", exc_info=True)


# ─────────────────────────────────────────────────────────────────────────────
# Processamento de arquivos
# ─────────────────────────────────────────────────────────────────────────────

def process_files(
    files: List[Path],
    arts: ModelArtifacts,
    state: ManagerState,
    generate_html: bool = True,
) -> List[dict]:
    if not files:
        print(f"  {_c('Nenhum arquivo novo para analisar.', _C.DIM)}")
        return []

    results = []
    for i, path in enumerate(files, 1):
        _section(f"Arquivo {i}/{len(files)}: {path.name}")
        print(f"  {_c('Analisando …', _C.DIM)}", flush=True)

        try:
            result = analyze_file(
                path, arts,
                on_incident=_on_incident,
                on_progress=_on_progress,
            )
            print()  # nova linha após barra de progresso
            results.append(result)
            log.info(
                f"Arquivo '{path.name}': {result['rows']:,} fluxos | "
                f"{len(result['incidents'])} incidentes | {result['elapsed_s']:.1f}s"
            )
        except Exception as e:
            log.error(f"Erro ao processar '{path.name}': {e}", exc_info=True)
            print(f"  {_c(f'ERRO: {e}', _C.RED)}")

    state.mark_analyzed([r["filename"] for r in results])

    if results:
        _print_summary(results)
        _export_incidents(results)
        _export_staging(results)
        if generate_html:
            _generate_report(results, arts)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Comandos
# ─────────────────────────────────────────────────────────────────────────────

def cmd_watch(interval: int = 60, background: bool = False) -> None:
    """Monitora COLLECTOR_DIR continuamente (daemon)."""
    if background:
        from IDS.modules.utils import run_background
        cmd = [sys.executable, __file__, "watch", "--interval", str(interval)]
        proc = run_background(cmd, log_file=Config.LOG_APP)
        print(f"  Detector iniciado em background — PID {proc.pid}")
        print(f"  Log: {Config.LOG_APP}")
        return

    Config.ensure_dirs()
    arts  = ModelArtifacts()
    arts.load()
    state = ManagerState()

    _stop = [False]
    def _sig(sig, _):
        print(f"\n  {_c('Encerrando detector …', _C.YELLOW)}")
        _stop[0] = True
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT,  _sig)

    log.info(f"Detector em modo WATCH — intervalo={interval}s | dir={Config.COLLECTOR_DIR}")
    print(f"\n  {_c('Detector ativo', _C.GREEN, _C.BOLD)} — intervalo={interval}s | "
          f"Ctrl+C para encerrar\n")

    while not _stop[0]:
        files = scan_new_files(state)
        if files:
            process_files(files, arts, state)
        else:
            print(f"  {_c('Aguardando novos arquivos …', _C.DIM)} "
                  f"[{datetime.now().strftime('%H:%M:%S')}]", end="\r")
        for _ in range(interval):
            if _stop[0]:
                break
            time.sleep(1)

    log.info("Detector encerrado.")


def cmd_batch() -> None:
    """Processa todos os arquivos pendentes de uma vez."""
    Config.ensure_dirs()
    arts  = ModelArtifacts()
    arts.load()
    state = ManagerState()
    files = scan_new_files(state)
    print(f"  {len(files)} arquivo(s) para processar.")
    process_files(files, arts, state)


def cmd_file(path: Path) -> None:
    """Processa um único arquivo."""
    Config.ensure_dirs()
    arts  = ModelArtifacts()
    arts.load()
    state = ManagerState()
    if not path.exists():
        print(f"  {_c(f'Arquivo não encontrado: {path}', _C.RED)}")
        sys.exit(1)
    process_files([path], arts, state)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="SecurityIA — Motor de Detecção de Incidentes IDS",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("watch", help="Monitoramento contínuo do diretório de captura")
    w.add_argument("--interval",   type=int, default=60,
                   help="Intervalo de varredura em segundos (padrão: 60)")
    w.add_argument("--background", action="store_true",
                   help="Executa em segundo plano")

    sub.add_parser("batch", help="Processa todos os arquivos pendentes")

    fp = sub.add_parser("file", help="Processa um arquivo específico")
    fp.add_argument("path", type=Path, help="Caminho do arquivo .parquet")

    args = p.parse_args()

    if args.cmd == "watch":
        cmd_watch(interval=args.interval, background=args.background)
    elif args.cmd == "batch":
        cmd_batch()
    elif args.cmd == "file":
        cmd_file(args.path)


if __name__ == "__main__":
    main()
