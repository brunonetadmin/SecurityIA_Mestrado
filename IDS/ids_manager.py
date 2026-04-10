#!/usr/bin/env python3
"""
ids_manager.py — Interface Principal do Sistema IDS

Frontend interativo que orquestra os módulos do sistema:
  - ids_system.py  : detecção e análise de incidentes
  - ids_learn.py   : re-treinamento e avaliação do modelo
  - ids_report.py  : geração de relatórios HTML/TXT

Uso interativo (recomendado):
  python3 ids_manager.py

Uso direto pelo console (opcional — todos os comandos disponíveis via --help):
  python3 ids_manager.py analyze
  python3 ids_manager.py retrain
  python3 ids_manager.py both
  python3 ids_manager.py evaluate v1 "7 dias de coleta"
  python3 ids_manager.py benchmark
  python3 ids_manager.py report-eval
  python3 ids_manager.py status

Autor: Bruno Cavalcante Barbosa — UFAL
"""

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import warnings
warnings.filterwarnings('ignore')

try:
    import pyarrow.parquet as pq
except ImportError as exc:
    sys.exit(f"[ERRO] {exc}. Execute: pip install pyarrow")

from ids_config import IDSConfig
from ids_system import (
    ACTION, CONF_HIGH, CONF_MEDIUM, FEATURE_COLUMNS, META_COLUMNS,
    MITRE, SEVERITY, BENIGN_LABEL,
    ModelArtifacts, ManagerState,
    analyze_file, scan_new_files,
)
from ids_learn import load_evaluator, run_retraining
from ids_report import generate_report, list_reports


# ──────────────────────────────────────────────────────────────────────────────
# Utilitários de terminal (ANSI, formatação, entrada do usuário)
# ──────────────────────────────────────────────────────────────────────────────

# Códigos ANSI por chave de cor.
_ANSI: Dict[str, str] = {
    'R': '\033[91m', 'Y': '\033[93m', 'B': '\033[94m',
    'G': '\033[92m', 'C': '\033[96m', 'W': '\033[1m',
    'D': '\033[2m',  'X': '\033[0m',
}


def _c(text: str, *codes: str) -> str:
    """Aplica cores ANSI ao texto se o stdout for um TTY; retorna texto limpo em pipes."""
    if not sys.stdout.isatty():
        return text
    prefix = ''.join(_ANSI.get(code, '') for code in codes)
    return f"{prefix}{text}{_ANSI['X']}"


def _progress_bar(done: int, total: int, width: int = 30) -> str:
    """Barra de progresso textual: [████░░░] 53.0% 420/800."""
    if not total:
        return ''
    filled = int(done / total * width)
    return f"[{'█' * filled}{'░' * (width - filled)}] {done / total * 100:5.1f}%  {done:,}/{total:,}"


def _sev_color(label: str) -> str:
    """Retorna o label de severidade com a cor ANSI correspondente."""
    color_map = {'CRÍTICA': 'R', 'ALTA': 'Y', 'MÉDIA': 'B', 'BAIXA': 'C', 'INFO': 'D'}
    return _c(f'{label:7s}', color_map.get(label, 'X'), 'W')


def _sep(width: int = 68) -> str:
    return '─' * width


def _section(title: str) -> None:
    print(f"\n{_sep()}\n  {_c(title, 'W')}\n{_sep()}")


def _prompt(text: str) -> str:
    return input(f"  {text}").strip()


def _choice(prompt: str, options: List[str]) -> str:
    """Lê uma opção válida do usuário, repetindo até obter uma entrada aceita."""
    while True:
        value = _prompt(prompt)
        if value in options:
            return value
        print(f"  {_c('Opção inválida.', 'R')} Escolha entre: {', '.join(options)}")


BANNER = r"""
  ██╗██████╗ ███████╗
  ██║██╔══██╗██╔════╝
  ██║██║  ██║███████╗
  ██║██║  ██║╚════██║
  ██║██████╔╝███████║
  ╚═╝╚═════╝ ╚══════╝"""


# ──────────────────────────────────────────────────────────────────────────────
# Callbacks de progresso para analyze_file (exibição em tempo real)
# ──────────────────────────────────────────────────────────────────────────────

def _make_callbacks(total_rows: int) -> Tuple[Any, Any]:
    """
    Cria os dois callbacks usados por analyze_file para exibição em tempo real:
      on_incident : imprime cada incidente detectado em uma linha colorida.
      on_progress : atualiza a barra de progresso na mesma linha (\\r).
    """

    def on_incident(inc: dict) -> None:
        ts      = f"[{inc['flow_start']}]"
        attack  = inc['attack']
        conf    = inc['confidence']
        src     = f"{inc['src_ip']}:{inc['src_port']}"
        dst     = f"{inc['dst_ip']}:{inc['dst_port']}"
        proto   = inc['protocol']
        sev_lbl = inc['severity']
        print(
            f"  {_c(ts, 'D')}  {_sev_color(sev_lbl)}  "
            f"{_c(f'{attack:<32}', 'W')}  conf={conf:.3f}  "
            f"{_c(src, 'C')} → {_c(dst, 'Y')}  {proto}"
        )

    def on_progress(done: int, total: int, n_inc: int) -> None:
        bar   = _progress_bar(done, total)
        label = _c(f"{n_inc} incidente(s)", 'R' if n_inc else 'D')
        print(f"  Progresso: {bar}  {label}", end='\r', flush=True)

    return on_incident, on_progress


# ──────────────────────────────────────────────────────────────────────────────
# Operações principais (chamadas pelo menu e pelo CLI direto)
# ──────────────────────────────────────────────────────────────────────────────

def op_analyze(state: ManagerState, arts: ModelArtifacts, files: List[Path]) -> List[Dict]:
    """
    Executa a análise de incidentes sobre os arquivos fornecidos, exibindo cada
    incidente em tempo real e uma barra de progresso por batch.

    Retorna a lista de resultados de analyze_file().
    """
    _section("Análise de Incidentes em Andamento")
    print(
        f"\n  {_c('Legenda:', 'W')}  "
        f"{_c('CRÍTICA', 'R', 'W')}  {_c('ALTA   ', 'Y', 'W')}  "
        f"{_c('MÉDIA  ', 'B', 'W')}  {_c('BAIXA  ', 'C', 'W')}"
    )
    print(
        f"  Formato: [hh:mm:ss]  SEVERIDADE  "
        f"Tipo de Ataque                   conf=X.XXX  ORIGEM:P → DESTINO:P  PROTO\n"
    )

    t0      = time.time()
    results = []

    for idx, path in enumerate(files, 1):
        pf = pq.ParquetFile(str(path))
        total = pf.metadata.num_rows
        has_meta = all(c in pf.schema_arrow.names for c in META_COLUMNS)

        print(f"\n  {_c(f'[{idx}/{len(files)}]', 'C', 'W')} {_c(path.name, 'W')}")
        print(f"  Fluxos: {total:,}  |  Metadados IP: {'sim' if has_meta else 'não'}")
        print(f"  {_sep()}")

        on_incident, on_progress = _make_callbacks(total)
        result = analyze_file(path, arts, on_incident=on_incident, on_progress=on_progress)
        results.append(result)

        # Limpa a linha da barra de progresso e exibe resumo do arquivo
        n_inc   = len(result['incidents'])
        elapsed = result['elapsed_s']
        rate    = total / elapsed if elapsed > 0 else 0
        print(' ' * 80, end='\r')
        print(
            f"  {_c('✔', 'G')} {elapsed:.1f}s  |  {rate:,.0f} fluxos/s  |  "
            f"Normal={_c(str(result['normal']), 'G')}  |  "
            f"Incidentes={_c(str(n_inc), 'R' if n_inc else 'G')}"
        )

    total_inc = sum(len(r['incidents']) for r in results)
    total_time = time.time() - t0
    print(
        f"\n  {_c('✔ Análise global concluída:', 'G', 'W')}  "
        f"{total_time:.1f}s  |  {len(files)} arquivo(s)  |  "
        f"{_c(str(total_inc) + ' incidente(s)', 'R' if total_inc else 'G', 'W')}"
    )

    state.mark_analyzed([f.name for f in files])
    return results


def op_retrain(results: List[Dict], files: List[Path], state: ManagerState) -> bool:
    """
    Prepara o dataset anotado e executa o fine-tuning do modelo.
    Atualiza o estado persistido se bem-sucedido.
    """
    _section("Re-Treinamento do Modelo")

    if not results:
        print(f"  {_c('[AVISO]', 'Y')} Nenhum resultado de análise disponível.")
        print("  Execute a opção [3] Análise + Re-Treinamento para melhores resultados.")
        return False

    def _step_callback(step: int, msg: str) -> None:
        label = _c(f'Passo {step}/3', 'C', 'W')
        icon  = _c('[AVISO]', 'Y') if 'AVISO' in msg or 'Erro' in msg else ''
        print(f"\n  {label} — {icon} {msg}")

    ok = run_retraining(results, on_step=_step_callback)

    if ok:
        state.mark_trained([f.name for f in files])
        # Descarta o singleton para que o próximo acesso recarregue o modelo atualizado
        ModelArtifacts.reset()
        print(f"\n  {_c('✔ Re-treinamento concluído. Modelo será recarregado na próxima análise.', 'G', 'W')}")
    else:
        print(f"\n  {_c('[AVISO]', 'Y')} Re-treinamento não concluído. Verifique os logs.")

    return ok


def op_generate_report(results: List[Dict], version: str, retrained: bool) -> Tuple[Path, Path]:
    """Gera e exibe os caminhos do relatório HTML e do resumo TXT."""
    _section("Gerando Relatório")
    html_path, txt_path = generate_report(results, version, retrained)
    print(f"\n  {_c('✔', 'G')} HTML : {html_path}")
    print(f"  {_c('✔', 'G')} TXT  : {txt_path}")
    return html_path, txt_path


def print_terminal_summary(results: List[Dict], retrained: bool = False) -> None:
    """Exibe um resumo estruturado no terminal, agrupado por tipo de ataque e IP destino."""
    from collections import defaultdict as _dd

    all_inc     = [inc for r in results for inc in r['incidents']]
    total_flows = sum(r.get('rows', 0) for r in results)
    total_inc   = len(all_inc)
    atk_rate    = total_inc / total_flows * 100 if total_flows > 0 else 0.0

    print(f"\n{'═' * 68}")
    print(f"  {_c('RESUMO FINAL DA ANÁLISE', 'W')}")
    print(f"{'═' * 68}")
    print(f"  Arquivos analisados : {len(results)}")
    print(f"  Fluxos totais       : {total_flows:,}")
    print(f"  Incidentes          : {_c(str(total_inc), 'R' if total_inc else 'G', 'W')}")
    print(f"  Taxa de ataque      : {atk_rate:.2f}%")
    if retrained:
        print(f"  Re-treinamento      : {_c('Concluído', 'G', 'W')}")

    if not all_inc:
        print(f"\n  {_c('✔ Nenhum incidente. Tráfego dentro do padrão normal.', 'G', 'W')}")
        print(f"\n{'═' * 68}\n")
        return

    # Agrupa por (tipo de ataque → IP destino → contagem)
    groups: Dict = _dd(lambda: _dd(int))
    for inc in all_inc:
        groups[inc['attack']][inc['dst_ip']] += 1

    # Ordena por severidade decrescente, depois por volume
    sorted_attacks = sorted(
        groups.items(),
        key=lambda kv: (-SEVERITY.get(kv[0], (0,))[0], -sum(kv[1].values())),
    )

    print(f"\n  {_sep()}")
    print(f"  {_c('Detalhamento por Tipo de Ataque e Destino', 'W')}")
    print(f"  {_sep()}\n")

    for attack, dst_counts in sorted_attacks:
        _, sev_lbl, _ = SEVERITY.get(attack, (0, 'ALTA', ''))
        total_this    = sum(dst_counts.values())

        print(f"  {_sev_color(sev_lbl)}  {_c(attack, 'W')}  ({_c(str(total_this), 'W')} total)")

        for dst_ip, count in sorted(dst_counts.items(), key=lambda x: -x[1]):
            plural   = 'incidente' if count == 1 else 'incidentes'
            dst_fmt  = _c(f"'{dst_ip}'", 'Y') if dst_ip != '—' else _c(dst_ip, 'D')
            action   = ACTION.get(attack, 'Investigar')
            print(
                f"    {_c('→', 'D')}  {_c(str(count), 'W')} {plural} do tipo "
                f"{_c(attack, 'C')} com destino para {dst_fmt}"
            )

        print(f"    {_c('⚡ Ação recomendada:', 'D')} {ACTION.get(attack, 'Investigar')}\n")

    # Totais por arquivo
    print(f"  {_sep()}")
    print(f"  {_c('Por arquivo de captura:', 'W')}\n")
    for result in results:
        n   = len(result.get('incidents', []))
        bar = _c('█' * min(28, int(n / max(1, total_inc) * 28)), 'R') if n else _c('░' * 8, 'D')
        print(f"  {result['filename']:<46}  {_c(str(n), 'R' if n else 'G', 'W'):>5}  {bar}")

    print(f"\n{'═' * 68}\n")


# ──────────────────────────────────────────────────────────────────────────────
# Submenus
# ──────────────────────────────────────────────────────────────────────────────

def submenu_avaliacao() -> None:
    """Submenu de avaliação de Continual Learning, delegado ao ids_model_evaluator."""
    ev = load_evaluator()
    if ev is None:
        return

    while True:
        _section("Menu de Avaliação do Modelo")
        print()
        print("  [1]  Criar Benchmark Congelado")
        print("         └─ Separa 10% do dataset original (executar uma única vez)")
        print()
        print("  [2]  Avaliar Versão Atual")
        print("         └─ Calcula métricas no benchmark e registra FM/BWT/McNemar")
        print()
        print("  [3]  Ver Histórico de Avaliações")
        print("         └─ Lista versões, F1-macro, Kappa e Forgetting Measure")
        print()
        print("  [4]  Gerar Relatório de Evolução")
        print("         └─ HTML com gráficos, heatmap de classes e testes estatísticos")
        print()
        print("  [0]  Voltar")
        print()

        choice = _choice("  Escolha: ", ['0', '1', '2', '3', '4'])

        if choice == '0':
            break
        elif choice == '1':
            print()
            ev.create_benchmark(force=False)
        elif choice == '2':
            print()
            version = _prompt("Identificador da versão (ex: v1): ") or 'v?'
            label   = _prompt("Descrição (ex: 7 dias de coleta): ") or version
            notes   = _prompt("Notas adicionais (opcional): ")
            ev.evaluate_model(version, label, notes)
        elif choice == '3':
            history = ev._load_history()
            if not history:
                print(f"\n  {_c('Nenhuma avaliação registrada.', 'D')}")
            else:
                print(f"\n  {'Versão':<8} {'Label':<28} {'F1-Macro':<12} {'Kappa':<10} {'FM':<10} Data")
                print('  ' + '─' * 74)
                for h in history:
                    fm  = h.get('forgetting_measure')
                    fms = f'{fm:.4f}' if fm is not None else '    —   '
                    print(
                        f"  {h['version_id']:<8} {h['label']:<28} "
                        f"{h['f1_macro']:<12.6f} {h.get('cohen_kappa', 0.0):<10.6f} "
                        f"{fms:<10} {h['evaluated_at'][:16]}"
                    )
        elif choice == '4':
            ev.generate_evolution_report()

        _prompt("\n  [Enter] para continuar...")


def submenu_relatorios() -> None:
    """Submenu dedicado à gestão e visualização de relatórios gerados."""
    while True:
        _section("Menu de Relatórios")
        print()
        print("  [1]  Listar relatórios gerados")
        print("         └─ Exibe todos os relatórios em ordem cronológica inversa")
        print()
        print("  [2]  Ver último relatório")
        print("         └─ Exibe caminho completo e metadados do relatório mais recente")
        print()
        print("  [3]  Gerar Relatório de Evolução do Modelo")
        print("         └─ HTML com métricas de Continual Learning (requer avaliações)")
        print()
        print("  [0]  Voltar")
        print()

        choice = _choice("  Escolha: ", ['0', '1', '2', '3'])

        if choice == '0':
            break
        elif choice == '1':
            reports = list_reports()
            if not reports:
                print(f"\n  {_c('Nenhum relatório gerado ainda.', 'D')}")
                print(f"  Execute a análise (opção [1] ou [3] do menu principal) para gerar.")
            else:
                print(f"\n  {len(reports)} relatório(s) encontrado(s):\n")
                for rpt in reports[:20]:
                    sz = rpt.stat().st_size
                    print(f"    {rpt.name}  {sz / 1024:.0f} KB")
                if len(reports) > 20:
                    print(f"    ... e mais {len(reports) - 20} relatório(s).")
        elif choice == '2':
            reports = list_reports()
            if not reports:
                print(f"\n  {_c('Nenhum relatório disponível.', 'D')}")
            else:
                last = reports[0]
                print(f"\n  {_c('Último relatório:', 'W')}")
                print(f"    Arquivo : {last.name}")
                print(f"    Caminho : {last}")
                print(f"    Tamanho : {last.stat().st_size / 1024:.0f} KB")
                txt = last.with_suffix('.txt')
                if txt.exists():
                    print(f"    Resumo  : {txt}")
        elif choice == '3':
            ev = load_evaluator()
            if ev:
                ev.generate_evolution_report()

        _prompt("\n  [Enter] para continuar...")


def submenu_configuracoes() -> None:
    """Submenu de visualização do estado do sistema e configurações ativas."""
    while True:
        _section("Menu de Configurações")
        print()
        print("  [1]  Exibir configurações ativas")
        print("  [2]  Status do diretório do coletor")
        print("  [0]  Voltar")
        print()

        choice = _choice("  Escolha: ", ['0', '1', '2'])

        if choice == '0':
            break
        elif choice == '1':
            print(f"\n{IDSConfig.summary()}")
            print(f"\n  Para alterar as configurações, edite: {_c('ids_config.py', 'W')}")
        elif choice == '2':
            collector = IDSConfig.COLLECTOR_DIR
            print(f"\n  Diretório: {collector}")
            if collector.exists():
                files = sorted(collector.glob('captura_*.parquet'))
                if files:
                    state = ManagerState()
                    total_size = sum(f.stat().st_size for f in files)
                    print(f"  {len(files)} arquivo(s) | {total_size / 1024 ** 3:.2f} GiB total\n")
                    for f in files:
                        status = _c('[Analisado]', 'G') if state.was_analyzed(f.name) else _c('[Novo]     ', 'Y')
                        print(f"    {status}  {f.name}  {f.stat().st_size / 1024 ** 3:.2f} GiB")
                else:
                    print(f"  {_c('Nenhum arquivo de captura encontrado.', 'D')}")
                    print(f"  Execute: python3 ids_coletor.py")
            else:
                print(f"  {_c('[AVISO]', 'Y')} Diretório não existe: {collector}")

        _prompt("\n  [Enter] para continuar...")


def menu_status() -> None:
    """Exibe um painel de status geral do sistema."""
    _section("Status do Sistema")

    state    = ManagerState()
    new_files = scan_new_files(state)
    all_files = (
        sorted(IDSConfig.COLLECTOR_DIR.glob('captura_*.parquet'))
        if IDSConfig.COLLECTOR_DIR.exists() else []
    )

    model_ok = (IDSConfig.MODEL_DIR / IDSConfig.MODEL_FILENAME).exists()
    bench_ok = (IDSConfig.EVALUATION_DIR / 'benchmark' / 'benchmark_meta.json').exists()

    print(f"\n  {_c('● Modelo treinado', 'G' if model_ok else 'R')}    : "
          f"{'Disponível' if model_ok else 'NÃO ENCONTRADO'}")
    print(f"  {_c('● Benchmark', 'G' if bench_ok else 'Y')}          : "
          f"{'Criado' if bench_ok else 'Não criado (Avaliação → [1])'}")
    print(f"  {_c('● Arquivos no coletor', 'W')} : {len(all_files)} total  |  "
          f"{_c(str(len(new_files)) + ' novo(s)', 'Y' if new_files else 'G')}")

    if all_files:
        total_gb = sum(f.stat().st_size for f in all_files) / 1024 ** 3
        print(f"  {_c('● Volume total', 'W')}        : {total_gb:.2f} GiB")

    try:
        import json
        with open(IDSConfig.MODEL_DIR / IDSConfig.MODEL_INFO_FILENAME) as f:
            info = json.load(f)
        print(f"  {_c('● Features selecionadas', 'W')}: {len(info.get('selected_features', []))}")
        print(f"  {_c('● Classes do modelo', 'W')}   : {len(info.get('label_mapping', {}))}")
    except Exception:
        pass

    reports = list_reports()
    print(f"  {_c('● Relatórios gerados', 'W')}  : {len(reports)}")
    if reports:
        print(f"    Último : {reports[0].name}")


# ──────────────────────────────────────────────────────────────────────────────
# Menu Principal
# ──────────────────────────────────────────────────────────────────────────────

def _print_header(new_count: int) -> None:
    """Imprime o banner e o cabeçalho do menu principal."""
    print(f"\n{'═' * 52}")
    print(BANNER)
    print("═" * 52)
    print("  Sistema de Detecção de Intrusão — v2.0")
    print("  Universidade Federal de Alagoas")
    print(f"  Coletor   : {IDSConfig.COLLECTOR_DIR}")
    print(f"  Relatórios: {IDSConfig.IDS_REPORTS_DIR}")
    if new_count:
        print(f"\n  {_c(f'● {new_count} arquivo(s) novo(s) disponível(is)', 'Y', 'W')}")
    print("═" * 52)


def _print_main_menu() -> None:
    """Imprime as opções do menu principal."""
    print()
    print(f"  {_c('# Menu Principal', 'W')}")
    print()
    print("  [1]  Análise de Incidentes")
    print("         └─ Detecta ataques nos arquivos novos e exibe incidentes em tempo real")
    print()
    print("  [2]  Re-Treinamento do Modelo")
    print("         └─ Fine-tuning com dados capturados (recomenda-se análise prévia)")
    print()
    print("  [3]  Análise + Re-Treinamento")
    print("         └─ Executa detecção e atualiza o modelo em sequência")
    print()
    print("  [4]  Avaliação do Modelo  →")
    print("         └─ Métricas de Continual Learning: FM, BWT, McNemar")
    print()
    print("  [5]  Relatórios  →")
    print("         └─ Listar, visualizar e exportar relatórios gerados")
    print()
    print("  [6]  Status do Sistema")
    print("         └─ Visão geral de arquivos, modelo e relatórios")
    print()
    print("  [7]  Configurações  →")
    print("         └─ Configurações ativas e estado do diretório de coleta")
    print()
    print("  [0]  Sair")
    print()


def _handle_analysis_choice(
    choice: str,
    state: ManagerState,
    new_files: List[Path],
) -> None:
    """Processa as escolhas [1], [2] e [3] do menu principal."""
    if not new_files:
        print(f"\n  {_c('Nenhum arquivo novo no diretório de coleta.', 'Y')}")
        print(f"  Diretório: {IDSConfig.COLLECTOR_DIR}")
        print(f"  Execute ids_coletor.py para iniciar a captura.")
        _prompt("\n  [Enter] para voltar...")
        return

    # Lista os arquivos disponíveis para o operador confirmar
    print(f"\n  {_c('Arquivos novos disponíveis:', 'W')}\n")
    for idx, f in enumerate(new_files, 1):
        sz = f.stat().st_size
        try:
            rows = pq.ParquetFile(str(f)).metadata.num_rows
            rows_str = f"{rows:,} fluxos"
        except Exception:
            rows_str = "?"
        print(f"    [{idx:2d}] {f.name:<46}  {sz / 1024 ** 3:.2f} GiB  {rows_str}")

    total_gb = sum(f.stat().st_size for f in new_files) / 1024 ** 3
    print(f"\n  Total: {len(new_files)} arquivo(s) | {total_gb:.2f} GiB")

    do_analysis = choice in ('1', '3')
    do_training = choice in ('2', '3')
    results:   List[Dict] = []
    retrained: bool       = False

    # Carrega artefatos do modelo uma única vez por sessão
    _section("Carregando Modelo")
    arts = ModelArtifacts()
    try:
        arts.load()
    except FileNotFoundError as exc:
        print(f"\n  {_c('[ERRO]', 'R')} {exc}")
        print("  Execute o treinamento inicial antes de usar o IDS Manager.")
        _prompt("\n  [Enter] para voltar...")
        return

    version = arts.version_tag()

    if do_analysis:
        results = op_analyze(state, arts, new_files)

    if do_training:
        retrained = op_retrain(results, new_files, state)

    if do_analysis and results:
        print_terminal_summary(results, retrained)
        op_generate_report(results, version, retrained)
    elif not do_analysis and retrained:
        print(f"\n  {_c('✔ Re-treinamento concluído com sucesso.', 'G', 'W')}")

    _prompt("\n  [Enter] para voltar ao menu...")


def main_menu() -> None:
    """Loop principal do menu interativo."""
    state = ManagerState()

    while True:
        new_files = scan_new_files(state)
        _print_header(len(new_files))
        _print_main_menu()

        choice = _choice("  Escolha: ", ['0', '1', '2', '3', '4', '5', '6', '7'])

        if choice == '0':
            print(f"\n  {_c('Até logo.', 'D')}\n")
            break
        elif choice in ('1', '2', '3'):
            _handle_analysis_choice(choice, state, new_files)
        elif choice == '4':
            submenu_avaliacao()
        elif choice == '5':
            submenu_relatorios()
        elif choice == '6':
            menu_status()
            _prompt("\n  [Enter] para voltar...")
        elif choice == '7':
            submenu_configuracoes()


# ──────────────────────────────────────────────────────────────────────────────
# CLI direto (uso opcional pelo console, sem menu interativo)
# ──────────────────────────────────────────────────────────────────────────────

def cli_direct(args: argparse.Namespace) -> None:
    """Executa uma operação diretamente, sem menu interativo."""
    state = ManagerState()

    if args.command == 'status':
        menu_status()
        return

    if args.command in ('analyze', 'retrain', 'both'):
        new_files = scan_new_files(state)
        if not new_files:
            print("Nenhum arquivo novo no diretório de coleta.")
            return

        arts = ModelArtifacts()
        arts.load()
        version = arts.version_tag()
        results: List[Dict] = []
        retrained = False

        if args.command in ('analyze', 'both'):
            results = op_analyze(state, arts, new_files)

        if args.command in ('retrain', 'both'):
            retrained = op_retrain(results, new_files, state)

        if results:
            print_terminal_summary(results, retrained)
            op_generate_report(results, version, retrained)
        return

    if args.command == 'evaluate':
        ev = load_evaluator()
        if ev:
            ev.evaluate_model(args.version, args.label, getattr(args, 'notes', ''))
        return

    if args.command == 'benchmark':
        ev = load_evaluator()
        if ev:
            ev.create_benchmark(force=getattr(args, 'force', False))
        return

    if args.command == 'report-eval':
        ev = load_evaluator()
        if ev:
            ev.generate_evolution_report()
        return


# ──────────────────────────────────────────────────────────────────────────────
# Ponto de entrada
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "IDS Manager — Interface principal do sistema de detecção de intrusão.\n"
            "Sem argumentos: abre o menu interativo.\n"
            "Com subcomando: executa a operação diretamente (modo console)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('analyze',     help='Análise de incidentes nos arquivos novos.')
    sub.add_parser('retrain',     help='Re-treinamento do modelo com dados capturados.')
    sub.add_parser('both',        help='Análise + re-treinamento em sequência.')
    sub.add_parser('status',      help='Exibe o status geral do sistema.')

    ev_p = sub.add_parser('evaluate', help='Avalia o modelo no benchmark congelado.')
    ev_p.add_argument('version', help='Identificador da versão (ex: v1)')
    ev_p.add_argument('label',   help='Descrição legível (ex: "7 dias de coleta")')
    ev_p.add_argument('--notes', default='', help='Notas adicionais para o histórico.')

    bm_p = sub.add_parser('benchmark', help='Cria o benchmark congelado de avaliação.')
    bm_p.add_argument('--force', action='store_true',
                      help='Recria o benchmark mesmo que já exista.')

    sub.add_parser('report-eval', help='Gera o relatório HTML de evolução do modelo.')

    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s [%(levelname)s] %(message)s')

    if args.command:
        cli_direct(args)
    else:
        main_menu()


if __name__ == '__main__':
    main()