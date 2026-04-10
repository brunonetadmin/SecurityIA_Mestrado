#!/usr/bin/env python3
"""
ids_report.py — Geração de Relatórios de Segurança

Responsabilidades:
  - Gerar relatório HTML completo a partir dos resultados de análise.
  - Gerar resumo executivo em TXT para armazenamento e consulta rápida.
  - Helpers HTML internos (badges, barras de progresso).

A nomenclatura dos arquivos é controlada por IDSConfig.report_filename(),
garantindo numeração sequencial global: relatorio_{NNN}_{versao}_{data}.html

Importado por: ids_manager.py

Autor: Bruno Cavalcante Barbosa — UFAL
"""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from ids_config import IDSConfig
from ids_system import ACTION, MITRE, SEVERITY


# ──────────────────────────────────────────────────────────────────────────────
# Helpers HTML
# ──────────────────────────────────────────────────────────────────────────────

def _badge(text: str, color: str) -> str:
    """Retorna um <span> estilizado como badge colorido."""
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:10px;font-size:11px;font-weight:600">{text}</span>'
    )


def _progress_bar(value: int, maximum: int, color: str = '#dc3545', height: int = 14) -> str:
    """Retorna uma barra de progresso HTML proporcional ao valor/máximo."""
    pct = min(100.0, (value / maximum * 100)) if maximum > 0 else 0.0
    return (
        f'<div style="display:flex;align-items:center;gap:6px">'
        f'<div style="flex:1;background:#f0f0f0;border-radius:4px;height:{height}px">'
        f'<div style="width:{pct:.1f}%;background:{color};height:100%;border-radius:4px"></div>'
        f'</div><span style="min-width:38px;font-size:12px">{value:,}</span></div>'
    )


# ──────────────────────────────────────────────────────────────────────────────
# Construção do HTML
# ──────────────────────────────────────────────────────────────────────────────

_CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;margin:0;
     padding:0;background:#f8f9fa;color:#212529}
.wrap{max-width:1100px;margin:0 auto;padding:24px 16px}
h1{font-size:24px;margin:0 0 4px}
h2{font-size:18px;border-bottom:2px solid #dee2e6;padding-bottom:6px;
   margin-top:36px;color:#1a1a2e}
h3{font-size:14px;color:#495057;margin:16px 0 8px}
.hdr{background:linear-gradient(135deg,#1a1a2e,#16213e);color:#fff;
     padding:28px 32px;border-radius:12px;margin-bottom:24px}
.hdr p{margin:4px 0;opacity:.85;font-size:14px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(148px,1fr));gap:12px;margin:16px 0}
.card{background:#fff;border-radius:10px;padding:16px;
      box-shadow:0 2px 6px rgba(0,0,0,.08);text-align:center}
.card .val{font-size:26px;font-weight:700;margin-bottom:4px}
.card .lbl{font-size:12px;color:#6c757d}
.card.d .val{color:#dc3545} .card.w .val{color:#fd7e14}
.card.i .val{color:#0d6efd} .card.g .val{color:#198754}
table{width:100%;border-collapse:collapse;font-size:13px;background:#fff;
      border-radius:8px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,.06);margin-bottom:20px}
th{background:#343a40;color:#fff;padding:9px 12px;text-align:left;
   font-weight:600;font-size:12px}
td{padding:8px 12px;border-bottom:1px solid #f0f0f0}
tr:hover td{background:#f8f9fa}
.rec{background:#fff3cd;border-left:4px solid #ffc107;padding:10px 14px;
     border-radius:0 6px 6px 0;margin:6px 0}
section{margin-bottom:36px}
footer{margin-top:36px;padding-top:14px;border-top:1px solid #dee2e6;
       font-size:11px;color:#6c757d;text-align:center}
"""

# Recomendações operacionais por tipo de ataque.
_RECOMMENDATIONS: Dict[str, str] = {
    'DDoS':       'Ativar mitigação DDoS na borda. Acionar provedor de trânsito para filtragem upstream.',
    'Heartbleed': 'Atualizar OpenSSL imediatamente. Revogar e re-emitir todos os certificados SSL/TLS.',
    'Infiltration': 'Isolar hosts afetados. Iniciar processo formal de resposta a incidentes.',
    'Bot':        'Bloquear IPs de C2 identificados. Verificar hosts para malware. Auditar DNS.',
    'SSH-Patator': 'Implementar fail2ban para SSH. Migrar para autenticação por chave pública.',
    'FTP-Patator': 'Substituir FTP por SFTP. Aplicar bloqueio progressivo de tentativas de login.',
    'Web Attack \u2013 Sql Injection': 'Parametrizar todas as queries SQL. Implementar WAF.',
    'Web Attack \u2013 XSS': 'Aplicar Content-Security-Policy. Sanitizar todos os inputs de usuário.',
    'DoS Hulk':   'Implementar rate-limiting HTTP. Habilitar proteção contra flood no servidor.',
    'DoS GoldenEye': 'Limitar conexões simultâneas por IP. Ajustar timeouts de conexão HTTP.',
}


def _section_files(results: List[Dict]) -> str:
    rows = ''.join(
        f'<tr>'
        f'<td style="font-family:monospace;font-size:12px">{r["filename"]}</td>'
        f'<td>{r.get("rows", 0):,}</td>'
        f'<td>{r.get("normal", 0):,}</td>'
        f'<td><strong style="color:#dc3545">{len(r.get("incidents", []))}</strong></td>'
        f'<td>{len(r.get("incidents", [])) / r.get("rows", 1) * 100:.2f}%</td>'
        f'</tr>'
        for r in results
    )
    return (
        '<table><tr><th>Arquivo</th><th>Total Fluxos</th><th>Normal</th>'
        f'<th>Incidentes</th><th>Taxa</th></tr>{rows}</table>'
    )


def _section_attack_table(by_attack: Counter, total_inc: int) -> str:
    if not by_attack:
        return '<p style="color:#6c757d">Nenhum incidente detectado.</p>'

    max_count = max(by_attack.values())
    rows = ''.join(
        (lambda a=attack, c=count: (
            f'<tr>'
            f'<td><strong>{a}</strong></td>'
            f'<td>{_badge(SEVERITY.get(a, (0, "INFO", "#6c757d"))[1], SEVERITY.get(a, (0, "INFO", "#6c757d"))[2])}</td>'
            f'<td>{_progress_bar(c, max_count, SEVERITY.get(a, (0, "INFO", "#6c757d"))[2])}</td>'
            f'<td>{c / total_inc * 100:.1f}%</td>'
            f'<td style="font-size:11px">{MITRE.get(a, ("—", "—"))[0]}</td>'
            f'<td style="font-size:11px;color:#6c757d">{MITRE.get(a, ("—", "—"))[1]}</td>'
            f'</tr>'
        ))()
        for attack, count in sorted(by_attack.items(), key=lambda x: -x[1])
    )
    return (
        '<table><tr><th>Tipo de Ataque</th><th>Severidade</th><th>Contagem</th>'
        f'<th>%</th><th>Tática MITRE</th><th>Técnica</th></tr>{rows}</table>'
    )


def _section_ip_table(by_ip: Counter, color: str) -> str:
    if not by_ip:
        return '<tr><td colspan="3">Metadados de IP não disponíveis (coletor v&lt;1.1).</td></tr>'
    max_v = max(by_ip.values())
    return ''.join(
        f'<tr>'
        f'<td style="font-family:monospace;font-size:12px">{ip}</td>'
        f'<td>{cnt}</td>'
        f'<td>{_progress_bar(cnt, max_v, color)}</td>'
        f'</tr>'
        for ip, cnt in by_ip.most_common(20)
    )


def _section_incidents(all_inc: List[dict]) -> str:
    """Tabela detalhada de incidentes, limitada a 2 000 linhas por performance."""
    shown = sorted(all_inc, key=lambda x: (-x['sev_int'], -x['confidence']))[:2000]
    note  = (
        f'<p style="color:#6c757d;font-size:12px">Exibindo {len(shown):,} de '
        f'{len(all_inc):,} incidentes.</p>'
    ) if len(all_inc) > 2000 else ''
    rows  = ''.join(
        f'<tr>'
        f'<td>{inc["flow_start"]}</td>'
        f'<td><strong>{inc["attack"]}</strong></td>'
        f'<td>{_badge(inc["severity"], inc["sev_color"])}</td>'
        f'<td>{_badge(inc["conf_level"], "#198754" if inc["conf_level"] == "ALTA" else "#fd7e14")} {inc["confidence"]:.3f}</td>'
        f'<td style="font-family:monospace;font-size:11px">{inc["src_ip"]}:{inc["src_port"]}</td>'
        f'<td style="font-family:monospace;font-size:11px">{inc["dst_ip"]}:{inc["dst_port"]}</td>'
        f'<td>{inc["protocol"]}</td>'
        f'<td style="font-size:11px">{inc["action"]}</td>'
        f'</tr>'
        for inc in shown
    )
    return (
        f'{note}<table>'
        '<tr><th>Hora</th><th>Ataque</th><th>Severidade</th>'
        '<th>Confiança</th><th>IP Origem</th><th>IP Destino</th>'
        f'<th>Proto</th><th>Ação</th></tr>{rows}</table>'
    )


def _section_heatmap(hour_dist: Dict[int, int]) -> str:
    """Mapa de calor de incidentes por hora do dia (UTC)."""
    max_v = max(hour_dist.values()) if hour_dist else 1
    cells = ''
    for h in range(24):
        v   = hour_dist.get(h, 0)
        pct = v / max_v if max_v else 0
        r   = int(220 * pct); g = int(53 * pct); b = int(69 * pct)
        bg  = f'rgba({r},{g},{b},{max(0.05, pct * 0.9):.2f})'
        cells += (
            f'<td title="{h:02d}h: {v}" style="background:{bg};text-align:center;'
            f'font-size:10px;padding:5px 2px;min-width:24px">'
            f'{h:02d}h<br><b>{v}</b></td>'
        )
    return f'<table style="border-collapse:collapse"><tr>{cells}</tr></table>'


def _section_recommendations(by_attack: Counter) -> str:
    items = [
        f'<div class="rec">'
        f'<strong>{_badge(SEVERITY.get(a, (0, "INFO", "#6c757d"))[1], SEVERITY.get(a, (0, "INFO", "#6c757d"))[2])} {a}</strong><br>'
        f'<span style="font-size:13px">{text}</span></div>'
        for a, text in _RECOMMENDATIONS.items()
        if a in by_attack
    ]
    return ''.join(items) if items else '<p style="color:#6c757d">Sem recomendações específicas.</p>'


# ──────────────────────────────────────────────────────────────────────────────
# Função pública principal
# ──────────────────────────────────────────────────────────────────────────────

def generate_report(
    results: List[Dict],
    model_version: str,
    retrained: bool = False,
) -> Tuple[Path, Path]:
    """
    Gera relatório HTML detalhado e resumo TXT a partir dos resultados de análise.

    A nomenclatura é determinada por IDSConfig.report_filename(), que atribui
    um número sequencial global único por sessão:
      relatorio_{NNN}_{versao}_{YYYYMMDD}.html / .txt

    Parâmetros
    ----------
    results       : Lista de resultados de analyze_file().
    model_version : Tag de versão do modelo (ex.: 'v1', 'v2').
    retrained     : Se True, inclui nota de re-treinamento no cabeçalho.

    Retorna
    -------
    Tuple(html_path, txt_path)
    """
    IDSConfig.IDS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    html_path = IDSConfig.report_filename(model_version, 'html')
    txt_path  = html_path.with_suffix('.txt')

    # Agrega todos os incidentes dos resultados
    all_inc     = [inc for r in results for inc in r.get('incidents', [])]
    total_flows = sum(r.get('rows', 0) for r in results)
    total_inc   = len(all_inc)
    atk_rate    = total_inc / total_flows * 100 if total_flows > 0 else 0.0

    by_attack = Counter(inc['attack'] for inc in all_inc)
    by_sev    = Counter(inc['severity'] for inc in all_inc)
    by_src    = Counter(inc['src_ip'] for inc in all_inc if inc['src_ip'] != '—')
    by_dst    = Counter(inc['dst_ip'] for inc in all_inc if inc['dst_ip'] != '—')
    by_conf   = Counter(inc['conf_level'] for inc in all_inc)

    hour_dist: Dict[int, int] = defaultdict(int)
    for inc in all_inc:
        ep = inc.get('flow_start_epoch', 0.0)
        if ep > 0:
            hour_dist[datetime.fromtimestamp(ep, tz=timezone.utc).hour] += 1

    filenames = [r['filename'] for r in results]
    now_str   = datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    periodo   = (
        f"{filenames[0]} — {filenames[-1]}"
        if len(filenames) > 1
        else filenames[0] if filenames else '—'
    )

    retrain_note = (
        '<p style="background:#d1e7dd;border-left:4px solid #198754;'
        'padding:10px 14px;border-radius:0 6px 6px 0;font-size:13px">'
        '✅ <strong>Re-treinamento concluído</strong> — modelo atualizado nesta sessão.</p>'
    ) if retrained else ''

    # ── Monta o HTML ──────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Relatório IDS — {periodo}</title>
<style>{_CSS}</style>
</head>
<body>
<div class="wrap">

<div class="hdr">
  <h1>🔒 Relatório de Análise de Segurança</h1>
  <p>Período: <strong>{periodo}</strong></p>
  <p>Modelo: <strong>{model_version}</strong>
     &nbsp;|&nbsp; Arquivo: <strong>{html_path.name}</strong>
     &nbsp;|&nbsp; Gerado: {now_str}</p>
  {retrain_note}
</div>

<section id="resumo">
  <h2>1. Resumo Executivo</h2>
  <div class="cards">
    <div class="card"><div class="val">{len(results)}</div><div class="lbl">Arquivos</div></div>
    <div class="card"><div class="val">{total_flows:,}</div><div class="lbl">Fluxos totais</div></div>
    <div class="card {'d' if total_inc > 0 else 'g'}">
      <div class="val">{total_inc:,}</div><div class="lbl">Incidentes</div></div>
    <div class="card {'d' if atk_rate > 5 else 'w' if atk_rate > 1 else 'g'}">
      <div class="val">{atk_rate:.2f}%</div><div class="lbl">Taxa de ataque</div></div>
    <div class="card d"><div class="val">{by_sev.get('CRÍTICA', 0):,}</div><div class="lbl">Críticos</div></div>
    <div class="card w"><div class="val">{by_sev.get('ALTA', 0):,}</div><div class="lbl">Altos</div></div>
    <div class="card i"><div class="val">{by_conf.get('ALTA', 0):,}</div><div class="lbl">Alta confiança</div></div>
    <div class="card"><div class="val">{len(by_attack)}</div><div class="lbl">Tipos únicos</div></div>
  </div>
</section>

<section id="arquivos">
  <h2>2. Arquivos Analisados</h2>
  {_section_files(results)}
</section>

<section id="ataques">
  <h2>3. Incidentes por Tipo de Ataque</h2>
  {_section_attack_table(by_attack, total_inc)}
</section>

<section id="temporal">
  <h2>4. Distribuição Temporal por Hora (UTC)</h2>
  {_section_heatmap(hour_dist)}
  <p style="font-size:12px;color:#6c757d;margin-top:6px">
    Intensidade da cor proporcional ao volume de incidentes na hora.</p>
</section>

<section id="ips">
  <h2>5. Top IPs Ofensores</h2>
  <table><tr><th>IP Origem</th><th>Incidentes</th><th>Distribuição</th></tr>
  {_section_ip_table(by_src, '#dc3545')}
  </table>
  <h2>6. Top IPs Alvos</h2>
  <table><tr><th>IP Destino</th><th>Incidentes</th><th>Distribuição</th></tr>
  {_section_ip_table(by_dst, '#0d6efd')}
  </table>
</section>

<section id="incidentes">
  <h2>7. Log Detalhado de Incidentes</h2>
  {_section_incidents(all_inc) if all_inc else '<p style="color:#6c757d">Nenhum incidente detectado.</p>'}
</section>

<section id="recomendacoes">
  <h2>8. Recomendações de Segurança</h2>
  {_section_recommendations(by_attack)}
</section>

<footer>
  IDS Manager v2.0 — UFAL &nbsp;|&nbsp; {html_path.name} &nbsp;|&nbsp; {now_str}
</footer>
</div>
</body>
</html>"""

    html_path.write_text(html, encoding='utf-8')

    # ── Resumo TXT ────────────────────────────────────────────────────────────
    atk_lines = ''.join(
        f"  {a:<44} {c:>7,}  ({c / total_inc * 100:.1f}%)\n"
        for a, c in sorted(by_attack.items(), key=lambda x: -x[1])
    )
    src_lines = ''.join(
        f"  {ip:<24} {c:>8,}\n"
        for ip, c in by_src.most_common(10)
    ) or "  Metadados de IP não disponíveis.\n"
    file_lines = ''.join(
        f"  {r['filename']:<50} {len(r.get('incidents', [])):>7,} incidente(s)\n"
        for r in results
    )

    txt = (
        f"╔══════════════════════════════════════════════════════════════════════╗\n"
        f"║         RELATÓRIO DE ANÁLISE DE SEGURANÇA — IDS MANAGER            ║\n"
        f"╚══════════════════════════════════════════════════════════════════════╝\n"
        f"Arquivo    : {html_path.name}\n"
        f"Gerado em  : {now_str}\n"
        f"Período    : {periodo}\n"
        f"Modelo     : {model_version}\n"
        f"{'Re-treinamento realizado nesta sessão.' if retrained else 'Sem re-treinamento nesta sessão.'}\n\n"
        f"─── RESUMO ──────────────────────────────────────────────────────────\n"
        f"  Arquivos       : {len(results)}\n"
        f"  Fluxos totais  : {total_flows:,}\n"
        f"  Incidentes     : {total_inc:,}\n"
        f"  Taxa de ataque : {atk_rate:.2f}%\n"
        f"  Críticos       : {by_sev.get('CRÍTICA', 0):,}\n"
        f"  Altos          : {by_sev.get('ALTA', 0):,}\n\n"
        f"─── POR TIPO DE ATAQUE ──────────────────────────────────────────────\n"
        f"{atk_lines}\n"
        f"─── TOP 10 IPs OFENSORES ────────────────────────────────────────────\n"
        f"{src_lines}\n"
        f"─── POR ARQUIVO ─────────────────────────────────────────────────────\n"
        f"{file_lines}"
        f"═════════════════════════════════════════════════════════════════════\n"
    )
    txt_path.write_text(txt, encoding='utf-8')

    return html_path, txt_path


def list_reports() -> List[Path]:
    """Retorna todos os relatórios HTML gerados, em ordem cronológica inversa."""
    d = IDSConfig.IDS_REPORTS_DIR
    if not d.exists():
        return []
    return sorted(d.glob('relatorio_*.html'), reverse=True)