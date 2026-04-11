#!/usr/bin/env python3
"""
IDS/ids_reports.py — Geração de Relatórios HTML e TXT do Sistema IDS

Relatório HTML gerado em Reports/ inclui:
  - Cards de métricas (fluxos totais, incidentes, taxa de ataque, críticos)
  - Tabela detalhada de incidentes por tipo de ataque com MITRE ATT&CK
  - Top IPs de origem e destino
  - Heatmap de atividade por hora do dia
  - Comparativo de performance do modelo (versão base vs. atual)
  - Recomendações operacionais por tipo de ameaça

Relatório TXT gerado como fallback legível por máquina.

Uso interno (chamado por ids_detector.py):
    from IDS.ids_reports import generate_report
    html_path, txt_path = generate_report(results, arts)
"""

import html
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import Config
from IDS.modules.incident_engine import ModelArtifacts, SEVERITY, CONF_HIGH


# ─────────────────────────────────────────────────────────────────────────────
# Helpers HTML
# ─────────────────────────────────────────────────────────────────────────────

def _h(s: str) -> str:
    """Escapa HTML."""
    return html.escape(str(s))


def _badge(text: str, color: str) -> str:
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:0.78em;font-weight:600;">{_h(text)}</span>'
    )


def _progress(value: int, maximum: int, color: str = "#dc3545", h: int = 12) -> str:
    pct = min(value / max(maximum, 1) * 100, 100)
    return (
        f'<div style="background:#2d2d2d;border-radius:6px;height:{h}px;overflow:hidden;">'
        f'<div style="width:{pct:.1f}%;background:{color};height:100%;"></div></div>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Seções do relatório
# ─────────────────────────────────────────────────────────────────────────────

def _section_header(
    results: List[dict],
    arts: ModelArtifacts,
    total_flows: int,
    total_inc: int,
    atk_rate: float,
    by_sev: dict,
    generated_at: str,
) -> str:
    return f"""
<div class="hdr">
  <div class="hdr-title">
    <span class="logo">🛡️</span> SecurityIA — Relatório de Detecção de Intrusão
  </div>
  <div class="hdr-meta">
    Gerado em {_h(generated_at)} &nbsp;|&nbsp; Modelo {_h(arts.version)} &nbsp;|&nbsp;
    {len(results)} arquivo(s) analisado(s)
  </div>
</div>
<div class="cards">
  <div class="card">
    <div class="val">{total_flows:,}</div><div class="lbl">Fluxos Totais</div>
  </div>
  <div class="card {'d' if total_inc > 0 else 'g'}">
    <div class="val">{total_inc:,}</div><div class="lbl">Incidentes</div>
  </div>
  <div class="card {'d' if atk_rate > 5 else 'w' if atk_rate > 1 else 'g'}">
    <div class="val">{atk_rate:.2f}%</div><div class="lbl">Taxa de Ataque</div>
  </div>
  <div class="card d">
    <div class="val">{by_sev.get('CRÍTICA', 0):,}</div><div class="lbl">Críticos</div>
  </div>
  <div class="card {'w' if by_sev.get('ALTA', 0) > 0 else 'g'}">
    <div class="val">{by_sev.get('ALTA', 0):,}</div><div class="lbl">Alta Severidade</div>
  </div>
  <div class="card">
    <div class="val">{by_sev.get('MÉDIA', 0):,}</div><div class="lbl">Média Severidade</div>
  </div>
</div>
"""


def _section_files(results: List[dict]) -> str:
    rows = ""
    for r in results:
        fps = r.get("flows_per_s", 0)
        rows += (
            f"<tr><td>{_h(r['filename'])}</td>"
            f"<td>{r['rows']:,}</td>"
            f"<td>{r['normal']:,}</td>"
            f"<td><b>{len(r['incidents'])}</b></td>"
            f"<td>{r['elapsed_s']:.1f}s ({fps:,.0f} fl/s)</td></tr>\n"
        )
    return f"""
<h2>📁 Arquivos Analisados</h2>
<table>
  <thead><tr><th>Arquivo</th><th>Total</th><th>Normal</th>
  <th>Incidentes</th><th>Tempo</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
"""


def _section_attacks(by_attack: Counter, total_inc: int) -> str:
    if not by_attack:
        return "<h2>📊 Distribuição de Ataques</h2><p>Nenhum incidente detectado.</p>"

    rows = ""
    for atk, cnt in sorted(by_attack.items(), key=lambda x: -x[1]):
        sev_int, sev_lbl, sev_col = SEVERITY.get(atk, (3, "ALTA", "#dc3545"))
        pct = cnt / max(total_inc, 1) * 100
        rows += (
            f"<tr>"
            f"<td>{_badge(sev_lbl, sev_col)}</td>"
            f"<td>{_h(atk)}</td>"
            f"<td>{cnt:,}</td>"
            f"<td>{pct:.1f}%</td>"
            f"<td>{_progress(cnt, total_inc, sev_col)}</td>"
            f"</tr>\n"
        )
    return f"""
<h2>📊 Distribuição de Ataques</h2>
<table>
  <thead><tr><th>Severidade</th><th>Tipo de Ataque</th>
  <th>Contagem</th><th>%</th><th>Proporção</th></tr></thead>
  <tbody>{rows}</tbody>
</table>
"""


def _section_incidents(incidents: List[dict], limit: int = 200) -> str:
    if not incidents:
        return "<h2>🔍 Incidentes Detalhados</h2><p>Nenhum incidente.</p>"

    shown = sorted(incidents, key=lambda x: -x["sev_int"])[:limit]
    rows  = ""
    for inc in shown:
        sev_col = inc["sev_color"]
        conf_color = "#28a745" if inc["confidence"] >= CONF_HIGH else "#fd7e14"
        rows += (
            f"<tr>"
            f"<td>{_badge(inc['severity'], sev_col)}</td>"
            f"<td><b>{_h(inc['attack'])}</b></td>"
            f"<td><span style='color:{conf_color};font-weight:600;'>"
            f"{inc['conf_pct']}</span></td>"
            f"<td>{_h(inc['src_ip'])}:{inc['src_port']}</td>"
            f"<td>{_h(inc['dst_ip'])}:{inc['dst_port']}</td>"
            f"<td>{_h(inc['protocol'])}</td>"
            f"<td>{_h(inc['flow_start'])}</td>"
            f"<td><code>{_h(inc['mitre_tech'])}</code><br>"
            f"<small>{_h(inc['mitre_tactic'])}</small></td>"
            f"<td><small>{_h(inc['action'])}</small></td>"
            f"</tr>\n"
        )
    note = (f"<p><small>Exibindo {len(shown):,} de {len(incidents):,} incidentes "
            f"(ordenados por severidade).</small></p>" if len(incidents) > limit else "")
    return f"""
<h2>🔍 Incidentes Detalhados</h2>
{note}
<table>
  <thead><tr>
    <th>Sev.</th><th>Tipo</th><th>Conf.</th>
    <th>Origem</th><th>Destino</th><th>Proto</th>
    <th>Hora</th><th>MITRE</th><th>Ação</th>
  </tr></thead>
  <tbody>{rows}</tbody>
</table>
"""


def _section_top_ips(incidents: List[dict]) -> str:
    if not incidents:
        return ""
    src_c = Counter(inc["src_ip"] for inc in incidents if inc["src_ip"] != "—")
    dst_c = Counter(inc["dst_ip"] for inc in incidents if inc["dst_ip"] != "—")

    def _ip_rows(c: Counter, limit=10) -> str:
        rows = ""
        total = sum(c.values())
        for ip, cnt in c.most_common(limit):
            pct = cnt / max(total, 1) * 100
            rows += (
                f"<tr><td><code>{_h(ip)}</code></td>"
                f"<td>{cnt:,}</td>"
                f"<td>{_progress(cnt, total, '#0d6efd')}</td></tr>\n"
            )
        return rows

    return f"""
<h2>🌐 Top IPs</h2>
<div style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem;">
  <div>
    <h3>Origens mais ativas</h3>
    <table><thead><tr><th>IP Origem</th><th>Incidentes</th><th>Proporção</th></tr></thead>
    <tbody>{_ip_rows(src_c)}</tbody></table>
  </div>
  <div>
    <h3>Destinos mais atacados</h3>
    <table><thead><tr><th>IP Destino</th><th>Incidentes</th><th>Proporção</th></tr></thead>
    <tbody>{_ip_rows(dst_c)}</tbody></table>
  </div>
</div>
"""


def _section_heatmap(incidents: List[dict]) -> str:
    """Heatmap de atividade por hora do dia."""
    hour_dist: Dict[int, int] = {h: 0 for h in range(24)}
    for inc in incidents:
        epoch = inc.get("flow_start_epoch", 0)
        if epoch > 0:
            try:
                from datetime import datetime, timezone
                h = datetime.fromtimestamp(epoch, tz=timezone.utc).hour
                hour_dist[h] = hour_dist.get(h, 0) + 1
            except Exception:
                pass

    if not any(hour_dist.values()):
        return ""

    mx   = max(hour_dist.values()) or 1
    cells = ""
    for h in range(24):
        cnt = hour_dist.get(h, 0)
        pct = cnt / mx
        alpha = 0.1 + 0.9 * pct
        cells += (
            f'<div style="background:rgba(220,53,69,{alpha:.2f});'
            f'padding:6px;border-radius:4px;text-align:center;font-size:0.75em;">'
            f'<div style="font-weight:600;">{h:02d}h</div>'
            f'<div style="color:#ccc;">{cnt}</div></div>'
        )
    return f"""
<h2>⏱️ Atividade por Hora (UTC)</h2>
<div style="display:grid;grid-template-columns:repeat(24,1fr);gap:4px;margin:1rem 0;">
  {cells}
</div>
"""


def _section_recommendations(by_attack: Counter) -> str:
    recs = {
        "DDoS":        "🚨 Ative mitigação DDoS no uplink. Considere scrubbing center.",
        "Infiltration":"🔒 Isole os hosts envolvidos. Faça análise forense imediata.",
        "Heartbleed":  "🔒 Atualize OpenSSL e invalide certificados comprometidos.",
        "Bot":         "🔒 Bloqueie C2 no firewall. Verifique integridade dos hosts.",
        "PortScan":    "📡 Revise regras de firewall. Considere rate-limiting de ICMP/TCP-SYN.",
        "FTP-Patator": "🔑 Ative 2FA para FTP. Considere migrar para SFTP.",
        "SSH-Patator": "🔑 Ative 2FA e fail2ban. Mude a porta padrão do SSH.",
    }
    lines = ""
    for atk in by_attack:
        rec = recs.get(atk)
        if rec:
            sev_int, sev_lbl, sev_col = SEVERITY.get(atk, (3, "ALTA", "#dc3545"))
            lines += (
                f'<div style="background:#1e1e2e;border-left:4px solid {sev_col};'
                f'padding:0.75rem 1rem;border-radius:0 6px 6px 0;margin:0.5rem 0;">'
                f'{_badge(atk, sev_col)} {rec}</div>'
            )
    if not lines:
        return ""
    return f"<h2>💡 Recomendações Operacionais</h2>{lines}"


def _section_model_info(arts: ModelArtifacts) -> str:
    return f"""
<h2>🤖 Informações do Modelo</h2>
<table>
  <tbody>
    <tr><th>Versão</th><td>{_h(arts.version)}</td></tr>
    <tr><th>Treinado em</th><td>{_h(arts.trained_at)}</td></tr>
    <tr><th>Classes</th><td>{len(arts.label_map)}</td></tr>
    <tr><th>Features</th><td>{len(arts.selected_features)}</td></tr>
    <tr><th>Classes reconhecidas</th>
        <td>{', '.join(_h(v) for v in sorted(arts.label_map.values()))}</td></tr>
  </tbody>
</table>
"""


# ─────────────────────────────────────────────────────────────────────────────
# CSS e HTML base
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
  background:#0d1117;color:#c9d1d9;font-size:14px;line-height:1.6}
.wrap{max-width:1400px;margin:0 auto;padding:1.5rem}
.hdr{background:linear-gradient(135deg,#161b22,#1f2937);padding:1.5rem 2rem;
  border-radius:12px;margin-bottom:1.5rem;border:1px solid #30363d}
.hdr-title{font-size:1.5rem;font-weight:700;color:#f0f6fc;margin-bottom:.4rem}
.logo{font-size:1.8rem}
.hdr-meta{color:#8b949e;font-size:.85rem}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  gap:1rem;margin-bottom:1.5rem}
.card{background:#161b22;border:1px solid #30363d;border-radius:10px;
  padding:1rem;text-align:center;transition:transform .15s}
.card:hover{transform:translateY(-2px)}
.card .val{font-size:1.8rem;font-weight:700;color:#f0f6fc}
.card .lbl{font-size:.75rem;color:#8b949e;margin-top:.2rem;text-transform:uppercase}
.card.g{border-color:#238636}.card.g .val{color:#3fb950}
.card.w{border-color:#9e6a03}.card.w .val{color:#d29922}
.card.d{border-color:#da3633}.card.d .val{color:#f85149}
h2{color:#f0f6fc;font-size:1.1rem;margin:2rem 0 .75rem;
  padding-bottom:.4rem;border-bottom:1px solid #21262d}
h3{color:#c9d1d9;font-size:.95rem;margin:1rem 0 .5rem}
table{width:100%;border-collapse:collapse;margin:.5rem 0 1.5rem;
  background:#161b22;border-radius:8px;overflow:hidden;
  border:1px solid #21262d}
thead{background:#21262d}
th,td{padding:.6rem .9rem;text-align:left;border-bottom:1px solid #21262d}
th{font-weight:600;color:#8b949e;font-size:.8rem;text-transform:uppercase}
tr:hover{background:#1c2128}
code{background:#161b22;padding:1px 5px;border-radius:4px;
  font-size:.85em;color:#79c0ff}
small{color:#8b949e}
@media(max-width:768px){.cards{grid-template-columns:repeat(2,1fr)}}
"""


def _build_html(body: str, title: str = "SecurityIA IDS Report") -> str:
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>{_h(title)}</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="wrap">
{body}
<div style="text-align:center;color:#21262d;font-size:.75rem;margin-top:3rem;
  padding-top:1rem;border-top:1px solid #21262d;">
  SecurityIA — PPGI/IC/UFAL &nbsp;|&nbsp; Bruno Cavalcante Barbosa &nbsp;|&nbsp;
  Orient. Prof. Dr. André Luiz Lins de Aquino
</div>
</div>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# Relatório TXT (fallback e machine-readable)
# ─────────────────────────────────────────────────────────────────────────────

def _build_txt(
    results: List[dict],
    arts: ModelArtifacts,
    total_flows: int,
    total_inc: int,
    atk_rate: float,
    by_attack: Counter,
    generated_at: str,
) -> str:
    lines = [
        "=" * 72,
        "  SecurityIA — Relatório de Detecção de Intrusão",
        f"  Gerado: {generated_at}  |  Modelo: {arts.version}",
        "=" * 72,
        f"  Fluxos analisados  : {total_flows:,}",
        f"  Incidentes         : {total_inc:,}",
        f"  Taxa de ataque     : {atk_rate:.2f}%",
        "",
        "  DISTRIBUIÇÃO POR TIPO DE ATAQUE:",
        "  " + "─" * 48,
    ]
    for atk, cnt in sorted(by_attack.items(), key=lambda x: -x[1]):
        sev_int, sev_lbl, _ = SEVERITY.get(atk, (3, "ALTA", ""))
        lines.append(f"  [{sev_lbl:^8s}]  {atk:<35s}  {cnt:>6,}")
    lines += ["", "  TOP 20 INCIDENTES (por severidade):", "  " + "─" * 68]
    all_inc = sorted(
        [inc for r in results for inc in r["incidents"]],
        key=lambda x: -x["sev_int"],
    )[:20]
    for inc in all_inc:
        lines.append(
            f"  {inc['severity']:<8s}  {inc['attack']:<35s}  "
            f"{inc['src_ip']}:{inc['src_port']} → "
            f"{inc['dst_ip']}:{inc['dst_port']}  "
            f"conf={inc['conf_pct']}  {inc['mitre_tech']}"
        )
    lines += ["", "=" * 72]
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Função pública
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(
    results: List[dict],
    arts: ModelArtifacts,
) -> Tuple[Path, Path]:
    """
    Gera relatório HTML e TXT em Config.REPORTS_DIR.
    Retorna (html_path, txt_path).
    """
    Config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    all_inc     = [inc for r in results for inc in r["incidents"]]
    total_flows = sum(r["rows"] for r in results)
    total_inc   = len(all_inc)
    atk_rate    = total_inc / max(total_flows, 1) * 100
    by_sev      = Counter(inc["severity"] for inc in all_inc)
    by_attack   = Counter()
    for r in results:
        for a, n in r["atk_counts"].items():
            by_attack[a] += n

    # ── HTML ──────────────────────────────────────────────────────────────────
    body = (
        _section_header(results, arts, total_flows, total_inc,
                        atk_rate, by_sev, generated_at)
        + _section_files(results)
        + _section_attacks(by_attack, total_inc)
        + _section_heatmap(all_inc)
        + _section_top_ips(all_inc)
        + _section_incidents(all_inc)
        + _section_recommendations(by_attack)
        + _section_model_info(arts)
    )
    html_path = Config.report_path(label=f"v{arts.version}", ext="html")
    html_path.write_text(_build_html(body), encoding="utf-8")

    # ── TXT ───────────────────────────────────────────────────────────────────
    txt_body  = _build_txt(results, arts, total_flows, total_inc,
                           atk_rate, by_attack, generated_at)
    txt_path  = html_path.with_suffix(".txt")
    txt_path.write_text(txt_body, encoding="utf-8")

    return html_path, txt_path
