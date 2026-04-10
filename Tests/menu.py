"""
menu.py — Menu Interativo Principal
=====================================
Dissertação: Detecção de Anomalias em Redes de Computadores Utilizando Aprendizado de Máquina
Autor  : Bruno Cavalcante Barbosa — bcb@ic.ufal.br
Orient.: Prof. Dr. André Luiz Lins de Aquino — PPGI/IC/UFAL

Uso
---
    python menu.py
"""

import os, sys, time, traceback
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from config import (
    BASE_DIR, DATASET_DIR, REPORTS_DIR, REPORT_DIRS, REPORT_NAMES,
    print_config, verificar_dataset,
)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITÁRIOS DE TERMINAL
# ═══════════════════════════════════════════════════════════════════════════════

def limpar() -> None:
    os.system("clear" if os.name != "nt" else "cls")


def linha(char="═", n=62) -> str:
    return char * n


def cabecalho() -> None:
    limpar()
    print(linha())
    print("  PPGI/IC/UFAL — Sistema de Análises para IDS")
    print("  Detecção de Anomalias com Bi-LSTM + Atenção de Bahdanau")
    print(linha())
    print("  Autor   : Bruno Cavalcante Barbosa — bcb@ic.ufal.br")
    print("  Orient. : Prof. Dr. André Luiz Lins de Aquino")
    print(linha())
    print(f"  Base    : {BASE_DIR}")
    print(f"  Dataset : {DATASET_DIR}")
    print(f"  Reports : {REPORTS_DIR}")
    print(linha())


def pausar() -> None:
    input("\n  Pressione ENTER para voltar ao menu...")


def status_dataset() -> str:
    csvs = list(DATASET_DIR.glob("*.csv"))
    if csvs:
        return f"✓ {len(csvs)} arquivo(s) CSV encontrado(s)"
    return "✗ Não encontrado"


def status_relatorio(analise_id: int) -> str:
    rdir = REPORT_DIRS[analise_id]
    md   = rdir / f"Relatorio_{analise_id}.md"
    figs = list((rdir / "figuras").glob("*.png")) if (rdir/"figuras").exists() else []
    tabs = list((rdir / "tabelas").glob("*.csv")) if (rdir/"tabelas").exists() else []
    if md.exists():
        return f"✓ {len(figs)} fig. | {len(tabs)} tab."
    return "✗ Não gerado"


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTOR DE ANÁLISE
# ═══════════════════════════════════════════════════════════════════════════════

def executar_analise(analise_id: int, dataset_disponivel: bool) -> bool:
    """Importa e executa a análise solicitada. Retorna True se bem-sucedida."""
    modulos = {
        1: ("analise_1_arquiteturas",   "executar"),
        2: ("analise_2_balanceamento",  "executar"),
        3: ("analise_3_teoria_informacao", "executar"),
        4: ("analise_4_otimizacao_validacao", "executar"),
    }
    mod_nome, fn_nome = modulos[analise_id]
    print(f"\n  Carregando módulo: {mod_nome}...")

    try:
        import importlib
        mod = importlib.import_module(mod_nome)
        importlib.reload(mod)       # garante versão mais recente em caso de re-execução
        fn  = getattr(mod, fn_nome)
        t0  = time.perf_counter()
        fn(dataset_disponivel=dataset_disponivel)
        elapsed = time.perf_counter() - t0
        print(f"\n  ✅ Concluído em {elapsed:.1f}s ({elapsed/60:.1f} min)")
        return True
    except Exception:
        print(f"\n  ❌ Erro na Análise {analise_id}:")
        traceback.print_exc()
        return False


def executar_todas(dataset_disponivel: bool) -> None:
    """Executa as 4 análises em sequência, reportando status individual."""
    resultados = {}
    t_total    = time.perf_counter()

    for i in range(1, 5):
        print(f"\n{'═'*62}")
        print(f"  Iniciando Análise {i}: {REPORT_NAMES[i]}")
        print(f"{'═'*62}")
        t0 = time.perf_counter()
        ok = executar_analise(i, dataset_disponivel)
        resultados[i] = ("✅" if ok else "❌", round(time.perf_counter()-t0, 1))

    elapsed = time.perf_counter() - t_total
    print(f"\n{'═'*62}")
    print("  RELATÓRIO DE EXECUÇÃO — TODAS AS ANÁLISES")
    print(f"{'═'*62}")
    for i, (icone, t_s) in resultados.items():
        nome = REPORT_NAMES[i][:45]
        print(f"  {icone} Análise {i}: {nome:<45} {t_s:6.1f}s")
    print(f"{'─'*62}")
    print(f"  Tempo total: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Relatórios: {REPORTS_DIR}")
    print(f"{'═'*62}")


# ═══════════════════════════════════════════════════════════════════════════════
# MENUS
# ═══════════════════════════════════════════════════════════════════════════════

def menu_dataset(dataset_disponivel: bool) -> bool:
    """Submenu para gerenciamento do dataset."""
    while True:
        cabecalho()
        print("  DATASET — CIC-IDS2018")
        print(linha("─"))
        print(f"  Status   : {status_dataset()}")
        print(f"  Diretório: {DATASET_DIR}")
        print(linha("─"))
        print("  [1] Verificar/baixar dataset automaticamente")
        print("  [2] Ver instruções de download manual")
        print("  [3] Voltar")
        print(linha("─"))
        op = input("  Opção: ").strip()
        if op == "1":
            dataset_disponivel = verificar_dataset(interativo=True)
            pausar()
        elif op == "2":
            from config import _instrucoes_manuais
            _instrucoes_manuais()
            pausar()
        elif op == "3":
            return dataset_disponivel
        else:
            print("  Opção inválida.")
            time.sleep(1)


def menu_relatorios() -> None:
    """Submenu para visualizar status dos relatórios."""
    while True:
        cabecalho()
        print("  STATUS DOS RELATÓRIOS")
        print(linha("─"))
        for i in range(1, 5):
            st = status_relatorio(i)
            print(f"  [{i}] Análise {i}: {REPORT_NAMES[i][:40]:<40} {st}")
        print(linha("─"))
        print("  [5] Abrir diretório de relatórios")
        print("  [6] Voltar")
        print(linha("─"))
        op = input("  Opção: ").strip()
        if op in ("1","2","3","4"):
            rdir = REPORT_DIRS[int(op)]
            print(f"\n  Arquivos em: {rdir}")
            for f in sorted(rdir.rglob("*")):
                if f.is_file():
                    size = f.stat().st_size
                    print(f"    {f.relative_to(rdir)} ({size/1024:.1f} KB)")
            pausar()
        elif op == "5":
            os.system(f"xdg-open {REPORTS_DIR} 2>/dev/null || open {REPORTS_DIR} 2>/dev/null || echo 'Abra: {REPORTS_DIR}'")
            pausar()
        elif op == "6":
            return
        else:
            print("  Opção inválida.")
            time.sleep(1)


def menu_configuracao() -> None:
    """Exibe configuração ativa."""
    cabecalho()
    print_config()
    print("\n  Dependências opcionais:")
    for pkg in ["scikit-optimize", "tabulate", "scipy"]:
        try:
            __import__(pkg.replace("-", "_"))
            print(f"    ✓ {pkg}")
        except ImportError:
            print(f"    ✗ {pkg}  (instale: pip install {pkg})")
    pausar()


# ═══════════════════════════════════════════════════════════════════════════════
# MENU PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

def menu_principal() -> None:
    dataset_disponivel = False

    # Verificação silenciosa ao iniciar
    csvs = list(DATASET_DIR.glob("*.csv"))
    if csvs:
        dataset_disponivel = True

    while True:
        cabecalho()
        print("  MENU PRINCIPAL")
        print(linha("─"))
        print(f"  Dataset: {status_dataset()}")
        print(linha("─"))
        print()
        print("  ── ANÁLISES INDIVIDUAIS ─────────────────────────────")
        for i in range(1, 5):
            st   = status_relatorio(i)
            nome = REPORT_NAMES[i]
            print(f"  [{i}] {nome}")
            print(f"      Status: {st}")
            print()
        print("  ── EXECUÇÃO CONJUNTA ────────────────────────────────")
        print("  [5] Executar TODAS as análises (sequencial)")
        print()
        print("  ── UTILITÁRIOS ──────────────────────────────────────")
        print("  [6] Gerenciar Dataset CIC-IDS2018")
        print("  [7] Ver relatórios gerados")
        print("  [8] Configuração do ambiente")
        print("  [0] Sair")
        print()
        print(linha("─"))
        op = input("  Opção: ").strip()

        if op in ("1","2","3","4"):
            analise_id = int(op)
            cabecalho()
            print(f"  Análise {analise_id}: {REPORT_NAMES[analise_id]}")
            print(linha("─"))
            if not dataset_disponivel:
                print("  ℹ Dataset não detectado. Análises usarão dados sintéticos.")
                print("    (Use opção 6 para baixar o dataset real)")
            print()

            # Confirmação
            conf = input("  Confirmar execução? [S/n]: ").strip().lower()
            if conf in ("", "s", "sim", "y", "yes"):
                executar_analise(analise_id, dataset_disponivel)
                pausar()

        elif op == "5":
            cabecalho()
            print("  EXECUTAR TODAS AS ANÁLISES")
            print(linha("─"))
            print("  Ordem de execução:")
            for i in range(1, 5):
                print(f"    {i}. {REPORT_NAMES[i]}")
            print()
            if not dataset_disponivel:
                print("  ℹ Dataset não detectado — análises usarão dados sintéticos.")
            print(f"  Tempo estimado: 20–60 min (CPU, {os.cpu_count()} núcleos)")
            print()
            conf = input("  Confirmar execução de todas? [S/n]: ").strip().lower()
            if conf in ("", "s", "sim", "y", "yes"):
                executar_todas(dataset_disponivel)
                pausar()

        elif op == "6":
            dataset_disponivel = menu_dataset(dataset_disponivel)

        elif op == "7":
            menu_relatorios()

        elif op == "8":
            menu_configuracao()

        elif op == "0":
            limpar()
            print("  Encerrando. Bons experimentos!")
            print(f"  Relatórios disponíveis em: {REPORTS_DIR}\n")
            sys.exit(0)

        else:
            print("  Opção inválida. Tente novamente.")
            time.sleep(1)


# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    menu_principal()
