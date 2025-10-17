"""
Script Principal - Executa todas as análises para justificativas do trabalho
Gera todos os gráficos, tabelas e evidências necessárias
"""

import os
import sys
import time
import subprocess
import pandas as pd
import numpy as np
from datetime import datetime

def create_output_directory():
    """Cria diretório para outputs"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"analise_trabalho_{timestamp}"
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, "graficos"))
        os.makedirs(os.path.join(output_dir, "tabelas"))
        os.makedirs(os.path.join(output_dir, "relatorios"))
    return output_dir

def install_requirements():
    """Instala pacotes necessários"""
    requirements = [
        'numpy', 'pandas', 'matplotlib', 'seaborn', 'scikit-learn',
        'tensorflow', 'imbalanced-learn', 'scipy', 'scikit-optimize'
    ]
    
    print("Verificando dependências...")
    for package in requirements:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"⚠ Instalando {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                         capture_output=True)

def run_architecture_comparison(output_dir):
    """Executa análise de comparação de arquiteturas"""
    print("\n" + "="*60)
    print("1. EXECUTANDO ANÁLISE DE ARQUITETURAS NEURAIS")
    print("="*60)
    
    # Importa e executa análise
    from script_comparacao_arquiteturas import NetworkArchitectureComparison
    
    comparator = NetworkArchitectureComparison(sequence_length=100, n_features=23)
    results = comparator.run_full_comparison()
    comparator.plot_results()
    
    # Move arquivos para diretório de output
    if os.path.exists('comparacao_arquiteturas.png'):
        os.rename('comparacao_arquiteturas.png', 
                 os.path.join(output_dir, 'graficos', 'comparacao_arquiteturas.png'))
    
    if os.path.exists('resultados_arquiteturas.csv'):
        os.rename('resultados_arquiteturas.csv', 
                 os.path.join(output_dir, 'tabelas', 'resultados_arquiteturas.csv'))
    
    print("✓ Análise de arquiteturas concluída")
    return results

def run_information_theory_analysis(output_dir):
    """Executa análise de teoria da informação"""
    print("\n" + "="*60)
    print("2. EXECUTANDO ANÁLISE DE TEORIA DA INFORMAÇÃO")
    print("="*60)
    
    from script_teoria_informacao import InformationTheoryAnalysis
    
    analyzer = InformationTheoryAnalysis()
    X, y, attack_types = analyzer.generate_network_dataset(n_samples=5000, n_features=23)
    
    eval_results, method_efficiency = analyzer.plot_information_theory_analysis(X, y)
    df_summary, df_top_features = analyzer.generate_summary_tables(X, y)
    
    # Move arquivos
    files_to_move = [
        ('analise_teoria_informacao.png', 'graficos'),
        ('comparacao_metodos_selecao.csv', 'tabelas'),
        ('top_caracteristicas.csv', 'tabelas'),
        ('analise_kl_divergence.csv', 'tabelas')
    ]
    
    for filename, subdir in files_to_move:
        if os.path.exists(filename):
            os.rename(filename, os.path.join(output_dir, subdir, filename))
    
    print("# Análise de Teoria da Informação Concluída com sucesso!")
    return eval_results, df_summary, df_top_features

def run_smote_enn_analysis(output_dir):
    """Executa análise SMOTE-ENN"""
    print("\n" + "="*60)
    print("3. EXECUTANDO ANÁLISE SMOTE-ENN")
    print("="*60)
    
    from script_smote_enn import SMOTEENNAnalysis
    
    analyzer = SMOTEENNAnalysis()
    X, y = analyzer.create_imbalanced_network_dataset(n_samples=8000)
    
    evaluation_results, X_test, y_test = analyzer.comprehensive_evaluation(X, y)
    
    # Análises específicas
    X_train, X_test_split, y_train, y_test_split = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    k_smote_results = analyzer.evaluate_k_neighbors_smote(X_train, y_train, X_test_split, y_test_split)
    k_enn_results = analyzer.evaluate_k_neighbors_enn(X_train, y_train, X_test_split, y_test_split)
    order_results = analyzer.analyze_order_dependency(X_train, y_train, X_test_split, y_test_split)
    
    