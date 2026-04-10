"""
Script adaptado para análise de métricas de teoria da informação
Utiliza a base real CSE-CIC-IDS2018 em vez de dados sintéticos
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import entropy, pearsonr
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings('ignore')

# Importa o módulo de carregamento de dados
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader_cse_cic_ids2018 import CSECICIDSDataLoader

# Configuração para gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class InformationTheoryAnalysis:
    def __init__(self):
        self.data_loader = CSECICIDSDataLoader()
        self.feature_names = None
        self.X = None
        self.y = None
        self.y_names = None
        
    def load_real_data(self, sample_size=10000):
        """Carrega dados reais da base CSE-CIC-IDS2018"""
        print("Carregando dados reais CSE-CIC-IDS2018 para análise de teoria da informação...")
        
        # Carrega dados com classificação binária para análise principal
        X, y, y_names, feature_names = self.data_loader.load_data(sample_size=sample_size)
        self.X, self.y = self.data_loader.get_binary_classification_data(X, y, y_names)
        self.y_names = y_names
        self.feature_names = feature_names
        
        print(f"Dados carregados: {self.X.shape[0]} amostras, {self.X.shape[1]} características")
        print(f"Distribuição binária: Normal={np.sum(self.y==0)}, Ataque={np.sum(self.y==1)}")
        
        return self.X, self.y
    
    def calculate_information_gain(self, X=None, y=None):
        """Calcula Information Gain para cada característica"""
        if X is None:
            X, y = self.X, self.y
            
        n_features = X.shape[1]
        ig_scores = []
        
        # Entropy da variável target
        target_entropy = entropy(np.bincount(y) / len(y), base=2)
        
        for i in range(n_features):
            feature = X[:, i]
            # Discretiza característica contínua usando percentis
            bins = np.percentile(feature, [0, 20, 40, 60, 80, 100])
            bins = np.unique(bins)  # Remove duplicatas
            
            if len(bins) > 1:
                digitized = np.digitize(feature, bins[1:-1])
                
                # Calcula entropy condicional
                conditional_entropy = 0
                for bin_val in np.unique(digitized):
                    mask = digitized == bin_val
                    if np.sum(mask) > 0:
                        subset_y = y[mask]
                        if len(np.unique(subset_y)) > 1:
                            subset_entropy = entropy(np.bincount(subset_y) / len(subset_y), base=2)
                            conditional_entropy += (np.sum(mask) / len(y)) * subset_entropy
                
                # Information Gain = H(Y) - H(Y|X)
                ig = target_entropy - conditional_entropy
            else:
                ig = 0.0
                
            ig_scores.append(ig)
        
        return np.array(ig_scores)
    
    def calculate_mutual_information(self, X=None, y=None):
        """Calcula Mutual Information para cada característica"""
        if X is None:
            X, y = self.X, self.y
        return mutual_info_classif(X, y, random_state=42)
    
    def calculate_correlation_scores(self, X=None, y=None):
        """Calcula correlação de Pearson para cada característica"""
        if X is None:
            X, y = self.X, self.y
            
        correlations = []
        for i in range(X.shape[1]):
            corr, _ = pearsonr(X[:, i], y)
            correlations.append(abs(corr))
        return np.array(correlations)
    
    def calculate_mrmr_score(self, X, y, selected_features):
        """Calcula score mRMR para um conjunto de características"""
        if len(selected_features) == 0:
            return 0
        
        # Relevância (MI com target)
        mi_scores = self.calculate_mutual_information(X[:, selected_features], y)
        relevance = np.mean(mi_scores)
        
        # Redundância (MI entre características)
        if len(selected_features) == 1:
            redundancy = 0
        else:
            redundancy_scores = []
            for i in range(len(selected_features)):
                for j in range(i+1, len(selected_features)):
                    # Usa mutual_info_regression para calcular MI entre duas variáveis contínuas
                    mi_ij = mutual_info_regression(X[:, [selected_features[i]]], 
                    X[:, selected_features[j]], 
                    random_state=42)[0]
                    redundancy_scores.append(mi_ij)
            redundancy = np.mean(redundancy_scores) if redundancy_scores else 0
        
        # mRMR = Relevance - Redundancy
        return relevance - redundancy
    
    def compare_feature_selection_methods(self, X=None, y=None):
        """Compara diferentes métodos de seleção de características"""
        if X is None:
            X, y = self.X, self.y
            
        methods = {
            'Information_Gain': self.calculate_information_gain(X, y),
            'Mutual_Information': self.calculate_mutual_information(X, y),
            'Pearson_Correlation': self.calculate_correlation_scores(X, y),
            'F_Statistics': SelectKBest(f_classif, k='all').fit(X, y).scores_
        }
        
        # Normaliza scores para comparação
        for method in methods:
            scores = methods[method]
            if np.max(scores) > np.min(scores):
                methods[method] = (scores - scores.min()) / (scores.max() - scores.min())
            else:
                methods[method] = np.zeros_like(scores)
        
        return methods
    
    def evaluate_feature_sets(self, X=None, y=None, top_k_list=[5, 10, 15, 20]):
        """Avalia performance com diferentes números de características"""
        if X is None:
            X, y = self.X, self.y
            
        methods = self.compare_feature_selection_methods(X, y)
        results = {}
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        for method_name, scores in methods.items():
            method_results = {'k_features': [], 'cv_score': [], 'mrmr_score': []}
            
            # Ordena características por score
            feature_order = np.argsort(scores)[::-1]
            
            for k in top_k_list:
                if k <= len(feature_order):
                    selected_features = feature_order[:k]
                    X_selected = X[:, selected_features]
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(clf, X_selected, y, cv=5, n_jobs=-1)
                    
                    # mRMR score
                    mrmr = self.calculate_mrmr_score(X, y, selected_features)
                    
                    method_results['k_features'].append(k)
                    method_results['cv_score'].append(cv_scores.mean())
                    method_results['mrmr_score'].append(mrmr)
            
            results[method_name] = method_results
        
        return results
    
    def analyze_kl_divergence_thresholds(self, X=None, y=None):
        """Analisa thresholds de divergência KL para detecção de anomalias"""
        if X is None:
            X, y = self.X, self.y
            
        # Separa classes
        X_normal = X[y == 0]
        X_attack = X[y == 1]
        
        kl_divergences = []
        features_analyzed = []
        
        for i in range(X.shape[1]):
            # Cria histogramas para cada classe
            hist_normal, bins = np.histogram(X_normal[:, i], bins=20, density=True)
            hist_attack, _ = np.histogram(X_attack[:, i], bins=bins, density=True)
            
            # Adiciona pequena constante para evitar log(0)
            hist_normal = hist_normal + 1e-10
            hist_attack = hist_attack + 1e-10
            
            # Normaliza
            hist_normal = hist_normal / hist_normal.sum()
            hist_attack = hist_attack / hist_attack.sum()
            
            # Calcula KL divergence
            kl_div = entropy(hist_attack, hist_normal, base=2)
            kl_divergences.append(kl_div)
            features_analyzed.append(self.feature_names[i] if i < len(self.feature_names) else f'Feature_{i}')
        
        return features_analyzed, kl_divergences
    
    def analyze_feature_groups(self):
        """Analisa importância por grupos de características específicos da base CSE-CIC-IDS2018"""
        feature_groups = self.data_loader.get_feature_groups()
        group_importance = {}
        
        mi_scores = self.calculate_mutual_information()
        
        for group_name, features in feature_groups.items():
            if features:
                # Encontra índices das features do grupo
                indices = [i for i, fname in enumerate(self.feature_names) if fname in features]
                if indices:
                    group_scores = mi_scores[indices]
                    group_importance[group_name] = {
                        'mean_mi': np.mean(group_scores),
                        'max_mi': np.max(group_scores),
                        'n_features': len(indices),
                        'top_feature': self.feature_names[indices[np.argmax(group_scores)]]
                    }
        
        return group_importance
    
    def plot_information_theory_analysis(self, X=None, y=None, output_filename_png='analise_teoria_informacao.png'):
        """Gera gráficos de análise de teoria da informação com dados reais"""
        if X is None:
            X, y = self.X, self.y
            
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # 1. Comparação de métodos de seleção
        methods = self.compare_feature_selection_methods(X, y)
        
        # Seleciona top 20 features para visualização
        top_k = min(20, len(self.feature_names))
        ig_scores = self.calculate_information_gain(X, y)
        top_indices = np.argsort(ig_scores)[-top_k:]
        
        # Heatmap de scores para top features
        methods_df = pd.DataFrame({k: v[top_indices] for k, v in methods.items()}, 
                                index=[self.feature_names[i] for i in top_indices])
        sns.heatmap(methods_df, annot=False, cmap='viridis', ax=axes[0, 0])
        axes[0, 0].set_title('Comparação de Métodos de Seleção\n(Top 20 Features)', 
                           fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Características')
        
        # 2. Top características por Information Gain
        top_features_ig = np.argsort(ig_scores)[-10:]
        
        axes[0, 1].barh(range(10), ig_scores[top_features_ig], color='#FF6B6B')
        axes[0, 1].set_yticks(range(10))
        axes[0, 1].set_yticklabels([self.feature_names[i][:30] + '...' if len(self.feature_names[i]) > 30 
                                   else self.feature_names[i] for i in top_features_ig])
        axes[0, 1].set_title('Top 10 Features - Information Gain', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Information Gain (bits)')
        
        # 3. Correlação IG vs MI
        mi_scores = self.calculate_mutual_information(X, y)
        axes[0, 2].scatter(ig_scores, mi_scores, alpha=0.7, s=60, color='#4ECDC4')
        axes[0, 2].set_xlabel('Information Gain')
        axes[0, 2].set_ylabel('Mutual Information')
        axes[0, 2].set_title('Correlação: IG vs MI', fontsize=14, fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Adiciona linha de tendência
        valid_mask = ~(np.isnan(ig_scores) | np.isnan(mi_scores))
        if np.sum(valid_mask) > 1:
            z = np.polyfit(ig_scores[valid_mask], mi_scores[valid_mask], 1)
            p = np.poly1d(z)
            axes[0, 2].plot(ig_scores[valid_mask], p(ig_scores[valid_mask]), "--", color='red', linewidth=2)
        
        # 4. Performance vs número de características
        eval_results = self.evaluate_feature_sets(X, y)
        
        for method_name, results in eval_results.items():
            axes[1, 0].plot(results['k_features'], results['cv_score'], 
                           marker='o', linewidth=2, label=method_name, markersize=8)
        
        axes[1, 0].set_title('Performance vs Número de Características', 
                           fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Número de Características')
        axes[1, 0].set_ylabel('Acurácia (CV)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. mRMR scores
        for method_name, results in eval_results.items():
            axes[1, 1].plot(results['k_features'], results['mrmr_score'], 
                           marker='s', linewidth=2, label=method_name, markersize=8)
        
        axes[1, 1].set_title('Score mRMR vs Número de Características', 
                           fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Número de Características')
        axes[1, 1].set_ylabel('Score mRMR')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. KL Divergence analysis
        features_analyzed, kl_divergences = self.analyze_kl_divergence_thresholds(X, y)
        
        # Seleciona top 15 para visualização
        top_kl_indices = np.argsort(kl_divergences)[-15:]
        top_kl_features = [features_analyzed[i][:20] + '...' if len(features_analyzed[i]) > 20 
                          else features_analyzed[i] for i in top_kl_indices]
        top_kl_values = [kl_divergences[i] for i in top_kl_indices]
        
        axes[1, 2].barh(range(15), top_kl_values, color='#45B7D1')
        axes[1, 2].set_yticks(range(15))
        axes[1, 2].set_yticklabels(top_kl_features, fontsize=10)
        axes[1, 2].set_title('Top 15 Features - Divergência KL\n(Normal vs Ataque)', 
                           fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('KL Divergence (bits)')
        axes[1, 2].axvline(x=2.0, color='red', linestyle='--', linewidth=2, 
                          label='Threshold (2.0 bits)')
        axes[1, 2].legend()
        
        # 7. Distribuições por classe (exemplo de características top)
        top_feature_idx = np.argsort(kl_divergences)[-1]
        
        X_normal = X[y == 0, top_feature_idx]
        X_attack = X[y == 1, top_feature_idx]
        
        axes[2, 0].hist(X_normal, bins=30, alpha=0.7, label='Normal', 
                       color='#4ECDC4', density=True)
        axes[2, 0].hist(X_attack, bins=30, alpha=0.7, label='Ataque', 
                       color='#FF6B6B', density=True)
        axes[2, 0].set_title(f'Distribuições - {features_analyzed[top_feature_idx]}', 
                           fontsize=14, fontweight='bold')
        axes[2, 0].set_xlabel('Valor da Característica (Normalizado)')
        axes[2, 0].set_ylabel('Densidade')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Análise por grupos de features
        group_importance = self.analyze_feature_groups()
        
        if group_importance:
            groups = list(group_importance.keys())
            mean_mis = [group_importance[g]['mean_mi'] for g in groups]
            
            axes[2, 1].bar(range(len(groups)), mean_mis, color='#96CEB4', alpha=0.7)
            axes[2, 1].set_title('Importância Média por Grupo de Features', 
                               fontsize=14, fontweight='bold')
            axes[2, 1].set_xlabel('Grupo de Features')
            axes[2, 1].set_ylabel('MI Médio')
            axes[2, 1].set_xticks(range(len(groups)))
            axes[2, 1].set_xticklabels([g.replace('_', '\n') for g in groups], rotation=45, ha='right')
            
            # Adiciona número de features em cada grupo
            for i, (group, info) in enumerate(group_importance.items()):
                axes[2, 1].text(i, mean_mis[i] + 0.001, f"n={info['n_features']}", 
                               ha='center', va='bottom', fontsize=9)
        
        # 9. Matriz de redundância entre top features
        top_features = np.argsort(ig_scores)[-10:]
        redundancy_matrix = np.zeros((10, 10))
        
        for i in range(10):
            for j in range(10):
                if i != j:
                    mi_ij = mutual_info_regression(X[:, [top_features[i]]], 
                             X[:, top_features[j]], 
                             random_state=42)[0]
                    redundancy_matrix[i, j] = mi_ij
        
        feature_labels = [self.feature_names[i][:15] + '...' if len(self.feature_names[i]) > 15 
                         else self.feature_names[i] for i in top_features]
        sns.heatmap(redundancy_matrix, 
                   xticklabels=feature_labels,
                   yticklabels=feature_labels,
                   annot=True, fmt='.2f', cmap='Reds', ax=axes[2, 2])
        axes[2, 2].set_title('Matriz de Redundância\n(Top 10 Features)', 
                           fontsize=14, fontweight='bold')
        axes[2, 2].tick_params(labelsize=9)
        
        plt.tight_layout()
        plt.savefig(output_filename_png, dpi=300, bbox_inches='tight')
        plt.show()
        
        return eval_results, group_importance
    
    def generate_summary_tables(self, X=None, y=None):
        """Gera tabelas resumo para o trabalho"""
        if X is None:
            X, y = self.X, self.y
        
        # Tabela 1: Comparação de métodos
        methods = self.compare_feature_selection_methods(X, y)
        eval_results = self.evaluate_feature_sets(X, y, top_k_list=[10, 15, 20])
        
        summary_data = []
        for method_name in methods.keys():
            if method_name in eval_results:
                results = eval_results[method_name]
                # Pega resultado com 15 características
                idx_15 = results['k_features'].index(15) if 15 in results['k_features'] else -1
                if idx_15 >= 0:
                    summary_data.append({
                        'Método': method_name.replace('_', ' '),
                        'Acurácia_10_feat': results['cv_score'][0] if len(results['cv_score']) > 0 else 0,
                        'Acurácia_15_feat': results['cv_score'][idx_15],
                        'Acurácia_20_feat': results['cv_score'][-1] if len(results['cv_score']) > 2 else 0,
                        'mRMR_Score': results['mrmr_score'][idx_15],
                        'Estabilidade': np.std(results['cv_score'])
                    })
        
        df_summary = pd.DataFrame(summary_data)
        
        # Tabela 2: Top características por Information Gain
        ig_scores = self.calculate_information_gain(X, y)
        mi_scores = self.calculate_mutual_information(X, y)
        features_analyzed, kl_divergences = self.analyze_kl_divergence_thresholds(X, y)
        
        top_features_data = []
        top_indices = np.argsort(ig_scores)[-10:][::-1]  # Top 10 em ordem decrescente
        
        for i, idx in enumerate(top_indices):
            top_features_data.append({
                'Ranking': i + 1,
                'Característica': features_analyzed[idx],
                'Information_Gain': ig_scores[idx],
                'Mutual_Information': mi_scores[idx],
                'KL_Divergence': kl_divergences[idx],
                'Score_Combinado': 0.5 * ig_scores[idx] + 0.3 * mi_scores[idx] + 0.2 * (kl_divergences[idx] / max(kl_divergences))
            })
        
        df_top_features = pd.DataFrame(top_features_data)
        
        # Tabela 3: Análise por grupos de features
        group_importance = self.analyze_feature_groups()
        
        group_data = []
        for group_name, info in group_importance.items():
            group_data.append({
                'Grupo': group_name.replace('_', ' ').title(),
                'MI_Médio': info['mean_mi'],
                'MI_Máximo': info['max_mi'],
                'Número_Features': info['n_features'],
                'Top_Feature': info['top_feature']
            })
        
        df_groups = pd.DataFrame(group_data).sort_values('MI_Médio', ascending=False)
        
        return df_summary, df_top_features, df_groups

def main():
    """Função principal"""
    print("=== ANÁLISE DE TEORIA DA INFORMAÇÃO - CSE-CIC-IDS2018 ===")
    print("Análise com dados reais da base CSE-CIC-IDS2018\n")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    script_path = Path(__file__).resolve()
    output_dir = script_path.parent.parent / 'Resultados'
    os.makedirs(output_dir, exist_ok=True)

    # Define os nomes de todos os arquivos de saída
    png_filename = output_dir / f"analise_teoria_informacao_real_{timestamp}.png"
    summary_csv = output_dir / f"comparacao_metodos_selecao_real_{timestamp}.csv"
    top_features_csv = output_dir / f"top_caracteristicas_real_{timestamp}.csv"
    kl_analysis_csv = output_dir / f"analise_kl_divergence_real_{timestamp}.csv"
    groups_csv = output_dir / f"analise_grupos_features_{timestamp}.csv"
    
    # Inicializa analisador
    analyzer = InformationTheoryAnalysis()
    
    # Carrega dados reais
    X, y = analyzer.load_real_data(sample_size=50000)  # Usa 50k amostras para análise
    
    # Executa análise completa
    print("\nExecutando análise de teoria da informação...")
    eval_results, group_importance = analyzer.plot_information_theory_analysis(X, y, png_filename)
    
    # Gera tabelas resumo
    print("\nGerando tabelas resumo...")
    df_summary, df_top_features, df_groups = analyzer.generate_summary_tables(X, y)
    
    print("\n=== TABELA 1: COMPARAÇÃO DE MÉTODOS ===")
    print(df_summary.round(4))
    
    print("\n=== TABELA 2: TOP 10 CARACTERÍSTICAS ===")
    print(df_top_features.round(4))
    
    print("\n=== TABELA 3: ANÁLISE POR GRUPOS ===")
    print(df_groups.round(4))
    
    # Salva resultados
    df_summary.to_csv(summary_csv, index=False)
    df_top_features.to_csv(top_features_csv, index=False)
    df_groups.to_csv(groups_csv, index=False)
    
    # Análise específica de KL divergence
    features_analyzed, kl_divergences = analyzer.analyze_kl_divergence_thresholds(X, y)
    threshold_analysis = pd.DataFrame({
        'Característica': features_analyzed,
        'KL_Divergence': kl_divergences,
        'Significativo_2bits': np.array(kl_divergences) > 2.0,
        'Significativo_1bit': np.array(kl_divergences) > 1.0
    })
    
    print(f"\n=== ANÁLISE DE THRESHOLD KL ===")
    print(f"Características com KL > 2.0 bits: {np.sum(threshold_analysis['Significativo_2bits'])}")
    print(f"Características com KL > 1.0 bits: {np.sum(threshold_analysis['Significativo_1bit'])}")
    
    threshold_analysis.to_csv(kl_analysis_csv, index=False)
    
    # Análise específica da base CSE-CIC-IDS2018
    print(f"\n=== INSIGHTS ESPECÍFICOS DA BASE CSE-CIC-IDS2018 ===")
    print(f"Total de features analisadas: {len(analyzer.feature_names)}")
    print(f"Taxa de desbalanceamento: {np.sum(y==1)/np.sum(y==0):.3f} (ataques/normal)")
    
    # Top features por grupo
    print("\nTop feature por grupo:")
    for group, info in group_importance.items():
        print(f"  {group}: {info['top_feature']} (MI={info['max_mi']:.4f})")
    
    print("\nArquivos salvos:")
    print(f"- {png_filename}")
    print(f"- {summary_csv}")
    print(f"- {top_features_csv}")
    print(f"- {kl_analysis_csv}")
    print(f"- {groups_csv}")

if __name__ == "__main__":
    main()