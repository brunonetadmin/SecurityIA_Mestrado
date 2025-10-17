"""
Script para análise de métricas de teoria da informação
Justifica a seleção de características e quantificação de anomalias
"""

import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.feature_selection import mutual_info_classif, SelectKBest, f_classif, mutual_info_regression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from scipy.stats import entropy, pearsonr
from scipy.spatial.distance import jensenshannon
import warnings
warnings.filterwarnings('ignore')

# Configuração para gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class InformationTheoryAnalysis:
    def __init__(self):
        self.feature_names = [
            'Flow_Duration', 'Total_Fwd_Packets', 'Flow_Bytes_s', 'Flow_Packets_s',
            'Fwd_Packet_Length_Mean', 'Bwd_Packet_Length_Mean', 'Flow_IAT_Mean',
            'Fwd_IAT_Total', 'Bwd_IAT_Total', 'Packet_Length_Variance',
            'Flow_IAT_Std', 'Active_Mean', 'Active_Std', 'Idle_Mean', 'Idle_Std',
            'TCP_Flag_Count', 'Protocol_Type', 'Service_Type', 'Flag_Type',
            'Source_Bytes', 'Destination_Bytes', 'Count', 'Same_Service_Rate'
        ]
        
    def generate_network_dataset(self, n_samples=10000, n_features=23):
        """Gera dataset sintético simulando características de rede"""
        np.random.seed(42)
        
        # Dataset base
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=15,
            n_redundant=5,
            n_clusters_per_class=3,
            class_sep=0.8,
            random_state=42
        )
        
        # Adiciona características específicas de rede
        # Simula diferentes tipos de ataque
        attack_types = np.random.choice([0, 1, 2, 3, 4], size=n_samples, 
                                       p=[0.6, 0.15, 0.1, 0.1, 0.05])  # Normal, DoS, Probe, R2L, U2R
        
        # Modifica características baseado no tipo de ataque
        for i in range(n_samples):
            if attack_types[i] == 1:  # DoS
                X[i, 0] *= 0.1  # Duração curta
                X[i, 1] *= 5.0  # Muitos pacotes
                X[i, 2] *= 10.0  # Bytes/s alto
            elif attack_types[i] == 2:  # Probe
                X[i, 3] *= 3.0  # Packets/s alto
                X[i, 6] *= 0.5  # IAT baixo
            elif attack_types[i] == 3:  # R2L
                X[i, 4] *= 2.0  # Tamanho de pacote específico
                X[i, 16] = 1.0  # TCP específico
            elif attack_types[i] == 4:  # U2R
                X[i, 19] *= 3.0  # Source bytes alto
                X[i, 21] *= 10.0  # Count alto
        
        # Adiciona ruído e correlações
        noise = np.random.normal(0, 0.1, X.shape)
        X = X + noise
        
        # Cria algumas características redundantes
        X[:, -3] = X[:, 0] + np.random.normal(0, 0.05, n_samples)  # Redundante com duração
        X[:, -2] = X[:, 1] * 0.8 + np.random.normal(0, 0.1, n_samples)  # Redundante com packets
        
        return X, y, attack_types
    
    def calculate_information_gain(self, X, y):
        """Calcula Information Gain para cada característica"""
        n_features = X.shape[1]
        ig_scores = []
        
        # Entropy da variável target
        target_entropy = entropy(np.bincount(y) / len(y), base=2)
        
        for i in range(n_features):
            feature = X[:, i]
            # Discretiza característica contínua
            bins = np.linspace(feature.min(), feature.max(), 10)
            digitized = np.digitize(feature, bins)
            
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
            ig_scores.append(ig)
        
        return np.array(ig_scores)
    
    def calculate_mutual_information(self, X, y):
        """Calcula Mutual Information para cada característica"""
        return mutual_info_classif(X, y, random_state=42)
    
    def calculate_correlation_scores(self, X, y):
        """Calcula correlação de Pearson para cada característica"""
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
    
    def compare_feature_selection_methods(self, X, y):
        """Compara diferentes métodos de seleção de características"""
        methods = {
            'Information_Gain': self.calculate_information_gain(X, y),
            'Mutual_Information': self.calculate_mutual_information(X, y),
            'Pearson_Correlation': self.calculate_correlation_scores(X, y),
            'F_Statistics': SelectKBest(f_classif, k='all').fit(X, y).scores_
        }
        
        # Normaliza scores para comparação
        for method in methods:
            scores = methods[method]
            methods[method] = (scores - scores.min()) / (scores.max() - scores.min())
        
        return methods
    
    def evaluate_feature_sets(self, X, y, top_k_list=[5, 10, 15, 20]):
        """Avalia performance com diferentes números de características"""
        methods = self.compare_feature_selection_methods(X, y)
        results = {}
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        for method_name, scores in methods.items():
            method_results = {'k_features': [], 'cv_score': [], 'mrmr_score': []}
            
            # Ordena características por score
            feature_order = np.argsort(scores)[::-1]
            
            for k in top_k_list:
                if k <= len(feature_order):
                    selected_features = feature_order[:k]
                    X_selected = X[:, selected_features]
                    
                    # Cross-validation score
                    cv_scores = cross_val_score(clf, X_selected, y, cv=5)
                    
                    # mRMR score
                    mrmr = self.calculate_mrmr_score(X, y, selected_features)
                    
                    method_results['k_features'].append(k)
                    method_results['cv_score'].append(cv_scores.mean())
                    method_results['mrmr_score'].append(mrmr)
            
            results[method_name] = method_results
        
        return results
    
    def analyze_kl_divergence_thresholds(self, X, y):
        """Analisa thresholds de divergência KL para detecção de anomalias"""
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
    
    def plot_information_theory_analysis(self, X, y, output_filename_png):
        """Gera gráficos de análise de teoria da informação"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # 1. Comparação de métodos de seleção
        methods = self.compare_feature_selection_methods(X, y)
        
        # Heatmap de scores
        methods_df = pd.DataFrame(methods, index=self.feature_names[:X.shape[1]])
        sns.heatmap(methods_df, annot=False, cmap='viridis', ax=axes[0, 0])
        axes[0, 0].set_title('Comparação de Métodos de Seleção\n(Scores Normalizados)', 
                           fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Características')
        
        # 2. Top características por Information Gain
        ig_scores = self.calculate_information_gain(X, y)
        top_features_ig = np.argsort(ig_scores)[-10:]
        
        axes[0, 1].barh(range(10), ig_scores[top_features_ig], color='#FF6B6B')
        axes[0, 1].set_yticks(range(10))
        axes[0, 1].set_yticklabels([self.feature_names[i] if i < len(self.feature_names) 
                                   else f'Feature_{i}' for i in top_features_ig])
        axes[0, 1].set_title('Top 10 - Information Gain', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Information Gain (bits)')
        
        # 3. Correlação IG vs MI
        mi_scores = self.calculate_mutual_information(X, y)
        axes[0, 2].scatter(ig_scores, mi_scores, alpha=0.7, s=60, color='#4ECDC4')
        axes[0, 2].set_xlabel('Information Gain')
        axes[0, 2].set_ylabel('Mutual Information')
        axes[0, 2].set_title('Correlação: IG vs MI', fontsize=14, fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Adiciona linha de tendência
        z = np.polyfit(ig_scores, mi_scores, 1)
        p = np.poly1d(z)
        axes[0, 2].plot(ig_scores, p(ig_scores), "--", color='red', linewidth=2)
        
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
        top_kl_features = [features_analyzed[i] for i in top_kl_indices]
        top_kl_values = [kl_divergences[i] for i in top_kl_indices]
        
        axes[1, 2].barh(range(15), top_kl_values, color='#45B7D1')
        axes[1, 2].set_yticks(range(15))
        axes[1, 2].set_yticklabels(top_kl_features, fontsize=10)
        axes[1, 2].set_title('Top 15 - Divergência KL\n(Normal vs Ataque)', 
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
        axes[2, 0].set_xlabel('Valor da Característica')
        axes[2, 0].set_ylabel('Densidade')
        axes[2, 0].legend()
        axes[2, 0].grid(True, alpha=0.3)
        
        # 8. Redundância entre características
        # Calcula MI entre pares de características top
        top_features = np.argsort(ig_scores)[-10:]
        redundancy_matrix = np.zeros((10, 10))
        
        for i in range(10):
            for j in range(10):
                if i != j:
                    mi_ij = mutual_info_regression(X[:, [top_features[i]]], 
                             X[:, top_features[j]], 
                             random_state=42)[0]
                    redundancy_matrix[i, j] = mi_ij
        
        feature_labels = [features_analyzed[i] for i in top_features]
        sns.heatmap(redundancy_matrix, 
                   xticklabels=feature_labels,
                   yticklabels=feature_labels,
                   annot=True, fmt='.2f', cmap='Reds', ax=axes[2, 1])
        axes[2, 1].set_title('Matriz de Redundância\n(Top 10 Características)', 
                           fontsize=14, fontweight='bold')
        
        # 9. Eficiência dos métodos
        # Calcula razão performance/complexidade
        method_efficiency = {}
        for method_name, results in eval_results.items():
            # Pega resultado com 15 características
            idx_15 = results['k_features'].index(15) if 15 in results['k_features'] else -1
            if idx_15 >= 0:
                performance = results['cv_score'][idx_15]
                # Assume complexidade baseada no número de cálculos necessários
                complexity = {
                    'Information_Gain': 1.0,
                    'Mutual_Information': 1.5,
                    'Pearson_Correlation': 0.5,
                    'F_Statistics': 0.8
                }.get(method_name, 1.0)
                
                efficiency = performance / complexity
                method_efficiency[method_name] = {
                    'performance': performance,
                    'complexity': complexity,
                    'efficiency': efficiency
                }
        
        methods_names = list(method_efficiency.keys())
        efficiencies = [method_efficiency[m]['efficiency'] for m in methods_names]
        performances = [method_efficiency[m]['performance'] for m in methods_names]
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        axes[2, 2].scatter(performances, efficiencies, s=200, alpha=0.7, c=colors[:len(methods_names)])
        
        for i, method in enumerate(methods_names):
            axes[2, 2].annotate(method.replace('_', '\n'), 
                               (performances[i], efficiencies[i]),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=10, fontweight='bold')
        
        axes[2, 2].set_title('Eficiência dos Métodos\n(Performance vs Complexidade)', 
                           fontsize=14, fontweight='bold')
        axes[2, 2].set_xlabel('Performance (Acurácia)')
        axes[2, 2].set_ylabel('Eficiência (Performance/Complexidade)')
        axes[2, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_filename_png, dpi=300, bbox_inches='tight')
        plt.show()
        
        return eval_results, method_efficiency
    
    def generate_summary_tables(self, X, y):
        """Gera tabelas resumo para o trabalho"""
        
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
        
        return df_summary, df_top_features

def main():
    """Função principal"""
    print("=== ANÁLISE DE TEORIA DA INFORMAÇÃO ===")
    print("Justificativas para seleção de características e quantificação de anomalias\n")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    script_path = Path(__file__).resolve()
    output_dir = script_path.parent.parent / 'Resultados'
    os.makedirs(output_dir, exist_ok=True)

    # Define os nomes de todos os arquivos de saída
    png_filename = output_dir / f"analise_teoria_informacao_{timestamp}.png"
    summary_csv = output_dir / f"comparacao_metodos_selecao_{timestamp}.csv"
    top_features_csv = output_dir / f"top_caracteristicas_{timestamp}.csv"
    kl_analysis_csv = output_dir / f"analise_kl_divergence_{timestamp}.csv"
    
    # Inicializa analisador
    analyzer = InformationTheoryAnalysis()
    
    # Gera dataset sintético
    print("Gerando dataset sintético de rede...")
    X, y, attack_types = analyzer.generate_network_dataset(n_samples=5000, n_features=23)
    
    print(f"Dataset gerado: {X.shape[0]} amostras, {X.shape[1]} características")
    print(f"Distribuição de classes: Normal={np.sum(y==0)}, Ataque={np.sum(y==1)}")
    
    # Executa análise completa
    print("\nExecutando análise de teoria da informação...")
    eval_results, method_efficiency = analyzer.plot_information_theory_analysis(X, y, png_filename)
    
    # Gera tabelas resumo
    print("\nGerando tabelas resumo...")
    df_summary, df_top_features = analyzer.generate_summary_tables(X, y)
    
    print("\n=== TABELA 1: COMPARAÇÃO DE MÉTODOS ===")
    print(df_summary.round(4))
    
    print("\n=== TABELA 2: TOP 10 CARACTERÍSTICAS ===")
    print(df_top_features.round(4))
    
    # Salva resultados
    df_summary.to_csv(summary_csv, index=False)
    df_top_features.to_csv(top_features_csv, index=False)
    
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
    
    print("\nArquivos salvos:")
    print(f"- {png_filename}")
    print(f"- {summary_csv}")
    print(f"- {top_features_csv}")
    print(f"- {kl_analysis_csv}")

if __name__ == "__main__":
    main()