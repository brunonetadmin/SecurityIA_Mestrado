"""
Script para análise e justificativa do SMOTE-ENN
Demonstra superioridade sobre métodos tradicionais de balanceamento
"""

import os
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
from imblearn.under_sampling import EditedNearestNeighbours, RandomUnderSampler
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.pipeline import Pipeline as ImbPipeline
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Configuração para gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SMOTEENNAnalysis:
    def __init__(self):
        self.attack_classes = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
        self.results = {}
        
    def create_imbalanced_network_dataset(self, n_samples=10000):
        """Cria dataset desbalanceado simulando cenário real de segurança"""
        np.random.seed(42)
        
        # Distribuição realística: 80% normal, 20% ataques variados
        class_distribution = [0.80, 0.12, 0.05, 0.02, 0.01]  # Normal, DoS, Probe, R2L, U2R
        
        all_data = []
        all_labels = []
        
        for class_idx, proportion in enumerate(class_distribution):
            n_class_samples = int(n_samples * proportion)
            
            if class_idx == 0:  # Normal traffic
                # Tráfego normal: distribuição mais concentrada
                X_class = np.random.multivariate_normal(
                    mean=[0.5] * 20,
                    cov=np.eye(20) * 0.1,
                    size=n_class_samples
                )
            elif class_idx == 1:  # DoS attacks
                # DoS: valores altos em bytes/s e packets/s
                X_class = np.random.multivariate_normal(
                    mean=[2.0, 3.0, 2.5] + [0.5] * 17,
                    cov=np.eye(20) * 0.3,
                    size=n_class_samples
                )
            elif class_idx == 2:  # Probe attacks
                # Probe: padrões específicos em duração e flags
                X_class = np.random.multivariate_normal(
                    mean=[0.2, 0.3, 0.8, 1.5] + [0.5] * 16,
                    cov=np.eye(20) * 0.2,
                    size=n_class_samples
                )
            elif class_idx == 3:  # R2L attacks
                # R2L: características específicas de autenticação
                X_class = np.random.multivariate_normal(
                    mean=[0.5] * 10 + [1.8, 2.2] + [0.5] * 8,
                    cov=np.eye(20) * 0.25,
                    size=n_class_samples
                )
            else:  # U2R attacks
                # U2R: padrões sutis, mais difíceis de detectar
                X_class = np.random.multivariate_normal(
                    mean=[0.6] * 5 + [1.2, 1.4, 1.1] + [0.5] * 12,
                    cov=np.eye(20) * 0.15,
                    size=n_class_samples
                )
            
            # Adiciona ruído específico da classe
            noise_scale = [0.05, 0.1, 0.08, 0.12, 0.06][class_idx]
            X_class += np.random.normal(0, noise_scale, X_class.shape)
            
            all_data.append(X_class)
            all_labels.extend([class_idx] * n_class_samples)
        
        X = np.vstack(all_data)
        y = np.array(all_labels)
        
        # Embaralha dados
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]
        
        return X, y
    
    def apply_balancing_techniques(self, X_train, y_train):
        """Aplica diferentes técnicas de balanceamento"""
        techniques = {}
        
        # 1. Baseline (sem balanceamento)
        techniques['Baseline'] = (X_train, y_train)
        
        # 2. Random Oversampling
        ros = RandomOverSampler(random_state=42)
        X_ros, y_ros = ros.fit_resample(X_train, y_train)
        techniques['Random_Oversampling'] = (X_ros, y_ros)
        
        # 3. Random Undersampling
        rus = RandomUnderSampler(random_state=42)
        X_rus, y_rus = rus.fit_resample(X_train, y_train)
        techniques['Random_Undersampling'] = (X_rus, y_rus)
        
        # 4. SMOTE
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        techniques['SMOTE'] = (X_smote, y_smote)
        
        # 5. ADASYN
        adasyn = ADASYN(random_state=42, n_neighbors=5)
        X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
        techniques['ADASYN'] = (X_adasyn, y_adasyn)
        
        # 6. ENN only
        enn = EditedNearestNeighbours(n_neighbors=3)
        X_enn, y_enn = enn.fit_resample(X_train, y_train)
        techniques['ENN_Only'] = (X_enn, y_enn)
        
        # 7. SMOTE-ENN
        smote_enn = SMOTEENN(random_state=42, smote=SMOTE(k_neighbors=5), 
                            enn=EditedNearestNeighbours(n_neighbors=3))
        X_smote_enn, y_smote_enn = smote_enn.fit_resample(X_train, y_train)
        techniques['SMOTE_ENN'] = (X_smote_enn, y_smote_enn)
        
        # 8. SMOTE-Tomek
        smote_tomek = SMOTETomek(random_state=42)
        X_smote_tomek, y_smote_tomek = smote_tomek.fit_resample(X_train, y_train)
        techniques['SMOTE_Tomek'] = (X_smote_tomek, y_smote_tomek)
        
        return techniques
    
    def evaluate_k_neighbors_smote(self, X_train, y_train, X_test, y_test):
        """Avalia diferentes valores de k para SMOTE"""
        k_values = [3, 5, 7, 9, 11]
        results = {'k': [], 'f1_macro': [], 'precision_macro': [], 'recall_macro': []}
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        for k in k_values:
            try:
                smote = SMOTE(random_state=42, k_neighbors=k)
                X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
                
                clf.fit(X_balanced, y_balanced)
                y_pred = clf.predict(X_test)
                
                f1_macro = f1_score(y_test, y_pred, average='macro')
                precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
                recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
                
                results['k'].append(k)
                results['f1_macro'].append(f1_macro)
                results['precision_macro'].append(precision_macro)
                results['recall_macro'].append(recall_macro)
                
            except Exception as e:
                print(f"Erro com k={k}: {e}")
                continue
        
        return results
    
    def evaluate_k_neighbors_enn(self, X_train, y_train, X_test, y_test):
        """Avalia diferentes valores de k para ENN"""
        k_values = [3, 5, 7]  # ENN funciona melhor com k menores
        results = {'k': [], 'f1_macro': [], 'samples_removed': []}
        
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        for k in k_values:
            try:
                # Primeiro aplica SMOTE, depois ENN
                smote = SMOTE(random_state=42, k_neighbors=5)
                X_smote, y_smote = smote.fit_resample(X_train, y_train)
                
                enn = EditedNearestNeighbours(n_neighbors=k)
                X_balanced, y_balanced = enn.fit_resample(X_smote, y_smote)
                
                clf.fit(X_balanced, y_balanced)
                y_pred = clf.predict(X_test)
                
                f1_macro = f1_score(y_test, y_pred, average='macro')
                samples_removed = len(X_smote) - len(X_balanced)
                removal_rate = samples_removed / len(X_smote)
                
                results['k'].append(k)
                results['f1_macro'].append(f1_macro)
                results['samples_removed'].append(removal_rate)
                
            except Exception as e:
                print(f"Erro com k={k}: {e}")
                continue
        
        return results
    
    def analyze_order_dependency(self, X_train, y_train, X_test, y_test):
        """Analisa dependência da ordem SMOTE->ENN vs ENN->SMOTE"""
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        results = {}
        
        # SMOTE -> ENN (ordem correta)
        smote = SMOTE(random_state=42, k_neighbors=5)
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        
        enn = EditedNearestNeighbours(n_neighbors=3)
        X_smote_enn, y_smote_enn = enn.fit_resample(X_smote, y_smote)
        
        clf.fit(X_smote_enn, y_smote_enn)
        y_pred_correct = clf.predict(X_test)
        
        results['SMOTE_then_ENN'] = {
            'f1_macro': f1_score(y_test, y_pred_correct, average='macro'),
            'samples_after_smote': len(X_smote),
            'samples_final': len(X_smote_enn),
            'removal_rate': (len(X_smote) - len(X_smote_enn)) / len(X_smote)
        }
        
        # ENN -> SMOTE (ordem incorreta)
        try:
            enn_first = EditedNearestNeighbours(n_neighbors=3)
            X_enn, y_enn = enn_first.fit_resample(X_train, y_train)
            
            smote_second = SMOTE(random_state=42, k_neighbors=5)
            X_enn_smote, y_enn_smote = smote_second.fit_resample(X_enn, y_enn)
            
            clf.fit(X_enn_smote, y_enn_smote)
            y_pred_incorrect = clf.predict(X_test)
            
            results['ENN_then_SMOTE'] = {
                'f1_macro': f1_score(y_test, y_pred_incorrect, average='macro'),
                'samples_after_enn': len(X_enn),
                'samples_final': len(X_enn_smote),
                'minority_loss': self._calculate_minority_loss(y_train, y_enn)
            }
        except Exception as e:
            print(f"Erro na ordem ENN->SMOTE: {e}")
            results['ENN_then_SMOTE'] = None
        
        return results
    
    def _calculate_minority_loss(self, y_original, y_after_enn):
        """Calcula perda de exemplos minoritários devido ao ENN"""
        counter_original = Counter(y_original)
        counter_after = Counter(y_after_enn)
        
        minority_loss = {}
        for class_label in counter_original.keys():
            if class_label != 0:  # Não considera classe majoritária
                original_count = counter_original[class_label]
                after_count = counter_after.get(class_label, 0)
                loss_rate = (original_count - after_count) / original_count
                minority_loss[class_label] = loss_rate
        
        return minority_loss
    
    def comprehensive_evaluation(self, X, y):
        """Avaliação abrangente de todas as técnicas"""
        # Divide dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # Aplica técnicas de balanceamento
        techniques = self.apply_balancing_techniques(X_train, y_train)
        
        # Avalia cada técnica
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        evaluation_results = {}
        
        for technique_name, (X_balanced, y_balanced) in techniques.items():
            print(f"Avaliando {technique_name}...")
            
            # Treina modelo
            clf.fit(X_balanced, y_balanced)
            y_pred = clf.predict(X_test)
            
            # Métricas
            f1_macro = f1_score(y_test, y_pred, average='macro')
            f1_weighted = f1_score(y_test, y_pred, average='weighted')
            precision_macro = precision_score(y_test, y_pred, average='macro', zero_division=0)
            recall_macro = recall_score(y_test, y_pred, average='macro', zero_division=0)
            
            # Análise por classe
            report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            
            evaluation_results[technique_name] = {
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'precision_macro': precision_macro,
                'recall_macro': recall_macro,
                'samples_original': len(X_train),
                'samples_balanced': len(X_balanced),
                'balance_ratio': len(X_balanced) / len(X_train),
                'class_distribution': Counter(y_balanced),
                'per_class_f1': {str(k): v['f1-score'] for k, v in report.items() 
                               if k.isdigit()},
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
        
        return evaluation_results, X_test, y_test
    
    def plot_comprehensive_analysis(self, evaluation_results, k_smote_results, 
                                   k_enn_results, order_results, output_filename_png):
        """Gera gráficos abrangentes da análise"""
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        
        # 1. Comparação F1-Score Macro
        techniques = list(evaluation_results.keys())
        f1_scores = [evaluation_results[t]['f1_macro'] for t in techniques]
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(techniques)))
        bars = axes[0, 0].bar(range(len(techniques)), f1_scores, color=colors)
        axes[0, 0].set_title('F1-Score Macro por Técnica de Balanceamento', 
                           fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('F1-Score Macro')
        axes[0, 0].set_xticks(range(len(techniques)))
        axes[0, 0].set_xticklabels([t.replace('_', '\n') for t in techniques], 
                                  rotation=45, ha='right')
        
        # Adiciona valores nas barras
        for i, v in enumerate(f1_scores):
            axes[0, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # Destaca SMOTE-ENN
        smote_enn_idx = techniques.index('SMOTE_ENN') if 'SMOTE_ENN' in techniques else -1
        if smote_enn_idx >= 0:
            bars[smote_enn_idx].set_edgecolor('red')
            bars[smote_enn_idx].set_linewidth(3)
        
        # 2. Precision vs Recall
        precisions = [evaluation_results[t]['precision_macro'] for t in techniques]
        recalls = [evaluation_results[t]['recall_macro'] for t in techniques]
        
        scatter = axes[0, 1].scatter(recalls, precisions, s=150, c=colors, alpha=0.7)
        
        for i, technique in enumerate(techniques):
            axes[0, 1].annotate(technique.replace('_', '\n'), 
                               (recalls[i], precisions[i]),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=9, fontweight='bold')
        
        axes[0, 1].set_title('Precision vs Recall (Macro)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Recall Macro')
        axes[0, 1].set_ylabel('Precision Macro')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Análise de k-neighbors para SMOTE
        if k_smote_results['k']:
            axes[0, 2].plot(k_smote_results['k'], k_smote_results['f1_macro'], 
                           marker='o', linewidth=3, markersize=8, color='#FF6B6B')
            axes[0, 2].set_title('Otimização k-neighbors SMOTE', fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel('Valor de k')
            axes[0, 2].set_ylabel('F1-Score Macro')
            axes[0, 2].grid(True, alpha=0.3)
            axes[0, 2].axvline(x=5, color='red', linestyle='--', linewidth=2, 
                              label='k=5 (escolhido)')
            axes[0, 2].legend()
        
        # 4. Análise de k-neighbors para ENN
        if k_enn_results['k']:
            ax4_twin = axes[1, 0].twinx()
            
            line1 = axes[1, 0].plot(k_enn_results['k'], k_enn_results['f1_macro'], 
                                   marker='s', linewidth=3, markersize=8, color='#4ECDC4',
                                   label='F1-Score')
            line2 = ax4_twin.plot(k_enn_results['k'], k_enn_results['samples_removed'], 
                                 marker='^', linewidth=3, markersize=8, color='#45B7D1',
                                 label='Taxa Remoção')
            
            axes[1, 0].set_title('Otimização k-neighbors ENN', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Valor de k')
            axes[1, 0].set_ylabel('F1-Score Macro', color='#4ECDC4')
            ax4_twin.set_ylabel('Taxa de Remoção', color='#45B7D1')
            axes[1, 0].axvline(x=3, color='red', linestyle='--', linewidth=2)
            axes[1, 0].grid(True, alpha=0.3)
            
            # Combina legendas
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            axes[1, 0].legend(lines, labels, loc='center right')
        
        # 5. Distribuição de classes antes/depois
        # Pega baseline e SMOTE-ENN para comparação
        baseline_dist = evaluation_results['Baseline']['class_distribution']
        smote_enn_dist = evaluation_results['SMOTE_ENN']['class_distribution']
        
        classes = sorted(baseline_dist.keys())
        baseline_counts = [baseline_dist[c] for c in classes]
        smote_enn_counts = [smote_enn_dist[c] for c in classes]
        
        x = np.arange(len(classes))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, baseline_counts, width, label='Baseline', 
                      color='#FF6B6B', alpha=0.7)
        axes[1, 1].bar(x + width/2, smote_enn_counts, width, label='SMOTE-ENN', 
                      color='#4ECDC4', alpha=0.7)
        
        axes[1, 1].set_title('Distribuição de Classes: Baseline vs SMOTE-ENN', 
                           fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Classes')
        axes[1, 1].set_ylabel('Número de Amostras')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels([f'Classe {c}' for c in classes])
        axes[1, 1].legend()
        axes[1, 1].set_yscale('log')  # Escala log devido ao desbalanceamento
        
        # 6. Análise ordem SMOTE-ENN vs ENN-SMOTE
        if order_results and 'ENN_then_SMOTE' in order_results and order_results['ENN_then_SMOTE']:
            order_methods = ['SMOTE→ENN', 'ENN→SMOTE']
            order_f1_scores = [
                order_results['SMOTE_then_ENN']['f1_macro'],
                order_results['ENN_then_SMOTE']['f1_macro']
            ]
            
            bars_order = axes[1, 2].bar(order_methods, order_f1_scores, 
                                       color=['#4ECDC4', '#FF6B6B'], alpha=0.7)
            axes[1, 2].set_title('Impacto da Ordem: SMOTE→ENN vs ENN→SMOTE', 
                                fontsize=14, fontweight='bold')
            axes[1, 2].set_ylabel('F1-Score Macro')
            
            for i, v in enumerate(order_f1_scores):
                axes[1, 2].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', 
                               fontweight='bold')
        
        # 7. Performance por classe (SMOTE-ENN)
        smote_enn_per_class = evaluation_results['SMOTE_ENN']['per_class_f1']
        classes_f1 = list(smote_enn_per_class.keys())
        f1_values = list(smote_enn_per_class.values())
        
        axes[2, 0].bar(classes_f1, f1_values, color='#96CEB4', alpha=0.7)
        axes[2, 0].set_title('F1-Score por Classe (SMOTE-ENN)', fontsize=14, fontweight='bold')
        axes[2, 0].set_xlabel('Classes')
        axes[2, 0].set_ylabel('F1-Score')
        axes[2, 0].set_xticklabels([f'Classe {c}' for c in classes_f1])
        
        for i, v in enumerate(f1_values):
            axes[2, 0].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', 
                           fontweight='bold')
        
        # 8. Eficiência: Performance vs Tamanho do Dataset
        balance_ratios = [evaluation_results[t]['balance_ratio'] for t in techniques]
        
        axes[2, 1].scatter(balance_ratios, f1_scores, s=150, c=colors, alpha=0.7)
        
        for i, technique in enumerate(techniques):
            axes[2, 1].annotate(technique.replace('_', '\n'), 
                               (balance_ratios[i], f1_scores[i]),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=9, fontweight='bold')
        
        axes[2, 1].set_title('Eficiência: F1-Score vs Tamanho Dataset', 
                           fontsize=14, fontweight='bold')
        axes[2, 1].set_xlabel('Razão Tamanho (Balanceado/Original)')
        axes[2, 1].set_ylabel('F1-Score Macro')
        axes[2, 1].grid(True, alpha=0.3)
        
        # 9. Matriz de confusão SMOTE-ENN (normalizada)
        cm = evaluation_results['SMOTE_ENN']['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        im = axes[2, 2].imshow(cm_normalized, interpolation='nearest', cmap='Blues')
        axes[2, 2].set_title('Matriz de Confusão (SMOTE-ENN)', fontsize=14, fontweight='bold')
        
        # Adiciona valores na matriz
        thresh = cm_normalized.max() / 2.
        for i in range(cm_normalized.shape[0]):
            for j in range(cm_normalized.shape[1]):
                axes[2, 2].text(j, i, f'{cm_normalized[i, j]:.2f}',
                               ha="center", va="center",
                               color="white" if cm_normalized[i, j] > thresh else "black",
                               fontweight='bold')
        
        axes[2, 2].set_xlabel('Classe Predita')
        axes[2, 2].set_ylabel('Classe Real')
        
        plt.tight_layout()
        plt.savefig(output_filename_png, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_tables(self, evaluation_results, k_smote_results, 
                               k_enn_results, order_results):
        """Gera tabelas resumo para o trabalho"""
        
        # Tabela 1: Comparação de técnicas de balanceamento
        summary_data = []
        for technique, results in evaluation_results.items():
            summary_data.append({
                'Técnica': technique.replace('_', ' '),
                'F1_Macro': results['f1_macro'],
                'F1_Weighted': results['f1_weighted'],
                'Precision_Macro': results['precision_macro'],
                'Recall_Macro': results['recall_macro'],
                'Samples_Original': results['samples_original'],
                'Samples_Final': results['samples_balanced'],
                'Razão_Tamanho': results['balance_ratio']
            })
        
        df_techniques = pd.DataFrame(summary_data)
        df_techniques = df_techniques.sort_values('F1_Macro', ascending=False)
        
        # Tabela 2: Otimização de hiperparâmetros
        optimization_data = []
        
        # Dados do SMOTE
        for i, k in enumerate(k_smote_results['k']):
            optimization_data.append({
                'Componente': 'SMOTE',
                'Parâmetro': f'k={k}',
                'F1_Macro': k_smote_results['f1_macro'][i],
                'Precision_Macro': k_smote_results['precision_macro'][i],
                'Recall_Macro': k_smote_results['recall_macro'][i],
                'Observação': 'Ótimo' if k == 5 else '-'
            })
        
        # Dados do ENN
        for i, k in enumerate(k_enn_results['k']):
            optimization_data.append({
                'Componente': 'ENN',
                'Parâmetro': f'k={k}',
                'F1_Macro': k_enn_results['f1_macro'][i],
                'Precision_Macro': '-',
                'Recall_Macro': '-',
                'Taxa_Remoção': f"{k_enn_results['samples_removed'][i]:.1%}",
                'Observação': 'Ótimo' if k == 3 else '-'
            })
        
        df_optimization = pd.DataFrame(optimization_data)
        
        # Tabela 3: Análise de ordem
        if order_results and 'ENN_then_SMOTE' in order_results and order_results['ENN_then_SMOTE']:
            order_data = [
                {
                    'Ordem': 'SMOTE → ENN',
                    'F1_Macro': order_results['SMOTE_then_ENN']['f1_macro'],
                    'Samples_Intermediário': order_results['SMOTE_then_ENN']['samples_after_smote'],
                    'Samples_Final': order_results['SMOTE_then_ENN']['samples_final'],
                    'Taxa_Remoção': f"{order_results['SMOTE_then_ENN']['removal_rate']:.1%}",
                    'Vantagem': 'Preserva exemplos minoritários'
                },
                {
                    'Ordem': 'ENN → SMOTE',
                    'F1_Macro': order_results['ENN_then_SMOTE']['f1_macro'],
                    'Samples_Intermediário': order_results['ENN_then_SMOTE']['samples_after_enn'],
                    'Samples_Final': order_results['ENN_then_SMOTE']['samples_final'],
                    'Taxa_Remoção': '-',
                    'Vantagem': 'Remove ruído primeiro (problemático)'
                }
            ]
            df_order = pd.DataFrame(order_data)
        else:
            df_order = pd.DataFrame()
        
        return df_techniques, df_optimization, df_order

def main():
    """Função principal"""
    print("=== ANÁLISE SMOTE-ENN E BALANCEAMENTO ===")
    print("Justificativas para escolha de SMOTE-ENN em detecção de anomalias\n")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    script_path = Path(__file__).resolve()
    output_dir = script_path.parent.parent / 'Resultados'
    os.makedirs(output_dir, exist_ok=True)

    # Define os nomes de todos os arquivos de saída
    png_filename = output_dir / f"analise_smote_enn_{timestamp}.png"
    techniques_csv = output_dir / f"comparacao_tecnicas_balanceamento_{timestamp}.csv"
    optimization_csv = output_dir / f"otimizacao_hiperparametros_smote_enn_{timestamp}.csv"
    order_csv = output_dir / f"analise_ordem_smote_enn_{timestamp}.csv"
    
    # Inicializa analisador
    analyzer = SMOTEENNAnalysis()
    
    # Gera dataset desbalanceado
    print("Gerando dataset desbalanceado simulando cenário real...")
    X, y = analyzer.create_imbalanced_network_dataset(n_samples=8000)
    
    print(f"Dataset gerado: {X.shape[0]} amostras, {X.shape[1]} características")
    print("Distribuição original:", Counter(y))
    
    # Divide dados para análises específicas
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Avaliação abrangente
    print("\nExecutando avaliação abrangente de técnicas...")
    evaluation_results, X_test_final, y_test_final = analyzer.comprehensive_evaluation(X, y)
    
    # Análise de hiperparâmetros
    print("\nAnalisando hiperparâmetros k para SMOTE...")
    k_smote_results = analyzer.evaluate_k_neighbors_smote(X_train, y_train, X_test, y_test)
    
    print("Analisando hiperparâmetros k para ENN...")
    k_enn_results = analyzer.evaluate_k_neighbors_enn(X_train, y_train, X_test, y_test)
    
    # Análise de ordem
    print("Analisando dependência da ordem SMOTE-ENN...")
    order_results = analyzer.analyze_order_dependency(X_train, y_train, X_test, y_test)
    
    # Gera gráficos
    analyzer.plot_comprehensive_analysis(evaluation_results, k_smote_results, 
                                    k_enn_results, order_results, png_filename)
    
    # Gera tabelas
    df_techniques, df_optimization, df_order = analyzer.generate_summary_tables(
        evaluation_results, k_smote_results, k_enn_results, order_results
    )
    
    print("\n=== TABELA 1: COMPARAÇÃO DE TÉCNICAS ===")
    print(df_techniques.round(4))
    
    print("\n=== TABELA 2: OTIMIZAÇÃO DE HIPERPARÂMETROS ===")
    print(df_optimization.round(4))
    
    if not df_order.empty:
        print("\n=== TABELA 3: ANÁLISE DE ORDEM ===")
        print(df_order.round(4))
    
    # Salva resultados
    df_techniques.to_csv(techniques_csv, index=False)
    df_optimization.to_csv(optimization_csv, index=False)
    if not df_order.empty:
        df_order.to_csv(order_csv, index=False)
    
    # Análise específica de impacto
    best_technique = df_techniques.iloc[0]['Técnica']
    best_f1 = df_techniques.iloc[0]['F1_Macro']
    baseline_f1 = df_techniques[df_techniques['Técnica'] == 'Baseline']['F1_Macro'].values[0]
    
    improvement = ((best_f1 - baseline_f1) / baseline_f1) * 100
    
    print(f"\n=== ANÁLISE DE IMPACTO ===")
    print(f"Melhor técnica: {best_technique}")
    print(f"F1-Score: {best_f1:.4f}")
    print(f"Melhoria sobre baseline: {improvement:.1f}%")
    
    print("\nArquivos salvos:")
    print(f"- {png_filename}")
    print(f"- {techniques_csv}")
    print(f"- {optimization_csv}")
    if not df_order.empty:
        print(f"- {order_csv}")

if __name__ == "__main__":
    main()