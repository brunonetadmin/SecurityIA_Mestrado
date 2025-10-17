"""
Script adaptado para comparação de arquiteturas neurais
Utiliza dados reais da base CSE-CIC-IDS2018 em sequências temporais
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from datetime import datetime
from keras.models import Sequential
from keras.layers import SimpleRNN, LSTM, Dense, Dropout, MultiHeadAttention, LayerNormalization, Bidirectional
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import time
import warnings
warnings.filterwarnings('ignore')

# Importa o módulo de carregamento de dados
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_loader_cse_cic_ids2018 import CSECICIDSDataLoader

# Configuração para gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NetworkArchitectureComparison:
    def __init__(self, sequence_length=100, n_features=None):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.results = {}
        self.data_loader = CSECICIDSDataLoader()
        
    def load_and_prepare_sequences(self, n_samples=10000):
        """Carrega dados reais e prepara sequências temporais"""
        print("Carregando dados CSE-CIC-IDS2018 para comparação de arquiteturas...")
        
        # Carrega dados
        X, y, y_names, feature_names = self.data_loader.load_data(sample_size=n_samples * 2)
        
        # Converte para classificação binária
        X, y = self.data_loader.get_binary_classification_data(X, y, y_names)
        
        # Atualiza número de features
        self.n_features = X.shape[1]
        
        # Cria sequências temporais
        print(f"Criando sequências temporais de tamanho {self.sequence_length}...")
        X_sequences, y_sequences = self.data_loader.create_temporal_sequences(
            X, y, self.sequence_length, step=self.sequence_length // 2
        )
        
        print(f"Sequências criadas: {X_sequences.shape}")
        print(f"Distribuição: Normal={np.sum(y_sequences==0)}, Ataque={np.sum(y_sequences==1)}")
        
        return X_sequences, y_sequences
    
    def create_simple_rnn(self):
        """Cria modelo RNN simples"""
        model = Sequential([
            SimpleRNN(64, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.3),
            SimpleRNN(32),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def create_lstm(self):
        """Cria modelo LSTM unidirecional"""
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model

    def create_bidirectional_lstm(self):
        """Cria modelo LSTM Bidirecional"""
        model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=(self.sequence_length, self.n_features)),
            Dropout(0.3),
            Bidirectional(LSTM(32)),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def create_transformer_like(self):
        """Cria modelo baseado em Transformer"""
        inputs = tf.keras.Input(shape=(self.sequence_length, self.n_features))
        
        # Projeção para dimensão adequada para atenção
        projected = Dense(64)(inputs)
        
        # Multi-head attention
        attention_output = MultiHeadAttention(num_heads=4, key_dim=16)(projected, projected)
        attention_output = LayerNormalization()(attention_output + projected)
        
        # Feed-forward
        ff_output = Dense(128, activation='relu')(attention_output)
        ff_output = Dense(64)(ff_output)
        ff_output = LayerNormalization()(ff_output + attention_output)
        
        # Pooling e classificação
        pooled = tf.keras.layers.GlobalAveragePooling1D()(ff_output)
        outputs = Dense(1, activation='sigmoid')(pooled)
        
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def evaluate_memory_degradation(self):
        """Avalia degradação de memória com dados reais em diferentes tamanhos de sequência"""
        sequence_lengths = [10, 25, 50, 100, 200]
        architectures = ['SimpleRNN', 'LSTM']
        results = {arch: [] for arch in architectures}
        
        print("\nAvaliando degradação de memória com sequências reais...")
        
        for seq_len in sequence_lengths:
            print(f"  Testando sequência de tamanho {seq_len}...")
            
            # Carrega dados com novo tamanho de sequência
            temp_comp = NetworkArchitectureComparison(sequence_length=seq_len)
            X, y = temp_comp.load_and_prepare_sequences(n_samples=2000)
            
            # Garante que temos amostras suficientes
            if len(X) < 100:
                print(f"    Pulando seq_len={seq_len} - amostras insuficientes")
                continue
                
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            for arch in architectures:
                try:
                    model = temp_comp.create_simple_rnn() if arch == 'SimpleRNN' else temp_comp.create_lstm()
                    
                    # Treina com callbacks para early stopping
                    early_stop = tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=3, restore_best_weights=True
                    )
                    
                    history = model.fit(
                        X_train, y_train, 
                        epochs=10, 
                        batch_size=32, 
                        validation_split=0.2, 
                        verbose=0,
                        callbacks=[early_stop]
                    )
                    
                    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
                    results[arch].append(accuracy)
                    
                    # Limpa memória
                    tf.keras.backend.clear_session()
                    
                except Exception as e:
                    print(f"    Erro com {arch} em seq_len={seq_len}: {e}")
                    results[arch].append(0.5)  # Valor padrão em caso de erro
        
        return sequence_lengths[:len(results['SimpleRNN'])], results
    
    def compare_computational_complexity(self):
        """Compara complexidade computacional com dados reais"""
        sequence_lengths = [50, 100, 200]
        times = {'SimpleRNN': [], 'LSTM': [], 'LSTM Bidirecional': [], 'Transformer': []}
        
        print("\nMedindo complexidade computacional com dados reais...")
        
        for seq_len in sequence_lengths:
            print(f"  Medindo tempo para sequência {seq_len}...")
            
            temp_comp = NetworkArchitectureComparison(sequence_length=seq_len)
            X, y = temp_comp.load_and_prepare_sequences(n_samples=1000)
            
            if len(X) < 100:
                print(f"    Pulando seq_len={seq_len} - amostras insuficientes")
                continue
            
            for arch_name in times.keys():
                try:
                    if arch_name == 'SimpleRNN': 
                        model = temp_comp.create_simple_rnn()
                    elif arch_name == 'LSTM': 
                        model = temp_comp.create_lstm()
                    elif arch_name == 'LSTM Bidirecional': 
                        model = temp_comp.create_bidirectional_lstm()
                    else: 
                        model = temp_comp.create_transformer_like()
                    
                    start_time = time.time()
                    model.fit(X, y, epochs=5, batch_size=32, verbose=0)
                    end_time = time.time()
                    
                    times[arch_name].append(end_time - start_time)
                    
                    # Limpa memória
                    tf.keras.backend.clear_session()
                    
                except Exception as e:
                    print(f"    Erro com {arch_name}: {e}")
                    times[arch_name].append(0)
        
        # Ajusta listas para mesmo tamanho
        valid_lengths = sequence_lengths[:len(times['SimpleRNN'])]
        
        return valid_lengths, times
    
    def run_full_comparison(self):
        """Executa comparação completa com dados reais"""
        print("=== Iniciando comparação completa com dados CSE-CIC-IDS2018 ===")
        
        # Carrega dados
        X, y = self.load_and_prepare_sequences(n_samples=5000)
        
        # Divide dados
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nDados de treinamento: {X_train.shape}")
        print(f"Dados de teste: {X_test.shape}")
        
        models = {
            'SimpleRNN': self.create_simple_rnn(),
            'LSTM': self.create_lstm(),
            'LSTM Bidirecional': self.create_bidirectional_lstm(),
            'Transformer': self.create_transformer_like()
        }
        
        print("\nTreinando modelos com dados reais...")
        
        for name, model in models.items():
            print(f"\nTreinando {name}...")
            
            try:
                # Callbacks para melhor treinamento
                early_stop = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True
                )
                reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=3, min_lr=0.00001
                )
                
                start_time = time.time()
                
                history = model.fit(
                    X_train, y_train, 
                    epochs=30, 
                    batch_size=64, 
                    validation_split=0.2, 
                    verbose=1,
                    callbacks=[early_stop, reduce_lr]
                )
                
                training_time = time.time() - start_time
                
                # Avaliação
                y_pred = (model.predict(X_test) > 0.5).astype(int)
                loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                
                print(f"\n--- Relatório de Classificação para {name} ---")
                print(classification_report(y_test, y_pred, target_names=['Normal', 'Ataque']))
                report = classification_report(y_test, y_pred, output_dict=True)
                
                self.results[name] = {
                    'accuracy': accuracy,
                    'training_time': training_time,
                    'history': history,
                    'predictions': y_pred,
                    'model': model,
                    'report': report,
                    'epochs_trained': len(history.history['loss'])
                }
                
            except Exception as e:
                print(f"Erro ao treinar {name}: {e}")
                self.results[name] = {
                    'accuracy': 0.5,
                    'training_time': 0,
                    'history': None,
                    'predictions': np.zeros_like(y_test),
                    'model': None,
                    'report': {'1': {'f1-score': 0, 'precision': 0, 'recall': 0}},
                    'epochs_trained': 0
                }
        
        return self.results
    
    def plot_results(self, output_filename_png):
        """Gera gráficos comparativos com resultados reais"""
        if not self.results:
            print("Execute a comparação antes de plotar os resultados.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        fig.suptitle('Comparativo de Arquiteturas - Base CSE-CIC-IDS2018', fontsize=18, fontweight='bold')
        
        # 1. Acurácia Final
        accuracies = [self.results[name]['accuracy'] for name in self.results.keys()]
        training_times = [self.results[name]['training_time'] for name in self.results.keys()]
        names = list(self.results.keys())
        colors = sns.color_palette("husl", len(names))

        axes[0, 0].bar(names, accuracies, color=colors)
        axes[0, 0].set_title('Acurácia Final (Dados Reais)', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Acurácia')
        axes[0, 0].set_ylim(min(accuracies) - 0.05, 1.0)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Tempo de Treinamento
        axes[0, 1].bar(names, training_times, color=colors)
        axes[0, 1].set_title('Tempo de Treinamento', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Tempo (segundos)')
        for i, v in enumerate(training_times):
            axes[0, 1].text(i, v, f'{v:.1f}s', ha='center', va='bottom', fontweight='bold')

        # 3. Degradação de Memória
        seq_lengths_mem, memory_results = self.evaluate_memory_degradation()
        if seq_lengths_mem:
            for arch, results in memory_results.items():
                if results:  # Verifica se há resultados
                    axes[0, 2].plot(seq_lengths_mem, results, marker='o', linewidth=2, 
                                   label=arch, markersize=8)
            axes[0, 2].set_title('Degradação de Memória (Dados Reais)', fontsize=14, fontweight='bold')
            axes[0, 2].set_xlabel('Tamanho da Sequência')
            axes[0, 2].set_ylabel('Acurácia')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Complexidade Computacional
        seq_lengths_comp, comp_times = self.compare_computational_complexity()
        if seq_lengths_comp:
            for arch, times in comp_times.items():
                if times:  # Verifica se há tempos registrados
                    axes[1, 0].plot(seq_lengths_comp, times, marker='s', linewidth=2, 
                                   label=arch, markersize=8)
            axes[1, 0].set_title('Complexidade Computacional', fontsize=14, fontweight='bold')
            axes[1, 0].set_xlabel('Tamanho da Sequência')
            axes[1, 0].set_ylabel('Tempo de Treinamento (s)')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            if max([max(t) for t in comp_times.values() if t]) > 0:
                axes[1, 0].set_yscale('log')

        # 5. Curvas de Aprendizado
        for name, result in self.results.items():
            if result['history'] is not None:
                axes[1, 1].plot(result['history'].history['val_accuracy'], '--', 
                               linewidth=2, label=f'{name} - Validação')
        axes[1, 1].set_title('Curvas de Aprendizado (Validação)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Épocas')
        axes[1, 1].set_ylabel('Acurácia')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # 6. F1-Score vs Tempo (Eficiência)
        f1_scores = [self.results[name]['report'].get('1', {}).get('f1-score', 0) 
                    for name in self.results.keys()]
        valid_indices = [i for i, f1 in enumerate(f1_scores) if f1 > 0]
        
        if valid_indices:
            valid_times = [training_times[i] for i in valid_indices]
            valid_f1 = [f1_scores[i] for i in valid_indices]
            valid_names = [names[i] for i in valid_indices]
            valid_colors = [colors[i] for i in valid_indices]
            
            axes[1, 2].scatter(valid_times, valid_f1, s=200, alpha=0.7, c=valid_colors)
            for i, name in enumerate(valid_names):
                axes[1, 2].annotate(name, (valid_times[i], valid_f1[i]), 
                                   xytext=(10, -15), textcoords='offset points', 
                                   fontsize=12, fontweight='bold')
        
        axes[1, 2].set_title('Eficiência: F1-Score (Ataque) vs Tempo', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Tempo de Treinamento (s)')
        axes[1, 2].set_ylabel('F1-Score para Classe "Ataque"')
        axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_filename_png, dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_summary_tables(self):
        """Gera tabelas resumo com resultados da comparação"""
        if not self.results:
            return pd.DataFrame(), pd.DataFrame()
        
        # Tabela 1: Resumo de Performance
        summary_data = []
        for name, result in self.results.items():
            if result['report'] and '1' in result['report']:
                report_ataque = result['report']['1']
                summary_data.append({
                    'Arquitetura': name,
                    'Acurácia': result['accuracy'],
                    'Tempo_Treinamento_s': result['training_time'],
                    'Épocas_Treinadas': result['epochs_trained'],
                    'Precisão_Ataque': report_ataque.get('precision', 0),
                    'Recall_Ataque': report_ataque.get('recall', 0),
                    'F1_Score_Ataque': report_ataque.get('f1-score', 0),
                    'Parâmetros_Aprox': self._estimate_parameters(name)
                })
        
        df_results = pd.DataFrame(summary_data)
        
        # Tabela 2: Análise de Eficiência
        efficiency_data = []
        for name, result in self.results.items():
            if result['accuracy'] > 0.5:  # Apenas modelos que funcionaram
                f1_ataque = result['report'].get('1', {}).get('f1-score', 0)
                efficiency = f1_ataque / (result['training_time'] + 1)  # +1 para evitar divisão por zero
                
                efficiency_data.append({
                    'Arquitetura': name,
                    'F1_Score_Ataque': f1_ataque,
                    'Tempo_Treinamento_s': result['training_time'],
                    'Eficiência': efficiency,
                    'Tempo_por_Época': result['training_time'] / max(result['epochs_trained'], 1)
                })
        
        df_efficiency = pd.DataFrame(efficiency_data).sort_values('Eficiência', ascending=False)
        
        return df_results, df_efficiency
    
    def _estimate_parameters(self, architecture):
        """Estima número aproximado de parâmetros para cada arquitetura"""
        # Estimativas baseadas na configuração padrão
        params_map = {
            'SimpleRNN': 64*self.n_features + 64*64 + 32*64 + 32*32 + 16*32 + 1*16,
            'LSTM': 4*(64*self.n_features + 64*64) + 4*(32*64 + 32*32) + 16*32 + 1*16,
            'LSTM Bidirecional': 2*(4*(64*self.n_features + 64*64) + 4*(32*64 + 32*32)) + 16*64 + 1*16,
            'Transformer': self.n_features*64 + 4*16*64*3 + 128*64 + 64*128 + 1*64
        }
        return params_map.get(architecture, 0)

def main():
    print("=== COMPARAÇÃO DE ARQUITETURAS NEURAIS - CSE-CIC-IDS2018 ===")
    print("Análise com dados reais de tráfego de rede\n")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    script_path = Path(__file__).resolve()
    output_dir = script_path.parent.parent / 'Resultados'
    
    # Garante que o diretório de resultados exista
    os.makedirs(output_dir, exist_ok=True)
    
    # Cria os nomes de arquivo completos
    png_filename = output_dir / f"comparacao_arquiteturas_real_{timestamp}.png"
    csv_filename = output_dir / f"resultados_arquiteturas_real_{timestamp}.csv"
    efficiency_csv = output_dir / f"eficiencia_arquiteturas_real_{timestamp}.csv"

    # Executa comparação
    comparator = NetworkArchitectureComparison(sequence_length=100)
    results = comparator.run_full_comparison()
    
    # Gera visualizações
    comparator.plot_results(png_filename)
    
    # Gera tabelas
    df_results, df_efficiency = comparator.generate_summary_tables()
    
    print("\n=== RESULTADOS COMPARATIVOS DETALHADOS ===")
    print(df_results.round(4))
    
    print("\n=== ANÁLISE DE EFICIÊNCIA ===")
    print(df_efficiency.round(4))
    
    # Salva resultados
    df_results.to_csv(csv_filename, index=False)
    df_efficiency.to_csv(efficiency_csv, index=False)
    
    # Análise específica para a base CSE-CIC-IDS2018
    print("\n=== INSIGHTS ESPECÍFICOS DA BASE CSE-CIC-IDS2018 ===")
    
    # Melhor arquitetura
    if not df_results.empty:
        best_arch = df_results.loc[df_results['F1_Score_Ataque'].idxmax()]
        print(f"\nMelhor arquitetura para detecção de ataques:")
        print(f"  {best_arch['Arquitetura']}")
        print(f"  F1-Score (Ataque): {best_arch['F1_Score_Ataque']:.4f}")
        print(f"  Precisão (Ataque): {best_arch['Precisão_Ataque']:.4f}")
        print(f"  Recall (Ataque): {best_arch['Recall_Ataque']:.4f}")
        
        # Comparação LSTM vs LSTM Bidirecional
        if 'LSTM' in df_results['Arquitetura'].values and 'LSTM Bidirecional' in df_results['Arquitetura'].values:
            lstm_simple = df_results[df_results['Arquitetura'] == 'LSTM'].iloc[0]
            lstm_bi = df_results[df_results['Arquitetura'] == 'LSTM Bidirecional'].iloc[0]
            
            improvement = ((lstm_bi['F1_Score_Ataque'] - lstm_simple['F1_Score_Ataque']) / 
                          lstm_simple['F1_Score_Ataque']) * 100
            
            print(f"\nMelhoria LSTM Bidirecional sobre LSTM simples:")
            print(f"  F1-Score: {improvement:.1f}%")
            print(f"  Custo computacional adicional: {lstm_bi['Tempo_Treinamento_s']/lstm_simple['Tempo_Treinamento_s']:.1f}x")
    
    print(f"\nResultados salvos em:")
    print(f"- {png_filename}")
    print(f"- {csv_filename}")
    print(f"- {efficiency_csv}")

if __name__ == "__main__":
    main()