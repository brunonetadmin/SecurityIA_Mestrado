"""
Script para comparação de arquiteturas neurais: LSTM Bidirecional vs LSTM vs RNN vs Transformer
Justifica a escolha de LSTM Bidirecional para detecção de anomalias de rede
"""

import os
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

# Configuração para gráficos
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class NetworkArchitectureComparison:
    def __init__(self, sequence_length=100, n_features=23):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.results = {}
        
    def generate_synthetic_network_data(self, n_samples=10000):
        """Gera dados sintéticos simulando tráfego de rede com anomalias que exigem contexto."""
        np.random.seed(42)
        
        normal_data, attack_data = [], []
        
        # Tráfego normal: padrões regulares
        for _ in range(n_samples // 2):
            base_pattern = np.random.normal(0.5, 0.1, self.n_features)
            sequence = [base_pattern * (0.8 + 0.4 * np.sin(2 * np.pi * t / 20)) + np.random.normal(0, 0.05, self.n_features) for t in range(self.sequence_length)]
            normal_data.append(sequence)
            
        # Tráfego de ataque: padrões anômalos
        for _ in range(n_samples // 2):
            attack_type = np.random.rand()
            if attack_type < 0.33:  # Ataque de Rajada (DoS)
                sequence = []
                for t in range(self.sequence_length):
                    spike = np.random.normal(2.0, 0.3, self.n_features) if t > 50 and t < 60 else np.random.normal(0.1, 0.05, self.n_features)
                    sequence.append(spike)
                attack_data.append(sequence)
            elif attack_type < 0.66: # Ataque "Low-and-Slow" seguido de exfiltração
                sequence = []
                for t in range(self.sequence_length):
                    if t < 80: # Fase lenta e sutil
                        point = np.random.normal(0.6, 0.05, self.n_features) * 1.1
                    else: # Exfiltração
                        point = np.random.normal(1.5, 0.2, self.n_features)
                    sequence.append(point)
                attack_data.append(sequence)
            else:  # Ataque com padrão exponencial
                base_pattern = np.random.exponential(0.3, self.n_features)
                sequence = [base_pattern + np.random.normal(0, 0.1, self.n_features) for t in range(self.sequence_length)]
                attack_data.append(sequence)
        
        X = np.array(normal_data + attack_data)
        y = np.array([0] * len(normal_data) + [1] * len(attack_data))
        
        return X, y
    
    def create_simple_rnn(self):
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
        inputs = tf.keras.Input(shape=(self.sequence_length, self.n_features))
        attention_output = MultiHeadAttention(num_heads=4, key_dim=32)(inputs, inputs)
        attention_output = LayerNormalization()(attention_output + inputs)
        ff_output = Dense(64, activation='relu')(attention_output)
        ff_output = Dense(self.n_features)(ff_output)
        ff_output = LayerNormalization()(ff_output + attention_output)
        pooled = tf.keras.layers.GlobalAveragePooling1D()(ff_output)
        outputs = Dense(1, activation='sigmoid')(pooled)
        model = tf.keras.Model(inputs, outputs)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def evaluate_memory_degradation(self):
        """Avalia degradação de memória. Focado em RNN vs LSTM."""
        sequence_lengths = [10, 25, 50, 100, 200, 500]
        architectures = ['SimpleRNN', 'LSTM']
        results = {arch: [] for arch in architectures}
        
        for seq_len in sequence_lengths:
            print(f"Testando degradação para sequência de tamanho {seq_len}...")
            temp_comp = NetworkArchitectureComparison(sequence_length=seq_len)
            X, y = temp_comp.generate_synthetic_network_data(n_samples=2000)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            for arch in architectures:
                model = temp_comp.create_simple_rnn() if arch == 'SimpleRNN' else temp_comp.create_lstm()
                history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2, verbose=0)
                _, accuracy = model.evaluate(X_test, y_test, verbose=0)
                results[arch].append(accuracy)
        
        return sequence_lengths, results
    
    def compare_computational_complexity(self):
        """Compara complexidade computacional das arquiteturas"""
        sequence_lengths = [50, 100, 200, 500, 1000]
        times = {'SimpleRNN': [], 'LSTM': [], 'LSTM Bidirecional': [], 'Transformer': []}
        
        for seq_len in sequence_lengths:
            print(f"Medindo tempo para sequência {seq_len}...")
            temp_comp = NetworkArchitectureComparison(sequence_length=seq_len)
            X, _ = temp_comp.generate_synthetic_network_data(n_samples=1000)
            
            for arch_name in times.keys():
                if arch_name == 'SimpleRNN': model = temp_comp.create_simple_rnn()
                elif arch_name == 'LSTM': model = temp_comp.create_lstm()
                elif arch_name == 'LSTM Bidirecional': model = temp_comp.create_bidirectional_lstm()
                else: model = temp_comp.create_transformer_like()
                
                start_time = time.time()
                model.fit(X, np.zeros(len(X)), epochs=5, batch_size=32, verbose=0)
                end_time = time.time()
                times[arch_name].append(end_time - start_time)
        
        return sequence_lengths, times
    
    def run_full_comparison(self):
        """Executa comparação completa"""
        print("Gerando dados sintéticos...")
        X, y = self.generate_synthetic_network_data()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        models = {
            'SimpleRNN': self.create_simple_rnn(),
            'LSTM': self.create_lstm(),
            'LSTM Bidirecional': self.create_bidirectional_lstm(),
            'Transformer': self.create_transformer_like()
        }
        
        print("Treinando modelos...")
        for name, model in models.items():
            print(f"Treinando {name}...")
            start_time = time.time()
            history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.2, verbose=0)
            training_time = time.time() - start_time
            
            y_pred = (model.predict(X_test) > 0.5).astype(int)
            loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
            
            print(f"\n--- Relatório de Classificação para {name} ---")
            print(classification_report(y_test, y_pred, target_names=['Normal', 'Ataque']))
            report = classification_report(y_test, y_pred, output_dict=True)
            
            self.results[name] = {
                'accuracy': accuracy, 'training_time': training_time,
                'history': history, 'predictions': y_pred,
                'model': model, 'report': report
            }
        
        return self.results
    
    def plot_results(self, output_filename_png):
        """Gera gráficos comparativos"""
        if not self.results:
            print("Execute a comparação antes de plotar os resultados.")
            return

        fig, axes = plt.subplots(2, 3, figsize=(20, 13))
        fig.suptitle('Comparativo de Arquiteturas para Detecção de Anomalias', fontsize=18, fontweight='bold')
        
        accuracies = [self.results[name]['accuracy'] for name in self.results.keys()]
        training_times = [self.results[name]['training_time'] for name in self.results.keys()]
        names = list(self.results.keys())
        colors = sns.color_palette("husl", len(names))

        axes[0, 0].bar(names, accuracies, color=colors)
        axes[0, 0].set_title('Acurácia Final', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Acurácia')
        axes[0, 0].set_ylim(min(accuracies) - 0.05, 1.0)
        for i, v in enumerate(accuracies):
            axes[0, 0].text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
        
        axes[0, 1].bar(names, training_times, color=colors)
        axes[0, 1].set_title('Tempo de Treinamento', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('Tempo (segundos)')
        for i, v in enumerate(training_times):
            axes[0, 1].text(i, v, f'{v:.1f}s', ha='center', va='bottom', fontweight='bold', color='white')

        seq_lengths_mem, memory_results = self.evaluate_memory_degradation()
        for arch, results in memory_results.items():
            axes[0, 2].plot(seq_lengths_mem, results, marker='o', linewidth=2, label=arch, markersize=8)
        axes[0, 2].set_title('Degradação de Memória', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Tamanho da Sequência'); axes[0, 2].set_ylabel('Acurácia'); axes[0, 2].legend(); axes[0, 2].grid(True, alpha=0.3)
        
        seq_lengths_comp, comp_times = self.compare_computational_complexity()
        for arch, times in comp_times.items():
            axes[1, 0].plot(seq_lengths_comp, times, marker='s', linewidth=2, label=arch, markersize=8)
        axes[1, 0].set_title('Complexidade Computacional', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Tamanho da Sequência'); axes[1, 0].set_ylabel('Tempo de Treinamento (s)'); axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3); axes[1, 0].set_yscale('log')

        for name, result in self.results.items():
            axes[1, 1].plot(result['history'].history['val_accuracy'], '--', linewidth=2, label=f'{name} - Validação')
        axes[1, 1].set_title('Curvas de Aprendizado (Validação)', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Épocas'); axes[1, 1].set_ylabel('Acurácia'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)

        f1_scores = [self.results[name]['report']['1']['f1-score'] for name in self.results.keys()]
        axes[1, 2].scatter(training_times, f1_scores, s=200, alpha=0.7, c=colors)
        for i, name in enumerate(names):
            axes[1, 2].annotate(name, (training_times[i], f1_scores[i]), xytext=(10, -15), textcoords='offset points', fontsize=12, fontweight='bold')
        axes[1, 2].set_title('Eficiência: F1-Score (Ataque) vs Tempo', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Tempo de Treinamento (s)'); axes[1, 2].set_ylabel('F1-Score para Classe "Ataque"'); axes[1, 2].grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_filename_png, dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("=== COMPARAÇÃO DE ARQUITETURAS NEURAIS ===")
    print("Comparação Detalhada de Arquiteturas Neurais para a detecção de anomalias\n")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    script_path = Path(__file__).resolve()
    output_dir = script_path.parent.parent / 'Results'
    
    # Garante que o diretório de resultados exista
    os.makedirs(output_dir, exist_ok=True)
    
    # Cria os nomes de arquivo completos
    png_filename = output_dir / f"Test_Comparacao_arquiteturas_{timestamp}.png"
    csv_filename = output_dir / f"Test_Resultados_arquiteturas_{timestamp}.csv"

    comparator = NetworkArchitectureComparison(sequence_length=100, n_features=23)
    results = comparator.run_full_comparison()
    comparator.plot_results(png_filename)
    
    df_results_list = []
    for name, result in results.items():
        report_ataque = result['report']['1']
        df_results_list.append({
            'Arquitetura': name,
            'Acurácia': result['accuracy'],
            'Tempo_Treinamento_s': result['training_time'],
            'Precisão_Ataque': report_ataque['precision'],
            'Recall_Ataque': report_ataque['recall'],
            'F1_Score_Ataque': report_ataque['f1-score']
        })
    df_results = pd.DataFrame(df_results_list)
    
    print("\n=== RESULTADOS COMPARATIVOS DETALHADOS ===")
    print(df_results.round(4))
    
    df_results.to_csv(csv_filename, index=False)
    print(f"\nResultados salvos em '{csv_filename}'")
    print(f"Gráficos salvos em '{png_filename}'")

if __name__ == "__main__":
    main()