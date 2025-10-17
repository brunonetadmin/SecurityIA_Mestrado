"""
Módulo centralizado para carregamento e pré-processamento da base CSE-CIC-IDS2018
Garante consistência no uso dos dados entre todos os scripts de análise
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class CSECICIDSDataLoader:
    """
    Classe responsável pelo carregamento e pré-processamento consistente
    da base de dados CSE-CIC-IDS2018 para todos os experimentos.
    """
    
    def __init__(self, base_path='../Base/CSE-CIC-IDS2018'):
        self.base_path = Path(base_path)
        self.feature_columns = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.attack_mapping = {
            'Benign': 'Normal',
            'Bot': 'Bot',
            'DDoS attack-HOIC': 'DDoS',
            'DDoS attack-LOIC-UDP': 'DDoS', 
            'DDoS attacks-LOIC-HTTP': 'DDoS',
            'DoS attacks-GoldenEye': 'DoS',
            'DoS attacks-Hulk': 'DoS',
            'DoS attacks-SlowHTTPTest': 'DoS',
            'DoS attacks-Slowloris': 'DoS',
            'FTP-BruteForce': 'Brute Force',
            'SSH-Bruteforce': 'Brute Force',
            'Infilteration': 'Infiltration',
            'SQL Injection': 'Web Attack',
            'Brute Force -Web': 'Web Attack',
            'Brute Force -XSS': 'Web Attack'
        }
        
    def load_data(self, sample_size=None, attack_types_filter=None):
        """
        Carrega os dados da base CSE-CIC-IDS2018.
        
        Parameters:
        -----------
        sample_size : int, optional
            Número de amostras a serem carregadas (para testes rápidos)
        attack_types_filter : list, optional
            Lista de tipos de ataque a serem incluídos
            
        Returns:
        --------
        X : np.ndarray
            Features normalizadas
        y : np.ndarray
            Labels codificados
        y_names : np.ndarray
            Nomes dos ataques (para análise)
        feature_names : list
            Lista com nomes das features
        """
        print("Carregando base CSE-CIC-IDS2018...")
        
        # Lista todos os arquivos CSV no diretório
        csv_files = list(self.base_path.glob("*.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"Nenhum arquivo CSV encontrado em {self.base_path}")
        
        # Carrega e concatena todos os arquivos
        df_list = []
        for file_path in csv_files[:3]:  # Limita a 3 arquivos para processamento inicial
            print(f"  Carregando {file_path.name}...")
            try:
                df_temp = pd.read_csv(file_path, low_memory=False)
                df_list.append(df_temp)
            except Exception as e:
                print(f"  Erro ao carregar {file_path.name}: {e}")
                continue
        
        if not df_list:
            raise ValueError("Não foi possível carregar nenhum arquivo CSV")
            
        df = pd.concat(df_list, ignore_index=True)
        print(f"Total de amostras carregadas: {len(df)}")
        
        # Remove colunas com valores infinitos ou NaN excessivos
        df = df.replace([np.inf, -np.inf], np.nan)
        null_counts = df.isnull().sum()
        cols_to_drop = null_counts[null_counts > len(df) * 0.5].index
        df = df.drop(columns=cols_to_drop)
        
        # Identifica coluna de label
        label_column = None
        for col in ['Label', 'label', 'Attack', 'attack']:
            if col in df.columns:
                label_column = col
                break
        
        if label_column is None:
            raise ValueError("Coluna de label não encontrada no dataset")
        
        # Mapeia tipos de ataque
        df[label_column] = df[label_column].map(lambda x: self.attack_mapping.get(x, x))
        
        # Filtra tipos de ataque se especificado
        if attack_types_filter:
            df = df[df[label_column].isin(attack_types_filter)]
        
        # Amostragem se especificado
        if sample_size and len(df) > sample_size:
            # Amostragem estratificada para manter proporções
            df = df.groupby(label_column).apply(
                lambda x: x.sample(n=min(len(x), int(sample_size * len(x) / len(df))), 
                                 random_state=42)
            ).reset_index(drop=True)
        
        # Separa features e labels
        y_names = df[label_column].values
        feature_columns = [col for col in df.columns if col != label_column]
        
        # Remove colunas não numéricas das features
        numeric_columns = df[feature_columns].select_dtypes(include=[np.number]).columns
        self.feature_columns = list(numeric_columns)
        
        X = df[self.feature_columns].values
        
        # Trata valores faltantes
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Codifica labels
        y = self.label_encoder.fit_transform(y_names)
        
        # Normaliza features
        X = self.scaler.fit_transform(X)
        
        print(f"Shape final: X={X.shape}, y={y.shape}")
        print(f"Distribuição de classes: {pd.Series(y_names).value_counts()}")
        
        return X, y, y_names, self.feature_columns
    
    def create_temporal_sequences(self, X, y, sequence_length=100, step=50):
        """
        Cria sequências temporais para modelos LSTM.
        
        Parameters:
        -----------
        X : np.ndarray
            Features originais
        y : np.ndarray
            Labels originais
        sequence_length : int
            Tamanho de cada sequência
        step : int
            Passo entre sequências
            
        Returns:
        --------
        X_sequences : np.ndarray
            Features em formato de sequências (n_sequences, sequence_length, n_features)
        y_sequences : np.ndarray
            Labels correspondentes
        """
        sequences = []
        labels = []
        
        for i in range(0, len(X) - sequence_length + 1, step):
            sequences.append(X[i:i + sequence_length])
            # Label da sequência é o label majoritário
            labels.append(np.bincount(y[i:i + sequence_length]).argmax())
        
        return np.array(sequences), np.array(labels)
    
    def get_binary_classification_data(self, X, y, y_names):
        """
        Converte para classificação binária (Normal vs Ataque).
        
        Parameters:
        -----------
        X : np.ndarray
            Features
        y : np.ndarray
            Labels multi-classe
        y_names : np.ndarray
            Nomes das classes
            
        Returns:
        --------
        X : np.ndarray
            Features (inalterado)
        y_binary : np.ndarray
            Labels binários (0=Normal, 1=Ataque)
        """
        y_binary = (y_names != 'Normal').astype(int)
        return X, y_binary
    
    def get_attack_severity_weights(self, y_names):
        """
        Retorna pesos baseados na severidade dos ataques para análise ponderada.
        
        Parameters:
        -----------
        y_names : np.ndarray
            Nomes dos ataques
            
        Returns:
        --------
        weights : np.ndarray
            Pesos de severidade
        """
        severity_mapping = {
            'Normal': 0.0,
            'Bot': 0.8,
            'DDoS': 0.9,
            'DoS': 0.7,
            'Brute Force': 0.6,
            'Infiltration': 0.9,
            'Web Attack': 0.7
        }
        
        weights = np.array([severity_mapping.get(attack, 0.5) for attack in y_names])
        return weights
    
    def get_feature_groups(self):
        """
        Retorna grupos de features para análise de importância.
        
        Returns:
        --------
        feature_groups : dict
            Dicionário com grupos de features
        """
        if self.feature_columns is None:
            raise ValueError("Dados devem ser carregados primeiro")
            
        feature_groups = {
            'flow_basic': [],
            'packet_length': [],
            'iat': [],
            'flags': [],
            'flow_bytes_packets': [],
            'subflow': [],
            'misc': []
        }
        
        for feature in self.feature_columns:
            feature_lower = feature.lower()
            if 'flow' in feature_lower and 'bytes' not in feature_lower and 'packet' not in feature_lower:
                feature_groups['flow_basic'].append(feature)
            elif 'length' in feature_lower or 'len' in feature_lower:
                feature_groups['packet_length'].append(feature)
            elif 'iat' in feature_lower:
                feature_groups['iat'].append(feature)
            elif 'flag' in feature_lower:
                feature_groups['flags'].append(feature)
            elif 'bytes' in feature_lower or 'packet' in feature_lower:
                feature_groups['flow_bytes_packets'].append(feature)
            elif 'subflow' in feature_lower:
                feature_groups['subflow'].append(feature)
            else:
                feature_groups['misc'].append(feature)
        
        return feature_groups

# Funções auxiliares para compatibilidade com scripts existentes
def load_cse_cic_ids2018(sample_size=None, binary=True, sequence_length=None):
    """
    Função de conveniência para carregamento rápido.
    
    Parameters:
    -----------
    sample_size : int, optional
        Número de amostras
    binary : bool
        Se True, retorna classificação binária
    sequence_length : int, optional
        Se fornecido, retorna sequências para LSTM
        
    Returns:
    --------
    X, y : dados processados
    """
    loader = CSECICIDSDataLoader()
    X, y, y_names, feature_names = loader.load_data(sample_size=sample_size)
    
    if binary:
        X, y = loader.get_binary_classification_data(X, y, y_names)
    
    if sequence_length:
        X, y = loader.create_temporal_sequences(X, y, sequence_length)
    
    return X, y
