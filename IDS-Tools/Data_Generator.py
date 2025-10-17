#!/usr/bin/env python3
"""
##############################################################################################
# Gerador de Dados de Teste para Sistema IDS
# Simula coleta de tráfego de rede com diferentes padrões de ataque
#
# Este script gera dados sintéticos de tráfego de rede seguindo o padrão
# do dataset CIC-IDS2018 para teste do Sistema IDS.
# 
# Autor: Bruno Cavalcante Barbosa
# UFAL - Universidade Federal de Alagoas
##############################################################################################
"""

import numpy as np
import pandas as pd
import argparse
import json
from datetime import datetime, timedelta
import os
import sys
from pathlib import Path

class RealisticNetworkTrafficSimulator:
    """
    Simulador de tráfego de rede baseado em padrões reais do CSE-CIC-IDS2018
    """
    
    def __init__(self, base_dataset_path='Base/'):
        self.base_dataset_path = base_dataset_path
        self.traffic_patterns = {}
        self.feature_distributions = {}
        self.correlation_matrices = {}
        self.feature_names = []
        self.attack_types = {}
        
        # Carrega e analisa padrões do dataset real
        self._analyze_real_patterns()
    
    def _analyze_real_patterns(self):
        """Analisa padrões reais do dataset CSE-CIC-IDS2018"""
        print("Analisando padrões reais do dataset...")
        
        try:
            # Carrega uma amostra do dataset para análise
            X, y, feature_names, label_mapping = load_cic_ids2018_dataset(
                self.base_dataset_path, 
                sample_fraction=0.1  # 10% para análise rápida
            )
            
            self.feature_names = feature_names
            self.attack_types = {str(v): k for k, v in label_mapping.items()}
            
            # Análise por tipo de ataque
            for attack_code, attack_name in self.attack_types.items():
                print(f"Analisando padrão: {attack_name}")
                
                # Filtra amostras do tipo de ataque
                mask = y == int(attack_code)
                X_attack = X[mask]
                
                if len(X_attack) > 0:
                    # Calcula distribuições estatísticas
                    self.feature_distributions[attack_name] = self._calculate_distributions(X_attack)
                    
                    # Calcula matriz de correlação
                    self.correlation_matrices[attack_name] = np.corrcoef(X_attack.T)
                    
                    # Extrai padrões característicos
                    self.traffic_patterns[attack_name] = self._extract_patterns(X_attack)
            
            print("✓ Análise de padrões concluída")
            
        except Exception as e:
            print(f"⚠️  Erro ao analisar dataset: {e}")
            print("Usando padrões padrão...")
            self._use_default_patterns()
    
    def _calculate_distributions(self, X):
        """Calcula distribuições estatísticas das features"""
        distributions = {}
        
        for i, feature in enumerate(self.feature_names[:X.shape[1]]):
            data = X[:, i]
            
            # Remove outliers extremos (além de 3 desvios padrão)
            mean = np.mean(data)
            std = np.std(data)
            mask = np.abs(data - mean) <= 3 * std
            data_clean = data[mask]
            
            # Testa diferentes distribuições
            distributions[feature] = {
                'mean': np.mean(data_clean),
                'std': np.std(data_clean),
                'median': np.median(data_clean),
                'min': np.min(data_clean),
                'max': np.max(data_clean),
                'q25': np.percentile(data_clean, 25),
                'q75': np.percentile(data_clean, 75),
                'skewness': self._calculate_skewness(data_clean),
                'kurtosis': self._calculate_kurtosis(data_clean)
            }
            
            # Determina melhor distribuição
            if abs(distributions[feature]['skewness']) < 0.5:
                distributions[feature]['distribution'] = 'normal'
            elif distributions[feature]['min'] >= 0 and distributions[feature]['skewness'] > 1:
                distributions[feature]['distribution'] = 'lognormal'
            else:
                distributions[feature]['distribution'] = 'gamma'
        
        return distributions
    
    def _calculate_skewness(self, data):
        """Calcula assimetria (skewness) dos dados"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calcula curtose (kurtosis) dos dados"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _extract_patterns(self, X):
        """Extrai padrões característicos do tráfego"""
        patterns = {
            'temporal_patterns': {},
            'packet_patterns': {},
            'flow_patterns': {},
            'flag_patterns': {}
        }
        
        # Identifica índices de features por categoria
        for i, feature in enumerate(self.feature_names[:X.shape[1]]):
            data = X[:, i]
            
            if 'duration' in feature or 'time' in feature or 'iat' in feature:
                patterns['temporal_patterns'][feature] = {
                    'typical_range': [np.percentile(data, 10), np.percentile(data, 90)],
                    'peak_value': np.percentile(data, 95)
                }
            elif 'packet' in feature:
                patterns['packet_patterns'][feature] = {
                    'typical_range': [np.percentile(data, 10), np.percentile(data, 90)],
                    'mode': self._calculate_mode(data)
                }
            elif 'flow' in feature or 'bytes' in feature:
                patterns['flow_patterns'][feature] = {
                    'typical_range': [np.percentile(data, 10), np.percentile(data, 90)],
                    'burst_threshold': np.percentile(data, 95)
                }
            elif 'flag' in feature:
                patterns['flag_patterns'][feature] = {
                    'probability': np.mean(data > 0),
                    'typical_count': np.mean(data[data > 0]) if np.any(data > 0) else 0
                }
        
        return patterns
    
    def _calculate_mode(self, data):
        """Calcula moda dos dados"""
        values, counts = np.unique(data, return_counts=True)
        return values[np.argmax(counts)]
    
    def generate_realistic_sample(self, attack_type='NORMAL'):
        """Gera amostra realística baseada em padrões reais"""
        if attack_type not in self.feature_distributions:
            print(f"⚠️  Tipo de ataque desconhecido: {attack_type}")
            attack_type = 'NORMAL'
        
        sample = np.zeros(len(self.feature_names))
        distributions = self.feature_distributions[attack_type]
        patterns = self.traffic_patterns[attack_type]
        
        # Gera valores baseados nas distribuições reais
        for i, feature in enumerate(self.feature_names):
            if feature in distributions:
                dist_info = distributions[feature]
                
                if dist_info['distribution'] == 'normal':
                    value = np.random.normal(dist_info['mean'], dist_info['std'])
                elif dist_info['distribution'] == 'lognormal':
                    # Converte parâmetros para lognormal
                    mean = dist_info['mean']
                    std = dist_info['std']
                    if mean > 0:
                        sigma = np.sqrt(np.log(1 + (std/mean)**2))
                        mu = np.log(mean) - sigma**2/2
                        value = np.random.lognormal(mu, sigma)
                    else:
                        value = 0
                else:  # gamma
                    if dist_info['std'] > 0:
                        shape = (dist_info['mean'] / dist_info['std']) ** 2
                        scale = dist_info['std'] ** 2 / dist_info['mean']
                        value = np.random.gamma(shape, scale)
                    else:
                        value = dist_info['mean']
                
                # Aplica limites realistas
                value = np.clip(value, dist_info['min'], dist_info['max'])
                sample[i] = value
        
        # Aplica correlações entre features
        if attack_type in self.correlation_matrices:
            sample = self._apply_correlations(sample, attack_type)
        
        # Garante consistência lógica
        sample = self._ensure_logical_consistency(sample)
        
        return sample
    
    def _apply_correlations(self, sample, attack_type):
        """Aplica correlações entre features"""
        corr_matrix = self.correlation_matrices[attack_type]
        
        # Implementação simplificada: ajusta features altamente correlacionadas
        for i in range(len(sample)):
            for j in range(i+1, len(sample)):
                if abs(corr_matrix[i, j]) > 0.7:  # Alta correlação
                    # Ajusta feature j baseado em i
                    adjustment = corr_matrix[i, j] * (sample[i] - self.feature_distributions[attack_type][self.feature_names[i]]['mean'])
                    sample[j] += adjustment * 0.5  # Aplicação parcial
        
        return sample
    
    def _ensure_logical_consistency(self, sample):
        """Garante consistência lógica entre features"""
        # Exemplos de regras de consistência
        for i, feature in enumerate(self.feature_names):
            # Valores não negativos
            if sample[i] < 0:
                sample[i] = 0
            
            # Regras específicas
            if 'packet' in feature and 'fwd' in feature:
                # Pacotes forward devem ser consistentes com total
                total_idx = next((j for j, f in enumerate(self.feature_names) if 'total' in f and 'packet' in f), None)
                if total_idx is not None and sample[i] > sample[total_idx]:
                    sample[i] = sample[total_idx] * 0.6
            
            if 'min' in feature:
                # Min deve ser menor que max
                max_feature = feature.replace('min', 'max')
                max_idx = next((j for j, f in enumerate(self.feature_names) if f == max_feature), None)
                if max_idx is not None and sample[i] > sample[max_idx]:
                    sample[i], sample[max_idx] = sample[max_idx], sample[i]
        
        return sample
    
    def generate_dataset_from_real_patterns(self, n_samples, attack_distribution=None):
        """Gera dataset completo baseado em padrões reais"""
        if attack_distribution is None:
            # Usa distribuição do dataset original
            attack_distribution = self._get_original_distribution()
        
        print(f"Gerando {n_samples} amostras baseadas em padrões reais...")
        
        samples = []
        labels = []
        
        for i in range(n_samples):
            if i % 1000 == 0:
                print(f"Progresso: {i}/{n_samples} ({i/n_samples*100:.1f}%)")
            
            # Escolhe tipo de ataque
            attack_type = np.random.choice(
                list(attack_distribution.keys()),
                p=list(attack_distribution.values())
            )
            
            # Gera amostra realística
            sample = self.generate_realistic_sample(attack_type)
            
            samples.append(sample)
            labels.append(attack_type)
        
        # Converte para DataFrame
        df = pd.DataFrame(samples, columns=self.feature_names)
        df['attack_type'] = labels
        df['label'] = df['attack_type'].map({v: k for k, v in self.attack_types.items()})
        
        print(f"✓ Dataset gerado com {len(df)} amostras")
        
        return df
    
    def _get_original_distribution(self):
        """Obtém distribuição original do dataset"""
        # Valores aproximados do CSE-CIC-IDS2018
        return {
            'NORMAL': 0.831,
            'DOS_HULK': 0.046,
            'PORTSCAN': 0.039,
            'DDOS': 0.035,
            'DOS_GOLDENEYE': 0.018,
            'BRUTE_FORCE_FTP': 0.011,
            'BRUTE_FORCE_SSH': 0.008,
            'DOS_SLOWLORIS': 0.004,
            'DOS_SLOWHTTPTEST': 0.003,
            'BOTNET': 0.002,
            'WEB_ATTACK': 0.002,
            'INFILTRATION': 0.001,
            'HEARTBLEED': 0.0001
        }
    
    def validate_generated_data(self, generated_df, original_stats=None):
        """Valida dados gerados comparando com estatísticas originais"""
        print("\nValidando dados gerados...")
        
        validation_results = {
            'statistical_tests': {},
            'distribution_comparison': {},
            'logical_consistency': {}
        }
        
        # Testes estatísticos
        for feature in self.feature_names[:10]:  # Primeiras 10 features
            if feature in generated_df.columns:
                generated_data = generated_df[feature].values
                
                # Teste de normalidade
                from scipy import stats
                _, p_value = stats.normaltest(generated_data)
                validation_results['statistical_tests'][feature] = {
                    'normality_p_value': p_value,
                    'is_normal': p_value > 0.05
                }
        
        # Comparação de distribuições
        for attack_type in generated_df['attack_type'].unique():
            mask = generated_df['attack_type'] == attack_type
            subset = generated_df[mask]
            
            validation_results['distribution_comparison'][attack_type] = {
                'sample_count': len(subset),
                'percentage': len(subset) / len(generated_df) * 100
            }
        
        # Verificação de consistência lógica
        consistency_checks = {
            'non_negative_values': np.all(generated_df.select_dtypes(include=[np.number]) >= 0),
            'finite_values': np.all(np.isfinite(generated_df.select_dtypes(include=[np.number]))),
            'no_missing_values': generated_df.isnull().sum().sum() == 0
        }
        
        validation_results['logical_consistency'] = consistency_checks
        
        # Relatório de validação
        print("\n📊 Resultados da Validação:")
        print(f"✓ Valores não negativos: {consistency_checks['non_negative_values']}")
        print(f"✓ Valores finitos: {consistency_checks['finite_values']}")
        print(f"✓ Sem valores faltantes: {consistency_checks['no_missing_values']}")
        
        print("\nDistribuição de ataques:")
        for attack, info in validation_results['distribution_comparison'].items():
            print(f"  {attack}: {info['sample_count']} ({info['percentage']:.2f}%)")
        
        return validation_results
    
    
    """Simulador de tráfego de rede para geração de dados de teste"""
    
    def __init__(self):
        self.feature_names = [
            'flow_duration', 'total_fwd_packets', 'total_bwd_packets',
            'total_len_fwd_packets', 'total_len_bwd_packets',
            'fwd_packet_len_max', 'fwd_packet_len_min', 'fwd_packet_len_mean',
            'fwd_packet_len_std', 'bwd_packet_len_max', 'bwd_packet_len_min',
            'bwd_packet_len_mean', 'bwd_packet_len_std', 'flow_bytes_s',
            'flow_packets_s', 'flow_iat_mean', 'flow_iat_std', 'flow_iat_max',
            'flow_iat_min', 'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_std',
            'fwd_iat_max', 'fwd_iat_min', 'bwd_iat_total', 'bwd_iat_mean',
            'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min', 'fwd_psh_flags',
            'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags', 'fwd_header_len',
            'bwd_header_len', 'fwd_packets_s', 'bwd_packets_s', 'packet_len_min',
            'packet_len_max', 'packet_len_mean', 'packet_len_std', 'packet_len_var',
            'fin_flag_count', 'syn_flag_count', 'rst_flag_count', 'psh_flag_count',
            'ack_flag_count', 'urg_flag_count', 'cwe_flag_count', 'ece_flag_count',
            'down_up_ratio', 'avg_packet_size', 'avg_fwd_segment_size',
            'avg_bwd_segment_size', 'fwd_header_len_min', 'fwd_avg_bytes_bulk',
            'fwd_avg_packets_bulk', 'fwd_avg_bulk_rate', 'bwd_avg_bytes_bulk',
            'bwd_avg_packets_bulk', 'bwd_avg_bulk_rate', 'subflow_fwd_packets',
            'subflow_fwd_bytes', 'subflow_bwd_packets', 'subflow_bwd_bytes',
            'init_win_bytes_fwd', 'init_win_bytes_bwd', 'act_data_pkt_fwd',
            'min_seg_size_fwd', 'active_mean', 'active_std', 'active_max',
            'active_min', 'idle_mean', 'idle_std', 'idle_max', 'idle_min'
        ]
        
        self.attack_types = {
            'NORMAL': 0,
            'DOS_HULK': 1,
            'DOS_GOLDENEYE': 2,
            'DOS_SLOWLORIS': 3,
            'DOS_SLOWHTTPTEST': 4,
            'DDOS': 5,
            'PORTSCAN': 6,
            'BOTNET': 7,
            'INFILTRATION': 8,
            'BRUTE_FORCE_FTP': 9,
            'BRUTE_FORCE_SSH': 10,
            'WEB_ATTACK': 11,
            'HEARTBLEED': 12
        }
        
        self.traffic_patterns = self._define_traffic_patterns()
    
    def _define_traffic_patterns(self):
        """Define padrões de tráfego para cada tipo de ataque"""
        patterns = {
            'NORMAL': {
                'flow_duration': {'dist': 'normal', 'params': [45000, 15000]},
                'total_fwd_packets': {'dist': 'poisson', 'params': [8]},
                'total_bwd_packets': {'dist': 'poisson', 'params': [5]},
                'flow_bytes_s': {'dist': 'normal', 'params': [1200, 400]},
                'flow_packets_s': {'dist': 'normal', 'params': [0.15, 0.05]},
                'packet_len_mean': {'dist': 'normal', 'params': [420, 80]},
                'syn_flag_count': {'dist': 'poisson', 'params': [1]},
                'ack_flag_count': {'dist': 'poisson', 'params': [6]},
                'description': 'Tráfego normal de navegação web e aplicações'
            },
            
            'DOS_HULK': {
                'flow_duration': {'dist': 'normal', 'params': [1000, 300]},
                'total_fwd_packets': {'dist': 'poisson', 'params': [100]},
                'total_bwd_packets': {'dist': 'poisson', 'params': [2]},
                'flow_bytes_s': {'dist': 'normal', 'params': [50000, 10000]},
                'flow_packets_s': {'dist': 'normal', 'params': [100, 25]},
                'packet_len_mean': {'dist': 'normal', 'params': [500, 50]},
                'syn_flag_count': {'dist': 'poisson', 'params': [50]},
                'ack_flag_count': {'dist': 'poisson', 'params': [2]},
                'description': 'Ataque DoS com volume alto de requisições'
            },
            
            'DOS_GOLDENEYE': {
                'flow_duration': {'dist': 'normal', 'params': [800, 200]},
                'total_fwd_packets': {'dist': 'poisson', 'params': [80]},
                'total_bwd_packets': {'dist': 'poisson', 'params': [1]},
                'flow_bytes_s': {'dist': 'normal', 'params': [40000, 8000]},
                'flow_packets_s': {'dist': 'normal', 'params': [80, 20]},
                'packet_len_mean': {'dist': 'normal', 'params': [500, 50]},
                'syn_flag_count': {'dist': 'poisson', 'params': [40]},
                'ack_flag_count': {'dist': 'poisson', 'params': [1]},
                'description': 'Ataque DoS GoldenEye com requisições HTTP'
            },
            
            'DOS_SLOWLORIS': {
                'flow_duration': {'dist': 'normal', 'params': [60000, 15000]},
                'total_fwd_packets': {'dist': 'poisson', 'params': [20]},
                'total_bwd_packets': {'dist': 'poisson', 'params': [1]},
                'flow_bytes_s': {'dist': 'normal', 'params': [10, 5]},
                'flow_packets_s': {'dist': 'normal', 'params': [0.001, 0.0005]},
                'packet_len_mean': {'dist': 'normal', 'params': [100, 20]},
                'syn_flag_count': {'dist': 'poisson', 'params': [10]},
                'ack_flag_count': {'dist': 'poisson', 'params': [1]},
                'description': 'Ataque Slowloris com conexões lentas'
            },
            
            'PORTSCAN': {
                'flow_duration': {'dist': 'normal', 'params': [100, 50]},
                'total_fwd_packets': {'dist': 'poisson', 'params': [2]},
                'total_bwd_packets': {'dist': 'poisson', 'params': [0]},
                'flow_bytes_s': {'dist': 'normal', 'params': [1000, 300]},
                'flow_packets_s': {'dist': 'normal', 'params': [20, 5]},
                'packet_len_mean': {'dist': 'normal', 'params': [60, 10]},
                'syn_flag_count': {'dist': 'poisson', 'params': [2]},
                'ack_flag_count': {'dist': 'poisson', 'params': [0]},
                'description': 'Varredura de portas - múltiplas tentativas de conexão'
            },
            
            'BRUTE_FORCE_SSH': {
                'flow_duration': {'dist': 'normal', 'params': [5000, 1500]},
                'total_fwd_packets': {'dist': 'poisson', 'params': [15]},
                'total_bwd_packets': {'dist': 'poisson', 'params': [8]},
                'flow_bytes_s': {'dist': 'normal', 'params': [1500, 400]},
                'flow_packets_s': {'dist': 'normal', 'params': [3, 1]},
                'packet_len_mean': {'dist': 'normal', 'params': [120, 25]},
                'syn_flag_count': {'dist': 'poisson', 'params': [5]},
                'ack_flag_count': {'dist': 'poisson', 'params': [10]},
                'description': 'Tentativas de força bruta em SSH'
            },
            
            'BRUTE_FORCE_FTP': {
                'flow_duration': {'dist': 'normal', 'params': [3000, 800]},
                'total_fwd_packets': {'dist': 'poisson', 'params': [12]},
                'total_bwd_packets': {'dist': 'poisson', 'params': [6]},
                'flow_bytes_s': {'dist': 'normal', 'params': [1200, 300]},
                'flow_packets_s': {'dist': 'normal', 'params': [4, 1]},
                'packet_len_mean': {'dist': 'normal', 'params': [100, 20]},
                'syn_flag_count': {'dist': 'poisson', 'params': [4]},
                'ack_flag_count': {'dist': 'poisson', 'params': [8]},
                'description': 'Tentativas de força bruta em FTP'
            },
            
            'WEB_ATTACK': {
                'flow_duration': {'dist': 'normal', 'params': [2000, 600]},
                'total_fwd_packets': {'dist': 'poisson', 'params': [25]},
                'total_bwd_packets': {'dist': 'poisson', 'params': [15]},
                'flow_bytes_s': {'dist': 'normal', 'params': [5000, 1200]},
                'flow_packets_s': {'dist': 'normal', 'params': [10, 3]},
                'packet_len_mean': {'dist': 'normal', 'params': [200, 50]},
                'syn_flag_count': {'dist': 'poisson', 'params': [8]},
                'ack_flag_count': {'dist': 'poisson', 'params': [20]},
                'description': 'Ataques web - SQL injection, XSS, etc.'
            },
            
            'BOTNET': {
                'flow_duration': {'dist': 'normal', 'params': [30000, 8000]},
                'total_fwd_packets': {'dist': 'poisson', 'params': [50]},
                'total_bwd_packets': {'dist': 'poisson', 'params': [30]},
                'flow_bytes_s': {'dist': 'normal', 'params': [800, 200]},
                'flow_packets_s': {'dist': 'normal', 'params': [2, 0.5]},
                'packet_len_mean': {'dist': 'normal', 'params': [300, 60]},
                'syn_flag_count': {'dist': 'poisson', 'params': [10]},
                'ack_flag_count': {'dist': 'poisson', 'params': [40]},
                'description': 'Comunicação de botnet - C&C e exfiltração'
            },
            
            'INFILTRATION': {
                'flow_duration': {'dist': 'normal', 'params': [120000, 30000]},
                'total_fwd_packets': {'dist': 'poisson', 'params': [200]},
                'total_bwd_packets': {'dist': 'poisson', 'params': [150]},
                'flow_bytes_s': {'dist': 'normal', 'params': [600, 150]},
                'flow_packets_s': {'dist': 'normal', 'params': [1.5, 0.3]},
                'packet_len_mean': {'dist': 'normal', 'params': [250, 50]},
                'syn_flag_count': {'dist': 'poisson', 'params': [20]},
                'ack_flag_count': {'dist': 'poisson', 'params': [180]},
                'description': 'Tentativas de infiltração e movimento lateral'
            },
            
            'DDOS': {
                'flow_duration': {'dist': 'normal', 'params': [500, 150]},
                'total_fwd_packets': {'dist': 'poisson', 'params': [150]},
                'total_bwd_packets': {'dist': 'poisson', 'params': [1]},
                'flow_bytes_s': {'dist': 'normal', 'params': [80000, 20000]},
                'flow_packets_s': {'dist': 'normal', 'params': [200, 50]},
                'packet_len_mean': {'dist': 'normal', 'params': [600, 100]},
                'syn_flag_count': {'dist': 'poisson', 'params': [75]},
                'ack_flag_count': {'dist': 'poisson', 'params': [1]},
                'description': 'Ataque DDoS distribuído'
            },
            
            'HEARTBLEED': {
                'flow_duration': {'dist': 'normal', 'params': [1500, 400]},
                'total_fwd_packets': {'dist': 'poisson', 'params': [10]},
                'total_bwd_packets': {'dist': 'poisson', 'params': [5]},
                'flow_bytes_s': {'dist': 'normal', 'params': [2000, 500]},
                'flow_packets_s': {'dist': 'normal', 'params': [5, 1]},
                'packet_len_mean': {'dist': 'normal', 'params': [200, 40]},
                'syn_flag_count': {'dist': 'poisson', 'params': [3]},
                'ack_flag_count': {'dist': 'poisson', 'params': [7]},
                'description': 'Exploração da vulnerabilidade Heartbleed'
            }
        }
        
        return patterns
    
    def _generate_value(self, dist_info):
        """Gera valor baseado na distribuição especificada"""
        if dist_info['dist'] == 'normal':
            mean, std = dist_info['params']
            return max(0, np.random.normal(mean, std))
        elif dist_info['dist'] == 'poisson':
            lam = dist_info['params'][0]
            return max(0, np.random.poisson(lam))
        elif dist_info['dist'] == 'uniform':
            low, high = dist_info['params']
            return np.random.uniform(low, high)
        else:
            return 0
    
    def generate_traffic_sample(self, attack_type='NORMAL', noise_level=0.1):
        """Gera uma amostra de tráfego baseada no tipo de ataque"""
        if attack_type not in self.traffic_patterns:
            print(f"Tipo de ataque desconhecido: {attack_type}. Usando NORMAL.")
            attack_type = 'NORMAL'
        
        pattern = self.traffic_patterns[attack_type]
        sample = {}
        
        # Gera características principais baseadas no padrão
        for feature, dist_info in pattern.items():
            if feature != 'description':
                sample[feature] = self._generate_value(dist_info)
        
        # Gera características derivadas e complementares
        sample.update(self._generate_derived_features(sample, attack_type))
        
        # Completa características restantes
        for feature in self.feature_names:
            if feature not in sample:
                sample[feature] = self._generate_default_feature(feature, attack_type)
        
        # Adiciona ruído realístico
        if noise_level > 0:
            sample = self._add_noise(sample, noise_level)
        
        # Garante valores não negativos
        for key in sample:
            if isinstance(sample[key], (int, float)):
                sample[key] = max(0, sample[key])
        
        return sample
    
    def _generate_derived_features(self, sample, attack_type):
        """Gera características derivadas baseadas nas principais"""
        derived = {}
        
        # Características de comprimento de pacotes
        if 'packet_len_mean' in sample:
            derived['total_len_fwd_packets'] = sample.get('total_fwd_packets', 1) * sample['packet_len_mean']
            derived['total_len_bwd_packets'] = sample.get('total_bwd_packets', 1) * sample['packet_len_mean'] * 0.7
            derived['fwd_packet_len_mean'] = sample['packet_len_mean']
            derived['bwd_packet_len_mean'] = sample['packet_len_mean'] * 0.7
            derived['fwd_packet_len_max'] = sample['packet_len_mean'] * 2
            derived['fwd_packet_len_min'] = sample['packet_len_mean'] * 0.3
            derived['bwd_packet_len_max'] = sample['packet_len_mean'] * 1.5
            derived['bwd_packet_len_min'] = sample['packet_len_mean'] * 0.2
            derived['packet_len_std'] = sample['packet_len_mean'] * 0.3
            derived['packet_len_var'] = derived['packet_len_std'] ** 2
            derived['packet_len_max'] = derived['fwd_packet_len_max']
            derived['packet_len_min'] = derived['bwd_packet_len_min']
        
        # Características de Inter-Arrival Time
        if 'flow_duration' in sample and 'total_fwd_packets' in sample:
            total_packets = sample['total_fwd_packets'] + sample.get('total_bwd_packets', 0)
            if total_packets > 1:
                derived['flow_iat_mean'] = sample['flow_duration'] / total_packets
                derived['flow_iat_std'] = derived['flow_iat_mean'] * 0.5
                derived['flow_iat_max'] = derived['flow_iat_mean'] * 5
                derived['flow_iat_min'] = derived['flow_iat_mean'] * 0.1
                derived['fwd_iat_mean'] = derived['flow_iat_mean']
                derived['fwd_iat_std'] = derived['flow_iat_std']
                derived['fwd_iat_max'] = derived['flow_iat_max']
                derived['fwd_iat_min'] = derived['flow_iat_min']
                derived['fwd_iat_total'] = sample['flow_duration'] * 0.6
                derived['bwd_iat_mean'] = derived['flow_iat_mean'] * 1.2
                derived['bwd_iat_std'] = derived['flow_iat_std'] * 1.1
                derived['bwd_iat_max'] = derived['flow_iat_max'] * 1.3
                derived['bwd_iat_min'] = derived['flow_iat_min'] * 0.8
                derived['bwd_iat_total'] = sample['flow_duration'] * 0.4
        
        # Características de header e flags
        derived['fwd_header_len'] = sample.get('total_fwd_packets', 1) * 20
        derived['bwd_header_len'] = sample.get('total_bwd_packets', 1) * 20
        derived['fwd_header_len_min'] = 20
        
        # Características de flags TCP
        if 'syn_flag_count' not in sample:
            derived['syn_flag_count'] = 1 if attack_type == 'NORMAL' else np.random.poisson(2)
        if 'ack_flag_count' not in sample:
            derived['ack_flag_count'] = sample.get('total_fwd_packets', 1) + sample.get('total_bwd_packets', 0)
        
        derived['fin_flag_count'] = 1 if attack_type == 'NORMAL' else np.random.poisson(0.5)
        derived['rst_flag_count'] = 0 if attack_type == 'NORMAL' else np.random.poisson(0.2)
        derived['psh_flag_count'] = np.random.poisson(2)
        derived['urg_flag_count'] = np.random.poisson(0.1)
        derived['cwe_flag_count'] = np.random.poisson(0.05)
        derived['ece_flag_count'] = np.random.poisson(0.05)
        derived['fwd_psh_flags'] = np.random.poisson(1)
        derived['bwd_psh_flags'] = np.random.poisson(1)
        derived['fwd_urg_flags'] = np.random.poisson(0.1)
        derived['bwd_urg_flags'] = np.random.poisson(0.1)
        
        # Características de tamanho e segmento
        if 'packet_len_mean' in sample:
            derived['avg_packet_size'] = sample['packet_len_mean']
            derived['avg_fwd_segment_size'] = sample['packet_len_mean']
            derived['avg_bwd_segment_size'] = sample['packet_len_mean'] * 0.7
        
        # Características de proporção
        fwd_bytes = derived.get('total_len_fwd_packets', 1)
        bwd_bytes = derived.get('total_len_bwd_packets', 1)
        derived['down_up_ratio'] = bwd_bytes / fwd_bytes if fwd_bytes > 0 else 0
        
        # Características de janela TCP
        derived['init_win_bytes_fwd'] = np.random.normal(8192, 2000)
        derived['init_win_bytes_bwd'] = np.random.normal(8192, 2000)
        
        # Características de bulk
        derived['fwd_avg_bytes_bulk'] = 0
        derived['fwd_avg_packets_bulk'] = 0
        derived['fwd_avg_bulk_rate'] = 0
        derived['bwd_avg_bytes_bulk'] = 0
        derived['bwd_avg_packets_bulk'] = 0
        derived['bwd_avg_bulk_rate'] = 0
        
        # Características de subflow
        derived['subflow_fwd_packets'] = sample.get('total_fwd_packets', 1)
        derived['subflow_fwd_bytes'] = derived.get('total_len_fwd_packets', 1)
        derived['subflow_bwd_packets'] = sample.get('total_bwd_packets', 1)
        derived['subflow_bwd_bytes'] = derived.get('total_len_bwd_packets', 1)
        
        # Características de atividade
        derived['act_data_pkt_fwd'] = max(0, sample.get('total_fwd_packets', 1) - 1)
        derived['min_seg_size_fwd'] = 20
        
        # Características de tempo ativo/inativo
        derived['active_mean'] = sample.get('flow_duration', 1000) * 0.3
        derived['active_std'] = derived['active_mean'] * 0.5
        derived['active_max'] = derived['active_mean'] * 2
        derived['active_min'] = derived['active_mean'] * 0.1
        derived['idle_mean'] = sample.get('flow_duration', 1000) * 0.7
        derived['idle_std'] = derived['idle_mean'] * 0.3
        derived['idle_max'] = derived['idle_mean'] * 1.5
        derived['idle_min'] = derived['idle_mean'] * 0.2
        
        return derived
    
    def _generate_default_feature(self, feature, attack_type):
        """Gera valor padrão para características não especificadas"""
        if 'flag' in feature or 'count' in feature:
            return max(0, np.random.poisson(0.5))
        elif 'ratio' in feature:
            return np.random.uniform(0.1, 10)
        elif 'min' in feature:
            return max(0, np.random.normal(20, 10))
        elif 'max' in feature:
            return max(0, np.random.normal(2000, 500))
        elif 'mean' in feature:
            return max(0, np.random.normal(500, 150))
        elif 'std' in feature:
            return max(0, np.random.normal(200, 50))
        elif 'bytes' in feature:
            return max(0, np.random.normal(1000, 300))
        elif 'packets' in feature:
            return max(0, np.random.poisson(5))
        elif 'len' in feature:
            return max(0, np.random.normal(20, 5))
        elif 'duration' in feature:
            return max(0, np.random.normal(10000, 3000))
        elif 'time' in feature or 'iat' in feature:
            return max(0, np.random.normal(1000, 300))
        else:
            return max(0, np.random.normal(0, 100))
    
    def _add_noise(self, sample, noise_level):
        """Adiciona ruído realístico aos dados"""
        noisy_sample = sample.copy()
        
        for key, value in sample.items():
            if isinstance(value, (int, float)) and value > 0:
                # Adiciona ruído gaussiano proporcional ao valor
                noise = np.random.normal(0, value * noise_level)
                noisy_sample[key] = max(0, value + noise)
        
        return noisy_sample
    
    def generate_dataset(self, n_samples, attack_distribution=None, noise_level=0.1):
        """Gera conjunto de dados completo"""
        if attack_distribution is None:
            # Distribuição padrão baseada em redes reais
            attack_distribution = {
                'NORMAL': 0.75,
                'DOS_HULK': 0.05,
                'DOS_GOLDENEYE': 0.03,
                'DOS_SLOWLORIS': 0.02,
                'PORTSCAN': 0.04,
                'BRUTE_FORCE_SSH': 0.03,
                'BRUTE_FORCE_FTP': 0.02,
                'WEB_ATTACK': 0.02,
                'BOTNET': 0.01,
                'INFILTRATION': 0.01,
                'DDOS': 0.01,
                'HEARTBLEED': 0.01
            }
        
        print(f"Gerando {n_samples} amostras de tráfego de rede...")
        print("Distribuição de ataques:")
        for attack, prob in attack_distribution.items():
            count = int(n_samples * prob)
            print(f"  {attack}: {count} amostras ({prob*100:.1f}%)")
        
        samples = []
        labels = []
        timestamps = []
        
        base_time = datetime.now()
        
        for i in range(n_samples):
            # Progresso
            if i % 1000 == 0:
                print(f"Gerado: {i}/{n_samples} amostras ({i/n_samples*100:.1f}%)")
            
            # Escolhe tipo de ataque
            attack_type = np.random.choice(
                list(attack_distribution.keys()),
                p=list(attack_distribution.values())
            )
            
            # Gera amostra
            sample = self.generate_traffic_sample(attack_type, noise_level)
            
            # Adiciona metadados
            sample['attack_type'] = attack_type
            sample['label'] = self.attack_types[attack_type]
            sample['timestamp'] = (base_time + timedelta(seconds=i)).isoformat()
            sample['sample_id'] = f"SAMPLE_{i+1:06d}"
            
            samples.append(sample)
            labels.append(attack_type)
            timestamps.append(sample['timestamp'])
        
        print(f"Geração concluída: {len(samples)} amostras")
        
        # Converte para DataFrame
        df = pd.DataFrame(samples)
        
        # Reordena colunas
        metadata_cols = ['sample_id', 'timestamp', 'attack_type', 'label']
        feature_cols = [col for col in df.columns if col not in metadata_cols]
        df = df[metadata_cols + sorted(feature_cols)]
        
        return df
    
    def save_dataset(self, df, filename, include_metadata=True):
        """Salva dataset em arquivo CSV"""
        print(f"Salvando dataset em: {filename}")
        
        if not include_metadata:
            # Remove colunas de metadados para uso no IDS
            metadata_cols = ['sample_id', 'timestamp', 'attack_type', 'label']
            feature_cols = [col for col in df.columns if col not in metadata_cols]
            df_clean = df[feature_cols]
            df_clean.to_csv(filename, index=False, float_format='%.6f')
        else:
            df.to_csv(filename, index=False, float_format='%.6f')
        
        print(f"Dataset salvo com {len(df)} amostras")
        return filename
    
    def generate_dataset_info(self, df, filename):
        """Gera arquivo de informações sobre o dataset"""
        info = {
            'generation_info': {
                'timestamp': datetime.now().isoformat(),
                'total_samples': len(df),
                'total_features': len([col for col in df.columns if col not in ['sample_id', 'timestamp', 'attack_type', 'label']])
            },
            'attack_distribution': df['attack_type'].value_counts().to_dict(),
            'feature_statistics': {},
            'attack_descriptions': {}
        }
        
        # Estatísticas das características
        feature_cols = [col for col in df.columns if col not in ['sample_id', 'timestamp', 'attack_type', 'label']]
        for col in feature_cols[:10]:  # Primeiras 10 características
            if df[col].dtype in ['float64', 'int64']:
                info['feature_statistics'][col] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
        
        # Descrições dos ataques
        for attack_type in self.traffic_patterns:
            info['attack_descriptions'][attack_type] = self.traffic_patterns[attack_type].get('description', 'Sem descrição')
        
        # Salva informações
        info_filename = filename.replace('.csv', '_info.json')
        with open(info_filename, 'w') as f:
            json.dump(info, f, indent=4)
        
        print(f"Informações do dataset salvas em: {info_filename}")
        return info_filename
    
    def print_dataset_summary(self, df):
        """Imprime resumo do dataset gerado"""
        print("\n" + "="*60)
        print("RESUMO DO DATASET GERADO")
        print("="*60)
        
        print(f"Total de amostras: {len(df)}")
        print(f"Total de características: {len([col for col in df.columns if col not in ['sample_id', 'timestamp', 'attack_type', 'label']])}")
        
        print(f"\nDistribuição de ataques:")
        attack_counts = df['attack_type'].value_counts()
        for attack, count in attack_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {attack}: {count} amostras ({percentage:.1f}%)")
        
        print(f"\nEstatísticas de características principais:")
        main_features = ['flow_duration', 'total_fwd_packets', 'total_bwd_packets', 'flow_bytes_s', 'flow_packets_s']
        for feature in main_features:
            if feature in df.columns:
                print(f"  {feature}:")
                print(f"    Média: {df[feature].mean():.2f}")
                print(f"    Desvio: {df[feature].std():.2f}")
                print(f"    Min: {df[feature].min():.2f}")
                print(f"    Max: {df[feature].max():.2f}")
        
        print(f"\nArquivos gerados:")
        print(f"  - {filename} (dataset principal)")
        print(f"  - {filename.replace('.csv', '_info.json')} (informações)")
        print(f"  - {filename.replace('.csv', '_clean.csv')} (sem metadados)")

def main():
    """Função principal do gerador de dados"""
    parser = argparse.ArgumentParser(description='Gerador de Dados de Teste para Sistema IDS')
    parser.add_argument('--samples', '-n', type=int, default=1000,
                       help='Número de amostras a gerar (padrão: 1000)')
    parser.add_argument('--output', '-o', type=str, default='network_traffic_test.csv',
                       help='Nome do arquivo de saída (padrão: network_traffic_test.csv)')
    parser.add_argument('--noise', type=float, default=0.1,
                       help='Nível de ruído (0.0 a 1.0, padrão: 0.1)')
    parser.add_argument('--normal-ratio', type=float, default=0.75,
                       help='Proporção de tráfego normal (padrão: 0.75)')
    parser.add_argument('--attack-types', nargs='+', 
                       default=['DOS_HULK', 'PORTSCAN', 'BRUTE_FORCE_SSH', 'WEB_ATTACK'],
                       help='Tipos de ataque a incluir')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Modo interativo para configuração')
    
    args = parser.parse_args()
    
    # Modo interativo
    if args.interactive:
        print("="*60)
        print("GERADOR DE DADOS DE TESTE - SISTEMA IDS")
        print("="*60)
        
        # Solicita número de amostras
        try:
            n_samples = int(input(f"Número de amostras [{args.samples}]: ") or args.samples)
        except ValueError:
            n_samples = args.samples
        
        # Solicita nome do arquivo
        filename = input(f"Nome do arquivo [{args.output}]: ") or args.output
        
        # Solicita nível de ruído
        try:
            noise_level = float(input(f"Nível de ruído (0.0-1.0) [{args.noise}]: ") or args.noise)
        except ValueError:
            noise_level = args.noise
        
        # Solicita proporção de tráfego normal
        try:
            normal_ratio = float(input(f"Proporção de tráfego normal (0.0-1.0) [{args.normal_ratio}]: ") or args.normal_ratio)
        except ValueError:
            normal_ratio = args.normal_ratio
        
        print("\nTipos de ataque disponíveis:")
        simulator = NetworkTrafficSimulator()
        for i, attack_type in enumerate(simulator.attack_types.keys()):
            if attack_type != 'NORMAL':
                print(f"  {i+1}. {attack_type}")
        
        attack_choice = input("Escolha tipos de ataque (números separados por vírgula) ou Enter para usar padrão: ")
        if attack_choice:
            try:
                attack_indices = [int(x.strip()) for x in attack_choice.split(',')]
                attack_list = list(simulator.attack_types.keys())
                selected_attacks = [attack_list[i] for i in attack_indices if 0 < i < len(attack_list)]
                args.attack_types = selected_attacks
            except:
                print("Erro na seleção. Usando tipos padrão.")
        
        args.samples = n_samples
        args.output = filename
        args.noise = noise_level
        args.normal_ratio = normal_ratio
    
    # Cria simulador
    simulator = NetworkTrafficSimulator()
    
    # Configura distribuição de ataques
    remaining_prob = 1.0 - args.normal_ratio
    attack_prob = remaining_prob / len(args.attack_types)
    
    attack_distribution = {'NORMAL': args.normal_ratio}
    for attack_type in args.attack_types:
        if attack_type in simulator.attack_types:
            attack_distribution[attack_type] = attack_prob
    
    # Normaliza probabilidades
    total_prob = sum(attack_distribution.values())
    attack_distribution = {k: v/total_prob for k, v in attack_distribution.items()}
    
    print("="*60)
    print("INICIANDO GERAÇÃO DE DADOS")
    print("="*60)
    print(f"Amostras: {args.samples}")
    print(f"Arquivo: {args.output}")
    print(f"Ruído: {args.noise}")
    print(f"Distribuição configurada:")
    for attack, prob in attack_distribution.items():
        print(f"  {attack}: {prob:.2%}")
    print()
    
    # Gera dataset
    df = simulator.generate_dataset(
        n_samples=args.samples,
        attack_distribution=attack_distribution,
        noise_level=args.noise
    )
    
    # Salva arquivos
    filename = args.output
    simulator.save_dataset(df, filename, include_metadata=True)
    
    # Salva versão limpa (sem metadados) para uso no IDS
    clean_filename = filename.replace('.csv', '_clean.csv')
    simulator.save_dataset(df, clean_filename, include_metadata=False)
    
    # Gera informações
    info_filename = simulator.generate_dataset_info(df, filename)
    
    # Imprime resumo
    simulator.print_dataset_summary(df)
    
    print("\n" + "="*60)
    print("GERAÇÃO CONCLUÍDA COM SUCESSO!")
    print("="*60)
    print(f"Arquivos gerados:")
    print(f"  📄 {filename} - Dataset completo com metadados")
    print(f"  📄 {clean_filename} - Dataset limpo para IDS")
    print(f"  📄 {info_filename} - Informações do dataset")
    print(f"\nO arquivo {clean_filename} está pronto para uso no Sistema IDS!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nGeração interrompida pelo usuário.")
    except Exception as e:
        print(f"\nErro durante geração: {e}")
        import traceback
        traceback.print_exc()
