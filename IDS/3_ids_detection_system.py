#!/usr/bin/env python3
"""
##############################################################################################
# Sistema IDS de Detecção de Intrusão
# Baseado em Arquitetura LSTM Híbrida Treinada
# 
# Este script implementa o sistema IDS que utiliza o modelo treinado
# para análise de tráfego de rede em tempo real, identificando
# possíveis ataques e anomalias.
# 
# Autor: Bruno Cavalcante Barbosa
# UFAL - Universidade Federal de Alagoas
##############################################################################################
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
import joblib
import json
import argparse
from datetime import datetime
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class IDSDetectionSystem:
    """Sistema IDS para detecção de intrusão"""
    
    def __init__(self, model_path='trained_ids_model.h5'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.model_info = None
        self.attack_types = {}
        self.selected_features = []
        self.confidence_threshold = 0.8
        self.detection_stats = {
            'total_analyzed': 0,
            'normal_detected': 0,
            'attacks_detected': 0,
            'high_confidence_detections': 0,
            'low_confidence_detections': 0
        }
        
        # Carrega modelo e preprocessors
        self._load_model_components()
    
    """Carrega modelo treinado e componentes associados"""
    def _load_model_components(self):
        try:
            print("Carregando modelo treinado...")
            
            # Verifica se arquivos existem
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Modelo não encontrado: {self.model_path}")
            
            # Carrega modelo
            self.model = load_model(self.model_path)
            print(f"Modelo carregado: {self.model_path}")
            
            # Carrega preprocessors
            scaler_path = self.model_path.replace('.h5', '_scaler.pkl')
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)
                print(f"Scaler carregado: {scaler_path}")
            
            label_encoder_path = self.model_path.replace('.h5', '_label_encoder.pkl')
            if os.path.exists(label_encoder_path):
                self.label_encoder = joblib.load(label_encoder_path)
                print(f"Label encoder carregado: {label_encoder_path}")
            
            # Carrega informações do modelo
            info_path = 'model_info.json'
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    self.model_info = json.load(f)
                
                self.attack_types = self.model_info.get('attack_types', {})
                self.selected_features = self.model_info.get('selected_features', [])
                print(f"Informações do modelo carregadas: {info_path}")
            
            # Configurações padrão se não houver arquivo de info
            if not self.attack_types:
                self.attack_types = {
                    '0': 'NORMAL', '1': 'DOS_HULK', '2': 'DOS_GOLDENEYE', '3': 'DOS_SLOWLORIS',
                    '4': 'DOS_SLOWHTTPTEST', '5': 'DDOS', '6': 'PORTSCAN', '7': 'BOTNET',
                    '8': 'INFILTRATION', '9': 'BRUTE_FORCE_FTP', '10': 'BRUTE_FORCE_SSH',
                    '11': 'WEB_ATTACK', '12': 'HEARTBLEED'
                }
            
            print(f"✓ Sistema IDS inicializado com {len(self.attack_types)} classes de detecção")
            
        except Exception as e:
            print(f"Erro ao carregar componentes do modelo: {e}")
            print("Certifique-se de que o modelo foi treinado executando o script de treinamento.")
            sys.exit(1)
    
    """Pré-processa dados de entrada para o modelo"""
    def _preprocess_data(self, df):    
        try:
            # Seleciona características se especificadas
            if self.selected_features:
                # Verifica se todas as características necessárias estão presentes
                missing_features = [f for f in self.selected_features if f not in df.columns]
                if missing_features:
                    print(f"Características ausentes: {missing_features}")
                    # Cria características ausentes com valores padrão
                    for feature in missing_features:
                        df[feature] = 0
                
                # Seleciona apenas as características do modelo
                df_features = df[self.selected_features]
            else:
                # Remove colunas não numéricas
                df_features = df.select_dtypes(include=[np.number])
            
            # Normaliza usando o scaler treinado
            if self.scaler:
                X_scaled = self.scaler.transform(df_features)
            else:
                X_scaled = df_features.values
            
            # Reshape para LSTM (samples, timesteps, features)
            X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
            
            return X_reshaped
            
        except Exception as e:
            print(f"Erro no pré-processamento: {e}")
            return None
    
    """Analisa amostra usando o modelo treinado"""
    def _analyze_sample(self, X_processed):    
        try:
            # Predição
            predictions = self.model.predict(X_processed, verbose=0)
            
            # Obtém classes e probabilidades
            predicted_classes = np.argmax(predictions, axis=1)
            confidence_scores = np.max(predictions, axis=1)
            
            # Converte para tipos de ataque
            attack_predictions = []
            for pred_class in predicted_classes:
                attack_type = self.attack_types.get(str(pred_class), f'UNKNOWN_{pred_class}')
                attack_predictions.append(attack_type)
            
            return attack_predictions, confidence_scores, predictions
            
        except Exception as e:
            print(f"Erro na análise: {e}")
            return None, None, None
    
    """Gera alerta de segurança"""
    def _generate_alert(self, attack_type, confidence, sample_id, additional_info=None):    
        severity = self._get_attack_severity(attack_type)
        
        alert = {
            'timestamp': datetime.now().isoformat(),
            'sample_id': sample_id,
            'attack_type': attack_type,
            'confidence': confidence,
            'severity': severity,
            'status': 'ALERT' if attack_type != 'NORMAL' else 'NORMAL',
            'recommended_action': self._get_recommended_action(attack_type),
            'additional_info': additional_info or {}
        }
        
        return alert
    
    """Determina severidade do ataque"""
    def _get_attack_severity(self, attack_type):    
        severity_map = {
            'NORMAL': 0,
            'PORTSCAN': 1,
            'BRUTE_FORCE_FTP': 2,
            'BRUTE_FORCE_SSH': 2,
            'WEB_ATTACK': 3,
            'BOTNET': 4,
            'DOS_HULK': 4,
            'DOS_GOLDENEYE': 4,
            'DOS_SLOWLORIS': 4,
            'DOS_SLOWHTTPTEST': 4,
            'DDOS': 5,
            'INFILTRATION': 5,
            'HEARTBLEED': 5
        }
        return severity_map.get(attack_type, 3)
    
    """Retorna ação recomendada para o tipo de ataque"""
    def _get_recommended_action(self, attack_type):    
        action_map = {
            'NORMAL': 'ALLOW',
            'PORTSCAN': 'LOG_AND_MONITOR',
            'BRUTE_FORCE_FTP': 'BLOCK_IP_TEMPORARY',
            'BRUTE_FORCE_SSH': 'BLOCK_IP_TEMPORARY',
            'WEB_ATTACK': 'BLOCK_REQUEST',
            'BOTNET': 'QUARANTINE_HOST',
            'DOS_HULK': 'RATE_LIMIT_SOURCE',
            'DOS_GOLDENEYE': 'RATE_LIMIT_SOURCE',
            'DOS_SLOWLORIS': 'RATE_LIMIT_SOURCE',
            'DOS_SLOWHTTPTEST': 'RATE_LIMIT_SOURCE',
            'DDOS': 'EMERGENCY_BLOCK',
            'INFILTRATION': 'QUARANTINE_HOST',
            'HEARTBLEED': 'EMERGENCY_BLOCK'
        }
        return action_map.get(attack_type, 'INVESTIGATE')
    
    """Analisa arquivo de dados de rede"""
    def analyze_file(self, file_path):    
        try:
            print(f"📄 Carregando arquivo: {file_path}")
            
            # Carrega dados
            df = pd.read_csv(file_path)
            print(f"✓ Arquivo carregado: {len(df)} amostras")
            
            # Pré-processa dados
            X_processed = self._preprocess_data(df)
            if X_processed is None:
                return None
            
            print(f"✓ Dados pré-processados: {X_processed.shape}")
            
            # Analisa amostras
            print("🔍 Analisando tráfego de rede...")
            start_time = time.time()
            
            attack_predictions, confidence_scores, raw_predictions = self._analyze_sample(X_processed)
            
            if attack_predictions is None:
                return None
            
            analysis_time = time.time() - start_time
            
            # Cria relatório de resultados
            results = []
            alerts = []
            
            for i, (attack_type, confidence) in enumerate(zip(attack_predictions, confidence_scores)):
                sample_id = f"SAMPLE_{i+1:06d}"
                
                # Atualiza estatísticas
                self.detection_stats['total_analyzed'] += 1
                if attack_type == 'NORMAL':
                    self.detection_stats['normal_detected'] += 1
                else:
                    self.detection_stats['attacks_detected'] += 1
                
                if confidence >= self.confidence_threshold:
                    self.detection_stats['high_confidence_detections'] += 1
                else:
                    self.detection_stats['low_confidence_detections'] += 1
                
                # Cria resultado
                result = {
                    'sample_id': sample_id,
                    'attack_type': attack_type,
                    'confidence': confidence,
                    'severity': self._get_attack_severity(attack_type),
                    'timestamp': datetime.now().isoformat()
                }
                results.append(result)
                
                # Gera alerta se necessário
                if attack_type != 'NORMAL' and confidence >= self.confidence_threshold:
                    alert = self._generate_alert(attack_type, confidence, sample_id)
                    alerts.append(alert)
            
            # Cria relatório final
            report = {
                'analysis_info': {
                    'file_analyzed': file_path,
                    'total_samples': len(df),
                    'analysis_time': analysis_time,
                    'samples_per_second': len(df) / analysis_time,
                    'timestamp': datetime.now().isoformat()
                },
                'detection_summary': {
                    'normal_traffic': self.detection_stats['normal_detected'],
                    'attacks_detected': self.detection_stats['attacks_detected'],
                    'high_confidence': self.detection_stats['high_confidence_detections'],
                    'low_confidence': self.detection_stats['low_confidence_detections'],
                    'attack_rate': self.detection_stats['attacks_detected'] / self.detection_stats['total_analyzed'] * 100
                },
                'results': results,
                'alerts': alerts,
                'statistics': self.detection_stats
            }
            
            print(f"✓ Análise concluída em {analysis_time:.2f} segundos")
            print(f"✓ Velocidade: {len(df) / analysis_time:.1f} amostras/segundo")
            
            return report
            
        except Exception as e:
            print(f"❌ Erro na análise do arquivo: {e}")
            return None
    
    """Mostra o relatório de detecção"""
    def print_detection_report(self, report):
        
        if not report:
            print("❌ Nenhum relatório para exibir")
            return
        
        print("\n" + "="*80)
        print("RELATÓRIO DE DETECÇÃO DE INTRUSÃO")
        print("="*80)
        
        # Informações da análise
        info = report['analysis_info']
        print(f"📄 Arquivo analisado: {info['file_analyzed']}")
        print(f"