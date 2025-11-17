#!/usr/bin/env python3
"""
    Enhanced Feature Engineering Module for CICIoT Dataset
    Temporal, Behavioral, and Network Context Features for 40/60 Distribution
"""

import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    VarianceThreshold, f_classif, mutual_info_classif, 
    SelectKBest, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class EnhancedCICIoTFeatureEngineer:
    """Enhanced Feature Engineering with temporal and behavioral features"""
    
    def __init__(self, config):
        self.config = config
        self.feature_names = None
        self.selected_features = None
        self.scaler = StandardScaler()
        
        # Enhanced critical features for 40/60 distribution
        self.critical_features = [
            'Protocol Type', 'Duration', 'Rate', 'Srate', 'Drate',
            'syn_flag_number', 'fin_flag_number', 'rst_flag_number',
            'ack_flag_number', 'TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS',
            'DNS', 'Tot sum', 'Tot size', 'IAT', 'Variance'
        ]
        
        # New behavioral pattern features
        self.behavioral_features = []
        self.temporal_features = []
        self.network_context_features = []
    
    def clean_and_preprocess(self, df):
        """Enhanced data cleaning and preprocessing"""
        print("Step 1: Enhanced data cleaning and preprocessing...")
        
        # Handle infinite and missing values
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN with median for numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col != 'label':
                df[col] = df[col].fillna(df[col].median())
        
        print(f"After cleaning: {df.shape}")
        return df
    
    def engineer_temporal_features(self, df):
        """Engineer temporal and timing-based features"""
        print("Step 1.5: Engineering temporal features...")
        
        temporal_features = []
        
        # Enhanced timing features
        if 'Duration' in df.columns:
            # Duration-based features
            df['Duration_log'] = np.log1p(df['Duration'])
            df['Duration_squared'] = df['Duration'] ** 2
            temporal_features.extend(['Duration_log', 'Duration_squared'])
        
        if 'IAT' in df.columns:
            # Inter-arrival time features
            df['IAT_log'] = np.log1p(df['IAT'])
            df['IAT_variance'] = df['IAT'] * df.get('Variance', 1)
            temporal_features.extend(['IAT_log', 'IAT_variance'])
        
        # Rate-based temporal features
        if 'Rate' in df.columns and 'Srate' in df.columns:
            df['Rate_ratio'] = df['Rate'] / (df['Srate'] + 1e-6)
            df['Rate_deviation'] = np.abs(df['Rate'] - df['Srate'])
            temporal_features.extend(['Rate_ratio', 'Rate_deviation'])
        
        # Flow pattern features
        if 'flow_duration' in df.columns:
            df['flow_intensity'] = df.get('Tot size', 0) / (df['flow_duration'] + 1e-6)
            df['flow_efficiency'] = df.get('Number', 1) / (df['flow_duration'] + 1e-6)
            temporal_features.extend(['flow_intensity', 'flow_efficiency'])
        
        self.temporal_features = temporal_features
        print(f"Added {len(temporal_features)} temporal features")
        return df
    
    def engineer_behavioral_features(self, df):
        """Engineer behavioral pattern features"""
        print("Step 1.6: Engineering behavioral features...")
        
        behavioral_features = []
        
        # Statistical behavior patterns
        stats_cols = ['Rate', 'Srate', 'IAT', 'Tot size']
        available_stats = [col for col in stats_cols if col in df.columns]
        
        if len(available_stats) >= 2:
            # Correlation patterns
            for i, col1 in enumerate(available_stats):
                for col2 in available_stats[i+1:]:
                    corr_name = f'{col1}_{col2}_corr'
                    df[corr_name] = df[col1] * df[col2]
                    behavioral_features.append(corr_name)
        
        # Protocol switching behavior
        protocol_cols = [col for col in df.columns if col in ['TCP', 'UDP', 'ICMP', 'HTTP', 'HTTPS', 'DNS']]
        if len(protocol_cols) >= 2:
            df['protocol_diversity'] = df[protocol_cols].sum(axis=1)
            df['protocol_switching'] = (df[protocol_cols] > 0).sum(axis=1)
            behavioral_features.extend(['protocol_diversity', 'protocol_switching'])
        
        # Packet size behavior
        if 'Tot size' in df.columns and 'Number' in df.columns:
            df['avg_packet_size'] = df['Tot size'] / (df['Number'] + 1e-6)
            df['packet_size_variance'] = df['Tot size'] * df.get('Variance', 1)
            behavioral_features.extend(['avg_packet_size', 'packet_size_variance'])
        
        # Connection behavior patterns
        flag_cols = [col for col in df.columns if 'flag' in col.lower()]
        if len(flag_cols) >= 2:
            df['flag_complexity'] = df[flag_cols].sum(axis=1)
            df['abnormal_flags'] = (df[flag_cols] > df[flag_cols].mean()).sum(axis=1)
            behavioral_features.extend(['flag_complexity', 'abnormal_flags'])
        
        # Entropy-like features for randomness detection
        if 'Header_Length' in df.columns:
            df['header_entropy'] = -df['Header_Length'] * np.log2(df['Header_Length'] + 1e-6)
            behavioral_features.append('header_entropy')
        
        self.behavioral_features = behavioral_features
        print(f"Added {len(behavioral_features)} behavioral features")
        return df
    
    def engineer_network_context_features(self, df):
        """Engineer network context and topology features"""
        print("Step 1.7: Engineering network context features...")
        
        network_features = []
        
        # Communication pattern features
        if 'Rate' in df.columns and 'Duration' in df.columns:
            df['communication_density'] = df['Rate'] * df['Duration']
            df['burst_indicator'] = np.where(df['Rate'] > df['Rate'].quantile(0.8), 1, 0)
            network_features.extend(['communication_density', 'burst_indicator'])
        
        # Network load indicators
        if 'Tot size' in df.columns and 'IAT' in df.columns:
            df['network_load'] = df['Tot size'] / (df['IAT'] + 1e-6)
            df['congestion_indicator'] = np.where(df['IAT'] > df['IAT'].quantile(0.8), 1, 0)
            network_features.extend(['network_load', 'congestion_indicator'])
        
        # Connection quality features
        if 'rst_count' in df.columns and 'ack_flag_number' in df.columns:
            df['connection_stability'] = df['ack_flag_number'] / (df['rst_count'] + 1)
            df['connection_quality'] = 1 / (df['rst_count'] + 1)
            network_features.extend(['connection_stability', 'connection_quality'])
        
        # Service type indicators
        service_indicators = []
        if 'HTTP' in df.columns:
            service_indicators.append('HTTP')
        if 'HTTPS' in df.columns:
            service_indicators.append('HTTPS')
        if 'DNS' in df.columns:
            service_indicators.append('DNS')
        
        if len(service_indicators) >= 2:
            df['service_diversity'] = df[service_indicators].sum(axis=1)
            df['secure_ratio'] = df.get('HTTPS', 0) / (df.get('HTTP', 0) + df.get('HTTPS', 0) + 1e-6)
            network_features.extend(['service_diversity', 'secure_ratio'])
        
        # Anomaly detection features
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'label']
        
        if len(numeric_cols) >= 3:
            # Z-score based anomaly detection
            z_scores = np.abs(stats.zscore(df[numeric_cols[:3]], nan_policy='omit'))
            df['anomaly_score'] = np.mean(z_scores, axis=1)
            df['high_anomaly'] = np.where(df['anomaly_score'] > 2.0, 1, 0)
            network_features.extend(['anomaly_score', 'high_anomaly'])
        
        self.network_context_features = network_features
        print(f"Added {len(network_features)} network context features")
        return df
    
    def remove_low_variance_features(self, X, feature_names):
        """Enhanced low variance feature removal"""
        print("Step 2: Removing low variance features...")
        
        # Use slightly lower threshold for more features
        selector = VarianceThreshold(threshold=self.config.variance_threshold * 0.5)
        X_selected = selector.fit_transform(X)
        
        # Get selected feature names
        selected_mask = selector.get_support()
        selected_features = [feature_names[i] for i, mask in enumerate(selected_mask) if mask]
        
        print(f"Removed {len(feature_names) - len(selected_features)} low-variance features")
        print(f"Remaining features: {len(selected_features)}")
        
        return X_selected, selected_features
    
    def statistical_feature_selection(self, X, y, feature_names):
        """Enhanced statistical feature selection"""
        print("Step 3: Enhanced statistical feature selection...")
        
        # CORRECTION CRITIQUE: Utiliser exactement n_features_final
        target_features = self.config.n_features_final
        
        # F-test selection
        f_selector = SelectKBest(f_classif, k=min(target_features, X.shape[1]))
        X_f_selected = f_selector.fit_transform(X, y)
        f_scores = f_selector.scores_
        f_selected_mask = f_selector.get_support()
        
        # Mutual Information selection
        mi_scores = mutual_info_classif(X, y, random_state=self.config.seed)
        mi_indices = np.argsort(mi_scores)[::-1][:min(target_features, X.shape[1])]
        
        # Combine both methods mais limiter à target_features
        combined_mask = np.zeros(len(feature_names), dtype=bool)
        combined_mask[f_selected_mask] = True
        combined_mask[mi_indices] = True
        
        # CORRECTION: Limiter strictement à n_features_final
        if np.sum(combined_mask) > target_features:
            # Prendre les top features selon le score F + MI
            combined_scores = np.zeros(len(feature_names))
            combined_scores[f_selected_mask] += f_scores[f_selected_mask]
            for idx in mi_indices:
                combined_scores[idx] += mi_scores[idx]
            
            top_indices = np.argsort(combined_scores)[::-1][:target_features]
            combined_mask = np.zeros(len(feature_names), dtype=bool)
            combined_mask[top_indices] = True
        
        X_selected = X[:, combined_mask]
        selected_features = [feature_names[i] for i, mask in enumerate(combined_mask) if mask]
        
        # Store feature scores for analysis
        feature_scores = {}
        for i, fname in enumerate(feature_names):
            feature_scores[fname] = {
                'f_score': f_scores[i] if i < len(f_scores) else 0,
                'mi_score': mi_scores[i] if i < len(mi_scores) else 0,
                'selected': combined_mask[i]
            }
        
        print(f"Enhanced statistical selection: {len(selected_features)} features selected")
        return X_selected, selected_features, feature_scores
    
    def model_based_feature_selection(self, X, y, feature_names):
        """Enhanced model-based feature selection with ensemble"""
        print("Step 4: Enhanced model-based feature selection...")
        
        # CORRECTION CRITIQUE: Utiliser exactement n_features_final
        target_features = self.config.n_features_final
        
        # Use multiple models for robust feature selection
        models = [
            RandomForestClassifier(n_estimators=100, random_state=self.config.seed, class_weight='balanced'),
            RandomForestClassifier(n_estimators=50, max_depth=10, random_state=self.config.seed+1, class_weight='balanced'),
        ]
        
        all_importances = []
        
        # Use stratified cross-validation for robust feature importance
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.config.seed)
        
        for model in models:
            model_importances = np.zeros(X.shape[1])
            
            for train_idx, val_idx in skf.split(X, y):
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                model.fit(X_train_fold, y_train_fold)
                model_importances += model.feature_importances_
            
            model_importances /= skf.n_splits
            all_importances.append(model_importances)
        
        # Average importances across models
        ensemble_importances = np.mean(all_importances, axis=0)
        
        # CORRECTION: Sélectionner exactement target_features
        top_indices = np.argsort(ensemble_importances)[::-1][:target_features]
        X_selected = X[:, top_indices]
        selected_features = [feature_names[i] for i in top_indices]
        
        # Store feature importances
        importance_scores = {
            feature_names[i]: ensemble_importances[i] 
            for i in range(len(feature_names))
        }
        
        print(f"Enhanced model-based selection: {len(selected_features)} features selected")
        return X_selected, selected_features, importance_scores
    
    def apply_enhanced_domain_knowledge(self, selected_features, all_feature_names):
        """CORRECTION MAJEURE: Respecter strictement n_features_final"""
        print("Step 5: Applying enhanced domain knowledge...")
        
        # CORRECTION CRITIQUE: Ne jamais dépasser n_features_final
        target_count = self.config.n_features_final
        
        # Prioriser les features critiques disponibles
        critical_available = [f for f in self.critical_features if f in all_feature_names]
        
        # Commencer avec les features sélectionnées
        final_features = list(selected_features[:target_count])  # Limiter immédiatement
        
        # Si on n'a pas assez, compléter avec les features critiques manquantes
        if len(final_features) < target_count:
            for critical_feature in critical_available:
                if critical_feature not in final_features and len(final_features) < target_count:
                    final_features.append(critical_feature)
                    print(f"Added critical feature: {critical_feature}")
        
        # Si on a trop, garder seulement les top features
        if len(final_features) > target_count:
            # Prioriser les features critiques puis les autres
            prioritized = [f for f in final_features if f in critical_available]
            remaining = [f for f in final_features if f not in critical_available]
            
            if len(prioritized) > target_count:
                final_features = prioritized[:target_count]
            else:
                final_features = prioritized + remaining[:target_count - len(prioritized)]
        
        # VERIFICATION FINALE: S'assurer qu'on a exactement n_features_final
        final_features = final_features[:target_count]
        
        print(f"Final feature set: {len(final_features)} features (target: {target_count})")
        
        if len(final_features) != target_count:
            print(f"WARNING: Got {len(final_features)} features but expected {target_count}")
        
        return final_features
    
    def fit_transform(self, df):
        """Enhanced complete feature engineering pipeline"""
        print("="*60)
        print("ENHANCED FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        # Step 1: Clean data
        df_clean = self.clean_and_preprocess(df.copy())
        
        # Enhanced steps: Engineer new features
        df_enhanced = self.engineer_temporal_features(df_clean)
        df_enhanced = self.engineer_behavioral_features(df_enhanced)
        df_enhanced = self.engineer_network_context_features(df_enhanced)
        
        # Separate features and labels
        X = df_enhanced.drop('label', axis=1).values
        y = df_enhanced['label'].values
        self.feature_names = list(df_enhanced.drop('label', axis=1).columns)
        
        print(f"Starting with {X.shape[1]} features (including {len(self.temporal_features + self.behavioral_features + self.network_context_features)} engineered)")
        
        # Step 2: Remove low variance features
        X_var, features_var = self.remove_low_variance_features(X, self.feature_names)
        
        # Step 3: Enhanced statistical selection (limité à n_features_final)
        X_stat, features_stat, stat_scores = self.statistical_feature_selection(
            X_var, y, features_var
        )
        
        # Step 4: Enhanced model-based selection (limité à n_features_final)
        X_model, features_model, importance_scores = self.model_based_feature_selection(
            X_stat, y, features_stat
        )
        
        # Step 5: Apply enhanced domain knowledge (STRICTEMENT limité à n_features_final)
        final_features = self.apply_enhanced_domain_knowledge(features_model, self.feature_names)
        
        # VERIFICATION FINALE CRITIQUE
        if len(final_features) != self.config.n_features_final:
            print(f"CRITICAL ERROR: Expected {self.config.n_features_final} features but got {len(final_features)}")
            # Force la sélection exacte
            final_features = final_features[:self.config.n_features_final]
        
        # Get final feature indices from original enhanced dataframe
        available_final_features = [f for f in final_features if f in df_enhanced.columns]
        if len(available_final_features) != len(final_features):
            print(f"WARNING: Some features not available in dataframe")
            final_features = available_final_features
        
        # Extract final features from original enhanced dataframe
        X_final = df_enhanced[final_features].values
        
        # Update selected features
        self.selected_features = final_features
        
        # Fit scaler on final features
        X_scaled = self.scaler.fit_transform(X_final)
        
        print(f"Enhanced feature engineering complete: {X_scaled.shape[1]} features selected")
        print(f"Selected features: {self.selected_features}")
        
        # VERIFICATION FINALE
        if X_scaled.shape[1] != self.config.n_features_final:
            raise ValueError(f"CRITICAL: Final feature count mismatch. Expected {self.config.n_features_final}, got {X_scaled.shape[1]}")
        
        # Store enhanced feature analysis
        self.feature_analysis = {
            'statistical_scores': stat_scores,
            'importance_scores': importance_scores,
            'final_features': final_features,
            'temporal_features': self.temporal_features,
            'behavioral_features': self.behavioral_features,
            'network_context_features': self.network_context_features
        }
        
        return X_scaled, y
    
    def transform(self, df):
        """Transform new data using fitted enhanced feature engineering"""
        if self.selected_features is None:
            raise ValueError("Enhanced feature engineering not fitted yet. Call fit_transform first.")
        
        # Clean data
        df_clean = self.clean_and_preprocess(df.copy())
        
        # Apply same feature engineering
        df_enhanced = self.engineer_temporal_features(df_clean)
        df_enhanced = self.engineer_behavioral_features(df_enhanced)
        df_enhanced = self.engineer_network_context_features(df_enhanced)
        
        # Select and order features
        available_features = [f for f in self.selected_features if f in df_enhanced.columns]
        if len(available_features) != len(self.selected_features):
            missing = set(self.selected_features) - set(available_features)
            print(f"Warning: Missing features in new data: {missing}")
        
        X = df_enhanced[available_features].values
        X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def get_enhanced_feature_importance_report(self):
        """Generate enhanced feature importance report"""
        if not hasattr(self, 'feature_analysis'):
            return "Enhanced feature analysis not available. Run fit_transform first."
        
        report = []
        report.append("="*70)
        report.append("ENHANCED FEATURE IMPORTANCE REPORT")
        report.append("="*70)
        
        # Final selected features
        report.append(f"\nFinal Selected Features ({len(self.selected_features)}):")
        for i, feature in enumerate(self.selected_features, 1):
            importance = self.feature_analysis['importance_scores'].get(feature, 0)
            feature_type = self._get_feature_type(feature)
            report.append(f"{i:2d}. {feature:<30} (Importance: {importance:.4f}) [{feature_type}]")
        
        # Feature category breakdown
        categories = {
            'Temporal': self.feature_analysis['temporal_features'],
            'Behavioral': self.feature_analysis['behavioral_features'], 
            'Network Context': self.feature_analysis['network_context_features'],
            'Original Critical': [f for f in self.selected_features if f in self.critical_features]
        }
        
        for category, features in categories.items():
            included = [f for f in features if f in self.selected_features]
            if included:
                report.append(f"\n{category} Features Included ({len(included)}):")
                for feature in included:
                    importance = self.feature_analysis['importance_scores'].get(feature, 0)
                    report.append(f"  - {feature:<25} (Importance: {importance:.4f})")
        
        return "\n".join(report)
    
    def _get_feature_type(self, feature):
        """Get feature type for reporting"""
        if feature in self.temporal_features:
            return "Temporal"
        elif feature in self.behavioral_features:
            return "Behavioral"
        elif feature in self.network_context_features:
            return "Network"
        elif feature in self.critical_features:
            return "Critical"
        else:
            return "Original"
    
    # Backward compatibility methods
    def get_feature_importance_report(self):
        """Backward compatibility wrapper"""
        return self.get_enhanced_feature_importance_report()

# Backward compatibility
class CICIoTFeatureEngineer(EnhancedCICIoTFeatureEngineer):
    """Alias for backward compatibility"""
    pass