"""
Enterprise Model Performance Monitoring System
==============================================
Production-ready model monitoring with drift detection, performance tracking,
and automated alerting. Demonstrates MLOps and production ML expertise.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

class ModelMonitor:
    """
    Enterprise-grade model performance monitoring system.
    Tracks model health, data drift, and performance degradation.
    """
    
    def __init__(self, model, reference_data, target_column='income_flag'):
        """
        Initialize model monitor.
        
        Args:
            model: Trained ML model
            reference_data: Training/reference dataset
            target_column: Name of target variable
        """
        self.model = model
        self.reference_data = reference_data
        self.target_column = target_column
        self.monitoring_history = []
        
    def detect_data_drift(self, new_data, threshold=0.1):
        """
        Detect data drift using statistical tests.
        """
        drift_results = {}
        
        # Get numeric columns
        numeric_cols = new_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in self.reference_data.columns:
                # Kolmogorov-Smirnov test for distribution drift
                from scipy import stats
                
                ref_values = self.reference_data[col].dropna()
                new_values = new_data[col].dropna()
                
                # Perform KS test
                ks_stat, p_value = stats.ks_2samp(ref_values, new_values)
                
                drift_results[col] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'drift_detected': p_value < threshold,
                    'severity': 'HIGH' if p_value < 0.01 else 'MEDIUM' if p_value < 0.05 else 'LOW'
                }
        
        return drift_results
    
    def calculate_performance_metrics(self, X_test, y_true):
        """Calculate comprehensive performance metrics."""
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_proba),
            'sample_size': len(y_true)
        }
        
        return metrics
    
    def monitor_model_health(self, X_test, y_true, data_drift_threshold=0.05):
        """
        Comprehensive model health check.
        """
        print("RUNNING MODEL HEALTH CHECK...")
        print("=" * 50)
        
        # 1. Performance Metrics
        performance = self.calculate_performance_metrics(X_test, y_true)
        
        # 2. Data Drift Detection
        drift_results = self.detect_data_drift(X_test, data_drift_threshold)
        
        # 3. Prediction Distribution Analysis
        predictions = self.model.predict_proba(X_test)[:, 1]
        pred_stats = {
            'mean_prediction': predictions.mean(),
            'std_prediction': predictions.std(),
            'min_prediction': predictions.min(),
            'max_prediction': predictions.max()
        }
        
        # 4. Generate Health Report
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'performance_metrics': performance,
            'data_drift': drift_results,
            'prediction_statistics': pred_stats,
            'alerts': self._generate_alerts(performance, drift_results)
        }
        
        # Store in monitoring history
        self.monitoring_history.append(health_report)
        
        # Display results
        self._display_health_report(health_report)
        
        return health_report
    
    def _generate_alerts(self, performance, drift_results):
        """Generate automated alerts based on monitoring results."""
        alerts = []
        
        # Performance alerts
        if performance['accuracy'] < 0.8:
            alerts.append({
                'type': 'PERFORMANCE',
                'severity': 'HIGH',
                'message': f"Model accuracy dropped to {performance['accuracy']:.3f}"
            })
        
        if performance['roc_auc'] < 0.75:
            alerts.append({
                'type': 'PERFORMANCE',
                'severity': 'HIGH',
                'message': f"ROC AUC dropped to {performance['roc_auc']:.3f}"
            })
        
        # Drift alerts
        high_drift_features = [col for col, results in drift_results.items() 
                              if results['severity'] == 'HIGH']
        
        if high_drift_features:
            alerts.append({
                'type': 'DATA_DRIFT',
                'severity': 'HIGH',
                'message': f"High drift detected in features: {', '.join(high_drift_features)}"
            })
        
        return alerts
    
    def _display_health_report(self, report):
        """Display formatted health report."""
        print("MODEL PERFORMANCE:")
        perf = report['performance_metrics']
        print(f"   Accuracy: {perf['accuracy']:.3f}")
        print(f"   Precision: {perf['precision']:.3f}")
        print(f"   Recall: {perf['recall']:.3f}")
        print(f"   F1 Score: {perf['f1_score']:.3f}")
        print(f"   ROC AUC: {perf['roc_auc']:.3f}")
        
        print("\nDATA DRIFT ANALYSIS:")
        drift_count = sum(1 for r in report['data_drift'].values() if r['drift_detected'])
        print(f"   Features with drift: {drift_count}/{len(report['data_drift'])}")
        
        for feature, results in report['data_drift'].items():
            if results['drift_detected']:
                print(f"   WARNING: {feature}: {results['severity']} drift (p={results['p_value']:.4f})")
        
        print("\nALERTS:")
        if report['alerts']:
            for alert in report['alerts']:
                symbol = "HIGH" if alert['severity'] == 'HIGH' else "MEDIUM"
                print(f"   {symbol} {alert['type']}: {alert['message']}")
        else:
            print("   No alerts - Model is healthy!")
    
    def plot_monitoring_dashboard(self, save_path=None):
        """Create comprehensive monitoring dashboard."""
        if not self.monitoring_history:
            print("No monitoring history available. Run monitor_model_health() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Extract historical data
        timestamps = [datetime.fromisoformat(h['timestamp']) for h in self.monitoring_history]
        accuracies = [h['performance_metrics']['accuracy'] for h in self.monitoring_history]
        aucs = [h['performance_metrics']['roc_auc'] for h in self.monitoring_history]
        
        # 1. Accuracy over time
        axes[0, 0].plot(timestamps, accuracies, marker='o', linewidth=2, markersize=6)
        axes[0, 0].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='Threshold')
        axes[0, 0].set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. ROC AUC over time
        axes[0, 1].plot(timestamps, aucs, marker='s', linewidth=2, markersize=6, color='orange')
        axes[0, 1].axhline(y=0.75, color='red', linestyle='--', alpha=0.7, label='Threshold')
        axes[0, 1].set_title('ROC AUC Over Time', fontsize=14, fontweight='bold')
        axes[0, 1].set_ylabel('ROC AUC')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Drift heatmap (latest monitoring)
        if self.monitoring_history:
            latest_drift = self.monitoring_history[-1]['data_drift']
            drift_features = list(latest_drift.keys())[:10]  # Top 10 features
            drift_pvalues = [latest_drift[f]['p_value'] for f in drift_features]
            
            # Create heatmap data
            heatmap_data = np.array(drift_pvalues).reshape(1, -1)
            
            im = axes[1, 0].imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.1)
            axes[1, 0].set_title('Data Drift Detection (Latest)', fontsize=14, fontweight='bold')
            axes[1, 0].set_xticks(range(len(drift_features)))
            axes[1, 0].set_xticklabels(drift_features, rotation=45, ha='right')
            axes[1, 0].set_yticks([0])
            axes[1, 0].set_yticklabels(['p-value'])
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[1, 0])
            cbar.set_label('p-value (lower = more drift)')
        
        # 4. Alert summary
        alert_types = {}
        for history in self.monitoring_history:
            for alert in history['alerts']:
                alert_type = alert['type']
                alert_types[alert_type] = alert_types.get(alert_type, 0) + 1
        
        if alert_types:
            alert_names = list(alert_types.keys())
            alert_counts = list(alert_types.values())
            axes[1, 1].pie(alert_counts, labels=alert_names, autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Alert Distribution', fontsize=14, fontweight='bold')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Alerts\nModel Healthy!', 
                           ha='center', va='center', fontsize=16, fontweight='bold',
                           transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('System Status', fontsize=14, fontweight='bold')
        
        plt.suptitle('ENTERPRISE MODEL MONITORING DASHBOARD', 
                     fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Monitoring dashboard saved to {save_path}")
        
        plt.show()
    
    def export_monitoring_report(self, file_path):
        """Export monitoring history to JSON for reporting."""
        with open(file_path, 'w') as f:
            json.dump(self.monitoring_history, f, indent=2, default=str)
        print(f"Monitoring report exported to {file_path}")

def demonstrate_enterprise_monitoring():
    """Demo function showcasing enterprise monitoring capabilities."""
    print("ENTERPRISE MODEL MONITORING SYSTEM")
    print("=" * 60)
    print("CAPABILITIES DEMONSTRATED:")
    print("• Real-time model performance tracking")
    print("• Automated data drift detection") 
    print("• Performance degradation alerts")
    print("• Executive dashboard generation")
    print("• Production monitoring workflows")
    print("\nBUSINESS VALUE:")
    print("• Reduced model risk")
    print("• Proactive issue detection")
    print("• Automated reporting")
    print("• Regulatory compliance")
    print("• Cost savings through early intervention")

if __name__ == "__main__":
    demonstrate_enterprise_monitoring()
