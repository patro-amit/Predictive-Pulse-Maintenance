"""
Model Verification Script
Tests all 4 ML models to verify their predictions and accuracy
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "backend" / "models"
DATA_PATH = BASE_DIR / "data" / "predictive_maintenance_bigdata.csv"

def load_test_data():
    """Load and prepare test data"""
    print("📊 Loading BigData dataset...")
    df = pd.read_csv(DATA_PATH)
    
    # Split the same way as training (80-20 split with same random state)
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    
    print(f"✅ Test set: {len(test_df)} samples")
    return test_df

def verify_model(model_name, test_df):
    """Verify a single model's predictions"""
    model_path = MODELS_DIR / f"{model_name}.pkl"
    
    if not model_path.exists():
        print(f"❌ Model not found: {model_name}")
        return None
    
    print(f"\n{'='*70}")
    print(f"🔍 VERIFYING: {model_name.upper()}")
    print(f"{'='*70}")
    
    try:
        # Load model bundle
        bundle = joblib.load(model_path)
        model = bundle['model']
        scaler = bundle['scaler']
        features = bundle['features']
        stored_metrics = bundle.get('metrics', {})
        
        print(f"📦 Loaded: Model, Scaler, {len(features)} features")
        print(f"📈 Stored Accuracy: {stored_metrics.get('accuracy', 0)*100:.2f}%")
        
        # Prepare test data
        X_test = test_df[features]
        y_test = test_df['target']
        
        # Scale features
        X_test_scaled = scaler.transform(X_test)
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        tn, fp, fn, tp = cm.ravel()
        
        print(f"\n📊 VERIFICATION RESULTS:")
        print(f"   Accuracy:  {accuracy*100:.2f}% (stored: {stored_metrics.get('accuracy', 0)*100:.2f}%)")
        print(f"   Precision: {precision:.4f} (stored: {stored_metrics.get('precision', 0):.4f})")
        print(f"   Recall:    {recall:.4f} (stored: {stored_metrics.get('recall', 0):.4f})")
        print(f"   F1-Score:  {f1:.4f} (stored: {stored_metrics.get('f1', 0):.4f})")
        
        print(f"\n🔢 Confusion Matrix:")
        print(f"   True Negatives:  {tn} (stored: {stored_metrics.get('true_negatives', 0)})")
        print(f"   False Positives: {fp} (stored: {stored_metrics.get('false_positives', 0)})")
        print(f"   False Negatives: {fn} (stored: {stored_metrics.get('false_negatives', 0)})")
        print(f"   True Positives:  {tp} (stored: {stored_metrics.get('true_positives', 0)})")
        
        # Check prediction distribution
        n_needs_maintenance = np.sum(y_pred == 1)
        n_working_fine = np.sum(y_pred == 0)
        actual_maintenance = np.sum(y_test == 1)
        actual_working = np.sum(y_test == 0)
        
        print(f"\n📈 Prediction Distribution:")
        print(f"   Predicted 'Needs Maintenance': {n_needs_maintenance} ({n_needs_maintenance/len(y_pred)*100:.1f}%)")
        print(f"   Predicted 'Working Fine':      {n_working_fine} ({n_working_fine/len(y_pred)*100:.1f}%)")
        print(f"   Actual 'Needs Maintenance':    {actual_maintenance} ({actual_maintenance/len(y_test)*100:.1f}%)")
        print(f"   Actual 'Working Fine':         {actual_working} ({actual_working/len(y_test)*100:.1f}%)")
        
        # Check for bias
        if y_pred_proba is not None:
            avg_proba = np.mean(y_pred_proba)
            print(f"\n🎯 Average Prediction Probability: {avg_proba:.4f}")
            if avg_proba > 0.5:
                print(f"   ⚠️  Model is biased towards 'Needs Maintenance' (threshold issue)")
            elif avg_proba < 0.2:
                print(f"   ⚠️  Model is biased towards 'Working Fine'")
            else:
                print(f"   ✅ Model predictions are balanced")
        
        # Test with sample inputs
        print(f"\n🧪 Sample Predictions (first 10 test samples):")
        for i in range(min(10, len(X_test))):
            sample = X_test_scaled[i:i+1]
            pred = model.predict(sample)[0]
            actual = y_test.iloc[i]
            proba = model.predict_proba(sample)[0, 1] if hasattr(model, 'predict_proba') else None
            
            status = "✅" if pred == actual else "❌"
            pred_text = "NEEDS MAINTENANCE" if pred == 1 else "WORKING FINE"
            actual_text = "NEEDS MAINTENANCE" if actual == 1 else "WORKING FINE"
            proba_text = f"(prob: {proba:.3f})" if proba is not None else ""
            
            print(f"   Sample {i+1}: {status} Predicted: {pred_text} {proba_text} | Actual: {actual_text}")
        
        # Verification status
        acc_diff = abs(accuracy - stored_metrics.get('accuracy', 0))
        if acc_diff < 0.01:
            print(f"\n✅ MODEL VERIFICATION PASSED: Accuracy matches stored value")
        else:
            print(f"\n⚠️  MODEL VERIFICATION WARNING: Accuracy differs by {acc_diff*100:.2f}%")
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'prediction_distribution': {
                'needs_maintenance': n_needs_maintenance,
                'working_fine': n_working_fine
            },
            'avg_probability': avg_proba if y_pred_proba is not None else None
        }
        
    except Exception as e:
        print(f"❌ ERROR verifying {model_name}: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main verification routine"""
    print("="*70)
    print("🔬 MODEL VERIFICATION SYSTEM")
    print("="*70)
    print("Testing all 4 ML models against test dataset")
    print()
    
    # Load test data
    test_df = load_test_data()
    
    # Verify each model
    models = ['catboost', 'xgboost', 'gradient_boosting', 'random_forest']
    results = {}
    
    for model_name in models:
        result = verify_model(model_name, test_df)
        if result:
            results[model_name] = result
    
    # Summary
    print("\n" + "="*70)
    print("📊 VERIFICATION SUMMARY")
    print("="*70)
    
    if results:
        print(f"\n{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-"*70)
        for model_name, metrics in results.items():
            print(f"{model_name:<20} {metrics['accuracy']*100:>10.2f}% {metrics['precision']:>11.4f} {metrics['recall']:>11.4f} {metrics['f1']:>11.4f}")
        
        # Check for issues
        print("\n🔍 ISSUE DETECTION:")
        for model_name, metrics in results.items():
            issues = []
            
            # Check recall (should not be too low)
            if metrics['recall'] < 0.70:
                issues.append(f"Low recall ({metrics['recall']:.2%}) - may miss maintenance cases")
            
            # Check precision (should not be too low)
            if metrics['precision'] < 0.70:
                issues.append(f"Low precision ({metrics['precision']:.2%}) - many false alarms")
            
            # Check prediction bias
            if metrics['avg_probability'] is not None:
                if metrics['avg_probability'] > 0.6:
                    issues.append(f"Biased towards 'Needs Maintenance' (avg prob: {metrics['avg_probability']:.2%})")
                elif metrics['avg_probability'] < 0.15:
                    issues.append(f"Biased towards 'Working Fine' (avg prob: {metrics['avg_probability']:.2%})")
            
            if issues:
                print(f"\n⚠️  {model_name.upper()}:")
                for issue in issues:
                    print(f"   • {issue}")
            else:
                print(f"✅ {model_name.upper()}: No issues detected")
    
    print("\n" + "="*70)
    print("✅ VERIFICATION COMPLETE")
    print("="*70)

if __name__ == '__main__':
    main()
