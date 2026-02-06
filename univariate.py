# UNIVARIATE PIPELINE: Mann-Whitney Feature Selection + Nested CV
# Structure: 7 Repeats x 10 Outer (10% test) x 5 Inner
# Feature Selection: Mann-Whitney U test (p < 0.05)
# Inner CV: Tune C, l1_ratio (AUC criterion)
# Threshold: 0.5 fixed (balanced data: 52:48)

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score, f1_score
from scipy.stats import mannwhitneyu
import warnings
import time

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("UNIVARIATE PIPELINE: MANN-WHITNEY FEATURE SELECTION")
print("  Feature Selection: Mann-Whitney U test (p < 0.05)")
print("  Inner CV: Tune C, l1_ratio (AUC criterion)")
print("  Threshold: 0.5 fixed (balanced data)")
print("  Structure: 7 Repeats × 10 Outer (10% test) × 5 Inner")
print("="*80, flush=True)

# HELPER FUNCTIONS
def get_metrics(y_true, y_pred, y_prob):
    """
    Calculate classification metrics from predictions.
    Returns dict with AUC, Accuracy, Sensitivity, Specificity, PPV, NPV, F1.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'AUC': roc_auc_score(y_true, y_prob),
        'Acc': accuracy_score(y_true, y_pred),
        'Sens': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'Spec': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'F1': f1_score(y_true, y_pred)
    }

def mannwhitney_selection(X, y, alpha=0.05):
    """
    Univariate feature selection using Mann-Whitney U test.
    Tests each feature independently: Cancer vs Control.
    
    Parameters:
    -----------
    X : array, shape (n_samples, n_features)
    y : array, shape (n_samples,), binary labels (1=Cancer, 0=Control)
    alpha : float, significance threshold
    
    Returns:
    --------
    selected_idx : array of indices where p < alpha
    p_values : array of p-values for all features
    """
    n_features = X.shape[1]
    p_values = np.ones(n_features)
    
    cancer_mask = y == 1
    control_mask = y == 0
    
    for i in range(n_features):
        cancer_vals = X[cancer_mask, i]
        control_vals = X[control_mask, i]
        
        # Skip constant features
        if np.std(cancer_vals) == 0 and np.std(control_vals) == 0:
            continue
        
        try:
            _, p = mannwhitneyu(cancer_vals, control_vals, alternative='two-sided')
            p_values[i] = p
        except:
            continue
    
    # Select features with p < alpha
    selected_idx = np.where(p_values < alpha)[0]
    
    return selected_idx, p_values

# CONFIGURATION & DATA LOAD
N_REPEATS = 7
N_OUTER = 10
N_INNER = 5
ALPHA = 0.05
THRESHOLD = 0.5

# Grid search candidates
CANDIDATE_C = [0.001, 0.01, 0.05, 0.1, 0.5]
CANDIDATE_L1_RATIO = [0.0, 0.05, 0.1, 0.3]

# Load data
df = pd.read_excel('final_ov.xlsx')
y = (df['2 group class'] == 'Cancer').astype(int).values
feat_names = np.array([c for c in df.columns if c.startswith('hsa-')])
X = df[feat_names].values

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class distribution: {np.sum(y==1)} Cancer ({np.sum(y==1)/len(y)*100:.1f}%), "
      f"{np.sum(y==0)} Control ({np.sum(y==0)/len(y)*100:.1f}%)")
print(f"\nGrid Search:")
print(f"  C: {CANDIDATE_C}")
print(f"  l1_ratio: {CANDIDATE_L1_RATIO}")
print(f"  Threshold: {THRESHOLD} (fixed)", flush=True)

# EXAMPLE: Feature Selection on Full Data
print(f"\n{'#'*80}")
print("EXAMPLE: MANN-WHITNEY FEATURE SELECTION (Full Data)")
print(f"{'#'*80}", flush=True)

selected_idx_example, p_values_example = mannwhitney_selection(X, y, alpha=ALPHA)

print(f"\nTotal features: {X.shape[1]}")
print(f"Selected (p < {ALPHA}): {len(selected_idx_example)}")

# Sort by p-value
sorted_order = np.argsort(p_values_example)

print(f"\n{'='*80}")
print(f"TOP 30 FEATURES BY P-VALUE")
print(f"{'='*80}")
print(f"\n{'Rank':<5} | {'Biomarker':<25} | {'P-value':<12} | {'Pass p<0.05':<10}")
print(f"{'-'*60}")

for i, idx in enumerate(sorted_order[:30]):
    pval = p_values_example[idx]
    passed = "✓" if pval < ALPHA else "✗"
    print(f"{i+1:<5} | {feat_names[idx]:<25} | {pval:.6f}     | {passed}")

# Save full feature info
feature_info_df = pd.DataFrame({
    'Biomarker': feat_names,
    'P_value': p_values_example,
    'Pass_P05': p_values_example < ALPHA
})
feature_info_df = feature_info_df.sort_values('P_value')
feature_info_df.to_csv('mannwhitney_all_features.csv', index=False)
print(f"\n✓ Saved: mannwhitney_all_features.csv")

# MAIN NESTED CV LOOP
print(f"\n{'#'*80}")
print("STARTING NESTED CV...")
print(f"{'#'*80}", flush=True)

# Global accumulators
all_results = []
feature_selection_counts = np.zeros(X.shape[1])
all_best_c = []
all_best_l1 = []
all_n_selected = []

total_start = time.time()

for repeat in range(N_REPEATS):
    print(f"\n{'='*80}")
    print(f"REPEAT {repeat+1}/{N_REPEATS}")
    print(f"{'='*80}", flush=True)
    
    outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=42 + repeat)
    
    for outer_fold, (trainval_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"  Outer Fold {outer_fold+1}/{N_OUTER}...", end=" ", flush=True)
        
        # OUTER SPLIT: 90% train+val, 10% test
        X_trainval, X_test = X[trainval_idx], X[test_idx]
        y_trainval, y_test = y[trainval_idx], y[test_idx]
        
        # FEATURE SELECTION: Mann-Whitney (on train+val only)
        selected_idx, p_vals = mannwhitney_selection(X_trainval, y_trainval, alpha=ALPHA)
        
        n_selected = len(selected_idx)
        all_n_selected.append(n_selected)
        
        # Fallback if no features selected
        if n_selected == 0:
            print("WARNING: No significant features. Using top 50 by p-value.", flush=True)
            selected_idx = np.argsort(p_vals)[:50]
            n_selected = 50
        
        # Record feature selection
        feature_selection_counts[selected_idx] += 1
        
        X_trainval_selected = X_trainval[:, selected_idx]
        X_test_selected = X_test[:, selected_idx]
        
        # INNER CV: Tune C, l1_ratio (AUC criterion)
        inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True, 
                                   random_state=42 + repeat * 100 + outer_fold)
        
        best_c = None
        best_l1 = None
        best_inner_auc = -1
        
        for c in CANDIDATE_C:
            for l1 in CANDIDATE_L1_RATIO:
                inner_aucs = []
                
                for train_idx, val_idx in inner_cv.split(X_trainval_selected, y_trainval):
                    X_train = X_trainval_selected[train_idx]
                    y_train = y_trainval[train_idx]
                    X_val = X_trainval_selected[val_idx]
                    y_val = y_trainval[val_idx]
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_train_sc = scaler.fit_transform(X_train)
                    X_val_sc = scaler.transform(X_val)
                    
                    # Train model
                    model = LogisticRegression(
                        penalty='elasticnet', solver='saga',
                        l1_ratio=l1, C=c,
                        max_iter=5000, random_state=42
                    )
                    model.fit(X_train_sc, y_train)
                    
                    # Evaluate on validation (AUC)
                    val_prob = model.predict_proba(X_val_sc)[:, 1]
                    val_auc = roc_auc_score(y_val, val_prob)
                    inner_aucs.append(val_auc)
                
                mean_auc = np.mean(inner_aucs)
                
                if mean_auc > best_inner_auc:
                    best_inner_auc = mean_auc
                    best_c = c
                    best_l1 = l1
        
        all_best_c.append(best_c)
        all_best_l1.append(best_l1)
        
        # FINAL MODEL: Train on all train+val data
        scaler_final = StandardScaler()
        X_trainval_sc = scaler_final.fit_transform(X_trainval_selected)
        X_test_sc = scaler_final.transform(X_test_selected)
        
        final_model = LogisticRegression(
            penalty='elasticnet', solver='saga',
            l1_ratio=best_l1, C=best_c,
            max_iter=5000, random_state=42
        )
        final_model.fit(X_trainval_sc, y_trainval)
        
        # EVALUATE ON TEST SET
        y_test_prob = final_model.predict_proba(X_test_sc)[:, 1]
        y_test_pred = (y_test_prob >= THRESHOLD).astype(int)
        
        metrics = get_metrics(y_test, y_test_pred, y_test_prob)
        metrics['Repeat'] = repeat + 1
        metrics['Outer_Fold'] = outer_fold + 1
        metrics['N_Selected'] = n_selected
        metrics['Best_C'] = best_c
        metrics['Best_L1'] = best_l1
        
        all_results.append(metrics)
        
        print(f"N={n_selected}, C={best_c}, L1={best_l1}, "
              f"AUC={metrics['AUC']:.3f}, Sens={metrics['Sens']:.3f}, Spec={metrics['Spec']:.3f}", flush=True)

# FINAL SUMMARY
results_df = pd.DataFrame(all_results)

print(f"\n{'#'*80}")
print("FINAL RESULTS (7 Repeats × 10 Outer Folds = 70 evaluations)")
print(f"{'#'*80}")

print(f"\nFeature Selection (Mann-Whitney p < {ALPHA}):")
print(f"  Selected features: {np.mean(all_n_selected):.1f} ± {np.std(all_n_selected):.1f}")

print(f"\nHyperparameters (by Inner CV):")
print(f"  Best C: {pd.Series(all_best_c).value_counts().sort_index().to_dict()}")
print(f"  Best l1_ratio: {pd.Series(all_best_l1).value_counts().sort_index().to_dict()}")

print(f"\nThreshold: {THRESHOLD} (fixed)")

print(f"\nPerformance (Mean ± SD):")
print(f"  AUC:         {results_df['AUC'].mean():.4f} ± {results_df['AUC'].std():.4f}")
print(f"  Sensitivity: {results_df['Sens'].mean():.4f} ± {results_df['Sens'].std():.4f}")
print(f"  Specificity: {results_df['Spec'].mean():.4f} ± {results_df['Spec'].std():.4f}")
print(f"  Accuracy:    {results_df['Acc'].mean():.4f} ± {results_df['Acc'].std():.4f}")
print(f"  PPV:         {results_df['PPV'].mean():.4f} ± {results_df['PPV'].std():.4f}")
print(f"  NPV:         {results_df['NPV'].mean():.4f} ± {results_df['NPV'].std():.4f}")
print(f"  F1:          {results_df['F1'].mean():.4f} ± {results_df['F1'].std():.4f}")

sens_mean = results_df['Sens'].mean()
spec_mean = results_df['Spec'].mean()
print(f"\n  |Sens - Spec|: {abs(sens_mean - spec_mean):.4f}")

# TOP FEATURES BY SELECTION FREQUENCY
print(f"\n{'#'*80}")
print("TOP FEATURES BY SELECTION FREQUENCY (with p-values)")
print(f"{'#'*80}")

total_folds = N_REPEATS * N_OUTER
ranking_idx = np.argsort(feature_selection_counts)[::-1]

print(f"\n{'Rank':<5} | {'Biomarker':<25} | {'Count':<6} | {'Freq':<8} | {'P-value':<12}")
print(f"{'-'*70}")

for i, idx in enumerate(ranking_idx[:30]):
    if feature_selection_counts[idx] == 0:
        break
    freq = feature_selection_counts[idx] / total_folds * 100
    pval = p_values_example[idx]  # P-value from full data
    print(f"{i+1:<5} | {feat_names[idx]:<25} | {int(feature_selection_counts[idx]):<6} | {freq:.1f}%    | {pval:.6f}")

# SAVE RESULTS
results_df.to_csv('mannwhitney_pipeline_results.csv', index=False)

top_features_df = pd.DataFrame({
    'Rank': range(1, len(ranking_idx) + 1),
    'Biomarker': feat_names[ranking_idx],
    'Selection_Count': feature_selection_counts[ranking_idx].astype(int),
    'Frequency_Percent': (feature_selection_counts[ranking_idx] / total_folds * 100).round(2),
    'P_value': p_values_example[ranking_idx]
})
top_features_df = top_features_df[top_features_df['Selection_Count'] > 0]
top_features_df.to_csv('mannwhitney_top_features.csv', index=False)

print(f"\n✓ Saved: mannwhitney_pipeline_results.csv")
print(f"✓ Saved: mannwhitney_top_features.csv")
print(f"✓ Saved: mannwhitney_all_features.csv")
print(f"Total Runtime: {(time.time() - total_start)/60:.1f} min")



# MODEL COMPARISON: RF, SVM, XGBoost
# Feature Selection: Mann-Whitney U test (p < 0.05) - same as before
# Structure: 7 Repeats x 10 Outer (10% test) x 5 Inner
# Threshold: 0.5 fixed

# HELPER FUNCTIONS
def get_metrics(y_true, y_pred, y_prob):
    """Calculate classification metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'AUC': roc_auc_score(y_true, y_prob),
        'Acc': accuracy_score(y_true, y_pred),
        'Sens': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'Spec': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'PPV': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'F1': f1_score(y_true, y_pred)
    }

def mannwhitney_selection(X, y, alpha=0.05):
    """Univariate feature selection using Mann-Whitney U test."""
    n_features = X.shape[1]
    p_values = np.ones(n_features)
    
    cancer_mask = y == 1
    control_mask = y == 0
    
    for i in range(n_features):
        cancer_vals = X[cancer_mask, i]
        control_vals = X[control_mask, i]
        
        if np.std(cancer_vals) == 0 and np.std(control_vals) == 0:
            continue
        
        try:
            _, p = mannwhitneyu(cancer_vals, control_vals, alternative='two-sided')
            p_values[i] = p
        except:
            continue
    
    selected_idx = np.where(p_values < alpha)[0]
    return selected_idx, p_values

# MODEL CONFIGURATIONS
MODEL_CONFIGS = {
    'RF': {
        'model_class': RandomForestClassifier,
        'param_grid': {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, None],
            'min_samples_leaf': [1, 5, 9]
        },
        'fixed_params': {'random_state': 42, 'n_jobs': -1}
    },
    'SVM': {
        'model_class': SVC,
        'param_grid': {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        },
        'fixed_params': {'probability': True, 'random_state': 42}
    },
    'XGB': {
        'model_class': XGBClassifier,
        'param_grid': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.05, 0.1]
        },
        'fixed_params': {'random_state': 42, 'eval_metric': 'logloss', 'verbosity': 0}
    }
}

# CONFIGURATION & DATA LOAD
N_REPEATS = 7
N_OUTER = 10
N_INNER = 5
ALPHA = 0.05
THRESHOLD = 0.5

# Load data
df = pd.read_excel('final_ov.xlsx')
y = (df['2 group class'] == 'Cancer').astype(int).values
feat_names = np.array([c for c in df.columns if c.startswith('hsa-')])
X = df[feat_names].values

print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class distribution: {np.sum(y==1)} Cancer ({np.sum(y==1)/len(y)*100:.1f}%), "
      f"{np.sum(y==0)} Control ({np.sum(y==0)/len(y)*100:.1f}%)")
print(f"\nModels to compare: {list(MODEL_CONFIGS.keys())}", flush=True)

# MAIN NESTED CV LOOP
all_model_results = {name: [] for name in MODEL_CONFIGS.keys()}

total_start = time.time()

for repeat in range(N_REPEATS):
    print(f"\n{'='*80}")
    print(f"REPEAT {repeat+1}/{N_REPEATS}")
    print(f"{'='*80}", flush=True)
    
    outer_cv = StratifiedKFold(n_splits=N_OUTER, shuffle=True, random_state=42 + repeat)
    
    for outer_fold, (trainval_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        print(f"\n  Outer Fold {outer_fold+1}/{N_OUTER}", flush=True)
        
        # OUTER SPLIT
        X_trainval, X_test = X[trainval_idx], X[test_idx]
        y_trainval, y_test = y[trainval_idx], y[test_idx]
        
        # FEATURE SELECTION: Mann-Whitney (same for all models)
        selected_idx, _ = mannwhitney_selection(X_trainval, y_trainval, alpha=ALPHA)
        
        n_selected = len(selected_idx)
        if n_selected == 0:
            selected_idx = np.argsort(_)[:50]
            n_selected = 50
        
        X_trainval_sel = X_trainval[:, selected_idx]
        X_test_sel = X_test[:, selected_idx]
        
        # Scale features
        scaler = StandardScaler()
        X_trainval_sc = scaler.fit_transform(X_trainval_sel)
        X_test_sc = scaler.transform(X_test_sel)
        
        # EVALUATE EACH MODEL
        for model_name, config in MODEL_CONFIGS.items():
            print(f"    {model_name}...", end=" ", flush=True)
            
            # Inner CV for hyperparameter tuning
            inner_cv = StratifiedKFold(n_splits=N_INNER, shuffle=True,
                                       random_state=42 + repeat * 100 + outer_fold)
            
            best_params = None
            best_inner_auc = -1
            
            # Generate all parameter combinations
            param_names = list(config['param_grid'].keys())
            param_values = list(config['param_grid'].values())
            
            from itertools import product
            param_combinations = list(product(*param_values))
            
            for param_combo in param_combinations:
                params = dict(zip(param_names, param_combo))
                params.update(config['fixed_params'])
                
                inner_aucs = []
                
                for train_idx, val_idx in inner_cv.split(X_trainval_sc, y_trainval):
                    X_train = X_trainval_sc[train_idx]
                    y_train = y_trainval[train_idx]
                    X_val = X_trainval_sc[val_idx]
                    y_val = y_trainval[val_idx]
                    
                    model = config['model_class'](**params)
                    model.fit(X_train, y_train)
                    
                    val_prob = model.predict_proba(X_val)[:, 1]
                    val_auc = roc_auc_score(y_val, val_prob)
                    inner_aucs.append(val_auc)
                
                mean_auc = np.mean(inner_aucs)
                
                if mean_auc > best_inner_auc:
                    best_inner_auc = mean_auc
                    best_params = params.copy()
            
            # Final model with best params
            final_model = config['model_class'](**best_params)
            final_model.fit(X_trainval_sc, y_trainval)
            
            # Evaluate on test
            y_test_prob = final_model.predict_proba(X_test_sc)[:, 1]
            y_test_pred = (y_test_prob >= THRESHOLD).astype(int)
            
            metrics = get_metrics(y_test, y_test_pred, y_test_prob)
            metrics['Repeat'] = repeat + 1
            metrics['Outer_Fold'] = outer_fold + 1
            metrics['N_Selected'] = n_selected
            metrics['Best_Params'] = str(best_params)
            
            all_model_results[model_name].append(metrics)
            
            print(f"AUC={metrics['AUC']:.3f}, Sens={metrics['Sens']:.3f}, Spec={metrics['Spec']:.3f}", flush=True)

# FINAL SUMMARY
print(f"\n{'#'*80}")
print("FINAL RESULTS COMPARISON")
print(f"{'#'*80}")

summary_data = []

for model_name in MODEL_CONFIGS.keys():
    results_df = pd.DataFrame(all_model_results[model_name])
    
    summary = {
        'Model': model_name,
        'AUC': f"{results_df['AUC'].mean():.4f} ± {results_df['AUC'].std():.4f}",
        'Sensitivity': f"{results_df['Sens'].mean():.4f} ± {results_df['Sens'].std():.4f}",
        'Specificity': f"{results_df['Spec'].mean():.4f} ± {results_df['Spec'].std():.4f}",
        'Accuracy': f"{results_df['Acc'].mean():.4f} ± {results_df['Acc'].std():.4f}",
        'F1': f"{results_df['F1'].mean():.4f} ± {results_df['F1'].std():.4f}",
        'AUC_mean': results_df['AUC'].mean(),
        'Sens_mean': results_df['Sens'].mean(),
        'Spec_mean': results_df['Spec'].mean()
    }
    summary_data.append(summary)
    
    # Save individual results
    results_df.to_csv(f'mannwhitney_{model_name}_results.csv', index=False)

# Print comparison table
print(f"\n{'Model':<8} | {'AUC':<18} | {'Sensitivity':<18} | {'Specificity':<18} | {'F1':<18}")
print("-" * 90)

for s in summary_data:
    print(f"{s['Model']:<8} | {s['AUC']:<18} | {s['Sensitivity']:<18} | {s['Specificity']:<18} | {s['F1']:<18}")

# Save summary
summary_df = pd.DataFrame(summary_data)
summary_df.to_csv('mannwhitney_model_comparison.csv', index=False)

print(f"\n{'#'*80}")
print("BEST MODEL BY METRIC")
print(f"{'#'*80}")

best_auc = max(summary_data, key=lambda x: x['AUC_mean'])
best_sens = max(summary_data, key=lambda x: x['Sens_mean'])
best_spec = max(summary_data, key=lambda x: x['Spec_mean'])

print(f"\n  Best AUC:         {best_auc['Model']} ({best_auc['AUC']})")
print(f"  Best Sensitivity: {best_sens['Model']} ({best_sens['Sensitivity']})")
print(f"  Best Specificity: {best_spec['Model']} ({best_spec['Specificity']})")

print(f"\n✓ Saved: mannwhitney_RF_results.csv")
print(f"✓ Saved: mannwhitney_SVM_results.csv")
print(f"✓ Saved: mannwhitney_XGB_results.csv")
print(f"✓ Saved: mannwhitney_model_comparison.csv")
print(f"Total Runtime: {(time.time() - total_start)/60:.1f} min")
