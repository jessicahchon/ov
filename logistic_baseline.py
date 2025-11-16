import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from collections import Counter
import time

df = pd.read_excel("final_ov.xlsx")

# Drop numeric columns whose values are all zero 
df_num = df.apply(pd.to_numeric, errors="ignore")
zero_only_cols_strict = [
    c for c in df_num.columns
    if pd.api.types.is_numeric_dtype(df_num[c])
       and df_num[c].notna().any()                # at least one non-NA
       and df_num[c].fillna(np.nan).dropna().eq(0).all()
]
df = df.drop(columns=zero_only_cols_strict)
print(f"[Strict] Dropped all-zero columns: {len(zero_only_cols_strict)}")
print("Current shape:", df.shape)

# ================= Schema-level track definitions (no fitting) =================
TARGET_COL   = "2 group class"    # labels
GROUP_COL    = "cohort"           # cohort/batch
FEATURE_PREF = "hsa-"             # miRNA prefix
CA125_COL    = "CA-125"

# --- make sure cohort is string (used for stratification later) ---
df[GROUP_COL] = pd.Series(df[GROUP_COL], copy=False).astype("string")

# --- collect feature columns (miRNA only / with CA-125) ---
miRNA_cols = [c for c in df.columns if c.startswith(FEATURE_PREF)]
assert len(miRNA_cols) > 0, "No miRNA columns with the given prefix."

trackA_features = miRNA_cols
trackB_features = miRNA_cols + ([CA125_COL] if CA125_COL in df.columns else [])
# de-dup while preserving order
trackB_features = list(dict.fromkeys(trackB_features))

print(f"- Track A features: {len(trackA_features)}")
print(f"- Track B features: {len(trackB_features)}")

def _norm_labels(y):
    s = pd.Series(y, copy=False)

    # unify a few empty-like tokens to NaN (dict only!)
    s = s.replace({None: np.nan, "": np.nan, " ": np.nan,
                   "NA": np.nan, "NaN": np.nan, "nan": np.nan}).astype("string")

    s_clean = (s.fillna("")
                 .str.lower()
                 .str.replace(r"[^a-z]+", " ", regex=True)
                 .str.replace(r"\s+", " ", regex=True)
                 .str.strip())

    cancer_aliases = {
        "cancer", "case", "malignant", "tumor", "tumour", "positive",
        "ovca", "ovarian cancer"
    }
    control_aliases = {
        "control", "controls", "benign", "negative", "healthy", "normal",
        "borderline", "controls borderline", "borderline controls", "control borderline"
    }

    def map_one(t):
        if t == "": return None
        if t in cancer_aliases:  return "Cancer"
        if t in control_aliases: return "Control"
        if any(k in t for k in ["cancer","malignant","tumor","tumour","positive"]):
            return "Cancer"
        if any(k in t for k in ["control","benign","negative","healthy","normal","borderline"]):
            return "Control"
        return None

    mapped = s_clean.map(map_one)

    if mapped.isna().any():
        bad = sorted(pd.unique(s[mapped.isna()].astype("string")).tolist())
        raise ValueError(f"Unrecognized labels after normalization: {bad}. Extend mapping rules.")

    mapped = mapped.astype("string")
    uniq = set(mapped.unique().tolist())
    if uniq != {"Cancer", "Control"}:
        raise ValueError(f"Non-binary labels after normalization: {sorted(uniq)}")
    return mapped

# keep original labels for audit, then normalize in a new column and overwrite (optional)
df["_orig_labels"] = df[TARGET_COL].astype("string")
df[TARGET_COL] = _norm_labels(df[TARGET_COL])

print("Label counts:", df[TARGET_COL].value_counts().to_dict())

# --- ensure feature dtypes are numeric (avoid later failures) ---
def _coerce_numericcols(df_in, cols):
    X = df_in[cols].copy()
    for c in cols:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce")
    # drop columns that became all-NaN after coercion
    all_nan = [c for c in cols if X[c].isna().all()]
    if all_nan:
        print(f"[warn] Dropping non-numeric/all-NaN columns: {all_nan}")
        X = X.drop(columns=all_nan)
    # fill any residual NaNs with column medians (simple, safe); scaler will handle centering later
    X = X.fillna(X.median(numeric_only=True))
    return X

# materialize numeric feature frames you will reuse later
X_trackA = _coerce_numericcols(df, trackA_features)
X_trackB = _coerce_numericcols(df, trackB_features)
# keep same column orders back to the names (in case any drop happened)
trackA_features = list(X_trackA.columns)
trackB_features = list(X_trackB.columns)

print(f"[post-coerce] Track A usable features: {len(trackA_features)}")
print(f"[post-coerce] Track B usable features: {len(trackB_features)}")

X = X_trackA.values
y = df[TARGET_COL].map({"Control": 0, "Cancer": 1}).values
feature_names = trackA_features

# ================= Track A: Nested CV with Confusion Matrix Visualization =================
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import time

# Configuration
N_REPEATS = 7
N_OUTER_FOLDS = 10
N_INNER_FOLDS = 5
RANDOM_STATE = 42

# Prepare data for Track A
X = X_trackA.values
y = df[TARGET_COL].map({"Control": 0, "Cancer": 1}).values
feature_names = trackA_features

print(f"\n{'='*80}")
print(f"Track A: Nested CV on Full Dataset")
print(f"{'='*80}")
print(f"\nDataset: {len(y)} samples")
print(f"  - Control: {(y==0).sum()}")
print(f"  - Cancer:  {(y==1).sum()}")
print(f"\nNested CV Configuration:")
print(f"  - Repeats: {N_REPEATS}")
print(f"  - Outer folds: {N_OUTER_FOLDS}")
print(f"  - Inner folds: {N_INNER_FOLDS}")
print(f"{'='*80}\n")

# Optimized parameter grid
param_grid = {
    'C': [0.08, 0.1, 0.15],
    'l1_ratio': [0.4, 0.45, 0.5],
} 

print(f"Hyperparameter Grid:")
print(f"  C: {param_grid['C']}")
print(f"  l1_ratio: {param_grid['l1_ratio']}")
print(f"  Total combinations: {len(param_grid['C']) * len(param_grid['l1_ratio'])}\n")

# Storage for results
all_metrics = []
all_feature_importance = []
all_selected_features = []
all_confusion_matrices = []
repeat_confusion_matrices = []  # Store summed CM for each repeat

start_time = time.time()

for repeat_idx in range(N_REPEATS):
    print(f"\n{'='*80}")
    print(f"Repeat {repeat_idx + 1}/{N_REPEATS}")
    print(f"{'='*80}")
    
    # Outer CV on FULL dataset
    outer_cv = StratifiedKFold(
        n_splits=N_OUTER_FOLDS, 
        shuffle=True, 
        random_state=RANDOM_STATE + repeat_idx
    )
    
    repeat_fold_cms = []  # Store CMs for this repeat
    
    for outer_fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y)):
        # Split data
        X_train_fold = X[train_idx]
        X_test_fold = X[test_idx]
        y_train_fold = y[train_idx]
        y_test_fold = y[test_idx]
        
        # Standardize
        scaler = StandardScaler()
        X_train_fold_scaled = scaler.fit_transform(X_train_fold)
        X_test_fold_scaled = scaler.transform(X_test_fold)
        
        # ========== Inner CV for Hyperparameter Tuning and Feature Selection ==========
        inner_cv = StratifiedKFold(
            n_splits=N_INNER_FOLDS, 
            shuffle=True, 
            random_state=100 + repeat_idx
        )
        
        # Model for feature selection
        feature_selection_model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            class_weight='balanced',
            max_iter=10000,
            random_state=RANDOM_STATE,
            tol=1e-4
        )
        
        # Grid search for feature selection
        grid_search_fs = GridSearchCV(
            estimator=feature_selection_model,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search_fs.fit(X_train_fold_scaled, y_train_fold)
        best_model_fs = grid_search_fs.best_estimator_
        
        # Get feature importance
        feature_importance = np.abs(best_model_fs.coef_[0])
        all_feature_importance.append(feature_importance)
        
        # Select top 15% features
        threshold = np.percentile(feature_importance, 85)
        selected_features_idx = np.where(feature_importance > threshold)[0]
        selected_features = [feature_names[i] for i in selected_features_idx]
        all_selected_features.append(selected_features)
        
        # ========== Train Final Model with Selected Features ==========
        X_train_selected = X_train_fold_scaled[:, selected_features_idx]
        X_test_selected = X_test_fold_scaled[:, selected_features_idx]
        
        # Grid search with selected features
        final_model = LogisticRegression(
            penalty='elasticnet',
            solver='saga',
            class_weight='balanced',
            max_iter=10000,
            random_state=RANDOM_STATE,
            tol=1e-4
        )
        
        grid_search_final = GridSearchCV(
            estimator=final_model,
            param_grid=param_grid,
            cv=inner_cv,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=0
        )
        
        grid_search_final.fit(X_train_selected, y_train_fold)
        best_model_final = grid_search_final.best_estimator_
        
        # ========== Predictions ==========
        y_pred = best_model_final.predict(X_test_selected)
        y_pred_proba = best_model_final.predict_proba(X_test_selected)[:, 1]
        
        # Confusion matrix
        cm = confusion_matrix(y_test_fold, y_pred)
        repeat_fold_cms.append(cm)
        all_confusion_matrices.append(cm)
        
        # Metrics
        metrics = {
            'repeat': repeat_idx + 1,
            'fold': outer_fold_idx + 1,
            'accuracy': accuracy_score(y_test_fold, y_pred),
            'precision': precision_score(y_test_fold, y_pred, zero_division=0),
            'recall': recall_score(y_test_fold, y_pred, zero_division=0),
            'f1': f1_score(y_test_fold, y_pred, zero_division=0),
            'auc': roc_auc_score(y_test_fold, y_pred_proba),
            'best_C': best_model_final.C,
            'best_l1_ratio': best_model_final.l1_ratio,
            'n_selected_features': len(selected_features_idx),
            'best_cv_score': grid_search_final.best_score_,
            'n_test_samples': len(y_test_fold),
            'n_train_samples': len(y_train_fold)
        }
        all_metrics.append(metrics)
        
        print(f"  Fold {outer_fold_idx + 1:2d}/{N_OUTER_FOLDS}: "
              f"AUC={metrics['auc']:.3f}, "
              f"Acc={metrics['accuracy']:.3f}, "
              f"Test={len(y_test_fold):2d}, "
              f"Features={metrics['n_selected_features']:3d}"
              f"C={metrics['best_C']:.4f}, "
              f"l1={metrics['best_l1_ratio']:.2f}")
    
    # ========== Verification for Each Repeat ==========
    total_cm = np.sum(repeat_fold_cms, axis=0)
    repeat_confusion_matrices.append(total_cm)
    total_samples = np.sum(total_cm)
    
    # Calculate metrics from confusion matrix
    tn, fp, fn, tp = total_cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    print(f"\n  {'*'*60}")
    print(f"  *** Repeat {repeat_idx + 1} Summary ***")
    print(f"  {'*'*60}")
    print(f"  Total samples: {total_samples} (Expected: {len(y)}) {'✓' if total_samples == len(y) else '✗'}")
    print(f"\n  Combined Confusion Matrix:")
    print(f"              Predicted")
    print(f"              Control  Cancer")
    print(f"  True Control   {tn:3d}     {fp:3d}")
    print(f"       Cancer    {fn:3d}     {tp:3d}")
    print(f"\n  Performance:")
    print(f"    Accuracy:    {accuracy:.3f}")
    print(f"    Sensitivity: {sensitivity:.3f} (Recall)")
    print(f"    Specificity: {specificity:.3f}")
    print(f"    Precision:   {precision:.3f}")
    print(f"  {'*'*60}")

elapsed_time = time.time() - start_time
print(f"\n{'='*80}")
print(f"Total time: {elapsed_time:.1f}s ({elapsed_time/60:.1f}min)")
print(f"Average per fold: {elapsed_time/(N_REPEATS*N_OUTER_FOLDS):.1f}s")
print(f"{'='*80}\n")

# ========== Visualize Confusion Matrices for All Repeats ==========
print("\n" + "="*80)
print("CONFUSION MATRIX VISUALIZATION")
print("="*80 + "\n")

# Plot confusion matrices for all 7 repeats
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.ravel()

for repeat_idx, cm in enumerate(repeat_confusion_matrices):
    ax = axes[repeat_idx]
    
    # Create heatmap
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Control', 'Cancer'],
                yticklabels=['Control', 'Cancer'],
                cbar=False,
                annot_kws={'size': 14, 'weight': 'bold'})
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Set title with metrics
    ax.set_title(f'Repeat {repeat_idx + 1}\n'
                 f'Acc: {accuracy:.3f} | Sens: {sensitivity:.3f} | Spec: {specificity:.3f}\n'
                 f'Total: {np.sum(cm)}',
                 fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=11)

# Hide the 8th subplot
axes[7].axis('off')

plt.suptitle(f'Track A: Confusion Matrices for {N_REPEATS} Repeats\n'
             f'(Sum of {N_OUTER_FOLDS} Outer Folds per Repeat)', 
             fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.show()

# ========== Average Confusion Matrix Across All Repeats ==========
print("\n" + "="*80)
print("AVERAGE CONFUSION MATRIX (Across All Repeats)")
print("="*80)

avg_cm = np.mean(repeat_confusion_matrices, axis=0)
std_cm = np.std(repeat_confusion_matrices, axis=0)

print(f"\nAverage Confusion Matrix:")
print(f"              Predicted")
print(f"              Control        Cancer")
print(f"  True Control   {avg_cm[0,0]:.1f}±{std_cm[0,0]:.1f}     {avg_cm[0,1]:.1f}±{std_cm[0,1]:.1f}")
print(f"       Cancer    {avg_cm[1,0]:.1f}±{std_cm[1,0]:.1f}     {avg_cm[1,1]:.1f}±{std_cm[1,1]:.1f}")

# Plot average confusion matrix
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
sns.heatmap(avg_cm, annot=True, fmt='.1f', cmap='RdYlGn', ax=ax,
            xticklabels=['Control', 'Cancer'],
            yticklabels=['Control', 'Cancer'],
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 16, 'weight': 'bold'})

# Calculate average metrics
tn_avg, fp_avg, fn_avg, tp_avg = avg_cm.ravel()
accuracy_avg = (tp_avg + tn_avg) / (tp_avg + tn_avg + fp_avg + fn_avg)
sensitivity_avg = tp_avg / (tp_avg + fn_avg)
specificity_avg = tn_avg / (tn_avg + fp_avg)

ax.set_title(f'Average Confusion Matrix Across {N_REPEATS} Repeats\n'
             f'Acc: {accuracy_avg:.3f} | Sens: {sensitivity_avg:.3f} | Spec: {specificity_avg:.3f}',
             fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.show()

# ========== Performance Summary ==========
metrics_df = pd.DataFrame(all_metrics)

print("\n" + "="*80)
print("OVERALL PERFORMANCE SUMMARY")
print("="*80)

print("\nMetrics (Mean ± Std across all folds):")
for metric in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
    mean_val = metrics_df[metric].mean()
    std_val = metrics_df[metric].std()
    print(f"  {metric.upper():10s}: {mean_val:.4f} ± {std_val:.4f}")

print(f"\nFeatures selected: {metrics_df['n_selected_features'].mean():.1f} ± "
      f"{metrics_df['n_selected_features'].std():.1f}")

# Save results
output_file = 'trackA_nested_cv_results.csv'
metrics_df.to_csv(output_file, index=False)
print(f"\n{'='*80}")
print(f"Results saved to '{output_file}'")
print(f"{'='*80}\n")
