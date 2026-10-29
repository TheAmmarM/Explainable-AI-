"""
Comprehensive SHAP Analysis - Final Version for Paper
- Uses FULL Test Set (1631 samples) for rigor.
- Computes Stability (10 perturbations).
- Fixes "Label" error.
- Computes Honest "Concentration" instead of raw Sparsity.
"""

import pickle
import os
import shap
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss
)
from tqdm import tqdm
import time
import warnings

warnings.filterwarnings("ignore")

# ============================================================
# CONFIGURATION
# ============================================================
os.environ["OMP_NUM_THREADS"] = "8"
save_dir = r"D:\extra\BFRB dataset"
output_dir = r"D:\extra\BFRB dataset\SHAP"

# SETTINGS FOR PAPER
N_PERTURBATIONS = 10  # Standard for stability checks

# ============================================================
# 1. LOAD MODELS & DATA
# ============================================================
print("=" * 80)
print("COMPREHENSIVE SHAP ANALYSIS (FINAL)")
print("=" * 80)
print("\n[1/7] Loading models and data...")

with open(os.path.join(save_dir, "binary_model.pkl"), "rb") as f:
    binary_model = pickle.load(f)
with open(os.path.join(save_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)

feature_columns = meta["feature_columns"]

# Load Data
df = pd.read_csv(os.path.join(save_dir, "train.csv"))

# --- FIX 1: Handle Missing "Label" Column ---
# If 'label' doesn't exist, create it from 'gesture'
if "label" not in df.columns:
    # BFRB Gestures list
    BFRB_GESTURES = [
        "Above ear - pull hair", "Cheek - pinch skin",
        "Eyebrow - pull hair", "Eyelash - pull hair",
        "Forehead - pull hairline", "Forehead - scratch",
        "Neck - pinch skin", "Neck - scratch"
    ]
    df["gesture"] = df["gesture"].fillna("non_target")
    # 1 if in BFRB list, 0 otherwise
    df["label"] = df["gesture"].apply(lambda x: 1 if x in BFRB_GESTURES else 0)

# Feature Engineering (Standard)
if all(col in df.columns for col in ['acc_x', 'acc_y', 'acc_z']):
    df['imu_magnitude'] = np.sqrt(df['acc_x'] ** 2 + df['acc_y'] ** 2 + df['acc_z'] ** 2)
if all(col in df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
    df['gyro_magnitude'] = np.sqrt(df['gyro_x'] ** 2 + df['gyro_y'] ** 2 + df['gyro_z'] ** 2)

tof_cols = [col for col in df.columns if 'tof_' in col]
if tof_cols:
    df['tof_mean'] = df[tof_cols].mean(axis=1)

thermo_cols = [col for col in df.columns if 'thermo_' in col or 'thm_' in col]
if thermo_cols:
    df['thermo_mean'] = df[thermo_cols].mean(axis=1)

# Aggregation
numeric_features = df.select_dtypes(include=[np.number]).columns.drop(["sequence_id", "label"], errors="ignore")
agg_df = df.groupby("sequence_id")[numeric_features].agg(['mean', 'std', 'min', 'max', 'median', 'skew'])
agg_df.columns = ['_'.join(col) for col in agg_df.columns]
agg_df.reset_index(inplace=True)

# Prepare X and y
labels_df = df.groupby("sequence_id")["label"].first().reset_index()
full_df = pd.merge(agg_df, labels_df, on="sequence_id")

X = full_df[feature_columns]
y = full_df["label"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- SET N_SAMPLES TO FULL TEST SET ---
N_SAMPLES = len(X_test)

print(f"   Total samples: {len(X)}")
print(f"   Test samples: {len(X_test)}")
print(f"   Analyzing: {N_SAMPLES} samples (FULL TEST SET)")

# ============================================================
# 2. BASELINE MODEL METRICS
# ============================================================
print("\n[2/7] Computing baseline model metrics...")

y_pred = binary_model.predict(X_test)
y_proba = binary_model.predict_proba(X_test)[:, 1]

baseline_metrics = {
    'Accuracy': accuracy_score(y_test, y_pred),
    'Precision': precision_score(y_test, y_pred),
    'Recall': recall_score(y_test, y_pred),
    'F1-Score': f1_score(y_test, y_pred),
    'AUC-ROC': roc_auc_score(y_test, y_proba),
    'Log Loss': log_loss(y_test, y_proba)
}

print("   Baseline Performance:")
for metric, value in baseline_metrics.items():
    print(f"     {metric}: {value:.4f}")

# ============================================================
# 3. CREATE SHAP EXPLAINER
# ============================================================
print("\n[3/7] Creating SHAP explainer...")
explainer = shap.TreeExplainer(binary_model)

# ============================================================
# 4. EFFICIENCY METRICS
# ============================================================
print("\n[4/7] Computing efficiency metrics...")

# We time the first 100 samples to estimate speed, then run the rest
start_time = time.time()
# Run a batch to warm up and time
explainer.shap_values(X_test.iloc[:100])
end_time = time.time()

avg_time_per_sample = (end_time - start_time) / 100
total_estimated_time = avg_time_per_sample * N_SAMPLES

efficiency_metrics = {
    'Mean Time (s)': avg_time_per_sample,
    'Total Estimated Time (s)': total_estimated_time
}

print(f"   Mean Time per sample: {avg_time_per_sample:.6f}s")
print(f"   Est. Total Time for SHAP: {total_estimated_time:.2f}s")

# ============================================================
# 5. COMPUTE SHAP VALUES FOR ALL SAMPLES
# ============================================================
print("\n[5/7] Computing SHAP values for ALL samples...")
print("   (This creates the foundation for Fidelity/Sparsity)")

shap_values_all = explainer.shap_values(X_test)

# Handle format
if isinstance(shap_values_all, list):
    shap_values_array = shap_values_all[1]
elif len(shap_values_all.shape) == 3:
    shap_values_array = shap_values_all[:, :, 1]
else:
    shap_values_array = shap_values_all

# ============================================================
# 6. FIDELITY (Explanation Faithfulness)
# ============================================================
print("\n[6/7] Computing fidelity...")

fidelity_scores = []
# We compute fidelity for all samples (vectorized where possible would be faster, but loop is safer for logic)
# To save time, we can calculate this for the full set or a statistically significant subset (e.g., 500)
# Let's do all, but use a simple progress bar
for i in tqdm(range(N_SAMPLES), desc="   Fidelity Progress"):
    shap_vals = shap_values_array[i]

    # Top 10 important features
    top_features_idx = np.argsort(np.abs(shap_vals))[-10:]

    # Create masked sample (Set non-top features to mean)
    X_masked = X_test.iloc[[i]].copy()
    mask = np.ones(X_masked.shape[1], dtype=bool)
    mask[top_features_idx] = False

    # Replace non-top features with mean training values
    # (Optimized: pre-calculate means)
    if i == 0:
        means = X_train.mean().values

    X_masked.iloc[0, mask] = means[mask]

    # Predict
    orig_prob = binary_model.predict_proba(X_test.iloc[[i]])[0, 1]
    masked_prob = binary_model.predict_proba(X_masked)[0, 1]

    fidelity_scores.append(1 - abs(orig_prob - masked_prob))

fidelity_mean = np.mean(fidelity_scores)
print(f"   Fidelity (mean): {fidelity_mean:.4f}")

# ============================================================
# 7. STABILITY (Perturbation Robustness)
# ============================================================
print("\n[7/8] Computing stability (10 perturbations)...")
print("   (This takes the longest: ~4-5 minutes)")

stability_scores = []

# Loop through all samples
for i in tqdm(range(N_SAMPLES), desc="   Stability Progress"):
    original_shap = shap_values_array[i]
    perturbed_shaps = []

    # Create 10 perturbed versions of THIS sample
    # We can batch predict SHAP for these 10 at once for speed
    X_curr = X_test.iloc[[i]]
    X_batch_perturbed = []

    for _ in range(N_PERTURBATIONS):
        noise = np.random.normal(0, 0.01, size=X_curr.shape)
        # Add noise scaled by feature std
        if i == 0 and _ == 0:
            stds = X_train.std().values
        X_p = X_curr + noise * stds
        X_batch_perturbed.append(X_p)

    # Concat to single dataframe
    X_batch_df = pd.concat(X_batch_perturbed)

    # Run SHAP on batch of 10
    shap_batch = explainer.shap_values(X_batch_df)

    # Handle output format
    if isinstance(shap_batch, list):
        shap_batch = shap_batch[1]
    elif len(shap_batch.shape) == 3:
        shap_batch = shap_batch[:, :, 1]

    # Calculate Stability
    # Mean and Std of SHAP values across the 10 perturbations
    mean_shaps = np.mean(shap_batch, axis=0)
    std_shaps = np.std(shap_batch, axis=0)

    # Avoid div by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        cv = std_shaps / (np.abs(mean_shaps) + 1e-6)

    # Cap CV at 1.0, Stability = 1 - CV
    sample_stability = np.mean(1 - np.minimum(cv, 1.0))
    stability_scores.append(sample_stability)

stability_mean = np.mean(stability_scores)
print(f"   Stability (mean): {stability_mean:.4f}")

# ============================================================
# 8. SPARSITY & CONCENTRATION (HONEST METRIC)
# ============================================================
print("\n[8/9] Computing Concentration (Honest Sparsity)...")

# 1. Effective Sparsity (Count for 90% importance)
total_importance = np.sum(np.abs(shap_values_array), axis=1)
sorted_vals = np.sort(np.abs(shap_values_array), axis=1)[:, ::-1]
cumsum = np.cumsum(sorted_vals, axis=1)

n_features_needed = []
for i in range(N_SAMPLES):
    threshold = 0.90 * total_importance[i]
    if total_importance[i] == 0:
        n_features_needed.append(0)
    else:
        count = np.searchsorted(cumsum[i], threshold) + 1
        n_features_needed.append(count)

effective_sparsity = np.mean(n_features_needed)

# 2. Top-10 Concentration Ratio (Power of top 10)
top_10_sum = np.sum(sorted_vals[:, :10], axis=1)
concentration_ratio = np.mean(top_10_sum / (total_importance + 1e-9))

print(f"   Effective Sparsity (Features for 90%): {effective_sparsity:.2f}")
print(f"   Top-10 Concentration: {concentration_ratio:.1%}")

# ============================================================
# 9. SAVE & PRINT SUMMARY
# ============================================================
results = {
    'n_samples': N_SAMPLES,
    'baseline_metrics': baseline_metrics,
    'fidelity': fidelity_mean,
    'stability': stability_mean,
    'effective_sparsity': effective_sparsity,
    'concentration_ratio': concentration_ratio
}

# Save
with open(os.path.join(output_dir, "shap_final_paper_results.pkl"), "wb") as f:
    pickle.dump(results, f)

print("\n" + "=" * 80)
print("✅ FINAL PAPER RESULTS")
print("=" * 80)
print(f"Metric                 Value     Interpretation")
print(f"----------------------------------------------------------------")
print(f"Samples Analyzed       {N_SAMPLES}      Full Test Set (Statistical Rigor)")
print(f"Stability (10x)        {stability_mean:.3f}     Robustness to noise (Target > 0.8)")
print(f"Fidelity               {fidelity_mean:.3f}     Explanation Faithfulness")
print(f"Effective Sparsity     {effective_sparsity:.1f}      Features needed for 90% info")
print(f"Concentration (Top 10) {concentration_ratio:.1%}     Info held by just Top 10 features")
print("=" * 80)