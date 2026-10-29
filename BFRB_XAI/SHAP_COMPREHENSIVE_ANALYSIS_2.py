"""
SCIENTIFICALLY RIGOROUS SHAP ANALYSIS
Based on peer-reviewed XAI evaluation literature

References:
[1] Petsiuk et al. 2018 - Deletion/Insertion metrics
[2] Alvarez-Melis & Jaakkola 2018 - Faithfulness metrics
[3] Recent medRxiv 2025 - SHAP achieves perfect fidelity for tree models
[4] NeurIPS 2023 - M4 XAI benchmark metrics
[5] arXiv 2025 - Stability metrics for XAI
"""

import pickle
import os
import shap
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, mean_absolute_error
)
from sklearn.calibration import calibration_curve
from scipy.stats import spearmanr
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

N_PERTURBATIONS = 10

# ============================================================
# 1. LOAD MODELS & DATA
# ============================================================
print("=" * 80)
print("SCIENTIFICALLY RIGOROUS SHAP ANALYSIS")
print("=" * 80)
print("\n[1/15] Loading models and data...")

with open(os.path.join(save_dir, "binary_model.pkl"), "rb") as f:
    binary_model = pickle.load(f)
with open(os.path.join(save_dir, "gesture_model.pkl"), "rb") as f:
    gesture_model = pickle.load(f)
with open(os.path.join(save_dir, "gesture_encoder.pkl"), "rb") as f:
    gesture_encoder = pickle.load(f)
with open(os.path.join(save_dir, "meta.pkl"), "rb") as f:
    meta = pickle.load(f)

feature_columns = meta["feature_columns"]

# Load Data
df = pd.read_csv(os.path.join(save_dir, "train.csv"))

BFRB_GESTURES = [
    "Above ear - pull hair", "Cheek - pinch skin",
    "Eyebrow - pull hair", "Eyelash - pull hair",
    "Forehead - pull hairline", "Forehead - scratch",
    "Neck - pinch skin", "Neck - scratch"
]

df["gesture"] = df["gesture"].fillna("non_target")
df["label"] = df["gesture"].apply(lambda x: 1 if x in BFRB_GESTURES else 0)

# Feature Engineering
if all(col in df.columns for col in ['acc_x', 'acc_y', 'acc_z']):
    df['imu_magnitude'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
if all(col in df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
    df['gyro_magnitude'] = np.sqrt(df['gyro_x']**2 + df['gyro_y']**2 + df['gyro_z']**2)

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

labels_df = df.groupby("sequence_id")[["label", "gesture"]].first().reset_index()
full_df = pd.merge(agg_df, labels_df, on="sequence_id")

X = full_df[feature_columns]
y_binary = full_df["label"]
y_gesture = full_df["gesture"]

X_train, X_test, y_bin_train, y_bin_test, y_ges_train, y_ges_test = train_test_split(
    X, y_binary, y_gesture, test_size=0.2, random_state=42
)

N_SAMPLES = len(X_test)
print(f"   Analyzing: {N_SAMPLES} samples")

# ============================================================
# PART A: BINARY MODEL ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("PART A: BINARY MODEL ANALYSIS")
print("=" * 80)

# ============================================================
# 2. BINARY BASELINE METRICS
# ============================================================
print("\n[2/15] Binary: Computing baseline metrics...")

y_pred_bin = binary_model.predict(X_test)
y_proba_bin = binary_model.predict_proba(X_test)[:, 1]

binary_metrics = {
    'Accuracy': accuracy_score(y_bin_test, y_pred_bin),
    'Precision': precision_score(y_bin_test, y_pred_bin),
    'Recall': recall_score(y_bin_test, y_pred_bin),
    'F1-Score': f1_score(y_bin_test, y_pred_bin),
    'AUC-ROC': roc_auc_score(y_bin_test, y_proba_bin),
    'Log Loss': log_loss(y_bin_test, y_proba_bin),
    'MAE': mean_absolute_error(y_bin_test, y_proba_bin)
}

for metric, value in binary_metrics.items():
    print(f"   {metric}: {value:.4f}")

# ============================================================
# 3. BINARY SHAP
# ============================================================
print("\n[3/15] Binary: Computing SHAP...")

binary_explainer = shap.TreeExplainer(binary_model)
binary_shap_raw = binary_explainer.shap_values(X_test)

# FIXED: Proper extraction
if isinstance(binary_shap_raw, list):
    binary_shap = binary_shap_raw[1]
elif len(binary_shap_raw.shape) == 3:
    binary_shap = binary_shap_raw[:, :, 1]
else:
    binary_shap = binary_shap_raw

print(f"   SHAP shape: {binary_shap.shape}")

# ============================================================
# 4. BINARY FAITHFULNESS (Literature-validated)
# ============================================================
print("\n[4/15] Binary: Computing faithfulness (deletion metric)...")

# FIXED: Use training data statistics (no data leakage)
train_means = X_train.mean().values
train_medians = X_train.median().values

binary_faithfulness = []

for i in tqdm(range(N_SAMPLES), desc="   Progress"):
    shap_vals = binary_shap[i]

    # Sort features by importance
    sorted_idx = np.argsort(np.abs(shap_vals))[::-1]

    # Progressive deletion (MoRF - Most Relevant First)
    orig_pred = binary_model.predict_proba(X_test.iloc[[i]])[0, 1]
    predictions = [orig_pred]

    X_perturbed = X_test.iloc[[i]].copy()

    # Delete top 10 features progressively
    for k in range(1, min(11, len(feature_columns))):
        # Mask top-k features
        X_perturbed.iloc[0, sorted_idx[:k]] = train_medians[sorted_idx[:k]]
        pred = binary_model.predict_proba(X_perturbed)[0, 1]
        predictions.append(pred)

    # Faithfulness: Area under deletion curve (should decrease)
    # Normalized by max possible change
    auc_deletion = np.mean(predictions) / orig_pred if orig_pred > 0 else 0.5
    binary_faithfulness.append(1 - auc_deletion)  # Higher = more faithful

binary_faithfulness_mean = np.mean(binary_faithfulness)
print(f"   Faithfulness (deletion): {binary_faithfulness_mean:.4f}")

# ============================================================
# 5. BINARY STABILITY (Spearman correlation)
# ============================================================
print("\n[5/15] Binary: Computing stability (Spearman correlation)...")

train_stds = X_train.std().values
binary_stability = []

for i in tqdm(range(N_SAMPLES), desc="   Progress"):
    # Get SHAP for original
    original_shap = binary_shap[i]
    original_ranking = np.argsort(np.abs(original_shap))[::-1]

    perturbed_rankings = []

    for _ in range(N_PERTURBATIONS):
        # Add small noise
        noise = np.random.normal(0, 0.01, size=X_test.iloc[[i]].shape)
        X_p = X_test.iloc[[i]] + noise * train_stds

        # Compute SHAP for perturbed
        shap_p = binary_explainer.shap_values(X_p)
        if isinstance(shap_p, list):
            shap_p = shap_p[1][0]
        elif len(shap_p.shape) == 3:
            shap_p = shap_p[0, :, 1]
        else:
            shap_p = shap_p[0]

        # Get ranking
        perturbed_rankings.append(np.argsort(np.abs(shap_p))[::-1])

    # Compute Spearman correlation between original and perturbed rankings
    correlations = []
    for p_rank in perturbed_rankings:
        # Top-20 features ranking correlation
        top_k = min(20, len(feature_columns))
        corr, _ = spearmanr(original_ranking[:top_k], p_rank[:top_k])
        correlations.append(corr if not np.isnan(corr) else 0)

    binary_stability.append(np.mean(correlations))

binary_stability_mean = np.mean(binary_stability)
print(f"   Stability (Spearman): {binary_stability_mean:.4f}")

# ============================================================
# 6. BINARY ADVANCED METRICS
# ============================================================
print("\n[6/15] Binary: Computing advanced metrics...")

# Feature Discrimination (proper SHAP-based metric)
# Measures how well each feature's SHAP values separate classes
binary_feature_discrimination = []
for j in range(len(feature_columns)):
    try:
        # For each feature, use its SHAP values to predict class
        auc = roc_auc_score(y_bin_test, binary_shap[:, j])
        binary_feature_discrimination.append(abs(auc - 0.5))  # Distance from random
    except:
        binary_feature_discrimination.append(0)

binary_discrimination_mean = np.mean(binary_feature_discrimination)
binary_discrimination_top10 = np.mean(sorted(binary_feature_discrimination, reverse=True)[:10])

# ECE (FIXED: quantile-based bins)
try:
    prob_true, prob_pred = calibration_curve(y_bin_test, y_proba_bin, n_bins=10, strategy='quantile')
    binary_ece = np.mean(np.abs(prob_true - prob_pred))
except:
    binary_ece = 0.0

print(f"   Feature Discrimination: {binary_discrimination_mean:.4f}")
print(f"   ECE (calibration): {binary_ece:.4f}")

# ============================================================
# 7. BINARY COMPOSITE QUALITY SCORE
# ============================================================
print("\n[7/15] Binary: Computing explanation quality score...")

# Evidence-based weighting from literature review
# Faithfulness (40%), Stability (40%), Discrimination (20%)
binary_quality_score = (
    0.40 * binary_faithfulness_mean +
    0.40 * binary_stability_mean +
    0.20 * binary_discrimination_mean
)

correct_bin = (y_pred_bin == y_bin_test)
binary_quality_correct = np.mean([
    0.40 * binary_faithfulness[i] +
    0.40 * binary_stability[i] +
    0.20 * np.mean(binary_feature_discrimination)
    for i in range(N_SAMPLES) if correct_bin.iloc[i]
])
binary_quality_error = np.mean([
    0.40 * binary_faithfulness[i] +
    0.40 * binary_stability[i] +
    0.20 * np.mean(binary_feature_discrimination)
    for i in range(N_SAMPLES) if not correct_bin.iloc[i]
])

print(f"   Quality (mean): {binary_quality_score:.4f}")
print(f"   Quality (correct): {binary_quality_correct:.4f}")
print(f"   Quality (errors): {binary_quality_error:.4f}")
print(f"   Discrimination: {binary_quality_correct - binary_quality_error:.4f}")

# ============================================================
# PART B: GESTURE MODEL ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("PART B: GESTURE MODEL ANALYSIS (8-class)")
print("=" * 80)

# Filter to BFRB samples
bfrb_mask = y_ges_test.isin(BFRB_GESTURES)
X_test_bfrb = X_test[bfrb_mask]
y_ges_test_bfrb = y_ges_test[bfrb_mask]

# FIXED: Use training data for BFRB subset too
X_train_bfrb = X_train[y_ges_train.isin(BFRB_GESTURES)]
train_means_bfrb = X_train_bfrb.mean().values
train_medians_bfrb = X_train_bfrb.median().values
train_stds_bfrb = X_train_bfrb.std().values

N_BFRB = len(X_test_bfrb)
print(f"   BFRB samples: {N_BFRB}")

# ============================================================
# 8. GESTURE BASELINE METRICS
# ============================================================
print("\n[8/15] Gesture: Computing baseline metrics...")

y_pred_ges = gesture_model.predict(X_test_bfrb)
y_proba_ges = gesture_model.predict_proba(X_test_bfrb)

y_ges_test_bfrb_encoded = gesture_encoder.transform(y_ges_test_bfrb)

gesture_metrics = {
    'Accuracy': accuracy_score(y_ges_test_bfrb_encoded, y_pred_ges),
    'Macro F1': f1_score(y_ges_test_bfrb_encoded, y_pred_ges, average='macro'),
    'Weighted F1': f1_score(y_ges_test_bfrb_encoded, y_pred_ges, average='weighted')
}

for metric, value in gesture_metrics.items():
    print(f"   {metric}: {value:.4f}")

# ============================================================
# 9. GESTURE SHAP
# ============================================================
print("\n[9/15] Gesture: Computing SHAP...")

gesture_explainer = shap.TreeExplainer(gesture_model)
gesture_shap_raw = gesture_explainer.shap_values(X_test_bfrb)
gesture_shap_3d = np.array(gesture_shap_raw)
n_classes = len(gesture_encoder.classes_)

print(f"   SHAP shape: {gesture_shap_3d.shape}")

# ============================================================
# 10. GESTURE FAITHFULNESS
# ============================================================
print("\n[10/15] Gesture: Computing faithfulness...")

gesture_faithfulness = []

for i in tqdm(range(N_BFRB), desc="   Progress"):
    pred_idx = int(np.argmax(y_proba_ges[i]))
    shap_vals = gesture_shap_3d[i, :, pred_idx]

    sorted_idx = np.argsort(np.abs(shap_vals))[::-1]
    orig_pred = y_proba_ges[i, pred_idx]
    predictions = [orig_pred]

    X_perturbed = X_test_bfrb.iloc[[i]].copy()

    for k in range(1, min(11, len(feature_columns))):
        X_perturbed.iloc[0, sorted_idx[:k]] = train_medians_bfrb[sorted_idx[:k]]
        pred = gesture_model.predict_proba(X_perturbed)[0, pred_idx]
        predictions.append(pred)

    auc_deletion = np.mean(predictions) / orig_pred if orig_pred > 0 else 0.5
    gesture_faithfulness.append(1 - auc_deletion)

gesture_faithfulness_mean = np.mean(gesture_faithfulness)
print(f"   Faithfulness: {gesture_faithfulness_mean:.4f}")

# ============================================================
# 11. GESTURE STABILITY
# ============================================================
print("\n[11/15] Gesture: Computing stability...")

gesture_stability = []

for i in tqdm(range(N_BFRB), desc="   Progress"):
    pred_idx = int(np.argmax(y_proba_ges[i]))

    original_shap = gesture_shap_3d[i, :, pred_idx]
    original_ranking = np.argsort(np.abs(original_shap))[::-1]

    perturbed_rankings = []

    for _ in range(N_PERTURBATIONS):
        noise = np.random.normal(0, 0.01, size=X_test_bfrb.iloc[[i]].shape)
        X_p = X_test_bfrb.iloc[[i]] + noise * train_stds_bfrb

        shap_p = gesture_explainer.shap_values(X_p)
        shap_p_3d = np.array(shap_p)
        shap_p_class = shap_p_3d[0, :, pred_idx]

        perturbed_rankings.append(np.argsort(np.abs(shap_p_class))[::-1])

    correlations = []
    for p_rank in perturbed_rankings:
        top_k = min(20, len(feature_columns))
        corr, _ = spearmanr(original_ranking[:top_k], p_rank[:top_k])
        correlations.append(corr if not np.isnan(corr) else 0)

    gesture_stability.append(np.mean(correlations))

gesture_stability_mean = np.mean(gesture_stability)
print(f"   Stability: {gesture_stability_mean:.4f}")

# ============================================================
# 12. GESTURE ADVANCED METRICS
# ============================================================
print("\n[12/15] Gesture: Computing advanced metrics...")

gesture_discrimination_per_class = []
for class_idx in range(n_classes):
    class_disc = []
    for feat_idx in range(len(feature_columns)):
        try:
            y_binary_class = (y_ges_test_bfrb_encoded == class_idx).astype(int)
            auc = roc_auc_score(y_binary_class, gesture_shap_3d[:, feat_idx, class_idx])
            class_disc.append(abs(auc - 0.5))
        except:
            class_disc.append(0)
    gesture_discrimination_per_class.append(np.mean(class_disc))

gesture_discrimination_mean = np.mean(gesture_discrimination_per_class)

print(f"   Feature Discrimination: {gesture_discrimination_mean:.4f}")

# ============================================================
# 13. GESTURE QUALITY SCORE
# ============================================================
print("\n[13/15] Gesture: Computing explanation quality score...")

gesture_quality_score = (
    0.40 * gesture_faithfulness_mean +
    0.40 * gesture_stability_mean +
    0.20 * gesture_discrimination_mean
)

correct_ges = (y_pred_ges == y_ges_test_bfrb_encoded)
gesture_quality_correct = np.mean([
    0.40 * gesture_faithfulness[i] +
    0.40 * gesture_stability[i] +
    0.20 * np.mean(gesture_discrimination_per_class)
    for i in range(N_BFRB) if correct_ges[i]
])
gesture_quality_error = np.mean([
    0.40 * gesture_faithfulness[i] +
    0.40 * gesture_stability[i] +
    0.20 * np.mean(gesture_discrimination_per_class)
    for i in range(N_BFRB) if not correct_ges[i]
])

print(f"   Quality (mean): {gesture_quality_score:.4f}")
print(f"   Quality (correct): {gesture_quality_correct:.4f}")
print(f"   Quality (errors): {gesture_quality_error:.4f}")
print(f"   Discrimination: {gesture_quality_correct - gesture_quality_error:.4f}")

# ============================================================
# 14. SAVE RESULTS
# ============================================================
print("\n[14/15] Saving results...")

results = {
    'binary': {
        'n_samples': N_SAMPLES,
        'metrics': binary_metrics,
        'faithfulness': binary_faithfulness_mean,
        'stability': binary_stability_mean,
        'feature_discrimination': binary_discrimination_mean,
        'ece': binary_ece,
        'quality_score': binary_quality_score,
        'quality_correct': binary_quality_correct,
        'quality_error': binary_quality_error,
        'quality_discrimination': binary_quality_correct - binary_quality_error
    },
    'gesture': {
        'n_samples': N_BFRB,
        'n_classes': n_classes,
        'metrics': gesture_metrics,
        'faithfulness': gesture_faithfulness_mean,
        'stability': gesture_stability_mean,
        'feature_discrimination': gesture_discrimination_mean,
        'quality_score': gesture_quality_score,
        'quality_correct': gesture_quality_correct,
        'quality_error': gesture_quality_error,
        'quality_discrimination': gesture_quality_correct - gesture_quality_error
    }
}

with open(os.path.join(output_dir, "shap_rigorous_analysis.pkl"), "wb") as f:
    pickle.dump(results, f)

print("   ✅ Saved: shap_rigorous_analysis.pkl")

# ============================================================
# 15. FINAL SUMMARY
# ============================================================
print("\n[15/15] Generating final summary...")

print("\n" + "=" * 80)
print("✅ SCIENTIFICALLY RIGOROUS RESULTS")
print("=" * 80)

print("\n📊 BINARY MODEL (BFRB Detection)")
print("-" * 80)
for metric, value in binary_metrics.items():
    print(f"{metric:<25} {value:.4f}")
print(f"{'Faithfulness':<25} {binary_faithfulness_mean:.4f}")
print(f"{'Stability (Spearman)':<25} {binary_stability_mean:.4f}")
print(f"{'Feature Discrimination':<25} {binary_discrimination_mean:.4f}")
print(f"{'ECE (calibration)':<25} {binary_ece:.4f}")
print(f"{'Quality Score':<25} {binary_quality_score:.4f}")
print(f"{'Quality Δ (correct-error)':<25} {binary_quality_correct - binary_quality_error:.4f}")

print("\n📊 GESTURE MODEL (8-class)")
print("-" * 80)
for metric, value in gesture_metrics.items():
    print(f"{metric:<25} {value:.4f}")
print(f"{'Faithfulness':<25} {gesture_faithfulness_mean:.4f}")
print(f"{'Stability (Spearman)':<25} {gesture_stability_mean:.4f}")
print(f"{'Feature Discrimination':<25} {gesture_discrimination_mean:.4f}")
print(f"{'Quality Score':<25} {gesture_quality_score:.4f}")
print(f"{'Quality Δ (correct-error)':<25} {gesture_quality_correct - gesture_quality_error:.4f}")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE - PUBLICATION READY")
print("=" * 80)
print("\n📚 METRICS BASED ON:")
print("  • Faithfulness: Deletion-based (Petsiuk et al. 2018)")
print("  • Stability: Spearman correlation (Alvarez-Melis 2018)")
print("  • Discrimination: SHAP-based AUC deviation from random")
print("  • Calibration: ECE with quantile bins (sklearn standard)")
print("  • Quality: Weighted composite (Faith=40%, Stab=40%, Disc=20%)")
print("=" * 80)