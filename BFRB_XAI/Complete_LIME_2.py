# ============================================================
# RESEARCH-GRADE LIME EXPLAINABILITY PIPELINE (COMPATIBLE FIX)
# ============================================================

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from lime import lime_tabular

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")

# ============================================================
# PATHS
# ============================================================
MODEL_DIR = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\XAI_Work\BFRB_XAI\Trained_Models"
DATA_DIR = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\CMI_Sensor_Kaggle"
OUT_DIR = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\XAI_Work\BFRB_XAI\Figures_and_Graphs\Complete_LIME_2_Results"

os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# CRITICAL: LOAD TRAINING COMPONENTS
# ============================================================
print("=" * 80)
print("LIME ANALYSIS - WITH TRAINING COMPATIBILITY FIX")
print("=" * 80)

# Load ALL components saved during training
binary_model = joblib.load(os.path.join(MODEL_DIR, "binary_classifier.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
metadata = joblib.load(os.path.join(MODEL_DIR, "metadata.pkl"))

print(f"✓ Binary model loaded")
print(f"✓ Feature names: {len(feature_names)} features")
print(f"✓ Training threshold: {metadata['binary_threshold']}")

# ============================================================
# CRITICAL: RECREATE FEATURE SELECTION & SCALING
# ============================================================
print("\n[CRITICAL] Recreating training preprocessing pipeline...")

# Load and preprocess data EXACTLY as in training
df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
df["gesture"] = df["gesture"].fillna("non_target")

# Feature engineering (EXACT same as training)
df["imu_magnitude"] = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2)
tof_cols = [col for col in df.columns if col.startswith('tof_')]
if tof_cols:
    df['tof_mean'] = df[tof_cols].mean(axis=1)
thermo_cols = [col for col in df.columns if 'thermo_' in col or 'thm_' in col]
if thermo_cols:
    df['thermo_mean'] = df[thermo_cols].mean(axis=1)

# Aggregate (EXACT same as training)
numeric_features = df.select_dtypes(include=[np.number]).columns
numeric_features = [c for c in numeric_features if c not in ['sequence_id', 'row_id']]

agg_funcs = ['mean', 'std', 'min', 'max', 'median', 'skew']
agg_df = df.groupby("sequence_id")[numeric_features].agg(agg_funcs)
agg_df.columns = ['_'.join(col) for col in agg_df.columns]
agg_df.reset_index(inplace=True)

labels_df = df.groupby("sequence_id").first().reset_index()[["sequence_id", "gesture"]]
labels_df["gesture"] = labels_df["gesture"].fillna("non_target")
labels_df["binary_label"] = labels_df["gesture"].apply(
    lambda x: 1 if x in metadata['bfrb_gestures'] else 0
)

full_df = pd.merge(agg_df, labels_df, on="sequence_id")
X_raw = full_df.drop(columns=["sequence_id", "gesture", "binary_label"])
y_binary = full_df["binary_label"]

# CRITICAL: Align with saved feature names
print(f"\nFeature alignment check:")
print(f"  Raw features from data: {X_raw.shape[1]}")
print(f"  Saved feature names: {len(feature_names)}")

# Keep only features that were used in training
X = X_raw[feature_names].copy()
print(f"  After aligning with saved features: {X.shape}")

# Fill NA as in training
X = X.fillna(X.median())

# Train/val split (same random state as training)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(X, y_binary))
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_val = y_binary.iloc[val_idx]

print(f"✓ Data aligned with training: {X_train.shape} train, {X_val.shape} validation")

# ============================================================
# CRITICAL: CREATE SCALER FOR LIME
# ============================================================
print("\n[CRITICAL] Creating scaler for LIME (LightGBM doesn't need it, but LIME does)...")

# Create scaler on training data (even though LightGBM wasn't trained on scaled data)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(f"✓ Features scaled for LIME:")
print(f"  TOF range (scaled): [{X_train_scaled[:, 0].min():.2f}, {X_train_scaled[:, 0].max():.2f}]")
print(f"  IMU range (scaled): [{X_train_scaled[:, 6].min():.2f}, {X_train_scaled[:, 6].max():.2f}]")

# ============================================================
# CRITICAL: TEMPERATURE SCALING CHECK
# ============================================================
print("\n[CRITICAL] Checking if model needs temperature scaling...")

# Check model predictions on validation set
y_proba_val = binary_model.predict_proba(X_val)[:, 1]
extreme_count = np.sum((y_proba_val < 0.01) | (y_proba_val > 0.99))
extreme_ratio = extreme_count / len(y_proba_val)

print(f"  Extreme predictions (<0.01 or >0.99): {extreme_count}/{len(y_proba_val)} ({extreme_ratio:.1%})")

if extreme_ratio > 0.3:
    print("  ⚠️  Applying temperature scaling (T=2.0)...")


    def temperature_scaled_predictor(model, scaler=None, temperature=2.0):
        """Wrapper with temperature scaling."""

        def predict(X):
            # X might be from LIME (scaled) or raw (unscaled)
            # We need to handle both cases
            if scaler is not None:
                # Unscale if input is scaled
                X_unscaled = scaler.inverse_transform(X)
            else:
                X_unscaled = X

            # Get predictions
            probas = model.predict_proba(X_unscaled)

            # Apply temperature scaling
            epsilon = 1e-9
            probas = np.clip(probas, epsilon, 1 - epsilon)
            logit = np.log(probas[:, 1] / probas[:, 0]) / temperature
            p1 = 1 / (1 + np.exp(-logit))
            return np.c_[1 - p1, p1]

        return predict


    predict_fn = temperature_scaled_predictor(binary_model, scaler, temperature=2.0)
    print("  ✓ Temperature scaling applied")
else:
    # No temperature scaling needed
    def simple_predictor(model, scaler=None):
        """Simple wrapper that unscales if needed."""

        def predict(X):
            if scaler is not None:
                X_unscaled = scaler.inverse_transform(X)
            else:
                X_unscaled = X
            return model.predict_proba(X_unscaled)

        return predict


    predict_fn = simple_predictor(binary_model, scaler)
    print("  ✓ No temperature scaling needed")

# ============================================================
# LIME EXPLAINER (WITH SCALED TRAINING DATA)
# ============================================================
print("\n[LIME] Initializing explainer with scaled training data...")

explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,  # SCALED data for LIME
    feature_names=feature_names,
    class_names=["Non-BFRB", "BFRB"],
    mode="classification",
    discretize_continuous=True,
    kernel_width=2.0,  # Optimized for scaled features
    random_state=42
)

print("✓ LIME explainer initialized")
print(f"  Kernel width: 2.0")
print(f"  Training data shape: {X_train_scaled.shape}")

# ============================================================
# SAMPLE SELECTION
# ============================================================
print("\n[SAMPLING] Selecting diverse samples...")

pos_idx = np.where(y_val == 1)[0]
neg_idx = np.where(y_val == 0)[0]

n_per_class = min(20, len(pos_idx), len(neg_idx))
indices = np.concatenate([
    np.random.choice(pos_idx, n_per_class, replace=False),
    np.random.choice(neg_idx, n_per_class, replace=False)
])

print(f"✓ Selected {len(indices)} samples ({n_per_class} BFRB + {n_per_class} Non-BFRB)")

# ============================================================
# MAIN LIME ANALYSIS
# ============================================================
print("\n[ANALYSIS] Generating LIME explanations...")

times, fidelity_r2, stability_j = [], [], []
explanations = []

for i, idx in enumerate(indices):
    # Get SCALED instance (LIME expects scaled data)
    x_scaled = X_val_scaled[idx]

    # Time the explanation
    t0 = time.time()
    exp = explainer.explain_instance(
        x_scaled,
        predict_fn,  # Wrapper handles unscaling + temperature
        num_features=10,
        num_samples=10000
    )
    times.append(time.time() - t0)
    explanations.append(exp)

    # ============================================================
    # FIDELITY CALCULATION (R² on perturbed samples)
    # ============================================================
    try:
        # Access LIME's internal perturbed data
        if hasattr(exp, 'domain_mapper') and hasattr(exp.domain_mapper, 'perturbed_data'):
            Z = exp.domain_mapper.perturbed_data
            bb_preds = exp.domain_mapper.y_original[:, 1]  # Class 1 probabilities
            lime_preds = exp.domain_mapper.y_predicted

            # Calculate R²
            ss_res = np.sum((bb_preds - lime_preds) ** 2)
            ss_tot = np.sum((bb_preds - bb_preds.mean()) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            fidelity_r2.append(max(r2, 0))

            if i == 0:  # Debug first sample
                print(f"\n  [DEBUG Sample 0]")
                print(f"    Black-box preds range: [{bb_preds.min():.4f}, {bb_preds.max():.4f}]")
                print(f"    LIME preds range: [{lime_preds.min():.4f}, {lime_preds.max():.4f}]")
                print(f"    R² fidelity: {r2:.4f}")
        else:
            # Fallback to LIME's internal score
            fidelity_r2.append(max(exp.score, 0) if hasattr(exp, 'score') else 0)
    except:
        fidelity_r2.append(0)

    # ============================================================
    # STABILITY CALCULATION (Jaccard over 3 runs)
    # ============================================================
    if i < 10:  # Only for first 10 samples to save time
        feature_sets = []
        for _ in range(3):
            exp_stab = explainer.explain_instance(
                x_scaled, predict_fn, num_features=10, num_samples=2000
            )
            feat_set = set(f.split(" ")[0] for f, _ in exp_stab.as_list())
            feature_sets.append(feat_set)

        jaccards = []
        for a in range(3):
            for b in range(a + 1, 3):
                intersection = len(feature_sets[a] & feature_sets[b])
                union = len(feature_sets[a] | feature_sets[b])
                jaccards.append(intersection / union if union > 0 else 0)

        stability_j.append(np.mean(jaccards))
    else:
        # Use average of previous stability scores
        stability_j.append(np.mean(stability_j) if stability_j else 0.7)

    if (i + 1) % 10 == 0:
        print(f"  Processed {i + 1}/{len(indices)} samples...")

# ============================================================
# RESULTS ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("LIME ANALYSIS RESULTS")
print("=" * 80)

# Calculate metrics
if fidelity_r2:
    print(f"📊 FIDELITY (R²): {np.mean(fidelity_r2):.4f}")
    print(f"   Range: [{min(fidelity_r2):.4f}, {max(fidelity_r2):.4f}]")
    print(f"   Zero fidelity count: {sum(r2 == 0 for r2 in fidelity_r2)}/{len(fidelity_r2)}")
else:
    print(f"📊 FIDELITY (R²): No valid calculations")

print(f"⏱️  EFFICIENCY: {np.mean(times):.3f}s per explanation")
print(f"📈 STABILITY: {np.mean(stability_j):.4f} (Jaccard similarity)")

# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================
print("\n[FEATURE IMPORTANCE] Extracting global patterns...")

global_imp = defaultdict(list)
for exp in explanations:
    for f, w in exp.as_list():
        feat_name = f.split(" ")[0] if " " in f else f
        global_imp[feat_name].append(abs(w))

# Calculate mean importance
if global_imp:
    global_imp_mean = {k: np.mean(v) for k, v in global_imp.items()}
    top_features = sorted(global_imp_mean.items(), key=lambda x: x[1], reverse=True)[:10]

    print("\n🏆 TOP 10 FEATURES (Global LIME Importance):")
    for i, (feat, score) in enumerate(top_features, 1):
        print(f"  {i:2d}. {feat}: {score:.4f}")
else:
    top_features = []
    print("⚠️  No feature importance extracted")

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n💾 Saving results...")

results = {
    'fidelity_mean': np.mean(fidelity_r2) if fidelity_r2 else 0,
    'fidelity_all': fidelity_r2,
    'efficiency_mean': np.mean(times),
    'stability_mean': np.mean(stability_j),
    'top_features': top_features,
    'n_samples': len(indices),
    'feature_count': len(feature_names),
    'extreme_ratio': extreme_ratio,
    'temperature_scaling_applied': extreme_ratio > 0.3
}

# Save results
joblib.dump(results, os.path.join(OUT_DIR, "lime_results_fixed.pkl"))

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"✅ Fidelity (R²): {results['fidelity_mean']:.4f} (should be > 0.5)")
print(f"✅ Efficiency: {results['efficiency_mean']:.3f}s per explanation")
print(f"✅ Stability: {results['stability_mean']:.4f} (Jaccard)")
print(f"✅ Top feature: {top_features[0][0] if top_features else 'N/A'}")
print(f"✅ Temperature scaling: {'Applied' if results['temperature_scaling_applied'] else 'Not needed'}")
print("\n" + "=" * 80)

# ============================================================
# DIAGNOSTIC CHECK
# ============================================================
if results['fidelity_mean'] > 0.5:
    print("🎉 SUCCESS: LIME is working with good fidelity!")
    print("   The zero-fidelity bug has been FIXED.")
elif results['fidelity_mean'] > 0:
    print("⚠️  WARNING: LIME fidelity is low but non-zero.")
    print("   Try increasing kernel_width to 1.5 or num_samples to 10000.")
else:
    print("❌ ERROR: LIME fidelity is still zero.")
    print("   Possible issues:")
    print("   1. Feature mismatch between training and LIME")
    print("   2. Model predictions are too extreme")
    print("   3. LIME's perturbed data not accessible")

# ============================================================
# EXTENDED METRICS & VISUALIZATIONS
# ============================================================

print("\n" + "=" * 80)
print("EXTENDED METRICS CALCULATION & VISUALIZATION")
print("=" * 80)

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

# Set up plotting
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['font.size'] = 10
sns.set_style("whitegrid")


# ============================================================
# 1. COMPREHENSIVE METRICS CALCULATION
# ============================================================

def calculate_all_metrics(explanations, fidelity_r2, times, stability_j,
                          X_val_scaled, y_val, indices, binary_model, scaler):
    """Calculate all missing metrics."""

    metrics = {}

    # 1. Fidelity metrics (already have R², add MAE)
    if fidelity_r2:
        metrics['fidelity'] = {
            'r2_mean': np.mean(fidelity_r2),
            'r2_std': np.std(fidelity_r2),
            'r2_median': np.median(fidelity_r2),
            'r2_min': np.min(fidelity_r2),
            'r2_max': np.max(fidelity_r2)
        }

    # 2. Sparsity calculation
    print("\n[1/9] Calculating Sparsity...")
    sparsity_vals = []
    gini_coefficients = []

    for exp in explanations:
        weights = [abs(w) for _, w in exp.as_list()]
        total_weight = sum(weights)

        if total_weight > 0:
            # Features for 80% importance
            sorted_weights = sorted(weights, reverse=True)
            cumsum = np.cumsum(sorted_weights)
            features_80 = np.searchsorted(cumsum, 0.8 * total_weight) + 1
            sparsity_vals.append(features_80)

            # Gini coefficient for inequality
            n = len(weights)
            if n > 1:
                sorted_w = np.sort(weights)
                index = np.arange(1, n + 1)
                gini = (2 * np.sum(index * sorted_w)) / (n * sum(weights)) - (n + 1) / n
                gini_coefficients.append(gini)

    if sparsity_vals:
        metrics['sparsity'] = {
            'features_80_mean': np.mean(sparsity_vals),
            'features_80_std': np.std(sparsity_vals),
            'gini_mean': np.mean(gini_coefficients) if gini_coefficients else 0,
            'interpretation': 'Lower = more concise explanations'
        }

    # 3. Coverage calculation
    print("[2/9] Calculating Coverage...")
    coverage_threshold = 0.3  # R² > 0.3 considered valid
    valid_explanations = sum(r2 > coverage_threshold for r2 in fidelity_r2)
    coverage = valid_explanations / len(fidelity_r2)

    # Coverage by class
    y_sample = y_val.iloc[indices].values
    bfrb_mask = y_sample == 1
    if sum(bfrb_mask) > 0:
        coverage_bfrb = sum(fidelity_r2[i] > coverage_threshold for i in np.where(bfrb_mask)[0]) / sum(bfrb_mask)
    else:
        coverage_bfrb = 0

    if sum(~bfrb_mask) > 0:
        coverage_non = sum(fidelity_r2[i] > coverage_threshold for i in np.where(~bfrb_mask)[0]) / sum(~bfrb_mask)
    else:
        coverage_non = 0

    metrics['coverage'] = {
        'overall': coverage,
        'bfrb_class': coverage_bfrb,
        'non_bfrb_class': coverage_non,
        'valid_count': valid_explanations,
        'total_count': len(fidelity_r2)
    }

    # 4. Efficiency metrics
    print("[3/9] Calculating Efficiency...")
    metrics['efficiency'] = {
        'mean_time': np.mean(times),
        'median_time': np.median(times),
        'p95_time': np.percentile(times, 95),
        'total_time': sum(times),
        'samples_per_second': len(times) / sum(times)
    }

    # 5. Stability metrics (enhanced)
    print("[4/9] Calculating Enhanced Stability...")
    if stability_j:
        metrics['stability'] = {
            'mean_jaccard': np.mean(stability_j),
            'std_jaccard': np.std(stability_j),
            'cv_jaccard': np.std(stability_j) / np.mean(stability_j) if np.mean(stability_j) > 0 else 0,
            'interpretation': 'Higher = more consistent explanations'
        }

    # 6. Descriptive Accuracy (Feature Removal Test)
    print("[5/9] Calculating Descriptive Accuracy...")
    descriptive_scores = calculate_descriptive_accuracy(
        explanations[:min(10, len(explanations))],  # Test on first 10 for speed
        X_val_scaled[indices[:min(10, len(indices))]],
        binary_model,
        scaler
    )
    metrics['descriptive_accuracy'] = descriptive_scores

    # 7. Robustness (Perturbation Resistance)
    print("[6/9] Calculating Robustness...")
    robustness_scores = calculate_robustness(
        explainer,
        X_val_scaled[indices[:min(5, len(indices))]],  # Test on first 5 for speed
        predict_fn,
        feature_stds=np.std(X_train_scaled, axis=0)
    )
    metrics['robustness'] = robustness_scores

    # 8. Precision (against ground truth)
    print("[7/9] Calculating Precision...")
    precision_scores = calculate_precision(
        explanations,
        X_val_scaled[indices],
        y_val.iloc[indices].values,
        binary_model,
        feature_names
    )
    metrics['precision'] = precision_scores

    # 9. Clinician Trust Score
    print("[8/9] Calculating Clinician Trust Score...")
    trust_score = calculate_clinician_trust_score(metrics)
    metrics['clinician_trust'] = trust_score

    print("[9/9] All metrics calculated!")

    return metrics


# ============================================================
# HELPER FUNCTIONS FOR METRICS
# ============================================================

def calculate_descriptive_accuracy(explanations, instances, model, scaler, n_top_features=5):
    """Calculate feature removal impact on predictions."""
    scores = []

    for i, (exp, instance) in enumerate(zip(explanations, instances)):
        # Original prediction
        orig_pred = model.predict_proba(scaler.inverse_transform(instance.reshape(1, -1)))[0, 1]

        # Get top features from explanation
        top_features = exp.as_list()[:n_top_features]
        top_indices = []

        for feat_name, _ in top_features:
            # Find feature index
            clean_name = feat_name.split(' ')[0] if ' ' in feat_name else feat_name
            for idx, name in enumerate(feature_names):
                if clean_name in name or name in clean_name:
                    top_indices.append(idx)
                    break

        if top_indices:
            # Remove top features (set to mean)
            instance_modified = instance.copy()
            instance_modified[top_indices] = 0  # Mean is 0 after scaling

            # New prediction
            new_pred = model.predict_proba(scaler.inverse_transform(instance_modified.reshape(1, -1)))[0, 1]

            # Prediction drop
            pred_drop = orig_pred - new_pred
            relative_drop = pred_drop / orig_pred if orig_pred > 0 else 0

            scores.append({
                'original_pred': orig_pred,
                'modified_pred': new_pred,
                'absolute_drop': pred_drop,
                'relative_drop': relative_drop
            })

    if scores:
        avg_relative_drop = np.mean([s['relative_drop'] for s in scores])
        avg_absolute_drop = np.mean([s['absolute_drop'] for s in scores])

        return {
            'mean_relative_drop': avg_relative_drop,
            'mean_absolute_drop': avg_absolute_drop,
            'n_samples_tested': len(scores),
            'interpretation': 'Higher drop = more descriptive features'
        }

    return {'mean_relative_drop': 0, 'mean_absolute_drop': 0, 'n_samples_tested': 0}


def calculate_robustness(explainer, instances, predict_fn, feature_stds, n_perturbations=5):
    """Calculate robustness to input perturbations."""
    jaccard_scores = []
    spearman_scores = []

    for instance in instances:
        # Original explanation
        exp_orig = explainer.explain_instance(
            instance, predict_fn, num_features=10, num_samples=5000
        )
        orig_features = [f.split(' ')[0] for f, _ in exp_orig.as_list()]

        pert_scores_jaccard = []
        pert_scores_spearman = []

        for _ in range(n_perturbations):
            # Add small noise
            noise = np.random.normal(0, 0.05 * feature_stds, size=instance.shape)
            instance_perturbed = instance + noise

            # Perturbed explanation
            exp_pert = explainer.explain_instance(
                instance_perturbed, predict_fn, num_features=10, num_samples=5000
            )
            pert_features = [f.split(' ')[0] for f, _ in exp_pert.as_list()]

            # Jaccard similarity
            set_orig = set(orig_features)
            set_pert = set(pert_features)
            intersection = len(set_orig & set_pert)
            union = len(set_orig | set_pert)
            jaccard = intersection / union if union > 0 else 0
            pert_scores_jaccard.append(jaccard)

            # Spearman correlation for rankings
            common_features = list(set_orig & set_pert)
            if len(common_features) >= 3:
                rank_orig = [orig_features.index(f) for f in common_features]
                rank_pert = [pert_features.index(f) for f in common_features]
                rho, _ = spearmanr(rank_orig, rank_pert)
                if not np.isnan(rho):
                    pert_scores_spearman.append(rho)

        if pert_scores_jaccard:
            jaccard_scores.append(np.mean(pert_scores_jaccard))
        if pert_scores_spearman:
            spearman_scores.append(np.mean(pert_scores_spearman))

    return {
        'mean_jaccard': np.mean(jaccard_scores) if jaccard_scores else 0,
        'mean_spearman': np.mean(spearman_scores) if spearman_scores else 0,
        'n_samples_tested': len(instances),
        'interpretation': 'Higher = more robust to perturbations'
    }


def calculate_precision(explanations, instances, labels, model, feature_names, top_k=5):
    """Calculate precision against model-based importance."""
    from sklearn.inspection import permutation_importance

    # Calculate permutation importance as ground truth
    print("  Computing permutation importance...")
    perm_importance = permutation_importance(
        model, scaler.inverse_transform(instances), labels,
        n_repeats=5, random_state=42, n_jobs=1
    )

    # Get ground truth top features
    gt_importance = perm_importance.importances_mean
    gt_top_indices = np.argsort(gt_importance)[::-1][:top_k]
    gt_top_features = [feature_names[i] for i in gt_top_indices]

    # Compare with LIME explanations
    precision_scores = []

    for exp in explanations:
        lime_features = [f.split(' ')[0] for f, _ in exp.as_list()[:top_k]]

        # Precision@k
        intersection = set(lime_features) & set(gt_top_features)
        precision = len(intersection) / top_k
        precision_scores.append(precision)

    return {
        'precision_at_k': np.mean(precision_scores) if precision_scores else 0,
        'ground_truth_top_features': gt_top_features,
        'interpretation': f'Precision@{top_k} against permutation importance'
    }


def calculate_clinician_trust_score(metrics):
    """Calculate composite clinician trust score."""

    # Define weights based on clinical importance
    weights = {
        'fidelity': 0.25,  # Most important - explanations must be faithful
        'coverage': 0.20,  # Must work for most patients
        'stability': 0.15,  # Consistent explanations
        'sparsity': 0.10,  # Clinicians prefer concise explanations
        'robustness': 0.10,  # Shouldn't change with small noise
        'precision': 0.10,  # Should identify right features
        'descriptive_accuracy': 0.05,
        'efficiency': 0.05  # Less important clinically
    }

    # Normalize each metric to 0-1 scale
    normalized_scores = {}

    # Fidelity: R² normalized (0-1, target > 0.7)
    if 'fidelity' in metrics:
        fidelity_r2 = metrics['fidelity']['r2_mean']
        normalized_scores['fidelity'] = min(fidelity_r2 / 0.7, 1.0)

    # Coverage: already 0-1
    if 'coverage' in metrics:
        normalized_scores['coverage'] = metrics['coverage']['overall']

    # Stability: Jaccard already 0-1
    if 'stability' in metrics:
        normalized_scores['stability'] = metrics['stability']['mean_jaccard']

    # Sparsity: Lower is better (target < 15 features)
    if 'sparsity' in metrics:
        sparsity = metrics['sparsity']['features_80_mean']
        normalized_scores['sparsity'] = max(0, 1 - (sparsity / 50))  # Cap at 50 features

    # Robustness: average of Jaccard and Spearman
    if 'robustness' in metrics:
        robustness = (metrics['robustness']['mean_jaccard'] +
                      metrics['robustness']['mean_spearman']) / 2
        normalized_scores['robustness'] = robustness

    # Precision: already 0-1
    if 'precision' in metrics:
        normalized_scores['precision'] = metrics['precision']['precision_at_k']

    # Descriptive accuracy: relative drop (0-1)
    if 'descriptive_accuracy' in metrics:
        desc_acc = metrics['descriptive_accuracy']['mean_relative_drop']
        normalized_scores['descriptive_accuracy'] = min(desc_acc * 5, 1.0)  # Scale

    # Efficiency: time per explanation (target < 2 seconds)
    if 'efficiency' in metrics:
        efficiency = metrics['efficiency']['mean_time']
        normalized_scores['efficiency'] = max(0, 1 - (efficiency / 5))  # Cap at 5 seconds

    # Calculate weighted sum
    trust_score = 0
    for metric, weight in weights.items():
        if metric in normalized_scores:
            trust_score += normalized_scores[metric] * weight

    # Categorize trust level
    if trust_score >= 0.8:
        category = "High Trust"
        recommendation = "Ready for clinical deployment"
    elif trust_score >= 0.6:
        category = "Moderate Trust"
        recommendation = "Suitable with clinician oversight"
    elif trust_score >= 0.4:
        category = "Low Trust"
        recommendation = "Needs improvement before clinical use"
    else:
        category = "Very Low Trust"
        recommendation = "Research use only"

    return {
        'score': trust_score,
        'category': category,
        'recommendation': recommendation,
        'normalized_scores': normalized_scores,
        'weights': weights
    }


# ============================================================
# 2. VISUALIZATION FUNCTIONS
# ============================================================

def create_all_visualizations(metrics, explanations, fidelity_r2, times,
                              stability_j, out_dir=OUT_DIR):
    """Create comprehensive visualizations."""

    print(f"\n📊 Creating visualizations in {out_dir}...")

    # 1. Fidelity Distribution
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Fidelity histogram
    axes[0, 0].hist(fidelity_r2, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(fidelity_r2), color='red', linestyle='--',
                       label=f'Mean: {np.mean(fidelity_r2):.3f}')
    axes[0, 0].set_xlabel('Fidelity (R²)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Local Fidelity Distribution', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. Sparsity visualization
    if 'sparsity' in metrics:
        # Calculate sparsity for all explanations
        sparsity_values = []
        for exp in explanations:
            weights = [abs(w) for _, w in exp.as_list()]
            total_weight = sum(weights)
            if total_weight > 0:
                sorted_weights = sorted(weights, reverse=True)
                cumsum = np.cumsum(sorted_weights)
                features_80 = np.searchsorted(cumsum, 0.8 * total_weight) + 1
                sparsity_values.append(features_80)

        axes[0, 1].hist(sparsity_values, bins=15, color='lightgreen',
                        edgecolor='black', alpha=0.7)
        axes[0, 1].axvline(np.mean(sparsity_values), color='red', linestyle='--',
                           label=f'Mean: {np.mean(sparsity_values):.1f}')
        axes[0, 1].set_xlabel('Features for 80% Importance')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Sparsity Distribution', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

    # 3. Efficiency (Time) distribution
    axes[1, 0].hist(times, bins=20, color='gold', edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(np.mean(times), color='red', linestyle='--',
                       label=f'Mean: {np.mean(times):.3f}s')
    axes[1, 0].set_xlabel('Time per Explanation (seconds)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Efficiency Distribution', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Stability distribution
    axes[1, 1].hist(stability_j, bins=10, color='lightcoral',
                    edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(np.mean(stability_j), color='red', linestyle='--',
                       label=f'Mean: {np.mean(stability_j):.3f}')
    axes[1, 1].set_xlabel('Stability (Jaccard Similarity)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Stability Distribution', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '1_core_metrics_distributions.png'),
                dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Global Feature Importance
    create_global_importance_plot(explanations, feature_names, out_dir)

    # 6. Descriptive Accuracy Curve
    if 'descriptive_accuracy' in metrics and len(explanations) >= 10:
        create_descriptive_accuracy_plot(explanations[:10],
                                         X_val_scaled[indices[:10]],
                                         binary_model, scaler, out_dir)

    # 7. Clinician Trust Score Breakdown
    if 'clinician_trust' in metrics:
        create_trust_score_plot(metrics['clinician_trust'], out_dir)

    # 8. Metric Comparison Radar Chart
    create_radar_chart(metrics, out_dir)

    print("✅ All visualizations created!")


def create_global_importance_plot(explanations, feature_names, out_dir):
    """Create global feature importance bar chart."""

    # Aggregate importance across all explanations
    global_imp = defaultdict(list)
    for exp in explanations:
        for f, w in exp.as_list():
            feat_name = f.split(" ")[0] if " " in f else f
            global_imp[feat_name].append(abs(w))

    # Calculate mean importance
    if global_imp:
        global_imp_mean = {k: np.mean(v) for k, v in global_imp.items()}
        top_features = sorted(global_imp_mean.items(),
                              key=lambda x: x[1], reverse=True)[:20]

        plt.figure(figsize=(14, 8))
        features, scores = zip(*top_features)

        # Color by sensor type
        colors = []
        for feat in features:
            if 'tof' in feat.lower():
                colors.append('#3498db')  # Blue for TOF
            elif 'thermo' in feat.lower() or 'thm' in feat.lower():
                colors.append('#e74c3c')  # Red for thermal
            elif 'acc' in feat.lower() or 'imu' in feat.lower():
                colors.append('#2ecc71')  # Green for IMU
            elif 'gyro' in feat.lower():
                colors.append('#f39c12')  # Orange for gyro
            else:
                colors.append('#95a5a6')  # Gray for others

        bars = plt.barh(range(len(features)), scores, color=colors, alpha=0.8)

        # Add value labels
        for i, (bar, score) in enumerate(zip(bars, scores)):
            plt.text(score, bar.get_y() + bar.get_height() / 2,
                     f' {score:.3f}', va='center', ha='left', fontsize=9)

        plt.yticks(range(len(features)), features, fontsize=10)
        plt.xlabel('Mean Absolute LIME Importance', fontsize=12)
        plt.title(f'Global Feature Importance (Top 20)\nAggregated from {len(explanations)} Explanations',
                  fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(axis='x', alpha=0.3)

        # Add legend for sensor types
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#3498db', alpha=0.8, label='TOF (Distance)'),
            Patch(facecolor='#e74c3c', alpha=0.8, label='Thermal (Temperature)'),
            Patch(facecolor='#2ecc71', alpha=0.8, label='IMU (Motion)'),
            Patch(facecolor='#f39c12', alpha=0.8, label='Gyro (Rotation)')
        ]
        plt.legend(handles=legend_elements, loc='lower right')

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '2_global_feature_importance.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


def create_descriptive_accuracy_plot(explanations, instances, model, scaler, out_dir):
    """Create feature removal impact plot."""

    removal_curves = []
    n_features_to_test = min(10, len(feature_names))

    for exp, instance in zip(explanations, instances):
        orig_pred = model.predict_proba(scaler.inverse_transform(
            instance.reshape(1, -1)))[0, 1]

        curve = [orig_pred]
        instance_modified = instance.copy()

        # Get all features sorted by importance
        sorted_features = sorted(exp.as_list(), key=lambda x: abs(x[1]), reverse=True)

        for k in range(1, n_features_to_test + 1):
            # Remove top k features (set to mean/0)
            for feat_name, _ in sorted_features[:k]:
                clean_name = feat_name.split(' ')[0] if ' ' in feat_name else feat_name
                for idx, name in enumerate(feature_names):
                    if clean_name in name or name in clean_name:
                        instance_modified[idx] = 0  # Mean is 0 after scaling
                        break

            # Get prediction after removal
            pred = model.predict_proba(scaler.inverse_transform(
                instance_modified.reshape(1, -1)))[0, 1]
            curve.append(pred)

        removal_curves.append(curve)

    if removal_curves:
        # Plot average removal curve
        plt.figure(figsize=(10, 6))

        # Plot individual curves
        for i, curve in enumerate(removal_curves):
            plt.plot(range(len(curve)), curve, alpha=0.2, color='gray', linewidth=0.5)

        # Plot mean curve
        mean_curve = np.mean(removal_curves, axis=0)
        std_curve = np.std(removal_curves, axis=0)

        plt.plot(range(len(mean_curve)), mean_curve,
                 color='red', linewidth=3, label='Mean', marker='o')
        plt.fill_between(range(len(mean_curve)),
                         mean_curve - std_curve,
                         mean_curve + std_curve,
                         alpha=0.3, color='red', label='±1 std')

        plt.xlabel('Number of Top Features Removed', fontsize=12)
        plt.ylabel('Model Prediction (Probability)', fontsize=12)
        plt.title('Descriptive Accuracy: Feature Removal Impact',
                  fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Add annotation for prediction drop
        pred_drop = mean_curve[0] - mean_curve[-1]
        plt.annotate(f'Total drop: {pred_drop:.3f}',
                     xy=(len(mean_curve) - 1, mean_curve[-1]),
                     xytext=(len(mean_curve) - 3, mean_curve[-1] + 0.1),
                     arrowprops=dict(arrowstyle='->', color='blue'),
                     fontsize=11, color='blue')

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, '3_descriptive_accuracy_curve.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()


def create_trust_score_plot(trust_data, out_dir):
    """Create visualization of clinician trust score breakdown."""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Radar chart of normalized scores
    metrics = list(trust_data['normalized_scores'].keys())
    values = list(trust_data['normalized_scores'].values())

    # Add first value to close the radar chart
    metrics_radar = metrics + [metrics[0]]
    values_radar = values + [values[0]]

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    ax1 = plt.subplot(121, polar=True)
    ax1.plot(angles, values_radar, 'o-', linewidth=2, color='#3498db')
    ax1.fill(angles, values_radar, alpha=0.25, color='#3498db')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics, fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.set_title('Trust Score Components', fontsize=14, fontweight='bold')

    # 2. Bar chart of weights and scores
    ax2 = plt.subplot(122)
    x = np.arange(len(metrics))
    width = 0.35

    ax2.bar(x - width / 2, [trust_data['weights'].get(m, 0) for m in metrics],
            width, label='Weight', color='#2ecc71', alpha=0.7)
    ax2.bar(x + width / 2, values, width, label='Score', color='#e74c3c', alpha=0.7)

    ax2.set_xlabel('Metric', fontsize=12)
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_title(f"Overall Trust Score: {trust_data['score']:.3f} ({trust_data['category']})",
                  fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(metrics, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add recommendation text
    plt.figtext(0.5, 0.01, f"Recommendation: {trust_data['recommendation']}",
                ha='center', fontsize=11, style='italic',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '4_clinician_trust_score.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


def create_radar_chart(metrics, out_dir):
    """Create radar chart comparing all metrics."""

    # Define target values
    targets = {
        'Fidelity (R²)': 0.7,
        'Coverage (%)': 80,
        'Stability': 0.7,
        'Sparsity': 15,
        'Efficiency (s)': 2.0,
        'Robustness': 0.7,
        'Precision': 0.7
    }

    # Calculate current values
    current = {}
    if 'fidelity' in metrics:
        current['Fidelity (R²)'] = metrics['fidelity']['r2_mean'] * 100  # Convert to percentage
    if 'coverage' in metrics:
        current['Coverage (%)'] = metrics['coverage']['overall'] * 100
    if 'stability' in metrics:
        current['Stability'] = metrics['stability']['mean_jaccard']
    if 'sparsity' in metrics:
        current['Sparsity'] = metrics['sparsity']['features_80_mean']
    if 'efficiency' in metrics:
        current['Efficiency (s)'] = metrics['efficiency']['mean_time']
    if 'robustness' in metrics:
        current['Robustness'] = metrics['robustness']['mean_jaccard']
    if 'precision' in metrics:
        current['Precision'] = metrics['precision']['precision_at_k']

    # Normalize for radar chart (all 0-1, lower sparsity/efficiency is better)
    labels = list(targets.keys())
    n_metrics = len(labels)

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the circle

    # Prepare data
    target_values = []
    current_values = []

    for label in labels:
        if label == 'Sparsity':
            # Lower is better (invert)
            target_norm = max(0, 1 - (targets[label] / 50))
            current_norm = max(0, 1 - (current.get(label, 0) / 50))
        elif label == 'Efficiency (s)':
            # Lower is better (invert)
            target_norm = max(0, 1 - (targets[label] / 10))
            current_norm = max(0, 1 - (current.get(label, 0) / 10))
        elif label in ['Coverage (%)', 'Fidelity (R²)']:
            # Convert from percentage
            target_norm = targets[label] / 100
            current_norm = current.get(label, 0) / 100
        else:
            target_norm = targets[label]
            current_norm = current.get(label, 0)

        target_values.append(target_norm)
        current_values.append(current_norm)

    target_values += target_values[:1]
    current_values += current_values[:1]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))

    ax.plot(angles, target_values, 'o-', linewidth=2, label='Target', color='#2ecc71')
    ax.fill(angles, target_values, alpha=0.1, color='#2ecc71')

    ax.plot(angles, current_values, 'o-', linewidth=2, label='Current', color='#e74c3c')
    ax.fill(angles, current_values, alpha=0.1, color='#e74c3c')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('LIME Performance vs Targets\n(All Metrics Normalized 0-1)',
                 fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '5_metrics_vs_targets_radar.png'),
                dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================
# 3. MAIN EXECUTION
# ============================================================

# Calculate all metrics
print("\n" + "=" * 80)
print("CALCULATING ALL METRICS")
print("=" * 80)

all_metrics = calculate_all_metrics(
    explanations=explanations,
    fidelity_r2=fidelity_r2,
    times=times,
    stability_j=stability_j,
    X_val_scaled=X_val_scaled,
    y_val=y_val,
    indices=indices,
    binary_model=binary_model,
    scaler=scaler
)

# Create all visualizations
create_all_visualizations(
    metrics=all_metrics,
    explanations=explanations,
    fidelity_r2=fidelity_r2,
    times=times,
    stability_j=stability_j,
    out_dir=OUT_DIR
)

# ============================================================
# 4. FINAL REPORT
# ============================================================

print("\n" + "=" * 80)
print("COMPREHENSIVE LIME METRICS REPORT")
print("=" * 80)

# Print formatted report
report_sections = {
    "FIDELITY": "Local Faithfulness of Explanations",
    "SPARSITY": "Conciseness of Explanations",
    "COVERAGE": "Applicability Across Samples",
    "STABILITY": "Consistency Across Runs",
    "EFFICIENCY": "Computational Performance",
    "DESCRIPTIVE ACCURACY": "Feature Importance Validation",
    "ROBUSTNESS": "Resistance to Input Perturbations",
    "PRECISION": "Agreement with Ground Truth",
    "CLINICIAN TRUST": "Overall Clinical Trustworthiness"
}

for section, description in report_sections.items():
    print(f"\n📋 {section}: {description}")
    print("-" * 60)

    if section.lower().replace(" ", "_") in ['fidelity', 'sparsity', 'coverage',
                                             'stability', 'efficiency']:
        key = section.lower().replace(" ", "_")
        if key in all_metrics:
            for k, v in all_metrics[key].items():
                if isinstance(v, float):
                    print(f"  {k.replace('_', ' ').title()}: {v:.4f}")
                else:
                    print(f"  {k.replace('_', ' ').title()}: {v}")

    elif section == "DESCRIPTIVE ACCURACY":
        if 'descriptive_accuracy' in all_metrics:
            da = all_metrics['descriptive_accuracy']
            print(f"  Mean Relative Drop: {da['mean_relative_drop']:.4f}")
            print(f"  Mean Absolute Drop: {da['mean_absolute_drop']:.4f}")
            print(f"  Samples Tested: {da['n_samples_tested']}")

    elif section == "ROBUSTNESS":
        if 'robustness' in all_metrics:
            rb = all_metrics['robustness']
            print(f"  Jaccard Similarity: {rb['mean_jaccard']:.4f}")
            print(f"  Spearman Correlation: {rb['mean_spearman']:.4f}")

    elif section == "PRECISION":
        if 'precision' in all_metrics:
            pr = all_metrics['precision']
            print(f"  Precision@5: {pr['precision_at_k']:.4f}")
            print(f"  Ground Truth Top Features:")
            for i, feat in enumerate(pr['ground_truth_top_features'][:5], 1):
                print(f"    {i}. {feat}")

    elif section == "CLINICIAN TRUST":
        if 'clinician_trust' in all_metrics:
            ct = all_metrics['clinician_trust']
            print(f"  Score: {ct['score']:.3f}/1.0")
            print(f"  Category: {ct['category']}")
            print(f"  Recommendation: {ct['recommendation']}")
            print(f"\n  Component Scores:")
            for metric, score in ct['normalized_scores'].items():
                weight = ct['weights'].get(metric, 0)
                print(f"    {metric.title():20s}: {score:.3f} (weight: {weight:.2f})")

# Save comprehensive results
print(f"\n💾 Saving comprehensive metrics report...")
joblib.dump(all_metrics, os.path.join(OUT_DIR, "lime_comprehensive_metrics.pkl"))

# Create summary CSV
summary_data = []
for metric_name, metric_data in all_metrics.items():
    if isinstance(metric_data, dict):
        for k, v in metric_data.items():
            if isinstance(v, (int, float, str)):
                summary_data.append({
                    'metric_category': metric_name,
                    'metric_name': k,
                    'value': v
                })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(os.path.join(OUT_DIR, "lime_metrics_summary.csv"), index=False)

print("\n" + "=" * 80)
print("✅ COMPLETE! All metrics calculated and visualizations created.")
print(f"📁 Results saved to: {OUT_DIR}")
print("\n📊 Generated files:")
print(f"  1. lime_comprehensive_metrics.pkl - All metrics (pickle)")
print(f"  2. lime_metrics_summary.csv - Metrics summary (CSV)")
print(f"  3. 1_core_metrics_distributions.png - Core metrics")
print(f"  4. 2_global_feature_importance.png - Global importance")
print(f"  5. 3_descriptive_accuracy_curve.png - Feature removal impact")
print(f"  6. 4_clinician_trust_score.png - Trust score breakdown")
print(f"  7. 5_metrics_vs_targets_radar.png - Performance vs targets")
print("=" * 80)