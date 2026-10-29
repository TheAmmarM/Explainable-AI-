# ============================================================
# COMPREHENSIVE LIME EXPLAINABILITY WITH RESEARCH METRICS
# BFRB Detection Model - Full XAI Evaluation
# ============================================================

import os
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import defaultdict
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, log_loss
)
from lime import lime_tabular
from lime.submodular_pick import SubmodularPick

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_DIR = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\XAI_Work\BFRB_XAI\Trained_Models"
DATA_FOLDER = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\CMI_Sensor_Kaggle"
OUTPUT_DIR = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\XAI_Work\BFRB_XAI\Figures_and_Graphs\Complete_LIME_Results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: LOAD MODELS AND DATA
# ============================================================
print("=" * 80)
print("LIME EXPLAINABILITY ANALYSIS - RESEARCH-GRADE EVALUATION")
print("=" * 80)
print("\n[1/10] Loading Models and Data...")

# Load models
binary_model = joblib.load(os.path.join(MODEL_DIR, "binary_classifier.pkl"))
gesture_model = joblib.load(os.path.join(MODEL_DIR, "gesture_classifier.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
metadata = joblib.load(os.path.join(MODEL_DIR, "metadata.pkl"))

print(f"✓ Binary model loaded")
print(f"✓ Gesture model loaded")
print(f"✓ Feature names: {len(feature_names)} features")

# Load and prepare data
df = pd.read_csv(os.path.join(DATA_FOLDER, "train.csv"))
df["gesture"] = df["gesture"].fillna("non_target")

# Feature engineering (match training)
if all(col in df.columns for col in ['acc_x', 'acc_y', 'acc_z']):
    df['imu_magnitude'] = np.sqrt(df['acc_x'] ** 2 + df['acc_y'] ** 2 + df['acc_z'] ** 2)

if all(col in df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
    df['gyro_magnitude'] = np.sqrt(df['gyro_x'] ** 2 + df['gyro_y'] ** 2 + df['gyro_z'] ** 2)

tof_cols = [col for col in df.columns if col.startswith('tof_')]
if tof_cols:
    df['tof_mean'] = df[tof_cols].mean(axis=1)

thermo_cols = [col for col in df.columns if 'thermo_' in col or 'thm_' in col]
if thermo_cols:
    df['thermo_mean'] = df[thermo_cols].mean(axis=1)

# Aggregate
numeric_features = df.select_dtypes(include=[np.number]).columns.drop("sequence_id", errors="ignore")
numeric_features = [c for c in numeric_features if c != 'row_id']

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
X = full_df.drop(columns=["sequence_id", "gesture", "binary_label"])
y_binary = full_df["binary_label"]

X = X[feature_names].fillna(X[feature_names].median())

# Train-val split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(X, y_binary))
X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]
y_val = y_binary.iloc[val_idx]

print(f"✓ Data prepared: {len(X_train)} train, {len(X_val)} validation samples")

# ============================================================
# BASELINE PERFORMANCE METRICS
# ============================================================
print("\n[2/10] Computing Baseline Model Performance...")

y_pred_proba = binary_model.predict_proba(X_val)[:, 1]
y_pred = (y_pred_proba > 0.45).astype(int)

baseline_metrics = {
    'Accuracy': accuracy_score(y_val, y_pred),
    'Precision': precision_score(y_val, y_pred),
    'Recall': recall_score(y_val, y_pred),
    'F1-Score': f1_score(y_val, y_pred),
    'AUC-ROC': roc_auc_score(y_val, y_pred_proba),
    'Log Loss': log_loss(y_val, y_pred_proba)
}

print("\nBASELINE MODEL PERFORMANCE:")
print("-" * 60)
for metric, value in baseline_metrics.items():
    print(f"  {metric:12s}: {value:.4f}")

# ============================================================
# STEP 2: INITIALIZE LIME EXPLAINER
# ============================================================
print("\n[3/10] Initializing LIME Explainer...")

explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=['Non-BFRB', 'BFRB'],
    mode='classification',
    discretize_continuous=True,
    random_state=42
)

print("✓ LIME explainer initialized with discretization enabled")

# ============================================================
# STEP 3: STRATIFIED SAMPLE SELECTION
# ============================================================
print("\n[4/10] Selecting Diverse Samples for Explanation...")

np.random.seed(42)
bfrb_samples = np.where(y_val == 1)[0]
non_bfrb_samples = np.where(y_val == 0)[0]

n_per_class = min(20, len(bfrb_samples), len(non_bfrb_samples))
sample_indices = np.concatenate([
    np.random.choice(bfrb_samples, n_per_class, replace=False),
    np.random.choice(non_bfrb_samples, n_per_class, replace=False)
])
n_samples = len(sample_indices)

print(f"✓ Selected {n_samples} samples ({n_per_class} BFRB + {n_per_class} Non-BFRB)")

# ============================================================
# STEP 4: GENERATE LIME EXPLANATIONS
# ============================================================
print("\n[5/10] Generating LIME Explanations...")

explanations = []
explanation_times = []
predicted_labels = []
true_labels = []
confidences = []

for i, idx in enumerate(sample_indices):
    sample = X_val.iloc[idx].values
    true_label = y_val.iloc[idx]

    # Time the explanation generation (Efficiency metric)
    start_time = time.time()
    exp = explainer.explain_instance(
        data_row=sample,
        predict_fn=binary_model.predict_proba,
        num_features=10,
        num_samples=5000
    )
    elapsed = time.time() - start_time

    explanations.append(exp)
    explanation_times.append(elapsed)

    # Get predictions
    pred_proba = binary_model.predict_proba(sample.reshape(1, -1))[0]
    pred_class = 1 if pred_proba[1] > 0.45 else 0

    predicted_labels.append(pred_class)
    true_labels.append(true_label)
    confidences.append(pred_proba[pred_class])

    if (i + 1) % 10 == 0:
        print(f"  Progress: {i + 1}/{n_samples} explanations generated...")

print(f"✓ Generated {len(explanations)} explanations")
print(f"  Average time per explanation: {np.mean(explanation_times):.3f}s")
print(f"  Total time: {np.sum(explanation_times):.2f}s")

# ============================================================
# METRIC 1: EFFICIENCY
# ============================================================
print("\n[6/10] Computing EFFICIENCY Metrics...")

efficiency_stats = {
    'Mean Time (s)': np.mean(explanation_times),
    'Median Time (s)': np.median(explanation_times),
    'Std Time (s)': np.std(explanation_times),
    'Min Time (s)': np.min(explanation_times),
    'Max Time (s)': np.max(explanation_times),
    'Total Time (s)': np.sum(explanation_times)
}

print("\nEFFICIENCY METRICS:")
print("-" * 60)
for metric, value in efficiency_stats.items():
    print(f"  {metric:20s}: {value:.4f}")

# Visualization
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(explanation_times, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(np.mean(explanation_times), color='red', linestyle='--',
           label=f'Mean: {np.mean(explanation_times):.3f}s')
ax.set_xlabel('Explanation Time (seconds)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('LIME Explanation Generation Time Distribution', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '1_efficiency_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()
print("✓ Efficiency plot saved")

# ============================================================
# METRIC 2: FIDELITY (FAITHFULNESS)
# ============================================================
print("\n[7/10] Computing FIDELITY Metrics...")

fidelity_scores = []

for i, exp in enumerate(explanations):
    sample = X_val.iloc[sample_indices[i]].values

    # Black-box prediction
    blackbox_pred = binary_model.predict_proba(sample.reshape(1, -1))[0]

    # LIME local model prediction
    lime_pred = exp.local_pred[0] if hasattr(exp, 'local_pred') else None

    if lime_pred is not None:
        # Fidelity: How close is LIME's surrogate to the black-box?
        fidelity = 1 - abs(blackbox_pred[1] - lime_pred)
        fidelity_scores.append(fidelity)

fidelity_mean = np.mean(fidelity_scores) if fidelity_scores else 0.0
fidelity_std = np.std(fidelity_scores) if fidelity_scores else 0.0

print("\nFIDELITY METRICS:")
print("-" * 60)
print(f"  Mean Fidelity: {fidelity_mean:.4f}")
print(f"  Std Fidelity:  {fidelity_std:.4f}")
if fidelity_scores:
    print(f"  Min Fidelity:  {np.min(fidelity_scores):.4f}")
    print(f"  Max Fidelity:  {np.max(fidelity_scores):.4f}")

# Visualization
if fidelity_scores:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(fidelity_scores, bins=20, color='forestgreen', alpha=0.7, edgecolor='black')
    ax.axvline(fidelity_mean, color='red', linestyle='--',
               label=f'Mean: {fidelity_mean:.4f}')
    ax.set_xlabel('Fidelity Score', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('LIME Fidelity: How Well Surrogate Matches Black-Box',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_fidelity_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Fidelity plot saved")

# ============================================================
# METRIC 3: STABILITY (CONSISTENCY)
# ============================================================
print("\n[8/10] Computing STABILITY Metrics...")

stability_scores = []
n_stability_tests = 5  # Generate 5 explanations per sample

# Test on subset for computational efficiency
stability_sample_indices = sample_indices[:min(10, len(sample_indices))]

for idx in stability_sample_indices:
    sample = X_val.iloc[idx].values
    feature_importances = []

    for _ in range(n_stability_tests):
        exp = explainer.explain_instance(
            data_row=sample,
            predict_fn=binary_model.predict_proba,
            num_features=10,
            num_samples=5000
        )

        # Extract feature importance dictionary
        importance_dict = {}
        for feat, weight in exp.as_list():
            feat_name = feat.split(' ')[0] if ' ' in feat else feat
            importance_dict[feat_name] = weight
        feature_importances.append(importance_dict)

    # Calculate variance in feature importance across runs
    all_features = set()
    for imp_dict in feature_importances:
        all_features.update(imp_dict.keys())

    feature_variances = []
    for feat in all_features:
        values = [imp_dict.get(feat, 0) for imp_dict in feature_importances]
        feature_variances.append(np.std(values))

    # Stability: Lower variance = higher stability
    stability = 1 - np.mean(feature_variances) if feature_variances else 0
    stability_scores.append(max(0, stability))  # Ensure non-negative

stability_mean = np.mean(stability_scores) if stability_scores else 0.0
stability_std = np.std(stability_scores) if stability_scores else 0.0

print("\nSTABILITY METRICS:")
print("-" * 60)
print(f"  Mean Stability: {stability_mean:.4f}")
print(f"  Std Stability:  {stability_std:.4f}")
print(f"  (Higher = more consistent explanations across runs)")

# Visualization
if stability_scores:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(range(len(stability_scores)), stability_scores, color='coral', alpha=0.7)
    ax.axhline(stability_mean, color='red', linestyle='--',
               label=f'Mean: {stability_mean:.4f}')
    ax.set_xlabel('Sample Index', fontsize=12)
    ax.set_ylabel('Stability Score', fontsize=12)
    ax.set_title('LIME Stability: Consistency Across Repeated Explanations',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3_stability_scores.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Stability plot saved")

# ============================================================
# METRIC 4: SPARSITY
# ============================================================
print("\n[9/10] Computing SPARSITY Metrics...")

sparsity_scores = []

for exp in explanations:
    feature_weights = exp.as_list()
    total_weight = sum(abs(w) for _, w in feature_weights)

    if total_weight > 0:
        # Cumulative weight distribution
        sorted_weights = sorted([abs(w) for _, w in feature_weights], reverse=True)
        cumsum = np.cumsum(sorted_weights)

        # How many features needed to reach 80% of total importance?
        threshold = 0.8 * total_weight
        n_features_80 = np.searchsorted(cumsum, threshold) + 1
        sparsity_scores.append(n_features_80)

sparsity_mean = np.mean(sparsity_scores) if sparsity_scores else 0.0

print("\nSPARSITY METRICS:")
print("-" * 60)
print(f"  Avg Features for 80% Importance: {sparsity_mean:.2f}")
print(f"  Total Available Features: {len(feature_names)}")
print(f"  Sparsity Ratio: {(sparsity_mean / len(feature_names)):.4f}")

# Visualization
if sparsity_scores:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sparsity_scores, bins=15, color='purple', alpha=0.7, edgecolor='black')
    ax.axvline(sparsity_mean, color='red', linestyle='--',
               label=f'Mean: {sparsity_mean:.2f}')
    ax.set_xlabel('Number of Features for 80% Importance', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('LIME Sparsity: Feature Concentration',
                 fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '4_sparsity_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Sparsity plot saved")

# ============================================================
# EXTRACT GLOBAL FEATURE IMPORTANCE
# ============================================================
print("\n[Processing] Extracting Global Feature Importance...")

global_importance = defaultdict(list)
for exp in explanations:
    for feature, weight in exp.as_list():
        # Extract clean feature name (remove LIME's discretization info)
        feat_name = feature.split(' ')[0] if ' ' in feature else feature
        # Map to original feature names
        matching_features = [f for f in feature_names if feat_name in f or f in feat_name]
        if matching_features:
            global_importance[matching_features[0]].append(abs(weight))
        else:
            global_importance[feat_name].append(abs(weight))

global_importance_mean = {
    feat: np.mean(scores) for feat, scores in global_importance.items() if scores
}

if not global_importance_mean:
    print("⚠ Warning: No feature importance extracted. Using fallback method...")
    # Fallback: Use first explanation's features
    for feature, weight in explanations[0].as_list():
        feat_name = feature.split(' ')[0] if ' ' in feature else feature
        global_importance_mean[feat_name] = abs(weight)

top_features = sorted(global_importance_mean.items(),
                      key=lambda x: x[1], reverse=True)[:20]
top_feature_names = [f for f, _ in top_features[:10]]

print(f"✓ Extracted importance for {len(global_importance_mean)} features")
print(f"  Top feature: {top_features[0][0] if top_features else 'N/A'}")

# ============================================================
# METRIC 5: DESCRIPTIVE ACCURACY
# ============================================================
print("\n[10/10] Computing DESCRIPTIVE ACCURACY...")

if len(top_feature_names) > 0:
    # Remove top features that exist in X_val
    features_to_remove = [f for f in top_feature_names if f in X_val.columns]

    if features_to_remove:
        # Create a copy with SAME feature shape
        X_val_reduced = X_val.copy()

        # Neutralize top features instead of dropping them
        for feat in features_to_remove:
            if feat in X_val_reduced.columns:
                X_val_reduced[feat] = X_val_reduced[feat].median()

        # Predict with intact feature space
        y_pred_reduced = (
                binary_model.predict_proba(X_val_reduced)[:, 1] > 0.45
        ).astype(int)

        original_accuracy = baseline_metrics['Accuracy']
        reduced_accuracy = accuracy_score(y_val, y_pred_reduced)
        accuracy_drop = original_accuracy - reduced_accuracy
    else:
        accuracy_drop = 0.0
        reduced_accuracy = baseline_metrics['Accuracy']
else:
    accuracy_drop = 0.0
    reduced_accuracy = baseline_metrics['Accuracy']

original_accuracy = baseline_metrics['Accuracy']

print("\nDESCRIPTIVE ACCURACY:")
print("-" * 60)
print(f"  Original Accuracy: {original_accuracy:.4f}")
print(f"  Accuracy without Top Features: {reduced_accuracy:.4f}")
print(f"  Accuracy Drop: {accuracy_drop:.4f}")
print(f"  Relative Drop: {(accuracy_drop / original_accuracy * 100):.2f}%")

if accuracy_drop > 0.1:
    print("  ✓ LIME correctly identified critical features (large drop)")
elif accuracy_drop > 0.05:
    print("  ✓ LIME identified moderately important features")
else:
    print("  ⚠ LIME may not have captured most important features")

# ============================================================
# GLOBAL FEATURE IMPORTANCE VISUALIZATION
# ============================================================
print("\n[Visualizations] Generating Global Importance Plots...")

if top_features:
    plt.figure(figsize=(12, 8))
    features, scores = zip(*top_features)
    colors = ['#2ecc71' if s > np.median(scores) else '#e74c3c' for s in scores]
    plt.barh(range(len(features)), scores, color=colors, alpha=0.8)
    plt.yticks(range(len(features)), features, fontsize=10)
    plt.xlabel('Mean Absolute LIME Importance', fontsize=12)
    plt.title(f'Global Feature Importance (LIME)\nAggregated from {n_samples} Explanations',
              fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '5_global_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Global importance plot saved")

# ============================================================
# LOCAL CASE STUDIES
# ============================================================
print("\n[Visualizations] Generating Local Case Study Plots...")

# Find interesting cases
correct_bfrb_idx = [i for i, (t, p) in enumerate(zip(true_labels, predicted_labels))
                    if t == 1 and p == 1]
correct_non_idx = [i for i, (t, p) in enumerate(zip(true_labels, predicted_labels))
                   if t == 0 and p == 0]
false_pos_idx = [i for i, (t, p) in enumerate(zip(true_labels, predicted_labels))
                 if t == 0 and p == 1]

cases = []
if correct_bfrb_idx:
    cases.append((correct_bfrb_idx[0], "Correct_BFRB_Detection"))
if correct_non_idx:
    cases.append((correct_non_idx[0], "Correct_NonBFRB_Detection"))
if false_pos_idx:
    cases.append((false_pos_idx[0], "False_Positive_Error"))

for idx, label in cases:
    fig = explanations[idx].as_pyplot_figure()
    fig.suptitle(f'LIME Explanation: {label}\nTrue={true_labels[idx]}, '
                 f'Pred={predicted_labels[idx]}, Conf={confidences[idx]:.3f}',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'6_case_study_{label}.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
print(f"✓ Saved {len(cases)} case study plots")

# ============================================================
# CLASS-SPECIFIC ANALYSIS
# ============================================================
print("\n[Visualizations] Generating Class-Specific Analysis...")

bfrb_importance = defaultdict(list)
non_bfrb_importance = defaultdict(list)

for i, exp in enumerate(explanations):
    true_label = true_labels[i]
    target_dict = bfrb_importance if true_label == 1 else non_bfrb_importance

    for feature, weight in exp.as_list():
        feat_name = feature.split(' ')[0] if ' ' in feature else feature
        target_dict[feat_name].append(abs(weight))

bfrb_mean = {f: np.mean(s) for f, s in bfrb_importance.items() if s}
non_bfrb_mean = {f: np.mean(s) for f, s in non_bfrb_importance.items() if s}

if bfrb_mean and non_bfrb_mean:
    bfrb_top = sorted(bfrb_mean.items(), key=lambda x: x[1], reverse=True)[:10]
    non_bfrb_top = sorted(non_bfrb_mean.items(), key=lambda x: x[1], reverse=True)[:10]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # BFRB
    features_b, scores_b = zip(*bfrb_top)
    axes[0].barh(range(len(features_b)), scores_b, color='steelblue', alpha=0.8)
    axes[0].set_yticks(range(len(features_b)))
    axes[0].set_yticklabels(features_b, fontsize=10)
    axes[0].set_xlabel('LIME Importance', fontsize=12)
    axes[0].set_title('Top 10 Features - BFRB Class', fontsize=13, fontweight='bold')
    axes[0].invert_yaxis()
    axes[0].grid(axis='x', alpha=0.3)

    # Non-BFRB
    features_n, scores_n = zip(*non_bfrb_top)
    axes[1].barh(range(len(features_n)), scores_n, color='coral', alpha=0.8)
    axes[1].set_yticks(range(len(features_n)))
    axes[1].set_yticklabels(features_n, fontsize=10)
    axes[1].set_xlabel('LIME Importance', fontsize=12)
    axes[1].set_title('Top 10 Features - Non-BFRB Class', fontsize=13, fontweight='bold')
    axes[1].invert_yaxis()
    axes[1].grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '7_class_specific_importance.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Class-specific plot saved")

# ============================================================
# FEATURE HEATMAP
# ============================================================
print("\n[Visualizations] Generating Feature Interaction Heatmap...")

if top_features:
    top_n_features = min(15, len(top_features))
    top_features_list = [f for f, _ in top_features[:top_n_features]]

    lime_matrix = np.zeros((min(30, len(explanations)), len(top_features_list)))

    for i, exp in enumerate(explanations[:30]):
        exp_dict = {feat.split(' ')[0]: weight for feat, weight in exp.as_list()}
        for j, feat in enumerate(top_features_list):
            lime_matrix[i, j] = exp_dict.get(feat, 0)

    plt.figure(figsize=(14, 10))
    sns.heatmap(lime_matrix,
                xticklabels=top_features_list,
                yticklabels=[f"S{i + 1}" for i in range(lime_matrix.shape[0])],
                cmap='RdYlGn', center=0, annot=False,
                cbar_kws={'label': 'LIME Weight'})
    plt.title('LIME Feature Weights Across Samples\n(Green=Pro-BFRB, Red=Anti-BFRB)',
              fontsize=14, fontweight='bold')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Samples', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '8_feature_heatmap.png'),
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Feature heatmap saved")

# ============================================================
# METRIC COMPARISON TABLE
# ============================================================
print("\n[Final Report] Generating Metric Comparison Table...")

metrics_summary = pd.DataFrame({
    'Metric': [
        'Efficiency (Mean Time)',
        'Fidelity (Mean)',
        'Stability (Mean)',
        'Sparsity (Avg Features)',
        'Descriptive Accuracy Drop',
        'Baseline Accuracy',
        'Baseline F1-Score',
        'Baseline AUC-ROC'
    ],
    'Value': [
        f"{efficiency_stats['Mean Time (s)']:.4f}s",
        f"{fidelity_mean:.4f}",
        f"{stability_mean:.4f}",
        f"{sparsity_mean:.2f} / {len(feature_names)}",
        f"{accuracy_drop:.4f} ({(accuracy_drop / original_accuracy * 100):.2f}%)",
        f"{baseline_metrics['Accuracy']:.4f}",
        f"{baseline_metrics['F1-Score']:.4f}",
        f"{baseline_metrics['AUC-ROC']:.4f}"
    ],
    'Interpretation': [
        'Time to generate one explanation',
        'How well surrogate matches black-box',
        'Consistency across repeated runs',
        'Features needed for 80% importance',
        'Impact of removing top features',
        'Original model performance',
        'Original model F1-score',
        'Original model AUC-ROC'
    ]
})

print("\n" + "=" * 80)
print("COMPREHENSIVE LIME EVALUATION SUMMARY")
print("=" * 80)
print(metrics_summary.to_string(index=False))
print("=" * 80)

# Save to CSV
metrics_summary.to_csv(os.path.join(OUTPUT_DIR, 'lime_metrics_summary.csv'), index=False)
print("\n✓ Metrics summary saved to CSV")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("LIME ANALYSIS COMPLETE")
print("=" * 80)
print(f"✓ Analyzed {n_samples} samples ({n_per_class} per class)")
print(f"✓ Generated {len(explanations)} explanations")
print(f"✓ Computed 5 research-grade metrics:")
print(f"    1. Efficiency:  {efficiency_stats['Mean Time (s)']:.3f}s per explanation")
print(f"    2. Fidelity:    {fidelity_mean:.4f} (surrogate accuracy)")
print(f"    3. Stability:   {stability_mean:.4f} (consistency)")
print(f"    4. Sparsity:    {sparsity_mean:.2f} features for 80% importance")
print(f"    5. Descriptive: {accuracy_drop:.4f} accuracy drop")
if top_features:
    print(f"\n✓ Top Feature: {top_features[0][0]}")
print(f"✓ All results saved to: {OUTPUT_DIR}")
print("=" * 80)

# Save final summary
final_summary = {
    'n_samples': n_samples,
    'n_explanations': len(explanations),
    'baseline_metrics': baseline_metrics,
    'efficiency': efficiency_stats,
    'fidelity_mean': fidelity_mean,
    'stability_mean': stability_mean,
    'sparsity_mean': sparsity_mean,
    'descriptive_accuracy_drop': accuracy_drop,
    'top_10_features': top_feature_names,
    'global_importance': dict(top_features) if top_features else {}
}

joblib.dump(final_summary, os.path.join(OUTPUT_DIR, 'lime_final_summary.pkl'))
print("\n✓ Final summary saved to lime_final_summary.pkl")