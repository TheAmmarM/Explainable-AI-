# ============================================================
# IMPROVED FULL LIME EXPLAINABILITY ANALYSIS
# SHAP-Comparable Metrics + Advanced Visualizations
# ============================================================

import os
import warnings
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from collections import defaultdict, Counter
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, log_loss, confusion_matrix, mean_absolute_error
)
from sklearn.calibration import calibration_curve
from lime import lime_tabular
from scipy.stats import entropy, spearmanr
from tqdm import tqdm

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_DIR = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\XAI_Work\BFRB_XAI\Trained_Models"
DATA_FOLDER = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\CMI_Sensor_Kaggle"
OUTPUT_DIR = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\XAI_Work\BFRB_XAI\Figures_and_Graphs\Full_LIME_Results"

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Binary"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Multiclass"), exist_ok=True)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def calculate_mae(y_true, y_pred_proba):
    """Mean Absolute Error for probability predictions"""
    return mean_absolute_error(y_true, y_pred_proba)


def calculate_ece(y_true, y_pred_proba, n_bins=10):
    """
    Expected Calibration Error (ECE)
    Measures how well predicted probabilities match actual outcomes
    """
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins, strategy='uniform')

    # Calculate bin weights
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bin_edges[1:-1])

    ece = 0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_accuracy = y_true[mask].mean()
            bin_confidence = y_pred_proba[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece += bin_weight * abs(bin_accuracy - bin_confidence)

    return ece


def calculate_lime_auc(explanations, X_val, y_val, model, feature_names):
    """
    LIME-AUC: How well do LIME feature importances predict model performance?
    Similar to SHAP-AUC
    """
    # Get global feature importance from LIME
    global_importance = defaultdict(list)
    for exp in explanations:
        for feature, weight in exp.as_list():
            feat_name = feature.split(' ')[0] if ' ' in feature else feature
            matching = [f for f in feature_names if feat_name in f or f in feat_name]
            if matching:
                global_importance[matching[0]].append(abs(weight))

    importance_scores = {feat: np.mean(scores) for feat, scores in global_importance.items()}

    # Rank features by importance
    ranked_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

    # Calculate AUC by progressively masking features
    auc_scores = []
    n_steps = min(20, len(ranked_features))

    for i in range(1, n_steps + 1):
        # Mask top i features
        X_masked = X_val.copy()
        for feat_name, _ in ranked_features[:i]:
            if feat_name in X_masked.columns:
                X_masked[feat_name] = X_masked[feat_name].median()

        # Predict and calculate accuracy
        if hasattr(model, 'predict_proba'):
            y_pred = (model.predict_proba(X_masked)[:, 1] > 0.45).astype(int)
        else:
            y_pred = model.predict(X_masked)

        acc = accuracy_score(y_val, y_pred)
        auc_scores.append(acc)

    # Calculate AUC (area under accuracy degradation curve)
    baseline_acc = accuracy_score(y_val, model.predict(X_val) if not hasattr(model, 'predict_proba')
    else (model.predict_proba(X_val)[:, 1] > 0.45).astype(int))

    lime_auc = np.trapz(auc_scores, dx=1.0) / (n_steps * baseline_acc)

    return lime_auc, auc_scores, ranked_features


def calculate_clinician_trust_detailed(explanations, y_true, y_pred, y_pred_proba, feature_names):
    """
    Enhanced Clinical Trust Score with discrimination metric
    """
    correct_mask = (y_true == y_pred)
    error_mask = ~correct_mask

    # Trust score for correct predictions
    trust_correct = 0.0
    if correct_mask.sum() > 0:
        # Higher confidence on correct = better
        trust_correct = y_pred_proba[correct_mask].mean()

    # Trust score for errors
    trust_errors = 0.0
    if error_mask.sum() > 0:
        # Lower confidence on errors = better (model knows it's uncertain)
        trust_errors = 1 - y_pred_proba[error_mask].mean()

    # Overall trust (weighted by frequency)
    n_correct = correct_mask.sum()
    n_errors = error_mask.sum()
    n_total = len(y_true)

    trust_mean = (n_correct * trust_correct + n_errors * trust_errors) / n_total

    # Discrimination: Can the model distinguish correct from incorrect?
    discrimination = abs(trust_correct - (1 - trust_errors))

    return {
        'trust_mean': trust_mean,
        'trust_correct': trust_correct,
        'trust_errors': 1 - trust_errors,  # Report as original confidence
        'discrimination': discrimination
    }


def extract_feature_importance(exp, feature_names):
    """Extract clean feature importance from LIME explanation"""
    importance_dict = {}
    for feature, weight in exp.as_list():
        feat_name = feature.split(' ')[0] if ' ' in feature else feature
        matching = [f for f in feature_names if feat_name in f or f in feat_name]
        if matching:
            importance_dict[matching[0]] = weight
        else:
            importance_dict[feat_name] = weight
    return importance_dict


# ============================================================
# STEP 1: LOAD MODELS AND DATA
# ============================================================
print("=" * 80)
print("IMPROVED LIME EXPLAINABILITY ANALYSIS")
print("SHAP-Comparable Metrics + Advanced Visualizations")
print("=" * 80)
print("\n[1/15] Loading models and data...")

# Load models
binary_model = joblib.load(os.path.join(MODEL_DIR, "binary_classifier.pkl"))
gesture_model = joblib.load(os.path.join(MODEL_DIR, "gesture_classifier.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
metadata = joblib.load(os.path.join(MODEL_DIR, "metadata.pkl"))

# Load data first
df = pd.read_csv(os.path.join(DATA_FOLDER, "train.csv"))
df["gesture"] = df["gesture"].fillna("non_target")

# Extract gesture classes
if 'gesture_classes' in metadata:
    gesture_classes = metadata['gesture_classes']
elif 'bfrb_gestures' in metadata:
    gesture_classes = metadata['bfrb_gestures']
else:
    gesture_classes = sorted([g for g in df['gesture'].unique() if g != 'non_target'])

# Feature engineering
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

gesture_to_idx = {g: i for i, g in enumerate(gesture_classes)}
labels_df["gesture_label"] = labels_df["gesture"].map(gesture_to_idx)

full_df = pd.merge(agg_df, labels_df, on="sequence_id")
X = full_df.drop(columns=["sequence_id", "gesture", "binary_label", "gesture_label"])
y_binary = full_df["binary_label"]
y_gesture = full_df["gesture_label"]

X = X[feature_names].fillna(X[feature_names].median())

# Train-val split
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(X, y_binary))
X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]
y_binary_val = y_binary.iloc[val_idx]
y_gesture_val = y_gesture.iloc[val_idx]

print(f"Analyzing: {len(X_val)} samples")

# ============================================================
# PART A: BINARY MODEL ANALYSIS
# ============================================================
print("=" * 80)
print("PART A: BINARY MODEL ANALYSIS")
print("=" * 80)

# ============================================================
# STEP 2: BINARY BASELINE METRICS
# ============================================================
print(f"[2/15] Binary: Computing baseline metrics...", end=' ')

y_binary_pred_proba = binary_model.predict_proba(X_val)[:, 1]
y_binary_pred = (y_binary_pred_proba > 0.45).astype(int)

binary_baseline = {
    'Accuracy': accuracy_score(y_binary_val, y_binary_pred),
    'Precision': precision_score(y_binary_val, y_binary_pred),
    'Recall': recall_score(y_binary_val, y_binary_pred),
    'F1-Score': f1_score(y_binary_val, y_binary_pred),
    'AUC-ROC': roc_auc_score(y_binary_val, y_binary_pred_proba),
    'Log Loss': log_loss(y_binary_val, y_binary_pred_proba),
    'MAE': calculate_mae(y_binary_val, y_binary_pred_proba)
}

print(f"Accuracy: {binary_baseline['Accuracy']:.4f} "
      f"Precision: {binary_baseline['Precision']:.4f} "
      f"Recall: {binary_baseline['Recall']:.4f} "
      f"F1-Score: {binary_baseline['F1-Score']:.4f} "
      f"AUC-ROC: {binary_baseline['AUC-ROC']:.4f} "
      f"Log Loss: {binary_baseline['Log Loss']:.4f} "
      f"MAE: {binary_baseline['MAE']:.4f}")

# ============================================================
# STEP 3: INITIALIZE BINARY LIME
# ============================================================
print(f"[3/15] Binary: Initializing LIME...", end=' ')

binary_explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=['Non-BFRB', 'BFRB'],
    mode='classification',
    discretize_continuous=True,
    random_state=42
)

print("Done")

# ============================================================
# STEP 4: GENERATE BINARY LIME EXPLANATIONS
# ============================================================
print(f"[4/15] Binary: Generating LIME explanations...")

binary_explanations = []
binary_times = []

for i in tqdm(range(len(X_val)), desc="Binary LIME", ncols=80):
    sample = X_val.iloc[i].values

    start_time = time.time()
    exp = binary_explainer.explain_instance(
        data_row=sample,
        predict_fn=binary_model.predict_proba,
        num_features=10,
        num_samples=2000
    )
    elapsed = time.time() - start_time

    binary_explanations.append(exp)
    binary_times.append(elapsed)

print(f"LIME generation time: {np.sum(binary_times) / 60:.2f} minutes")

# ============================================================
# STEP 5: BINARY FIDELITY
# ============================================================
print(f"[5/15] Binary: Computing fidelity...", end=' ')

binary_fidelity_scores = []
for i, exp in enumerate(binary_explanations):
    sample = X_val.iloc[i].values
    blackbox_pred = binary_model.predict_proba(sample.reshape(1, -1))[0]
    lime_pred = exp.local_pred[0] if hasattr(exp, 'local_pred') else None

    if lime_pred is not None:
        fidelity = 1 - abs(blackbox_pred[1] - lime_pred)
        binary_fidelity_scores.append(fidelity)

binary_fidelity = np.mean(binary_fidelity_scores)
print(f"Fidelity: {binary_fidelity:.4f}")

# ============================================================
# STEP 6: BINARY STABILITY
# ============================================================
print(f"[6/15] Binary: Computing stability...")

binary_stability_scores = []
n_stability_samples = min(100, len(X_val))
stability_indices = np.random.choice(len(X_val), n_stability_samples, replace=False)

for idx in tqdm(stability_indices, desc="Stability test", ncols=80):
    sample = X_val.iloc[idx].values
    feature_importances = []

    for _ in range(5):
        exp = binary_explainer.explain_instance(
            data_row=sample,
            predict_fn=binary_model.predict_proba,
            num_features=10,
            num_samples=2000
        )
        importance_dict = extract_feature_importance(exp, feature_names)
        feature_importances.append(importance_dict)

    # Calculate variance
    all_features = set()
    for imp_dict in feature_importances:
        all_features.update(imp_dict.keys())

    feature_variances = []
    for feat in all_features:
        values = [imp_dict.get(feat, 0) for imp_dict in feature_importances]
        feature_variances.append(np.std(values))

    stability = 1 - np.mean(feature_variances) if feature_variances else 0
    binary_stability_scores.append(max(0, stability))

binary_stability = np.mean(binary_stability_scores)
print(f"Stability: {binary_stability:.4f}")

# ============================================================
# STEP 7: BINARY ADVANCED METRICS
# ============================================================
print(f"[7/15] Binary: Computing advanced metrics...", end=' ')

# LIME-AUC
binary_lime_auc, binary_auc_curve, binary_ranked_features = calculate_lime_auc(
    binary_explanations, X_val, y_binary_val, binary_model, feature_names
)

# ECE
binary_ece = calculate_ece(y_binary_val, y_binary_pred_proba)

print(f"LIME-AUC: {binary_lime_auc:.4f} ECE: {binary_ece:.4f}")

# ============================================================
# STEP 8: BINARY CLINICIAN TRUST SCORE
# ============================================================
print(f"[8/15] Binary: Computing Clinician Trust Score...", end=' ')

binary_trust = calculate_clinician_trust_detailed(
    binary_explanations, y_binary_val, y_binary_pred, y_binary_pred_proba, feature_names
)

print(f"Trust (mean): {binary_trust['trust_mean']:.4f} "
      f"Trust (correct): {binary_trust['trust_correct']:.4f} "
      f"Trust (errors): {binary_trust['trust_errors']:.4f} "
      f"Discrimination: {binary_trust['discrimination']:.4f}")

# ============================================================
# PART B: GESTURE MODEL ANALYSIS
# ============================================================
print("=" * 80)
print("PART B: GESTURE MODEL ANALYSIS (8-class)")
print("=" * 80)

bfrb_mask = y_binary_val == 1
X_val_bfrb = X_val[bfrb_mask]
y_gesture_val_bfrb = y_gesture_val[bfrb_mask]

print(f"BFRB samples: {len(X_val_bfrb)}")

# ============================================================
# STEP 9: GESTURE BASELINE METRICS
# ============================================================
print(f"[9/15] Gesture: Computing baseline metrics...", end=' ')

y_gesture_pred = gesture_model.predict(X_val_bfrb)
y_gesture_pred_proba = gesture_model.predict_proba(X_val_bfrb)

gesture_baseline = {
    'Accuracy': accuracy_score(y_gesture_val_bfrb, y_gesture_pred),
    'Macro F1': f1_score(y_gesture_val_bfrb, y_gesture_pred, average='macro'),
    'Weighted F1': f1_score(y_gesture_val_bfrb, y_gesture_pred, average='weighted')
}

print(f"Accuracy: {gesture_baseline['Accuracy']:.4f} "
      f"Macro F1: {gesture_baseline['Macro F1']:.4f} "
      f"Weighted F1: {gesture_baseline['Weighted F1']:.4f}")

# ============================================================
# STEP 10: INITIALIZE GESTURE LIME
# ============================================================
print(f"[10/15] Gesture: Initializing LIME...", end=' ')

X_train_bfrb = X_train[y_binary.iloc[train_idx] == 1]

multiclass_explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train_bfrb.values,
    feature_names=feature_names,
    class_names=gesture_classes,
    mode='classification',
    discretize_continuous=True,
    random_state=42
)

print("Done")

# ============================================================
# STEP 11: GENERATE GESTURE LIME EXPLANATIONS
# ============================================================
print(f"[11/15] Gesture: Generating LIME explanations...")

multiclass_explanations = []
multiclass_times = []

for i in tqdm(range(len(X_val_bfrb)), desc="Gesture LIME", ncols=80):
    sample = X_val_bfrb.iloc[i].values

    start_time = time.time()
    exp = multiclass_explainer.explain_instance(
        data_row=sample,
        predict_fn=gesture_model.predict_proba,
        num_features=10,
        num_samples=2000,
        top_labels=len(gesture_classes)
    )
    elapsed = time.time() - start_time

    multiclass_explanations.append(exp)
    multiclass_times.append(elapsed)

print(f"LIME generation time: {np.sum(multiclass_times) / 60:.2f} minutes")

# ============================================================
# STEP 12: GESTURE FIDELITY
# ============================================================
print(f"[12/15] Gesture: Computing fidelity...", end=' ')

multiclass_fidelity_scores = []
for i, exp in enumerate(multiclass_explanations):
    sample = X_val_bfrb.iloc[i].values
    pred_class = y_gesture_pred[i]

    blackbox_pred = gesture_model.predict_proba(sample.reshape(1, -1))[0]

    if hasattr(exp, 'local_pred'):
        lime_pred = exp.local_pred
        if isinstance(lime_pred, (list, np.ndarray)) and len(lime_pred) > pred_class:
            fidelity = 1 - abs(blackbox_pred[pred_class] - lime_pred[pred_class])
            multiclass_fidelity_scores.append(fidelity)

multiclass_fidelity = np.mean(multiclass_fidelity_scores) if multiclass_fidelity_scores else 0
print(f"Fidelity: {multiclass_fidelity:.4f}")

# ============================================================
# STEP 13: GESTURE STABILITY
# ============================================================
print(f"[13/15] Gesture: Computing stability...")

multiclass_stability_scores = []
n_mc_stability = min(100, len(X_val_bfrb))
mc_stability_indices = np.random.choice(len(X_val_bfrb), n_mc_stability, replace=False)

for idx in tqdm(mc_stability_indices, desc="Stability test", ncols=80):
    sample = X_val_bfrb.iloc[idx].values
    feature_importances = []

    for _ in range(5):
        exp = multiclass_explainer.explain_instance(
            data_row=sample,
            predict_fn=gesture_model.predict_proba,
            num_features=10,
            num_samples=2000,
            top_labels=len(gesture_classes)
        )
        importance_dict = extract_feature_importance(exp, feature_names)
        feature_importances.append(importance_dict)

    all_features = set()
    for imp_dict in feature_importances:
        all_features.update(imp_dict.keys())

    feature_variances = []
    for feat in all_features:
        values = [imp_dict.get(feat, 0) for imp_dict in feature_importances]
        feature_variances.append(np.std(values))

    stability = 1 - np.mean(feature_variances) if feature_variances else 0
    multiclass_stability_scores.append(max(0, stability))

multiclass_stability = np.mean(multiclass_stability_scores)
print(f"Stability: {multiclass_stability:.4f}")

# ============================================================
# STEP 14: GESTURE ADVANCED METRICS
# ============================================================
print(f"[14/15] Gesture: Computing advanced metrics...", end=' ')

# LIME-AUC for multiclass
multiclass_lime_auc, _, _ = calculate_lime_auc(
    multiclass_explanations, X_val_bfrb, y_gesture_val_bfrb, gesture_model, feature_names
)

# ECE for multiclass (use max probability)
gesture_max_proba = y_gesture_pred_proba.max(axis=1)
gesture_correct = (y_gesture_pred == y_gesture_val_bfrb).astype(int)
multiclass_ece = calculate_ece(gesture_correct, gesture_max_proba)

print(f"LIME-AUC: {multiclass_lime_auc:.4f} ECE: {multiclass_ece:.4f}")

# ============================================================
# STEP 15: GESTURE CLINICIAN TRUST SCORE
# ============================================================
print(f"[15/15] Gesture: Computing Clinician Trust Score...", end=' ')

multiclass_trust = calculate_clinician_trust_detailed(
    multiclass_explanations, y_gesture_val_bfrb, y_gesture_pred, gesture_max_proba, feature_names
)

print(f"Trust (mean): {multiclass_trust['trust_mean']:.4f} "
      f"Trust (correct): {multiclass_trust['trust_correct']:.4f} "
      f"Trust (errors): {multiclass_trust['trust_errors']:.4f} "
      f"Discrimination: {multiclass_trust['discrimination']:.4f}")

# ============================================================
# ADVANCED VISUALIZATIONS
# ============================================================
print("\n[Visualizations] Generating advanced plots...")

# 1. DESCRIPTIVE ACCURACY CURVES (Binary)
fig, ax = plt.subplots(figsize=(10, 6))
n_features_range = range(1, len(binary_ranked_features[:20]) + 1)
ax.plot(n_features_range, binary_auc_curve, marker='o', linewidth=2, markersize=6)
ax.axhline(binary_baseline['Accuracy'], color='red', linestyle='--', label='Baseline Accuracy')
ax.set_xlabel('Number of Top Features Removed', fontsize=12)
ax.set_ylabel('Model Accuracy', fontsize=12)
ax.set_title('Binary LIME: Descriptive Accuracy Curve\n(Feature Removal Impact)',
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Binary", 'descriptive_accuracy_curve.png'), dpi=300)
plt.close()

# 2. SPARSITY PLOTS (Binary)
binary_sparsity_scores = []
for exp in binary_explanations:
    feature_weights = exp.as_list()
    total_weight = sum(abs(w) for _, w in feature_weights)

    if total_weight > 0:
        sorted_weights = sorted([abs(w) for _, w in feature_weights], reverse=True)
        cumsum = np.cumsum(sorted_weights)
        threshold = 0.8 * total_weight
        n_features_80 = np.searchsorted(cumsum, threshold) + 1
        binary_sparsity_scores.append(n_features_80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(binary_sparsity_scores, bins=15, color='purple', alpha=0.7, edgecolor='black')
axes[0].axvline(np.mean(binary_sparsity_scores), color='red', linestyle='--',
                label=f'Mean: {np.mean(binary_sparsity_scores):.2f}')
axes[0].set_xlabel('Number of Features for 80% Importance', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Binary LIME: Sparsity Distribution', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Box plot
axes[1].boxplot([binary_sparsity_scores], vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7))
axes[1].set_ylabel('Number of Features for 80% Importance', fontsize=11)
axes[1].set_title('Binary LIME: Sparsity Box Plot', fontsize=12, fontweight='bold')
axes[1].grid(alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Binary", 'sparsity_analysis.png'), dpi=300)
plt.close()

# 3. STABILITY HISTOGRAMS (Binary)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Stability distribution
axes[0].hist(binary_stability_scores, bins=20, color='forestgreen', alpha=0.7, edgecolor='black')
axes[0].axvline(binary_stability, color='red', linestyle='--',
                label=f'Mean: {binary_stability:.4f}')
axes[0].set_xlabel('Stability Score', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title('Binary LIME: Stability Distribution', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Fidelity distribution
axes[1].hist(binary_fidelity_scores, bins=20, color='orange', alpha=0.7, edgecolor='black')
axes[1].axvline(binary_fidelity, color='red', linestyle='--',
                label=f'Mean: {binary_fidelity:.4f}')
axes[1].set_xlabel('Fidelity Score', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title('Binary LIME: Fidelity Distribution', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Binary", 'stability_fidelity_histograms.png'), dpi=300)
plt.close()

# 4. Global Feature Importance (Binary)
binary_global_importance = defaultdict(list)
for exp in binary_explanations:
    importance_dict = extract_feature_importance(exp, feature_names)
    for feat, weight in importance_dict.items():
        binary_global_importance[feat].append(abs(weight))

binary_importance_mean = {feat: np.mean(scores) for feat, scores in binary_global_importance.items()}
binary_top_features = sorted(binary_importance_mean.items(), key=lambda x: x[1], reverse=True)[:20]

plt.figure(figsize=(12, 8))
features, scores = zip(*binary_top_features)
colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
plt.barh(range(len(features)), scores, color=colors, alpha=0.8)
plt.yticks(range(len(features)), features, fontsize=10)
plt.xlabel('Mean Absolute LIME Importance', fontsize=12)
plt.title(f'Binary LIME: Global Feature Importance\nAggregated from {len(X_val)} Explanations',
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Binary", 'global_feature_importance.png'), dpi=300)
plt.close()

# 5. Multiclass visualizations
multiclass_global_importance = defaultdict(list)
for exp in multiclass_explanations:
    importance_dict = extract_feature_importance(exp, feature_names)
    for feat, weight in importance_dict.items():
        multiclass_global_importance[feat].append(abs(weight))

multiclass_importance_mean = {feat: np.mean(scores) for feat, scores in multiclass_global_importance.items()}
multiclass_top_features = sorted(multiclass_importance_mean.items(), key=lambda x: x[1], reverse=True)[:20]

plt.figure(figsize=(12, 8))
features, scores = zip(*multiclass_top_features)
colors = plt.cm.plasma(np.linspace(0, 1, len(features)))
plt.barh(range(len(features)), scores, color=colors, alpha=0.8)
plt.yticks(range(len(features)), features, fontsize=10)
plt.xlabel('Mean Absolute LIME Importance', fontsize=12)
plt.title(f'Multiclass LIME: Global Feature Importance\nAggregated from {len(X_val_bfrb)} Explanations',
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Multiclass", 'global_feature_importance.png'), dpi=300)
plt.close()

print("✓ All visualizations saved")

# ============================================================
# SAVE RESULTS
# ============================================================
print("\n[Saving] Generating final summary...")

# Compile all results
complete_results = {
    'binary': {
        'baseline': binary_baseline,
        'fidelity': binary_fidelity,
        'stability': binary_stability,
        'lime_auc': binary_lime_auc,
        'ece': binary_ece,
        'trust': binary_trust,
        'top_features': [f for f, _ in binary_top_features[:10]],
        'explanations': binary_explanations
    },
    'multiclass': {
        'baseline': gesture_baseline,
        'fidelity': multiclass_fidelity,
        'stability': multiclass_stability,
        'lime_auc': multiclass_lime_auc,
        'ece': multiclass_ece,
        'trust': multiclass_trust,
        'top_features': [f for f, _ in multiclass_top_features[:10]],
        'explanations': multiclass_explanations
    }
}

joblib.dump(complete_results, os.path.join(OUTPUT_DIR, 'lime_complete_binary_gesture.pkl'))
print(f"✅ Saved: lime_complete_binary_gesture.pkl")

# ============================================================
# FINAL SUMMARY (SHAP-STYLE)
# ============================================================
print("\n" + "=" * 80)
print("✅ COMPLETE RESULTS FOR PAPER")
print("=" * 80)

print(f"📊 BINARY MODEL (BFRB Detection)")
print(f"Accuracy {binary_baseline['Accuracy']:.4f} "
      f"Precision {binary_baseline['Precision']:.4f} "
      f"Recall {binary_baseline['Recall']:.4f} "
      f"F1-Score {binary_baseline['F1-Score']:.4f} "
      f"AUC-ROC {binary_baseline['AUC-ROC']:.4f} "
      f"Log Loss {binary_baseline['Log Loss']:.4f} "
      f"MAE {binary_baseline['MAE']:.4f} "
      f"LIME-AUC {binary_lime_auc:.4f} "
      f"ECE {binary_ece:.4f} "
      f"Fidelity {binary_fidelity:.4f} "
      f"Stability {binary_stability:.4f} "
      f"Trust Score {binary_trust['trust_mean']:.4f} "
      f"Trust Discrim. {binary_trust['discrimination']:.4f}")

print(f"\n📊 GESTURE MODEL (8-class Classification)")
print(f"Accuracy {gesture_baseline['Accuracy']:.4f} "
      f"Macro F1 {gesture_baseline['Macro F1']:.4f} "
      f"Weighted F1 {gesture_baseline['Weighted F1']:.4f} "
      f"LIME-AUC {multiclass_lime_auc:.4f} "
      f"ECE {multiclass_ece:.4f} "
      f"Fidelity {multiclass_fidelity:.4f} "
      f"Stability {multiclass_stability:.4f} "
      f"Trust Score {multiclass_trust['trust_mean']:.4f} "
      f"Trust Discrim. {multiclass_trust['discrimination']:.4f}")

print("=" * 80)
print("✅ ANALYSIS COMPLETE")
print(f"Key Findings:")
print(f"Binary: • {binary_baseline['Accuracy'] * 100:.1f}% accuracy, {binary_baseline['AUC-ROC']:.3f} AUC "
      f"• {binary_fidelity:.3f} fidelity, {binary_stability:.3f} stability "
      f"• {binary_trust['trust_mean']:.3f} trust (Δ={binary_trust['discrimination']:.3f})")
print(f"Gesture: • {gesture_baseline['Accuracy'] * 100:.1f}% accuracy, {gesture_baseline['Macro F1']:.3f} macro F1 "
      f"• {multiclass_fidelity:.3f} fidelity, {multiclass_stability:.3f} stability "
      f"• {multiclass_trust['trust_mean']:.3f} trust (Δ={multiclass_trust['discrimination']:.3f})")
print("=" * 80)