# ============================================================
# COMPREHENSIVE FULL LIME EXPLAINABILITY ANALYSIS
# Binary + Multiclass Classification with Extended Metrics
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
    roc_auc_score, log_loss, confusion_matrix
)
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

def calculate_semantic_richness(explanations, feature_names):
    """
    Semantic Richness: Diversity of features used in explanations
    Higher = Model uses diverse features (good interpretability)
    """
    all_features_used = set()
    feature_usage_counts = Counter()

    for exp in explanations:
        exp_features = set()
        for feature, weight in exp.as_list():
            feat_name = feature.split(' ')[0] if ' ' in feature else feature
            # Map to original feature names
            matching = [f for f in feature_names if feat_name in f or f in feat_name]
            if matching:
                exp_features.add(matching[0])
                feature_usage_counts[matching[0]] += 1
            else:
                exp_features.add(feat_name)
                feature_usage_counts[feat_name] += 1
        all_features_used.update(exp_features)

    # Metrics
    n_unique_features = len(all_features_used)
    n_total_features = len(feature_names)
    coverage_ratio = n_unique_features / n_total_features

    # Distribution entropy (higher = more uniform usage)
    counts = np.array(list(feature_usage_counts.values()))
    usage_entropy = entropy(counts / counts.sum())

    # Normalized entropy (0-1 scale)
    max_entropy = np.log(len(feature_usage_counts))
    normalized_entropy = usage_entropy / max_entropy if max_entropy > 0 else 0

    return {
        'unique_features_used': n_unique_features,
        'total_features': n_total_features,
        'coverage_ratio': coverage_ratio,
        'usage_entropy': usage_entropy,
        'normalized_entropy': normalized_entropy,
        'semantic_richness_score': (coverage_ratio + normalized_entropy) / 2
    }


def calculate_clinical_trust_score(explanations, y_true, y_pred, feature_names):
    """
    Clinical Trust Score: Composite metric for medical decision support
    Components:
    1. Prediction consistency (are confident predictions usually correct?)
    2. Explanation stability (similar cases get similar explanations?)
    3. Feature medical relevance (do important features make clinical sense?)
    4. Error interpretability (are mistakes explainable?)
    """

    # 1. Prediction Consistency
    correct_predictions = (y_true == y_pred)
    consistency_score = correct_predictions.mean()

    # 2. Explanation Stability (feature rank correlation between similar samples)
    stability_correlations = []

    # Group by predicted class
    for pred_class in np.unique(y_pred):
        class_mask = (y_pred == pred_class)
        class_explanations = [exp for i, exp in enumerate(explanations) if class_mask[i]]

        if len(class_explanations) < 2:
            continue

        # Compare feature rankings between pairs of explanations
        for i in range(min(10, len(class_explanations))):
            for j in range(i + 1, min(10, len(class_explanations))):
                exp_i = class_explanations[i]
                exp_j = class_explanations[j]

                # Get feature rankings
                features_i = {feat.split(' ')[0]: abs(weight)
                              for feat, weight in exp_i.as_list()}
                features_j = {feat.split(' ')[0]: abs(weight)
                              for feat, weight in exp_j.as_list()}

                # Common features
                common_features = set(features_i.keys()) & set(features_j.keys())
                if len(common_features) >= 3:
                    ranks_i = [features_i[f] for f in common_features]
                    ranks_j = [features_j[f] for f in common_features]

                    corr, _ = spearmanr(ranks_i, ranks_j)
                    if not np.isnan(corr):
                        stability_correlations.append(corr)

    stability_score = np.mean(stability_correlations) if stability_correlations else 0.5

    # 3. Feature Medical Relevance (sensor features that make clinical sense)
    # TOF (proximity), thermal, and motion sensors are clinically relevant for BFRB
    clinical_features = [f for f in feature_names if any(
        sensor in f.lower() for sensor in ['tof', 'thm', 'thermal', 'imu', 'gyro']
    )]

    # Check if top features are clinically relevant
    top_features_used = set()
    for exp in explanations:
        for feature, weight in exp.as_list()[:5]:  # Top 5 per explanation
            feat_name = feature.split(' ')[0] if ' ' in feature else feature
            matching = [f for f in feature_names if feat_name in f or f in feat_name]
            if matching:
                top_features_used.add(matching[0])

    clinical_overlap = len(top_features_used & set(clinical_features))
    clinical_relevance_score = clinical_overlap / len(top_features_used) if top_features_used else 0

    # 4. Error Interpretability (can we explain why errors happened?)
    error_mask = ~correct_predictions
    if error_mask.sum() > 0:
        error_explanations = [exp for i, exp in enumerate(explanations) if error_mask[i]]
        # Errors should have lower confidence (more uncertainty = more interpretable)
        error_confidences = []
        for exp in error_explanations:
            if hasattr(exp, 'local_pred'):
                error_confidences.append(abs(exp.local_pred[0] - 0.5))  # Distance from uncertain

        # Lower confidence on errors = better (more honest uncertainty)
        error_interp_score = 1 - np.mean(error_confidences) if error_confidences else 0.5
    else:
        error_interp_score = 1.0  # No errors = perfect

    # Composite Clinical Trust Score (weighted average)
    weights = {
        'consistency': 0.35,  # Most important: are predictions correct?
        'stability': 0.25,  # Are similar cases treated similarly?
        'relevance': 0.25,  # Do features make clinical sense?
        'error_interp': 0.15  # Are errors understandable?
    }

    clinical_trust_score = (
            weights['consistency'] * consistency_score +
            weights['stability'] * stability_score +
            weights['relevance'] * clinical_relevance_score +
            weights['error_interp'] * error_interp_score
    )

    return {
        'consistency_score': consistency_score,
        'stability_score': stability_score,
        'clinical_relevance_score': clinical_relevance_score,
        'error_interpretability_score': error_interp_score,
        'clinical_trust_score': clinical_trust_score,
        'n_stability_comparisons': len(stability_correlations)
    }


def extract_feature_importance(exp, feature_names):
    """Extract clean feature importance from LIME explanation"""
    importance_dict = {}
    for feature, weight in exp.as_list():
        feat_name = feature.split(' ')[0] if ' ' in feature else feature
        # Map to original feature names
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
print("FULL LIME EXPLAINABILITY ANALYSIS")
print("Binary + Multiclass Classification with Extended Metrics")
print("=" * 80)
print("\n[1/12] Loading Models and Data...")

# Load models
binary_model = joblib.load(os.path.join(MODEL_DIR, "binary_classifier.pkl"))
gesture_model = joblib.load(os.path.join(MODEL_DIR, "gesture_classifier.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
metadata = joblib.load(os.path.join(MODEL_DIR, "metadata.pkl"))

print(f"✓ Binary model loaded")
print(f"✓ Gesture model loaded")
print(f"✓ Feature names: {len(feature_names)} features")

# Load data first (needed for fallback gesture_classes extraction)
df = pd.read_csv(os.path.join(DATA_FOLDER, "train.csv"))
df["gesture"] = df["gesture"].fillna("non_target")

# Check what's in metadata and extract gesture classes
if 'gesture_classes' in metadata:
    gesture_classes = metadata['gesture_classes']
elif 'bfrb_gestures' in metadata:
    # If only bfrb_gestures exists, use that as gesture classes
    gesture_classes = metadata['bfrb_gestures']
else:
    # Fallback: extract from data
    gesture_classes = sorted([g for g in df['gesture'].unique() if g != 'non_target'])
    print("⚠ Warning: gesture_classes not in metadata, extracted from data")

print(f"✓ Gesture classes: {gesture_classes}")
print(f"✓ Number of gesture classes: {len(gesture_classes)}")

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

# Create gesture label mapping
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

print(f"✓ Data prepared: {len(X_train)} train, {len(X_val)} validation samples")
print(f"  Binary - BFRB: {sum(y_binary_val == 1)}, Non-BFRB: {sum(y_binary_val == 0)}")

# ============================================================
# BASELINE PERFORMANCE METRICS
# ============================================================
print("\n[2/12] Computing Baseline Model Performance...")

# Binary model
y_binary_pred_proba = binary_model.predict_proba(X_val)[:, 1]
y_binary_pred = (y_binary_pred_proba > 0.45).astype(int)

binary_baseline = {
    'Accuracy': accuracy_score(y_binary_val, y_binary_pred),
    'Precision': precision_score(y_binary_val, y_binary_pred),
    'Recall': recall_score(y_binary_val, y_binary_pred),
    'F1-Score': f1_score(y_binary_val, y_binary_pred),
    'AUC-ROC': roc_auc_score(y_binary_val, y_binary_pred_proba),
    'Log Loss': log_loss(y_binary_val, y_binary_pred_proba)
}

# Multiclass model (only on BFRB samples)
bfrb_mask = y_binary_val == 1
X_val_bfrb = X_val[bfrb_mask]
y_gesture_val_bfrb = y_gesture_val[bfrb_mask]

y_gesture_pred_proba = gesture_model.predict_proba(X_val_bfrb)
y_gesture_pred = gesture_model.predict(X_val_bfrb)

multiclass_baseline = {
    'Accuracy': accuracy_score(y_gesture_val_bfrb, y_gesture_pred),
    'F1-Score (Weighted)': f1_score(y_gesture_val_bfrb, y_gesture_pred, average='weighted'),
    'F1-Score (Macro)': f1_score(y_gesture_val_bfrb, y_gesture_pred, average='macro')
}

print("\nBINARY MODEL BASELINE:")
print("-" * 60)
for metric, value in binary_baseline.items():
    print(f"  {metric:15s}: {value:.4f}")

print("\nMULTICLASS MODEL BASELINE:")
print("-" * 60)
for metric, value in multiclass_baseline.items():
    print(f"  {metric:20s}: {value:.4f}")

# ============================================================
# STEP 3: INITIALIZE LIME EXPLAINERS
# ============================================================
print("\n[3/12] Initializing LIME Explainers...")

binary_explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=['Non-BFRB', 'BFRB'],
    mode='classification',
    discretize_continuous=True,
    random_state=42
)

# Multiclass explainer (train only on BFRB samples)
X_train_bfrb = X_train[y_binary.iloc[train_idx] == 1]

multiclass_explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train_bfrb.values,
    feature_names=feature_names,
    class_names=gesture_classes,
    mode='classification',
    discretize_continuous=True,
    random_state=42
)

print("✓ Binary LIME explainer initialized")
print("✓ Multiclass LIME explainer initialized")

# ============================================================
# STEP 4: FULL VALIDATION SET SELECTION
# ============================================================
print("\n[4/12] Preparing Full Validation Set for LIME...")

# Use ALL validation samples for fair comparison with SHAP
binary_sample_indices = np.arange(len(X_val))
n_binary_samples = len(binary_sample_indices)

# For multiclass, use all BFRB samples
multiclass_sample_indices = np.arange(len(X_val_bfrb))
n_multiclass_samples = len(multiclass_sample_indices)

print(f"✓ Binary analysis: {n_binary_samples} samples (entire validation set)")
print(f"✓ Multiclass analysis: {n_multiclass_samples} samples (all BFRB samples)")
print(f"  Estimated time: ~{(n_binary_samples * 1.5 + n_multiclass_samples * 2) / 60:.1f} minutes")

# ============================================================
# BINARY CLASSIFICATION ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("BINARY CLASSIFICATION LIME ANALYSIS")
print("=" * 80)

# ============================================================
# STEP 5: GENERATE BINARY LIME EXPLANATIONS
# ============================================================
print("\n[5/12] Generating Binary LIME Explanations...")

binary_explanations = []
binary_times = []
binary_predicted = []
binary_true = []
binary_confidences = []

for i in tqdm(range(n_binary_samples), desc="Binary LIME"):
    idx = binary_sample_indices[i]
    sample = X_val.iloc[idx].values
    true_label = y_binary_val.iloc[idx]

    start_time = time.time()
    exp = binary_explainer.explain_instance(
        data_row=sample,
        predict_fn=binary_model.predict_proba,
        num_features=10,
        num_samples=2000  # Reduced from 5000 for speed
    )
    elapsed = time.time() - start_time

    binary_explanations.append(exp)
    binary_times.append(elapsed)

    pred_proba = binary_model.predict_proba(sample.reshape(1, -1))[0]
    pred_class = 1 if pred_proba[1] > 0.45 else 0

    binary_predicted.append(pred_class)
    binary_true.append(true_label)
    binary_confidences.append(pred_proba[pred_class])

binary_predicted = np.array(binary_predicted)
binary_true = np.array(binary_true)

print(f"✓ Generated {len(binary_explanations)} binary explanations")
print(f"  Total time: {np.sum(binary_times) / 60:.2f} minutes")
print(f"  Average time: {np.mean(binary_times):.3f}s per explanation")

# ============================================================
# STEP 6: BINARY METRICS COMPUTATION
# ============================================================
print("\n[6/12] Computing Binary LIME Metrics...")

# 1. EFFICIENCY
binary_efficiency = {
    'Mean Time (s)': np.mean(binary_times),
    'Median Time (s)': np.median(binary_times),
    'Std Time (s)': np.std(binary_times),
    'Total Time (s)': np.sum(binary_times)
}

# 2. FIDELITY
binary_fidelity_scores = []
for i, exp in enumerate(binary_explanations):
    sample = X_val.iloc[i].values
    blackbox_pred = binary_model.predict_proba(sample.reshape(1, -1))[0]
    lime_pred = exp.local_pred[0] if hasattr(exp, 'local_pred') else None

    if lime_pred is not None:
        fidelity = 1 - abs(blackbox_pred[1] - lime_pred)
        binary_fidelity_scores.append(fidelity)

binary_fidelity = {
    'Mean Fidelity': np.mean(binary_fidelity_scores),
    'Std Fidelity': np.std(binary_fidelity_scores),
    'Min Fidelity': np.min(binary_fidelity_scores),
    'Max Fidelity': np.max(binary_fidelity_scores)
}

# 3. STABILITY (on subset for computational efficiency)
print("  Computing stability (on 50 samples)...")
binary_stability_scores = []
stability_test_indices = np.random.choice(n_binary_samples, min(50, n_binary_samples), replace=False)

for idx in stability_test_indices:
    sample = X_val.iloc[idx].values
    feature_importances = []

    for _ in range(5):  # 5 runs per sample
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

binary_stability = {
    'Mean Stability': np.mean(binary_stability_scores),
    'Std Stability': np.std(binary_stability_scores)
}

# 4. SPARSITY
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

binary_sparsity = {
    'Mean Features (80%)': np.mean(binary_sparsity_scores),
    'Sparsity Ratio': np.mean(binary_sparsity_scores) / len(feature_names)
}

# 5. GLOBAL FEATURE IMPORTANCE
binary_global_importance = defaultdict(list)
for exp in binary_explanations:
    importance_dict = extract_feature_importance(exp, feature_names)
    for feat, weight in importance_dict.items():
        binary_global_importance[feat].append(abs(weight))

binary_importance_mean = {
    feat: np.mean(scores) for feat, scores in binary_global_importance.items()
}
binary_top_features = sorted(binary_importance_mean.items(),
                             key=lambda x: x[1], reverse=True)[:20]
binary_top_feature_names = [f for f, _ in binary_top_features[:10]]

# 6. DESCRIPTIVE ACCURACY
X_val_reduced = X_val.copy()
for feat in binary_top_feature_names:
    if feat in X_val_reduced.columns:
        X_val_reduced[feat] = X_val_reduced[feat].median()

y_pred_reduced = (binary_model.predict_proba(X_val_reduced)[:, 1] > 0.45).astype(int)
reduced_accuracy = accuracy_score(y_binary_val, y_pred_reduced)
binary_accuracy_drop = binary_baseline['Accuracy'] - reduced_accuracy

binary_descriptive = {
    'Original Accuracy': binary_baseline['Accuracy'],
    'Reduced Accuracy': reduced_accuracy,
    'Accuracy Drop': binary_accuracy_drop,
    'Relative Drop %': (binary_accuracy_drop / binary_baseline['Accuracy'] * 100)
}

# 7. SEMANTIC RICHNESS
binary_semantic = calculate_semantic_richness(binary_explanations, feature_names)

# 8. CLINICAL TRUST SCORE
binary_clinical = calculate_clinical_trust_score(
    binary_explanations, binary_true, binary_predicted, feature_names
)

print("✓ All binary metrics computed")

# ============================================================
# MULTICLASS CLASSIFICATION ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("MULTICLASS CLASSIFICATION LIME ANALYSIS")
print("=" * 80)

# ============================================================
# STEP 7: GENERATE MULTICLASS LIME EXPLANATIONS
# ============================================================
print("\n[7/12] Generating Multiclass LIME Explanations...")

multiclass_explanations = []
multiclass_times = []
multiclass_predicted = []
multiclass_true = []
multiclass_confidences = []

for i in tqdm(range(n_multiclass_samples), desc="Multiclass LIME"):
    idx = multiclass_sample_indices[i]
    sample = X_val_bfrb.iloc[idx].values
    true_label = y_gesture_val_bfrb.iloc[idx]

    start_time = time.time()
    exp = multiclass_explainer.explain_instance(
        data_row=sample,
        predict_fn=gesture_model.predict_proba,
        num_features=10,
        num_samples=2000,
        top_labels=len(gesture_classes)  # Explain all classes
    )
    elapsed = time.time() - start_time

    multiclass_explanations.append(exp)
    multiclass_times.append(elapsed)

    pred_proba = gesture_model.predict_proba(sample.reshape(1, -1))[0]
    pred_class = gesture_model.predict(sample.reshape(1, -1))[0]

    multiclass_predicted.append(pred_class)
    multiclass_true.append(true_label)
    multiclass_confidences.append(pred_proba[pred_class])

multiclass_predicted = np.array(multiclass_predicted)
multiclass_true = np.array(multiclass_true)

print(f"✓ Generated {len(multiclass_explanations)} multiclass explanations")
print(f"  Total time: {np.sum(multiclass_times) / 60:.2f} minutes")
print(f"  Average time: {np.mean(multiclass_times):.3f}s per explanation")

# ============================================================
# STEP 8: MULTICLASS METRICS COMPUTATION
# ============================================================
print("\n[8/12] Computing Multiclass LIME Metrics...")

# 1. EFFICIENCY
multiclass_efficiency = {
    'Mean Time (s)': np.mean(multiclass_times),
    'Median Time (s)': np.median(multiclass_times),
    'Std Time (s)': np.std(multiclass_times),
    'Total Time (s)': np.sum(multiclass_times)
}

# 2. FIDELITY (for predicted class)
multiclass_fidelity_scores = []
for i, exp in enumerate(multiclass_explanations):
    sample = X_val_bfrb.iloc[i].values
    pred_class = multiclass_predicted[i]

    blackbox_pred = gesture_model.predict_proba(sample.reshape(1, -1))[0]

    # Get LIME's prediction for this class
    if hasattr(exp, 'local_pred'):
        lime_pred = exp.local_pred
        if isinstance(lime_pred, (list, np.ndarray)) and len(lime_pred) > pred_class:
            fidelity = 1 - abs(blackbox_pred[pred_class] - lime_pred[pred_class])
            multiclass_fidelity_scores.append(fidelity)

multiclass_fidelity = {
    'Mean Fidelity': np.mean(multiclass_fidelity_scores) if multiclass_fidelity_scores else 0,
    'Std Fidelity': np.std(multiclass_fidelity_scores) if multiclass_fidelity_scores else 0
}

# 3. STABILITY (on subset)
print("  Computing stability (on 30 samples)...")
multiclass_stability_scores = []
mc_stability_test_indices = np.random.choice(n_multiclass_samples,
                                             min(30, n_multiclass_samples),
                                             replace=False)

for idx in mc_stability_test_indices:
    sample = X_val_bfrb.iloc[idx].values
    pred_class = multiclass_predicted[idx]
    feature_importances = []

    for _ in range(5):
        exp = multiclass_explainer.explain_instance(
            data_row=sample,
            predict_fn=gesture_model.predict_proba,
            num_features=10,
            num_samples=2000,
            top_labels=len(gesture_classes)
        )

        # Get explanation for predicted class
        if hasattr(exp, 'as_list'):
            importance_dict = extract_feature_importance(exp, feature_names)
            feature_importances.append(importance_dict)

    if len(feature_importances) > 0:
        all_features = set()
        for imp_dict in feature_importances:
            all_features.update(imp_dict.keys())

        feature_variances = []
        for feat in all_features:
            values = [imp_dict.get(feat, 0) for imp_dict in feature_importances]
            feature_variances.append(np.std(values))

        stability = 1 - np.mean(feature_variances) if feature_variances else 0
        multiclass_stability_scores.append(max(0, stability))

multiclass_stability = {
    'Mean Stability': np.mean(multiclass_stability_scores) if multiclass_stability_scores else 0,
    'Std Stability': np.std(multiclass_stability_scores) if multiclass_stability_scores else 0
}

# 4. SPARSITY
multiclass_sparsity_scores = []
for exp in multiclass_explanations:
    feature_weights = exp.as_list()
    total_weight = sum(abs(w) for _, w in feature_weights)

    if total_weight > 0:
        sorted_weights = sorted([abs(w) for _, w in feature_weights], reverse=True)
        cumsum = np.cumsum(sorted_weights)
        threshold = 0.8 * total_weight
        n_features_80 = np.searchsorted(cumsum, threshold) + 1
        multiclass_sparsity_scores.append(n_features_80)

multiclass_sparsity = {
    'Mean Features (80%)': np.mean(multiclass_sparsity_scores) if multiclass_sparsity_scores else 0,
    'Sparsity Ratio': (np.mean(multiclass_sparsity_scores) / len(feature_names)) if multiclass_sparsity_scores else 0
}

# 5. GLOBAL FEATURE IMPORTANCE (aggregated across all classes)
multiclass_global_importance = defaultdict(list)
for exp in multiclass_explanations:
    importance_dict = extract_feature_importance(exp, feature_names)
    for feat, weight in importance_dict.items():
        multiclass_global_importance[feat].append(abs(weight))

multiclass_importance_mean = {
    feat: np.mean(scores) for feat, scores in multiclass_global_importance.items()
}
multiclass_top_features = sorted(multiclass_importance_mean.items(),
                                 key=lambda x: x[1], reverse=True)[:20]
multiclass_top_feature_names = [f for f, _ in multiclass_top_features[:10]]

# 6. PER-CLASS FEATURE IMPORTANCE
print("  Extracting per-class feature importance...")
class_specific_importance = {cls: defaultdict(list)
                             for cls in range(len(gesture_classes))}

for i, exp in enumerate(multiclass_explanations):
    true_class = multiclass_true[i]
    pred_class = multiclass_predicted[i]

    # Get explanation for predicted class
    importance_dict = extract_feature_importance(exp, feature_names)
    for feat, weight in importance_dict.items():
        class_specific_importance[pred_class][feat].append(abs(weight))

class_importance_mean = {}
for cls in range(len(gesture_classes)):
    class_importance_mean[cls] = {
        feat: np.mean(scores)
        for feat, scores in class_specific_importance[cls].items()
        if len(scores) > 0
    }

# 7. DESCRIPTIVE ACCURACY
X_val_bfrb_reduced = X_val_bfrb.copy()
for feat in multiclass_top_feature_names:
    if feat in X_val_bfrb_reduced.columns:
        X_val_bfrb_reduced[feat] = X_val_bfrb_reduced[feat].median()

y_gesture_pred_reduced = gesture_model.predict(X_val_bfrb_reduced)
reduced_mc_accuracy = accuracy_score(y_gesture_val_bfrb, y_gesture_pred_reduced)
multiclass_accuracy_drop = multiclass_baseline['Accuracy'] - reduced_mc_accuracy

multiclass_descriptive = {
    'Original Accuracy': multiclass_baseline['Accuracy'],
    'Reduced Accuracy': reduced_mc_accuracy,
    'Accuracy Drop': multiclass_accuracy_drop,
    'Relative Drop %': (multiclass_accuracy_drop / multiclass_baseline['Accuracy'] * 100)
}

# 8. SEMANTIC RICHNESS
multiclass_semantic = calculate_semantic_richness(multiclass_explanations, feature_names)

# 9. CLINICAL TRUST SCORE
multiclass_clinical = calculate_clinical_trust_score(
    multiclass_explanations, multiclass_true, multiclass_predicted, feature_names
)

print("✓ All multiclass metrics computed")

# ============================================================
# STEP 9: VISUALIZATIONS - BINARY
# ============================================================
print("\n[9/12] Generating Binary Visualizations...")

# Efficiency
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(binary_times, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(np.mean(binary_times), color='red', linestyle='--',
           label=f'Mean: {np.mean(binary_times):.3f}s')
ax.set_xlabel('Explanation Time (seconds)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Binary LIME: Efficiency Distribution (Full Validation Set)',
             fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Binary", '1_efficiency.png'), dpi=300)
plt.close()

# Fidelity
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(binary_fidelity_scores, bins=30, color='forestgreen', alpha=0.7, edgecolor='black')
ax.axvline(binary_fidelity['Mean Fidelity'], color='red', linestyle='--',
           label=f"Mean: {binary_fidelity['Mean Fidelity']:.4f}")
ax.set_xlabel('Fidelity Score', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Binary LIME: Fidelity Distribution', fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Binary", '2_fidelity.png'), dpi=300)
plt.close()

# Global Importance
plt.figure(figsize=(12, 8))
features, scores = zip(*binary_top_features)
colors = ['#2ecc71' if s > np.median(scores) else '#e74c3c' for s in scores]
plt.barh(range(len(features)), scores, color=colors, alpha=0.8)
plt.yticks(range(len(features)), features, fontsize=10)
plt.xlabel('Mean Absolute LIME Importance', fontsize=12)
plt.title(f'Binary LIME: Global Feature Importance\nAggregated from {n_binary_samples} Explanations',
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Binary", '3_global_importance.png'), dpi=300)
plt.close()

# Class-Specific Analysis
bfrb_importance = defaultdict(list)
non_bfrb_importance = defaultdict(list)

for i, exp in enumerate(binary_explanations):
    true_label = binary_true[i]
    target_dict = bfrb_importance if true_label == 1 else non_bfrb_importance

    importance_dict = extract_feature_importance(exp, feature_names)
    for feat, weight in importance_dict.items():
        target_dict[feat].append(abs(weight))

bfrb_mean = {f: np.mean(s) for f, s in bfrb_importance.items() if s}
non_bfrb_mean = {f: np.mean(s) for f, s in non_bfrb_importance.items() if s}

bfrb_top = sorted(bfrb_mean.items(), key=lambda x: x[1], reverse=True)[:10]
non_bfrb_top = sorted(non_bfrb_mean.items(), key=lambda x: x[1], reverse=True)[:10]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

features_b, scores_b = zip(*bfrb_top)
axes[0].barh(range(len(features_b)), scores_b, color='steelblue', alpha=0.8)
axes[0].set_yticks(range(len(features_b)))
axes[0].set_yticklabels(features_b, fontsize=10)
axes[0].set_xlabel('LIME Importance', fontsize=12)
axes[0].set_title(f'BFRB Class (n={sum(binary_true == 1)})', fontsize=13, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

features_n, scores_n = zip(*non_bfrb_top)
axes[1].barh(range(len(features_n)), scores_n, color='coral', alpha=0.8)
axes[1].set_yticks(range(len(features_n)))
axes[1].set_yticklabels(features_n, fontsize=10)
axes[1].set_xlabel('LIME Importance', fontsize=12)
axes[1].set_title(f'Non-BFRB Class (n={sum(binary_true == 0)})', fontsize=13, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Binary", '4_class_specific.png'), dpi=300)
plt.close()

print("✓ Binary visualizations saved")

# ============================================================
# STEP 10: VISUALIZATIONS - MULTICLASS
# ============================================================
print("\n[10/12] Generating Multiclass Visualizations...")

# Efficiency
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(multiclass_times, bins=30, color='purple', alpha=0.7, edgecolor='black')
ax.axvline(np.mean(multiclass_times), color='red', linestyle='--',
           label=f'Mean: {np.mean(multiclass_times):.3f}s')
ax.set_xlabel('Explanation Time (seconds)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Multiclass LIME: Efficiency Distribution (All BFRB Samples)',
             fontsize=14, fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Multiclass", '1_efficiency.png'), dpi=300)
plt.close()

# Global Importance
plt.figure(figsize=(12, 8))
features, scores = zip(*multiclass_top_features)
colors = plt.cm.viridis(np.linspace(0, 1, len(features)))
plt.barh(range(len(features)), scores, color=colors, alpha=0.8)
plt.yticks(range(len(features)), features, fontsize=10)
plt.xlabel('Mean Absolute LIME Importance', fontsize=12)
plt.title(f'Multiclass LIME: Global Feature Importance\nAggregated from {n_multiclass_samples} Explanations',
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Multiclass", '2_global_importance.png'), dpi=300)
plt.close()

# Per-Class Feature Importance
n_classes = len(gesture_classes)
n_cols = 3
n_rows = int(np.ceil(n_classes / n_cols))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 4 * n_rows))
axes = axes.flatten() if n_classes > 1 else [axes]

for cls_idx in range(n_classes):
    if cls_idx in class_importance_mean and class_importance_mean[cls_idx]:
        top_class_features = sorted(class_importance_mean[cls_idx].items(),
                                    key=lambda x: x[1], reverse=True)[:10]

        if top_class_features:
            features, scores = zip(*top_class_features)
            axes[cls_idx].barh(range(len(features)), scores,
                               color=plt.cm.tab10(cls_idx), alpha=0.8)
            axes[cls_idx].set_yticks(range(len(features)))
            axes[cls_idx].set_yticklabels(features, fontsize=8)
            axes[cls_idx].set_xlabel('Importance', fontsize=10)
            axes[cls_idx].set_title(f'{gesture_classes[cls_idx]}\n(n={sum(multiclass_predicted == cls_idx)})',
                                    fontsize=11, fontweight='bold')
            axes[cls_idx].invert_yaxis()
            axes[cls_idx].grid(axis='x', alpha=0.3)
    else:
        axes[cls_idx].text(0.5, 0.5, f'{gesture_classes[cls_idx]}\n(No samples)',
                           ha='center', va='center', fontsize=12)
        axes[cls_idx].axis('off')

# Hide extra subplots
for idx in range(n_classes, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Multiclass", '3_per_class_importance.png'), dpi=300)
plt.close()

# Confusion Matrix
cm = confusion_matrix(multiclass_true, multiclass_predicted)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=gesture_classes,
            yticklabels=gesture_classes)
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('True', fontsize=12)
plt.title('Multiclass LIME: Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "Multiclass", '4_confusion_matrix.png'), dpi=300)
plt.close()

print("✓ Multiclass visualizations saved")

# ============================================================
# STEP 11: COMPREHENSIVE METRICS SUMMARY
# ============================================================
print("\n[11/12] Generating Comprehensive Metrics Summary...")

# Binary Summary
binary_summary = pd.DataFrame({
    'Metric Category': [
        'BASELINE - Accuracy',
        'BASELINE - F1-Score',
        'BASELINE - AUC-ROC',
        'EFFICIENCY - Mean Time (s)',
        'EFFICIENCY - Total Time (min)',
        'FIDELITY - Mean',
        'FIDELITY - Std',
        'STABILITY - Mean',
        'STABILITY - Std',
        'SPARSITY - Mean Features (80%)',
        'SPARSITY - Ratio',
        'DESCRIPTIVE - Accuracy Drop',
        'DESCRIPTIVE - Relative Drop %',
        'SEMANTIC - Coverage Ratio',
        'SEMANTIC - Normalized Entropy',
        'SEMANTIC - Richness Score',
        'CLINICAL - Consistency',
        'CLINICAL - Stability',
        'CLINICAL - Relevance',
        'CLINICAL - Error Interpretability',
        'CLINICAL - Trust Score'
    ],
    'Value': [
        f"{binary_baseline['Accuracy']:.4f}",
        f"{binary_baseline['F1-Score']:.4f}",
        f"{binary_baseline['AUC-ROC']:.4f}",
        f"{binary_efficiency['Mean Time (s)']:.4f}",
        f"{binary_efficiency['Total Time (s)'] / 60:.2f}",
        f"{binary_fidelity['Mean Fidelity']:.4f}",
        f"{binary_fidelity['Std Fidelity']:.4f}",
        f"{binary_stability['Mean Stability']:.4f}",
        f"{binary_stability['Std Stability']:.4f}",
        f"{binary_sparsity['Mean Features (80%)']:.2f}",
        f"{binary_sparsity['Sparsity Ratio']:.4f}",
        f"{binary_descriptive['Accuracy Drop']:.4f}",
        f"{binary_descriptive['Relative Drop %']:.2f}%",
        f"{binary_semantic['coverage_ratio']:.4f}",
        f"{binary_semantic['normalized_entropy']:.4f}",
        f"{binary_semantic['semantic_richness_score']:.4f}",
        f"{binary_clinical['consistency_score']:.4f}",
        f"{binary_clinical['stability_score']:.4f}",
        f"{binary_clinical['clinical_relevance_score']:.4f}",
        f"{binary_clinical['error_interpretability_score']:.4f}",
        f"{binary_clinical['clinical_trust_score']:.4f}"
    ]
})

# Multiclass Summary
multiclass_summary = pd.DataFrame({
    'Metric Category': [
        'BASELINE - Accuracy',
        'BASELINE - F1-Score (Weighted)',
        'BASELINE - F1-Score (Macro)',
        'EFFICIENCY - Mean Time (s)',
        'EFFICIENCY - Total Time (min)',
        'FIDELITY - Mean',
        'FIDELITY - Std',
        'STABILITY - Mean',
        'STABILITY - Std',
        'SPARSITY - Mean Features (80%)',
        'SPARSITY - Ratio',
        'DESCRIPTIVE - Accuracy Drop',
        'DESCRIPTIVE - Relative Drop %',
        'SEMANTIC - Coverage Ratio',
        'SEMANTIC - Normalized Entropy',
        'SEMANTIC - Richness Score',
        'CLINICAL - Consistency',
        'CLINICAL - Stability',
        'CLINICAL - Relevance',
        'CLINICAL - Error Interpretability',
        'CLINICAL - Trust Score'
    ],
    'Value': [
        f"{multiclass_baseline['Accuracy']:.4f}",
        f"{multiclass_baseline['F1-Score (Weighted)']:.4f}",
        f"{multiclass_baseline['F1-Score (Macro)']:.4f}",
        f"{multiclass_efficiency['Mean Time (s)']:.4f}",
        f"{multiclass_efficiency['Total Time (s)'] / 60:.2f}",
        f"{multiclass_fidelity['Mean Fidelity']:.4f}",
        f"{multiclass_fidelity['Std Fidelity']:.4f}",
        f"{multiclass_stability['Mean Stability']:.4f}",
        f"{multiclass_stability['Std Stability']:.4f}",
        f"{multiclass_sparsity['Mean Features (80%)']:.2f}",
        f"{multiclass_sparsity['Sparsity Ratio']:.4f}",
        f"{multiclass_descriptive['Accuracy Drop']:.4f}",
        f"{multiclass_descriptive['Relative Drop %']:.2f}%",
        f"{multiclass_semantic['coverage_ratio']:.4f}",
        f"{multiclass_semantic['normalized_entropy']:.4f}",
        f"{multiclass_semantic['semantic_richness_score']:.4f}",
        f"{multiclass_clinical['consistency_score']:.4f}",
        f"{multiclass_clinical['stability_score']:.4f}",
        f"{multiclass_clinical['clinical_relevance_score']:.4f}",
        f"{multiclass_clinical['error_interpretability_score']:.4f}",
        f"{multiclass_clinical['clinical_trust_score']:.4f}"
    ]
})

# Save summaries
binary_summary.to_csv(os.path.join(OUTPUT_DIR, "Binary", 'metrics_summary.csv'), index=False)
multiclass_summary.to_csv(os.path.join(OUTPUT_DIR, "Multiclass", 'metrics_summary.csv'), index=False)

print("\nBINARY LIME METRICS SUMMARY:")
print("=" * 80)
print(binary_summary.to_string(index=False))

print("\n\nMULTICLASS LIME METRICS SUMMARY:")
print("=" * 80)
print(multiclass_summary.to_string(index=False))

# ============================================================
# STEP 12: SAVE FINAL RESULTS
# ============================================================
print("\n[12/12] Saving Final Results...")

# Binary results
binary_results = {
    'n_samples': n_binary_samples,
    'baseline_metrics': binary_baseline,
    'efficiency': binary_efficiency,
    'fidelity': binary_fidelity,
    'stability': binary_stability,
    'sparsity': binary_sparsity,
    'descriptive_accuracy': binary_descriptive,
    'semantic_richness': binary_semantic,
    'clinical_trust': binary_clinical,
    'top_10_features': binary_top_feature_names,
    'global_importance': dict(binary_top_features),
    'explanations': binary_explanations
}

# Multiclass results
multiclass_results = {
    'n_samples': n_multiclass_samples,
    'baseline_metrics': multiclass_baseline,
    'efficiency': multiclass_efficiency,
    'fidelity': multiclass_fidelity,
    'stability': multiclass_stability,
    'sparsity': multiclass_sparsity,
    'descriptive_accuracy': multiclass_descriptive,
    'semantic_richness': multiclass_semantic,
    'clinical_trust': multiclass_clinical,
    'top_10_features': multiclass_top_feature_names,
    'global_importance': dict(multiclass_top_features),
    'class_specific_importance': {k: dict(v) for k, v in class_importance_mean.items()},
    'explanations': multiclass_explanations
}

joblib.dump(binary_results, os.path.join(OUTPUT_DIR, "Binary", 'full_lime_results.pkl'))
joblib.dump(multiclass_results, os.path.join(OUTPUT_DIR, "Multiclass", 'full_lime_results.pkl'))

print(f"✓ Binary results saved to: {os.path.join(OUTPUT_DIR, 'Binary')}")
print(f"✓ Multiclass results saved to: {os.path.join(OUTPUT_DIR, 'Multiclass')}")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("FULL LIME ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nBINARY CLASSIFICATION:")
print(f"  ✓ Samples analyzed: {n_binary_samples} (100% of validation set)")
print(f"  ✓ Computation time: {binary_efficiency['Total Time (s)'] / 60:.2f} minutes")
print(f"  ✓ Mean fidelity: {binary_fidelity['Mean Fidelity']:.4f}")
print(f"  ✓ Semantic richness: {binary_semantic['semantic_richness_score']:.4f}")
print(f"  ✓ Clinical trust score: {binary_clinical['clinical_trust_score']:.4f}")
print(f"  ✓ Top feature: {binary_top_features[0][0]}")

print(f"\nMULTICLASS CLASSIFICATION:")
print(f"  ✓ Samples analyzed: {n_multiclass_samples} (all BFRB samples)")
print(f"  ✓ Computation time: {multiclass_efficiency['Total Time (s)'] / 60:.2f} minutes")
print(f"  ✓ Mean fidelity: {multiclass_fidelity['Mean Fidelity']:.4f}")
print(f"  ✓ Semantic richness: {multiclass_semantic['semantic_richness_score']:.4f}")
print(f"  ✓ Clinical trust score: {multiclass_clinical['clinical_trust_score']:.4f}")
print(f"  ✓ Top feature: {multiclass_top_features[0][0]}")

print(f"\n✓ All results saved to: {OUTPUT_DIR}")
print("=" * 80)

print("\n✅ READY FOR SHAP COMPARISON!")
print("This LIME analysis used the full validation set for fair comparison with SHAP.")