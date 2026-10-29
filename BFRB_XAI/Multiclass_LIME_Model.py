# ============================================================
# COMPREHENSIVE LIME EXPLAINABILITY - MULTICLASS CLASSIFIER
# BFRB Gesture Recognition - Full XAI Evaluation
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
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from lime import lime_tabular

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_DIR = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\XAI_Work\BFRB_XAI\Trained_Models"
DATA_FOLDER = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\CMI_Sensor_Kaggle"
OUTPUT_DIR = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\XAI_Work\BFRB_XAI\Figures_and_Graphs\LIME_Multiclass_Results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: LOAD MODELS AND DATA
# ============================================================
print("=" * 80)
print("LIME EXPLAINABILITY - MULTICLASS GESTURE CLASSIFICATION")
print("=" * 80)
print("\n[1/12] Loading Models and Data...")

# Load models
binary_model = joblib.load(os.path.join(MODEL_DIR, "binary_classifier.pkl"))
gesture_model = joblib.load(os.path.join(MODEL_DIR, "gesture_classifier.pkl"))
gesture_encoder = joblib.load(os.path.join(MODEL_DIR, "gesture_encoder.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
metadata = joblib.load(os.path.join(MODEL_DIR, "metadata.pkl"))

BFRB_GESTURES = metadata['bfrb_gestures']

print(f"✓ Binary model loaded")
print(f"✓ Gesture model loaded")
print(f"✓ Feature names: {len(feature_names)} features")
print(f"✓ BFRB Gestures: {len(BFRB_GESTURES)} classes")
for i, g in enumerate(BFRB_GESTURES, 1):
    print(f"    {i}. {g}")

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
    lambda x: 1 if x in BFRB_GESTURES else 0
)

full_df = pd.merge(agg_df, labels_df, on="sequence_id")
X = full_df.drop(columns=["sequence_id", "gesture", "binary_label"])
y_gesture = full_df["gesture"]
y_binary = full_df["binary_label"]

X = X[feature_names].fillna(X[feature_names].median())

# Filter for BFRB samples only (multiclass)
X_bfrb = X[y_binary == 1]
y_bfrb = y_gesture[y_binary == 1]

print(f"\n✓ Total samples: {len(X)}")
print(f"✓ BFRB samples: {len(X_bfrb)}")
print(f"✓ Gesture distribution:")
gesture_counts = y_bfrb.value_counts()
for gesture, count in gesture_counts.items():
    print(f"    {gesture}: {count}")

# Train-val split for BFRB data
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(X_bfrb, y_bfrb))
X_train = X_bfrb.iloc[train_idx]
X_val = X_bfrb.iloc[val_idx]
y_train = y_bfrb.iloc[train_idx]
y_val = y_bfrb.iloc[val_idx]

print(f"\n✓ Training samples: {len(X_train)}")
print(f"✓ Validation samples: {len(X_val)}")

# ============================================================
# BASELINE MULTICLASS PERFORMANCE
# ============================================================
print("\n[2/12] Computing Baseline Multiclass Performance...")

# Encode labels
y_train_enc = gesture_encoder.transform(y_train)
y_val_enc = gesture_encoder.transform(y_val)

# Predictions
y_pred_enc = gesture_model.predict(X_val)
y_pred = gesture_encoder.inverse_transform(y_pred_enc)

baseline_metrics = {
    'Accuracy': accuracy_score(y_val_enc, y_pred_enc),
    'Macro F1': f1_score(y_val_enc, y_pred_enc, average='macro'),
    'Weighted F1': f1_score(y_val_enc, y_pred_enc, average='weighted'),
    'Macro Precision': precision_score(y_val_enc, y_pred_enc, average='macro'),
    'Macro Recall': recall_score(y_val_enc, y_pred_enc, average='macro')
}

print("\nBASELINE MULTICLASS PERFORMANCE:")
print("-" * 60)
for metric, value in baseline_metrics.items():
    print(f"  {metric:18s}: {value:.4f}")

print("\nPER-CLASS PERFORMANCE:")
print("-" * 60)
print(classification_report(y_val, y_pred, target_names=BFRB_GESTURES, digits=4))

# Confusion matrix
cm = confusion_matrix(y_val_enc, y_pred_enc)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=BFRB_GESTURES, yticklabels=BFRB_GESTURES)
plt.title('Baseline Multiclass Confusion Matrix', fontsize=14, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '0_baseline_confusion_matrix.png'), dpi=300)
plt.close()
print("✓ Baseline confusion matrix saved")

# ============================================================
# STEP 2: INITIALIZE LIME EXPLAINER FOR MULTICLASS
# ============================================================
print("\n[3/12] Initializing LIME Explainer for Multiclass...")

explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=BFRB_GESTURES,  # All 8 gesture classes
    mode='classification',
    discretize_continuous=True,
    random_state=42
)

print("✓ LIME explainer initialized")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Features: {len(feature_names)}")
print(f"  - Classes: {len(BFRB_GESTURES)}")

# ============================================================
# STEP 3: STRATIFIED SAMPLE SELECTION
# ============================================================
print("\n[4/12] Selecting Diverse Samples (Stratified by Gesture)...")

np.random.seed(42)

# Select samples from each gesture class
samples_per_class = 5  # 5 samples per gesture
sample_indices = []

for gesture in BFRB_GESTURES:
    gesture_indices = np.where(y_val == gesture)[0]
    if len(gesture_indices) >= samples_per_class:
        selected = np.random.choice(gesture_indices, samples_per_class, replace=False)
    else:
        selected = gesture_indices  # Take all if less than desired
    sample_indices.extend(selected)

sample_indices = np.array(sample_indices)
n_samples = len(sample_indices)

print(f"✓ Selected {n_samples} samples across {len(BFRB_GESTURES)} gesture classes")
print(f"  (~{samples_per_class} samples per gesture)")

# ============================================================
# STEP 4: GENERATE LIME EXPLANATIONS
# ============================================================
print("\n[5/12] Generating LIME Explanations for Multiclass...")

explanations = []
explanation_times = []
predicted_labels = []
true_labels = []
predicted_probas = []

for i, idx in enumerate(sample_indices):
    sample = X_val.iloc[idx].values
    true_label = y_val.iloc[idx]

    # Time the explanation
    start_time = time.time()
    exp = explainer.explain_instance(
        data_row=sample,
        predict_fn=gesture_model.predict_proba,
        num_features=10,
        num_samples=5000,
        top_labels=len(BFRB_GESTURES)  # Explain all classes
    )
    elapsed = time.time() - start_time

    explanations.append(exp)
    explanation_times.append(elapsed)

    # Get predictions
    pred_proba = gesture_model.predict_proba(sample.reshape(1, -1))[0]
    pred_class_idx = np.argmax(pred_proba)
    pred_class = BFRB_GESTURES[pred_class_idx]

    predicted_labels.append(pred_class)
    true_labels.append(true_label)
    predicted_probas.append(pred_proba)

    if (i + 1) % 10 == 0:
        print(f"  Progress: {i + 1}/{n_samples} explanations generated...")

print(f"✓ Generated {len(explanations)} explanations")
print(f"  Average time: {np.mean(explanation_times):.3f}s per explanation")

# ============================================================
# METRIC 1: EFFICIENCY
# ============================================================
print("\n[6/12] Computing EFFICIENCY Metrics...")

efficiency_stats = {
    'Mean Time (s)': np.mean(explanation_times),
    'Median Time (s)': np.median(explanation_times),
    'Std Time (s)': np.std(explanation_times),
    'Total Time (s)': np.sum(explanation_times)
}

print("\nEFFICIENCY METRICS:")
print("-" * 60)
for metric, value in efficiency_stats.items():
    print(f"  {metric:20s}: {value:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(explanation_times, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
ax.axvline(np.mean(explanation_times), color='red', linestyle='--',
           label=f'Mean: {np.mean(explanation_times):.3f}s')
ax.set_xlabel('Explanation Time (seconds)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Multiclass LIME - Explanation Time Distribution',
             fontsize=14, fontweight='bold')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '1_efficiency_distribution.png'), dpi=300)
plt.close()
print("✓ Efficiency plot saved")

# ============================================================
# METRIC 2: MULTICLASS FIDELITY
# ============================================================
print("\n[7/12] Computing MULTICLASS FIDELITY...")

fidelity_scores = []

for i, exp in enumerate(explanations):
    sample = X_val.iloc[sample_indices[i]].values

    # Black-box prediction (all classes)
    blackbox_pred = gesture_model.predict_proba(sample.reshape(1, -1))[0]

    # LIME prediction for top predicted class
    top_label = exp.top_labels[0]
    lime_pred = exp.local_pred[top_label] if hasattr(exp, 'local_pred') else None

    if lime_pred is not None:
        # Fidelity for the top class
        fidelity = 1 - abs(blackbox_pred[top_label] - lime_pred)
        fidelity_scores.append(fidelity)

fidelity_mean = np.mean(fidelity_scores) if fidelity_scores else 0.0

print("\nMULTICLASS FIDELITY:")
print("-" * 60)
print(f"  Mean Fidelity: {fidelity_mean:.4f}")
print(f"  (How well LIME approximates the model for top predicted class)")

# ============================================================
# METRIC 3: SPARSITY
# ============================================================
print("\n[8/12] Computing SPARSITY...")

sparsity_scores = []

for exp in explanations:
    # Get explanation for top class
    top_label = exp.top_labels[0]
    feature_weights = exp.as_list(label=top_label)

    total_weight = sum(abs(w) for _, w in feature_weights)
    if total_weight > 0:
        sorted_weights = sorted([abs(w) for _, w in feature_weights], reverse=True)
        cumsum = np.cumsum(sorted_weights)
        threshold = 0.8 * total_weight
        n_features_80 = np.searchsorted(cumsum, threshold) + 1
        sparsity_scores.append(n_features_80)

sparsity_mean = np.mean(sparsity_scores) if sparsity_scores else 0.0

print("\nSPARSITY METRICS:")
print("-" * 60)
print(f"  Avg Features for 80% Importance: {sparsity_mean:.2f}")
print(f"  Total Features: {len(feature_names)}")

# ============================================================
# GLOBAL FEATURE IMPORTANCE PER CLASS
# ============================================================
print("\n[9/12] Computing GLOBAL FEATURE IMPORTANCE PER CLASS...")

# Store importance per gesture class
class_importance = {gesture: defaultdict(list) for gesture in BFRB_GESTURES}

for i, exp in enumerate(explanations):
    true_gesture = true_labels[i]

    # Get explanation for the TRUE class
    try:
        true_class_idx = BFRB_GESTURES.index(true_gesture)
        feature_weights = exp.as_list(label=true_class_idx)

        for feature, weight in feature_weights:
            feat_name = feature.split(' ')[0] if ' ' in feature else feature
            class_importance[true_gesture][feat_name].append(abs(weight))
    except:
        pass

# Calculate mean importance per class
class_importance_mean = {}
for gesture in BFRB_GESTURES:
    class_importance_mean[gesture] = {
        feat: np.mean(scores)
        for feat, scores in class_importance[gesture].items()
        if scores
    }

# Get top features per class
class_top_features = {}
for gesture in BFRB_GESTURES:
    if class_importance_mean[gesture]:
        sorted_features = sorted(
            class_importance_mean[gesture].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        class_top_features[gesture] = sorted_features
    else:
        class_top_features[gesture] = []

print("\nTOP 5 FEATURES PER GESTURE CLASS:")
print("-" * 60)
for gesture in BFRB_GESTURES:
    print(f"\n{gesture}:")
    if class_top_features[gesture]:
        for feat, score in class_top_features[gesture][:5]:
            print(f"  - {feat}: {score:.4f}")
    else:
        print("  (No explanations generated)")

# ============================================================
# VISUALIZATION: FEATURE IMPORTANCE PER CLASS
# ============================================================
print("\n[10/12] Visualizing Per-Class Feature Importance...")

# Create subplot grid
n_classes = len(BFRB_GESTURES)
n_cols = 2
n_rows = (n_classes + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
axes = axes.flatten()

for i, gesture in enumerate(BFRB_GESTURES):
    ax = axes[i]

    if class_top_features[gesture]:
        features, scores = zip(*class_top_features[gesture])
        colors = plt.cm.viridis(np.linspace(0, 1, len(features)))

        ax.barh(range(len(features)), scores, color=colors, alpha=0.8)
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels(features, fontsize=9)
        ax.set_xlabel('LIME Importance', fontsize=10)
        ax.set_title(gesture, fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                transform=ax.transAxes, fontsize=12)
        ax.set_title(gesture, fontsize=11, fontweight='bold')

# Hide extra subplots
for i in range(n_classes, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '2_per_class_importance.png'), dpi=300)
plt.close()
print("✓ Per-class importance plot saved")

# ============================================================
# CASE STUDIES: ONE EXAMPLE PER CLASS
# ============================================================
print("\n[11/12] Generating Case Study Examples...")

for gesture in BFRB_GESTURES:
    # Find first correctly classified sample of this gesture
    correct_samples = [
        i for i, (t, p) in enumerate(zip(true_labels, predicted_labels))
        if t == gesture and p == gesture
    ]

    if correct_samples:
        idx = correct_samples[0]
        exp = explanations[idx]

        # Get explanation for this gesture
        gesture_idx = BFRB_GESTURES.index(gesture)

        # Create figure with top label explanation
        fig = exp.as_pyplot_figure(label=gesture_idx)
        fig.suptitle(f'LIME Explanation: {gesture}\n'
                     f'True: {true_labels[idx]}, Pred: {predicted_labels[idx]}\n'
                     f'Confidence: {predicted_probas[idx][gesture_idx]:.3f}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()

        safe_name = gesture.replace(' ', '_').replace('-', '_')
        plt.savefig(os.path.join(OUTPUT_DIR, f'3_case_study_{safe_name}.png'),
                    dpi=300, bbox_inches='tight')
        plt.close()

print(f"✓ Generated case studies for {len(BFRB_GESTURES)} gesture classes")

# ============================================================
# CONFUSION-BASED ANALYSIS
# ============================================================
print("\n[12/12] Analyzing Confusion Patterns...")

# Find most confused pairs
conf_pairs = []
for i in range(len(BFRB_GESTURES)):
    for j in range(len(BFRB_GESTURES)):
        if i != j and cm[i, j] > 0:
            conf_pairs.append((BFRB_GESTURES[i], BFRB_GESTURES[j], cm[i, j]))

conf_pairs_sorted = sorted(conf_pairs, key=lambda x: x[2], reverse=True)

print("\nMOST CONFUSED GESTURE PAIRS:")
print("-" * 60)
for true_g, pred_g, count in conf_pairs_sorted[:5]:
    print(f"  {true_g} → {pred_g}: {count} misclassifications")

# Analyze why confusion happens (feature differences)
if conf_pairs_sorted:
    most_confused_true, most_confused_pred, _ = conf_pairs_sorted[0]

    print(f"\n\nFeature Analysis: Why '{most_confused_true}' confused with '{most_confused_pred}'?")
    print("-" * 60)

    # Get features for both classes
    features_true = dict(class_top_features[most_confused_true][:10])
    features_pred = dict(class_top_features[most_confused_pred][:10])

    # Find shared vs unique features
    shared_features = set(features_true.keys()) & set(features_pred.keys())
    unique_true = set(features_true.keys()) - set(features_pred.keys())
    unique_pred = set(features_pred.keys()) - set(features_true.keys())

    print(f"\nShared Important Features ({len(shared_features)}):")
    for feat in list(shared_features)[:5]:
        print(f"  - {feat}: True={features_true.get(feat, 0):.4f}, "
              f"Pred={features_pred.get(feat, 0):.4f}")

    print(f"\nUnique to '{most_confused_true}' ({len(unique_true)}):")
    for feat in list(unique_true)[:3]:
        print(f"  - {feat}: {features_true[feat]:.4f}")

    print(f"\nUnique to '{most_confused_pred}' ({len(unique_pred)}):")
    for feat in list(unique_pred)[:3]:
        print(f"  - {feat}: {features_pred[feat]:.4f}")

# ============================================================
# PROBABILITY DISTRIBUTION VISUALIZATION
# ============================================================
print("\n[Visualization] Generating Probability Distribution Analysis...")

# For each class, show distribution of predicted probabilities
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
axes = axes.flatten()

for i, gesture in enumerate(BFRB_GESTURES):
    ax = axes[i]

    # Get probabilities for this class across all samples where it was TRUE
    gesture_idx = BFRB_GESTURES.index(gesture)
    probs = [
        predicted_probas[j][gesture_idx]
        for j, label in enumerate(true_labels)
        if label == gesture
    ]

    if probs:
        ax.hist(probs, bins=15, color='skyblue', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(probs), color='red', linestyle='--',
                   label=f'Mean: {np.mean(probs):.3f}')
        ax.set_xlabel('Predicted Probability', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{gesture}\n(n={len(probs)} samples)',
                     fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No samples', ha='center', va='center',
                transform=ax.transAxes)
        ax.set_title(gesture, fontsize=11, fontweight='bold')

for i in range(n_classes, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '4_probability_distributions.png'), dpi=300)
plt.close()
print("✓ Probability distribution plot saved")

# ============================================================
# FEATURE HEATMAP ACROSS ALL CLASSES
# ============================================================
print("\n[Visualization] Generating Cross-Class Feature Heatmap...")

# Create matrix: classes × features
all_features = set()
for gesture in BFRB_GESTURES:
    all_features.update(class_importance_mean[gesture].keys())

top_global_features = list(all_features)[:20]  # Top 20 overall

heatmap_data = np.zeros((len(BFRB_GESTURES), len(top_global_features)))

for i, gesture in enumerate(BFRB_GESTURES):
    for j, feat in enumerate(top_global_features):
        heatmap_data[i, j] = class_importance_mean[gesture].get(feat, 0)

plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data,
            xticklabels=top_global_features,
            yticklabels=BFRB_GESTURES,
            cmap='YlOrRd', annot=False, cbar_kws={'label': 'LIME Importance'})
plt.title('Feature Importance Across All Gesture Classes',
          fontsize=14, fontweight='bold')
plt.xlabel('Features', fontsize=12)
plt.ylabel('Gesture Classes', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, '5_cross_class_heatmap.png'), dpi=300)
plt.close()
print("✓ Cross-class heatmap saved")

# ============================================================
# SUMMARY METRICS TABLE
# ============================================================
print("\n[Final Report] Generating Summary Metrics...")

metrics_summary = pd.DataFrame({
    'Metric': [
        'Efficiency (Mean Time)',
        'Fidelity (Mean)',
        'Sparsity (Avg Features)',
        'Baseline Accuracy',
        'Baseline Macro F1',
        'Baseline Weighted F1',
        'Number of Classes',
        'Samples Explained'
    ],
    'Value': [
        f"{efficiency_stats['Mean Time (s)']:.4f}s",
        f"{fidelity_mean:.4f}",
        f"{sparsity_mean:.2f} / {len(feature_names)}",
        f"{baseline_metrics['Accuracy']:.4f}",
        f"{baseline_metrics['Macro F1']:.4f}",
        f"{baseline_metrics['Weighted F1']:.4f}",
        f"{len(BFRB_GESTURES)}",
        f"{n_samples}"
    ]
})

print("\n" + "=" * 80)
print("MULTICLASS LIME EVALUATION SUMMARY")
print("=" * 80)
print(metrics_summary.to_string(index=False))
print("=" * 80)

metrics_summary.to_csv(os.path.join(OUTPUT_DIR, 'multiclass_metrics_summary.csv'),
                       index=False)
print("\n✓ Metrics summary saved")

# ============================================================
# FINAL SUMMARY
# ============================================================
print("\n" + "=" * 80)
print("MULTICLASS LIME ANALYSIS COMPLETE")
print("=" * 80)
print(f"✓ Analyzed {n_samples} samples across {len(BFRB_GESTURES)} gesture classes")
print(f"✓ Generated {len(explanations)} multiclass explanations")
print(f"✓ Efficiency: {efficiency_stats['Mean Time (s)']:.3f}s per explanation")
print(f"✓ Fidelity: {fidelity_mean:.4f}")
print(f"✓ Sparsity: {sparsity_mean:.2f} features for 80% importance")
print(f"\n✓ Most confused pair: {conf_pairs_sorted[0][0]} → {conf_pairs_sorted[0][1]}")
print(f"✓ All results saved to: {OUTPUT_DIR}")
print("=" * 80)

# Save final summary
final_summary = {
    'n_samples': n_samples,
    'n_classes': len(BFRB_GESTURES),
    'class_names': BFRB_GESTURES,
    'baseline_metrics': baseline_metrics,
    'efficiency': efficiency_stats,
    'fidelity_mean': fidelity_mean,
    'sparsity_mean': sparsity_mean,
    'class_top_features': class_top_features,
    'confusion_pairs': conf_pairs_sorted[:5]
}

joblib.dump(final_summary, os.path.join(OUTPUT_DIR, 'multiclass_final_summary.pkl'))
print("\n✓ Final summary saved to multiclass_final_summary.pkl")