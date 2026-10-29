# Streamlined BFRB Detection Pipeline
import os
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score, confusion_matrix, accuracy_score, classification_report,
    precision_score, recall_score, precision_recall_fscore_support
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
warnings.filterwarnings('ignore')

# ---------------- CONFIGURATION ---------------- #
DATA_FOLDER = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\CMI_Sensor_Kaggle"
TRAIN_DATA_PATH = os.path.join(DATA_FOLDER, "train.csv")

# BFRB gesture labels
BFRB_GESTURES = [
    'Above ear - pull hair', 'Forehead - pull hairline', 'Forehead - scratch',
    'Eyebrow - pull hair', 'Eyelash - pull hair', 'Neck - scratch',
    'Neck - pinch skin', 'Cheek - pinch skin'
]
ALL_GESTURES = BFRB_GESTURES + ['non_target']

# ---------------- LOAD & PREPROCESS DATA ---------------- #
print(f"Loading data from: {TRAIN_DATA_PATH}")
df = pd.read_csv(TRAIN_DATA_PATH)
print(f"Successfully loaded data with shape: {df.shape}")

# Fill NaN gestures with 'non_target'
df["gesture"] = df["gesture"].fillna("non_target")

# === ENHANCED FEATURE ENGINEERING === #
# IMU magnitude: net acceleration proxy
if all(col in df.columns for col in ['acc_x', 'acc_y', 'acc_z']):
    df['imu_magnitude'] = np.sqrt(df['acc_x'] ** 2 + df['acc_y'] ** 2 + df['acc_z'] ** 2)
    print("Added IMU magnitude feature")

# Gyro magnitude: angular velocity energy (if columns exist)
if all(col in df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
    df['gyro_magnitude'] = np.sqrt(df['gyro_x'] ** 2 + df['gyro_y'] ** 2 + df['gyro_z'] ** 2)
    print("Added gyro magnitude feature")

# TOF sensors: robust aggregate
tof_cols = [col for col in df.columns if col.startswith('tof_')]
if tof_cols:
    df['tof_mean'] = df[tof_cols].mean(axis=1)
    print(f"Added TOF mean feature from {len(tof_cols)} TOF columns")

# Thermopile sensors: thermal field aggregate
thermo_cols = [col for col in df.columns if 'thermo_' in col or 'thm_' in col]
if thermo_cols:
    df['thermo_mean'] = df[thermo_cols].mean(axis=1)
    print(f"Added thermo mean feature from {len(thermo_cols)} thermo columns")

# Feature Engineering
numeric_features = df.select_dtypes(include=[np.number]).columns.drop("sequence_id", errors="ignore")
numeric_features = [c for c in numeric_features if c != 'row_id']
print(f"Found {len(numeric_features)} numeric features")

# Sequence-level aggregation with enhanced functions
agg_funcs = ['mean', 'std', 'min', 'max', 'median', 'skew']
agg_df = df.groupby("sequence_id")[numeric_features].agg(agg_funcs)
agg_df.columns = ['_'.join(col) for col in agg_df.columns]
agg_df.reset_index(inplace=True)

# Labels per sequence
labels_df = df.groupby("sequence_id").first().reset_index()[["sequence_id", "gesture"]]
labels_df["gesture"] = labels_df["gesture"].fillna("non_target")
labels_df["binary_label"] = labels_df["gesture"].apply(lambda x: 1 if x in BFRB_GESTURES else 0)

# Merge features with labels
full_df = pd.merge(agg_df, labels_df, on="sequence_id")
X = full_df.drop(columns=["sequence_id", "gesture", "binary_label"])
y_binary = full_df["binary_label"]
y_gesture = full_df["gesture"]

print(f"Final feature matrix shape: {X.shape}")
print(f"Binary class distribution: {Counter(y_binary)}")
print(f"Gesture class distribution: {Counter(y_gesture)}")

# Handle missing values using median (robust)
X = X.fillna(X.median())

# === FEATURE SELECTION === #
from sklearn.feature_selection import SelectKBest, mutual_info_classif

if X.shape[1] > 100:
    print("Performing feature selection...")
    n_features_to_select = min(100, X.shape[1])
    selector = SelectKBest(score_func=mutual_info_classif, k=n_features_to_select)
    X_selected = selector.fit_transform(X, y_binary)
    selected_features = X.columns[selector.get_support()].tolist()
    X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
    print(f"Selected {X.shape[1]} features")

# ---------------- TRAIN-VALIDATION SPLIT ---------------- #
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(X, y_binary))
X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
y_train_bin, y_val_bin = y_binary.iloc[train_idx], y_binary.iloc[val_idx]
y_train_gesture, y_val_gesture = y_gesture.iloc[train_idx], y_gesture.iloc[val_idx]

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")

# ---------------- BINARY CLASSIFICATION ---------------- #
print("\n=== TRAINING BINARY CLASSIFIER ===")

# Class weights for imbalanced data
class_counts = Counter(y_binary)
total = sum(class_counts.values())
class_weight = {cls: total / (2 * count) for cls, count in class_counts.items()}

binary_model = LGBMClassifier(
objective="binary",
    boosting_type="dart",  # DART = dropouts, reduces overfitting
    scale_pos_weight=1.68,
    # --- DART-specific ---
    drop_rate=0.1, # ↓ from 60 → less aggressive dropping
    skip_drop=0.15,  # ↓ from 0.5 → more consistent learning

    # --- Tree complexity ---
    max_depth=6,  # ↓ from 5 → shallower trees = less variance
    num_leaves=30,  # ↓ from 20 → fewer splits per tree

    # --- Learning & sampling ---
    learning_rate=0.03,  # ↑ slightly from 0.02 (DART tolerates it)
    feature_fraction=0.7,  # ↓ from 0.7 → more randomness
    bagging_fraction=0.7,  # ↓ from 0.65
    bagging_freq=5,

    # --- Regularization ---
    min_child_samples=200,  # ↑ from 200 → each leaf needs more data
    reg_alpha=1.5,  # ↑ from 2.0 → stronger L1
    reg_lambda=2.0,  # ↑ from 4.0 → stronger L2

    # --- Other ---
    n_estimators=5000,  # ↓ from 3000 → DART learns slower but regularizes
    # class_weight="balanced",           #or use scale_pos_weight if you want custom ratio
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

print("Training binary classifier (with early stopping)...")
from lightgbm.callback import early_stopping, log_evaluation

binary_model.fit(
    X_train, y_train_bin,
    eval_set=[(X_val, y_val_bin)],
    eval_metric='binary_logloss',
    callbacks=[
        early_stopping(stopping_rounds=100),
        log_evaluation(period=100)
    ]
)

# Binary predictions with best iteration and custom threshold
binary_threshold = 0.45  # Lower threshold to catch more BFRB cases
best_iteration = getattr(binary_model, 'best_iteration_', None)
if best_iteration:
    y_prob_bin_train = binary_model.predict_proba(X_train, num_iteration=best_iteration)[:, 1]
    y_prob_bin_val = binary_model.predict_proba(X_val, num_iteration=best_iteration)[:, 1]
else:
    y_prob_bin_train = binary_model.predict_proba(X_train)[:, 1]
    y_prob_bin_val = binary_model.predict_proba(X_val)[:, 1]

y_pred_bin_train = (y_prob_bin_train > binary_threshold).astype(int)
y_pred_bin_val = (y_prob_bin_val > binary_threshold).astype(int)

# Binary metrics
bin_train_acc = accuracy_score(y_train_bin, y_pred_bin_train)
bin_val_acc = accuracy_score(y_val_bin, y_pred_bin_val)
bin_train_f1 = f1_score(y_train_bin, y_pred_bin_train)
bin_val_f1 = f1_score(y_val_bin, y_pred_bin_val)
bin_precision = precision_score(y_val_bin, y_pred_bin_val)
bin_recall = recall_score(y_val_bin, y_pred_bin_val)

print(f"Binary Training Accuracy: {bin_train_acc:.4f}")
print(f"Binary Validation Accuracy: {bin_val_acc:.4f}")
print(f"Binary Training F1: {bin_train_f1:.4f}")
print(f"Binary Validation F1: {bin_val_f1:.4f}")
print(f"Binary Precision: {bin_precision:.4f}")
print(f"Binary Recall: {bin_recall:.4f}")

# ---------------- MULTICLASS CLASSIFICATION ---------------- #
print("\n=== TRAINING MULTICLASS CLASSIFIER ===")

# Filter for BFRB samples only
X_train_multi = X_train[y_train_bin == 1]
y_train_multi = y_train_gesture[y_train_bin == 1]
X_val_multi = X_val[y_val_bin == 1]
y_val_multi = y_val_gesture[y_val_bin == 1]

print(f"Multiclass training samples: {len(X_train_multi)}")
print(f"Multiclass validation samples: {len(X_val_multi)}")

gesture_model = None
gesture_encoder = LabelEncoder()
gesture_encoder.fit(BFRB_GESTURES)

if len(X_train_multi) > 0:
    y_train_multi_enc = gesture_encoder.transform(y_train_multi)

    gesture_model = LGBMClassifier(
        objective="multiclass",
        num_class=len(BFRB_GESTURES),
        boosting_type="gbdt",
        max_depth=6,  # ↓ shallower to reduce overfitting
        num_leaves=96,  # balanced with depth
        learning_rate=0.03,  # slightly faster convergence
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=40,  # ↓ less strict, improves minority class learning
        n_estimators=3000,  # ↑ rely on early stopping
        reg_alpha=1.0,  # ↑ stronger L1 regularization
        reg_lambda=2.0,  # ↑ stronger L2 regularization
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )

    print("Training gesture classifier (with early stopping)...")

    # Train with early stopping if validation data available
    if len(X_val_multi) > 0:
        y_val_multi_enc = gesture_encoder.transform(y_val_multi)
        gesture_model.fit(
            X_train_multi, y_train_multi_enc,
            eval_set=[(X_val_multi, y_val_multi_enc)],
            eval_metric='multi_logloss',
            callbacks=[
                early_stopping(stopping_rounds=80),
                log_evaluation(period=100)
            ]
        )
    else:
        gesture_model.fit(X_train_multi, y_train_multi_enc)

    # Multiclass predictions (only on BFRB samples)
    if len(X_val_multi) > 0:
        gesture_best_iteration = getattr(gesture_model, 'best_iteration_', None)
        if gesture_best_iteration:
            y_pred_multi_enc = gesture_model.predict(X_val_multi, num_iteration=gesture_best_iteration)
        else:
            y_pred_multi_enc = gesture_model.predict(X_val_multi)

        y_pred_multi = gesture_encoder.inverse_transform(y_pred_multi_enc)
        y_val_multi_enc = gesture_encoder.transform(y_val_multi)

        # Multiclass metrics
        multi_acc = accuracy_score(y_val_multi_enc, y_pred_multi_enc)
        multi_f1_macro = f1_score(y_val_multi_enc, y_pred_multi_enc, average='macro')
        multi_f1_weighted = f1_score(y_val_multi_enc, y_pred_multi_enc, average='weighted')
        multi_precision = precision_score(y_val_multi_enc, y_pred_multi_enc, average='macro')
        multi_recall = recall_score(y_val_multi_enc, y_pred_multi_enc, average='macro')

        print(f"Multiclass Accuracy: {multi_acc:.4f}")
        print(f"Multiclass F1 (Macro): {multi_f1_macro:.4f}")
        print(f"Multiclass F1 (Weighted): {multi_f1_weighted:.4f}")
        print(f"Multiclass Precision (Macro): {multi_precision:.4f}")
        print(f"Multiclass Recall (Macro): {multi_recall:.4f}")
    else:
        print("No BFRB samples in validation set for multiclass evaluation")
        y_pred_multi = []
        multi_acc = multi_f1_macro = multi_precision = multi_recall = 0.0
else:
    print("No BFRB samples in training set for multiclass training")
    y_pred_multi = []
    multi_acc = multi_f1_macro = multi_precision = multi_recall = 0.0

# ---------------- FULL PIPELINE EVALUATION ---------------- #
print("\n=== FULL PIPELINE EVALUATION ===")

# Create final predictions combining binary + multiclass
final_predictions = []
final_encoder = LabelEncoder()
final_encoder.fit(ALL_GESTURES)

for i in range(len(X_val)):
    if y_pred_bin_val[i] == 1 and gesture_model is not None:  # Predicted as BFRB
        sample_x = X_val.iloc[i:i + 1]
        gesture_best_iteration = getattr(gesture_model, 'best_iteration_', None)
        if gesture_best_iteration:
            gesture_pred_enc = gesture_model.predict(sample_x, num_iteration=gesture_best_iteration)[0]
        else:
            gesture_pred_enc = gesture_model.predict(sample_x)[0]
        gesture_pred = gesture_encoder.inverse_transform([gesture_pred_enc])[0]
        final_predictions.append(gesture_pred)
    else:
        final_predictions.append("non_target")

print(f"Using binary threshold: {binary_threshold}")
print(f"Binary predictions - Positive class: {sum(y_pred_bin_val)}/{len(y_pred_bin_val)}")

# Ground truth for full evaluation
y_val_gesture_mapped = y_val_gesture.apply(lambda x: x if x in BFRB_GESTURES else "non_target")
y_true_final = final_encoder.transform(y_val_gesture_mapped)
y_pred_final = final_encoder.transform(final_predictions)

final_acc = accuracy_score(y_true_final, y_pred_final)
final_f1_macro = f1_score(y_true_final, y_pred_final, average='macro')
final_f1_weighted = f1_score(y_true_final, y_pred_final, average='weighted')

print(f"Final Pipeline Accuracy: {final_acc:.4f}")
print(f"Final Pipeline F1 (Macro): {final_f1_macro:.4f}")
print(f"Final Pipeline F1 (Weighted): {final_f1_weighted:.4f}")

# ---------------- CLASSIFICATION REPORTS ---------------- #
print("\n" + "=" * 50)
print("BINARY CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(y_val_bin, y_pred_bin_val,
                            target_names=['Non-BFRB', 'BFRB'], digits=4))

if len(X_val_multi) > 0:
    print("\n" + "=" * 50)
    print("MULTICLASS CLASSIFICATION REPORT (BFRB only)")
    print("=" * 50)
    print(classification_report(y_val_multi_enc, y_pred_multi_enc,
                                target_names=BFRB_GESTURES, digits=4))

print("\n" + "=" * 50)
print("FULL PIPELINE CLASSIFICATION REPORT")
print("=" * 50)
print(classification_report(y_true_final, y_pred_final,
                            target_names=final_encoder.classes_, digits=4))

# ---------------- COST MATRIX CALCULATION ---------------- #
print("\n" + "=" * 50)
print("COST MATRIX ANALYSIS")
print("=" * 50)

# Define cost matrix (False Negative more costly for medical application)
# Cost matrix: [[TN_cost, FP_cost], [FN_cost, TP_cost]]
cost_matrix = np.array([[0, 1],  # Non-BFRB: TN=0, FP=1
                        [5, 0]])  # BFRB: FN=5 (high cost), TP=0

binary_cm = confusion_matrix(y_val_bin, y_pred_bin_val)
total_cost = np.sum(binary_cm * cost_matrix)
avg_cost_per_sample = total_cost / len(y_val_bin)

print(f"Total misclassification cost: {total_cost}")
print(f"Average cost per sample: {avg_cost_per_sample:.4f}")
print("Cost Matrix (rows=actual, cols=predicted):")
print("        Non-BFRB  BFRB")
print(f"Non-BFRB    {cost_matrix[0, 0]}      {cost_matrix[0, 1]}")
print(f"BFRB        {cost_matrix[1, 0]}      {cost_matrix[1, 1]}")

# ---------------- VISUALIZATIONS ---------------- #
plt.style.use('default')
plt.rcParams.update({'font.size': 10, 'figure.dpi': 100})

# Create output directory
save_dir = r"D:\extra\bfrb_results"
os.makedirs(save_dir, exist_ok=True)

# 1. Binary Confusion Matrix
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Binary confusion matrix
sns.heatmap(binary_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0],
            xticklabels=['Non-BFRB', 'BFRB'],
            yticklabels=['Non-BFRB', 'BFRB'])
axes[0, 0].set_title(f'Binary Confusion Matrix\nAccuracy: {bin_val_acc:.3f}, F1: {bin_val_f1:.3f}')
axes[0, 0].set_ylabel('True Label')
axes[0, 0].set_xlabel('Predicted Label')

# Multiclass confusion matrix
if len(X_val_multi) > 0:
    multi_cm = confusion_matrix(y_val_multi_enc, y_pred_multi_enc)
    sns.heatmap(multi_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[0, 1],
                xticklabels=BFRB_GESTURES, yticklabels=BFRB_GESTURES)
    axes[0, 1].set_title(f'Multiclass Confusion Matrix\nAccuracy: {multi_acc:.3f}, F1: {multi_f1_macro:.3f}')
    axes[0, 1].set_ylabel('True Label')
    axes[0, 1].set_xlabel('Predicted Label')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].tick_params(axis='y', rotation=45)
else:
    axes[0, 1].text(0.5, 0.5, 'No BFRB samples\nin validation set',
                    ha='center', va='center', transform=axes[0, 1].transAxes)
    axes[0, 1].set_title('Multiclass Confusion Matrix')

# Performance comparison
metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
binary_scores = [bin_val_acc, bin_val_f1, bin_precision, bin_recall]
multi_scores = [multi_acc, multi_f1_macro, multi_precision, multi_recall]

x = np.arange(len(metrics))
width = 0.35

axes[1, 0].bar(x - width / 2, binary_scores, width, label='Binary', color='lightblue')
axes[1, 0].bar(x + width / 2, multi_scores, width, label='Multiclass', color='lightcoral')
axes[1, 0].set_xlabel('Metrics')
axes[1, 0].set_ylabel('Score')
axes[1, 0].set_title('Binary vs Multiclass Performance')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(metrics)
axes[1, 0].legend()
axes[1, 0].set_ylim(0, 1)

# Training vs Validation comparison
train_val_metrics = ['Accuracy', 'F1-Score']
train_scores = [bin_train_acc, bin_train_f1]
val_scores = [bin_val_acc, bin_val_f1]

x = np.arange(len(train_val_metrics))
axes[1, 1].bar(x - width / 2, train_scores, width, label='Training', color='lightgreen')
axes[1, 1].bar(x + width / 2, val_scores, width, label='Validation', color='lightcoral')
axes[1, 1].set_xlabel('Metrics')
axes[1, 1].set_ylabel('Score')
axes[1, 1].set_title('Training vs Validation Performance (Binary)')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(train_val_metrics)
axes[1, 1].legend()
axes[1, 1].set_ylim(0, 1)

plt.tight_layout()
plt.savefig(os.path.join(save_dir, "comprehensive_results.png"), dpi=300, bbox_inches='tight')
plt.show()

# 2. Full Pipeline Confusion Matrix
plt.figure(figsize=(12, 8))
final_cm = confusion_matrix(y_true_final, y_pred_final)
sns.heatmap(final_cm, annot=True, fmt='d', cmap='Greens',
            xticklabels=final_encoder.classes_,
            yticklabels=final_encoder.classes_)
plt.title(f'Full Pipeline Confusion Matrix\nAccuracy: {final_acc:.3f}, Macro F1: {final_f1_macro:.3f}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, "full_pipeline_confusion_matrix.png"), dpi=300, bbox_inches='tight')
plt.show()

print(f"\nResults and plots saved to: {save_dir}")
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Binary Classification - Train Acc: {bin_train_acc:.3f}, Val Acc: {bin_val_acc:.3f}")
print(f"Binary Classification - Train F1: {bin_train_f1:.3f}, Val F1: {bin_val_f1:.3f}")
print(f"Multiclass Classification - Accuracy: {multi_acc:.3f}, F1: {multi_f1_macro:.3f}")
print(f"Full Pipeline - Accuracy: {final_acc:.3f}, F1: {final_f1_macro:.3f}")
print(f"Total Cost (Binary): {total_cost}, Avg Cost/Sample: {avg_cost_per_sample:.3f}")

# ---------------- MODEL SAVING ---------------- #
print("\n" + "=" * 60)
print("SAVING MODELS")
print("=" * 60)

model_dir = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\XAI_Work\BFRB_XAI\Trained_Models"
os.makedirs(model_dir, exist_ok=True)

# Save binary model
binary_model_path = os.path.join(model_dir, "binary_classifier.pkl")
joblib.dump(binary_model, binary_model_path)
print(f"✓ Binary classifier saved: {binary_model_path}")

# Save multiclass model (if exists)
if gesture_model is not None:
    gesture_model_path = os.path.join(model_dir, "gesture_classifier.pkl")
    joblib.dump(gesture_model, gesture_model_path)
    print(f"✓ Gesture classifier saved: {gesture_model_path}")

# Save encoders
gesture_encoder_path = os.path.join(model_dir, "gesture_encoder.pkl")
final_encoder_path = os.path.join(model_dir, "final_encoder.pkl")
joblib.dump(gesture_encoder, gesture_encoder_path)
joblib.dump(final_encoder, final_encoder_path)
print(f"✓ Encoders saved")

# Save feature names for LIME
feature_names_path = os.path.join(model_dir, "feature_names.pkl")
joblib.dump(X.columns.tolist(), feature_names_path)
print(f"✓ Feature names saved: {len(X.columns)} features")

# Save metadata
metadata = {
    'binary_threshold': binary_threshold,
    'bfrb_gestures': BFRB_GESTURES,
    'all_gestures': ALL_GESTURES,
    'feature_count': X.shape[1],
    'train_size': len(X_train),
    'val_size': len(X_val)
}
metadata_path = os.path.join(model_dir, "metadata.pkl")
joblib.dump(metadata, metadata_path)
print(f"✓ Metadata saved")

print("\n" + "=" * 60)
print(f"All models saved to: {model_dir}")
print("=" * 60)