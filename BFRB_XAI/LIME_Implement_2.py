# LIME_Implementation.py
# Simple LIME Explainability for BFRB Detection Models

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from lime import lime_tabular
from lime.submodular_pick import SubmodularPick

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
MODEL_DIR = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\XAI_Work\BFRB_XAI\Trained_Models"
DATA_FOLDER = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\CMI_Sensor_Kaggle"
OUTPUT_DIR = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\XAI_Work\BFRB_XAI\Figures_and_Graphs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# STEP 1: LOAD SAVED MODELS AND DATA
# ============================================================
print("=" * 60)
print("LOADING MODELS AND DATA")
print("=" * 60)

# Load models
binary_model = joblib.load(os.path.join(MODEL_DIR, "binary_model.pkl"))
gesture_model = joblib.load(os.path.join(MODEL_DIR, "gesture_model.pkl"))
feature_names = joblib.load(os.path.join(MODEL_DIR, "feature_names.pkl"))
metadata = joblib.load(os.path.join(MODEL_DIR, "metadata.pkl"))

print(f"✓ Binary model loaded")
print(f"✓ Gesture model loaded")
print(f"✓ Feature names loaded: {len(feature_names)} features")

# Load and prepare data (same preprocessing as training)
print("\nLoading validation data...")
df = pd.read_csv(os.path.join(DATA_FOLDER, "train.csv"))
df["gesture"] = df["gesture"].fillna("non_target")

# Feature engineering (same as training)
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

# Aggregate features
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

# Keep only the features used in training
X = X[feature_names].fillna(X[feature_names].median())

# Train-val split (same as training)
from sklearn.model_selection import StratifiedShuffleSplit

splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(splitter.split(X, y_binary))
X_train = X.iloc[train_idx]
X_val = X.iloc[val_idx]
y_val = y_binary.iloc[val_idx]

print(f"✓ Data prepared: {len(X_val)} validation samples")

# ============================================================
# STEP 2: INITIALIZE LIME EXPLAINER
# ============================================================
print("\n" + "=" * 60)
print("INITIALIZING LIME EXPLAINER")
print("=" * 60)

explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=feature_names,
    class_names=['Non-BFRB', 'BFRB'],
    mode='classification',
    discretize_continuous=True,
    random_state=42
)

print("✓ LIME explainer initialized")
print(f"  - Training samples: {len(X_train)}")
print(f"  - Features: {len(feature_names)}")
print(f"  - Discretization: Enabled (for interpretability)")

# ============================================================
# STEP 3: GENERATE EXPLANATIONS FOR INDIVIDUAL SAMPLES
# ============================================================
print("\n" + "=" * 60)
print("GENERATING INDIVIDUAL EXPLANATIONS")
print("=" * 60)

# Select diverse samples
np.random.seed(42)
n_samples = 10
sample_indices = np.random.choice(len(X_val), n_samples, replace=False)

explanations = []
print(f"\nExplaining {n_samples} samples...")

for i, idx in enumerate(sample_indices):
    sample = X_val.iloc[idx].values
    true_label = y_val.iloc[idx]

    # Generate explanation
    exp = explainer.explain_instance(
        data_row=sample,
        predict_fn=binary_model.predict_proba,
        num_features=10,  # Top 10 features
        num_samples=5000  # 5000 perturbations
    )

    explanations.append(exp)

    # Get prediction
    pred_proba = binary_model.predict_proba(sample.reshape(1, -1))[0]
    pred_class = 1 if pred_proba[1] > 0.45 else 0

    print(f"  Sample {i + 1}: True={true_label}, Pred={pred_class}, "
          f"Confidence={pred_proba[pred_class]:.3f}")

print(f"✓ Generated {len(explanations)} explanations")

# ============================================================
# STEP 4: VISUALIZE INDIVIDUAL EXPLANATIONS
# ============================================================
print("\n" + "=" * 60)
print("VISUALIZING EXPLANATIONS")
print("=" * 60)

# Save first 5 individual explanations
for i, exp in enumerate(explanations[:5]):
    fig = exp.as_pyplot_figure()
    true_label = y_val.iloc[sample_indices[i]]
    fig.suptitle(f'LIME Explanation - Sample {i + 1}\n'
                 f'True Label: {"BFRB" if true_label == 1 else "Non-BFRB"}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUTPUT_DIR, f'LIME_explanation_sample_{i + 1}.png'),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()
    print(f"  ✓ Saved explanation for sample {i + 1}")

# ============================================================
# STEP 5: GLOBAL FEATURE IMPORTANCE (AGGREGATION)
# ============================================================
print("\n" + "=" * 60)
print("COMPUTING GLOBAL FEATURE IMPORTANCE")
print("=" * 60)

# Aggregate importance scores across all explanations
feature_importance = {feat: [] for feat in feature_names}

for exp in explanations:
    for feature, weight in exp.as_list():
        # Extract clean feature name
        feat_name = feature.split(' ')[0] if ' ' in feature else feature
        if feat_name in feature_importance:
            feature_importance[feat_name].append(abs(weight))

# Calculate mean importance
global_importance = {
    feat: np.mean(scores) if scores else 0
    for feat, scores in feature_importance.items()
}

# Sort by importance
sorted_importance = sorted(
    global_importance.items(),
    key=lambda x: x[1],
    reverse=True
)

print("\nTop 15 Most Important Features (Global):")
print("-" * 60)
for rank, (feat, score) in enumerate(sorted_importance[:15], 1):
    print(f"{rank:2d}. {feat:45s} {score:.4f}")

# Visualize global importance
plt.figure(figsize=(12, 8))
top_n = 20
top_features = sorted_importance[:top_n]
features, scores = zip(*top_features)

colors = ['#2ecc71' if s > np.median(scores) else '#e74c3c' for s in scores]
plt.barh(range(len(features)), scores, color=colors, alpha=0.8)
plt.yticks(range(len(features)), features, fontsize=10)
plt.xlabel('Mean Absolute LIME Importance', fontsize=12)
plt.title(f'Global Feature Importance from LIME\n'
          f'Aggregated from {n_samples} Local Explanations',
          fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, 'LIME_global_importance.png'),
    dpi=300,
    bbox_inches='tight'
)
plt.show()
print("✓ Global importance plot saved")

# ============================================================
# STEP 6: SUBMODULAR PICK (REPRESENTATIVE SAMPLES)
# ============================================================
print("\n" + "=" * 60)
print("FINDING REPRESENTATIVE SAMPLES")
print("=" * 60)

try:
    sp_obj = SubmodularPick(
        explainer,
        X_val.values[:50],  # Use subset for speed
        binary_model.predict_proba,
        num_features=10,
        num_exps_desired=5
    )

    print(f"✓ Found {len(sp_obj.explanations)} representative samples")

    # Visualize representative explanations
    for i, exp in enumerate(sp_obj.explanations):
        fig = exp.as_pyplot_figure()
        fig.suptitle(f'Representative Sample {i + 1}\n(Selected by Submodular Pick)',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(
            os.path.join(OUTPUT_DIR, f'LIME_representative_{i + 1}.png'),
            dpi=300,
            bbox_inches='tight'
        )
        plt.close()
        print(f"  ✓ Saved representative sample {i + 1}")

except Exception as e:
    print(f"⚠ Submodular Pick failed: {e}")

# ============================================================
# STEP 7: COMPARE WITH MODEL FEATURE IMPORTANCE
# ============================================================
print("\n" + "=" * 60)
print("COMPARING LIME vs MODEL IMPORTANCE")
print("=" * 60)

# Get LightGBM feature importance
lgbm_importance = dict(zip(feature_names, binary_model.feature_importances_))
lgbm_sorted = sorted(lgbm_importance.items(), key=lambda x: x[1], reverse=True)

# Comparison plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# LIME
top_lime = sorted_importance[:10]
axes[0].barh(range(len(top_lime)), [s for _, s in top_lime], color='steelblue', alpha=0.8)
axes[0].set_yticks(range(len(top_lime)))
axes[0].set_yticklabels([f for f, _ in top_lime], fontsize=10)
axes[0].set_xlabel('LIME Importance', fontsize=12)
axes[0].set_title('Top 10 Features - LIME\n(Local Explanations)', fontsize=13, fontweight='bold')
axes[0].invert_yaxis()
axes[0].grid(axis='x', alpha=0.3)

# LightGBM
top_lgbm = lgbm_sorted[:10]
axes[1].barh(range(len(top_lgbm)), [s for _, s in top_lgbm], color='coral', alpha=0.8)
axes[1].set_yticks(range(len(top_lgbm)))
axes[1].set_yticklabels([f for f, _ in top_lgbm], fontsize=10)
axes[1].set_xlabel('LightGBM Gain', fontsize=12)
axes[1].set_title('Top 10 Features - LightGBM\n(Global Model)', fontsize=13, fontweight='bold')
axes[1].invert_yaxis()
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(
    os.path.join(OUTPUT_DIR, 'LIME_vs_LGBM.png'),
    dpi=300,
    bbox_inches='tight'
)
plt.show()
print("✓ Comparison plot saved")

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("LIME ANALYSIS COMPLETE")
print("=" * 60)
print(f"✓ Explained {n_samples} samples")
print(f"✓ Generated global importance from local explanations")
print(f"✓ Top feature: {sorted_importance[0][0]}")
print(f"✓ Results saved to: {OUTPUT_DIR}")
print("=" * 60)

# Save summary
summary = {
    'global_importance': sorted_importance,
    'top_10_features': [f for f, _ in sorted_importance[:10]],
    'n_samples_explained': n_samples,
    'methodology': {
        'num_perturbations': 5000,
        'num_features_shown': 10,
        'discretization': True
    }
}
joblib.dump(summary, os.path.join(OUTPUT_DIR, 'LIME_summary.pkl'))
print("\n✓ Summary saved to LIME_summary.pkl")