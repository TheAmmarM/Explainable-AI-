#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIME CALIBRATION TOOL - STANDALONE
===================================
Run this BEFORE the full analysis to find the optimal kernel_width.
It tests different widths to maximize Fidelity (R²).

Usage:
    python lime_calibration.py

Output:
    - Shows which kernel_width gives best R² scores
    - Tells you what to update in your config
"""

import numpy as np
import pandas as pd
import joblib
from lime import lime_tabular
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# ==============================================================================
# CONFIGURATION - UPDATE THESE PATHS!
# ==============================================================================

# Update these to match your system
MODEL_DIR = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\XAI_Work\BFRB_XAI\Trained_Models"
DATA_FOLDER = r"D:\Ammar\PythonProjects\Machine_Learning_Projects\CMI_Sensor_Kaggle"


# ==============================================================================
# CALIBRATION FUNCTION
# ==============================================================================

def prepare_data_for_calibration():
    """Prepare data using the same logic as the main analysis."""
    print("📂 Loading and preparing data...")

    # Load model and metadata
    model_path = Path(MODEL_DIR)
    binary_model = joblib.load(model_path / "binary_model.pkl")
    feature_names = joblib.load(model_path / "feature_names.pkl")
    metadata = joblib.load(model_path / "metadata.pkl")

    # Load raw data
    df = pd.read_csv(Path(DATA_FOLDER) / "train.csv")
    df["gesture"] = df["gesture"].fillna("non_target")

    # Feature engineering (same as main analysis)
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
    numeric_features = df.select_dtypes(include=[np.number]).columns
    numeric_features = [c for c in numeric_features if c not in ['sequence_id', 'row_id']]

    agg_funcs = ['mean', 'std', 'min', 'max', 'median', 'skew']
    agg_df = df.groupby("sequence_id")[numeric_features].agg(agg_funcs)
    agg_df.columns = ['_'.join(col) for col in agg_df.columns]
    agg_df.reset_index(inplace=True)

    # Labels
    labels_df = df.groupby("sequence_id").first().reset_index()[["sequence_id", "gesture"]]
    labels_df["binary_label"] = labels_df["gesture"].apply(
        lambda x: 1 if x in metadata['bfrb_gestures'] else 0
    )

    # Merge
    full_df = pd.merge(agg_df, labels_df, on="sequence_id")
    X = full_df.drop(columns=["sequence_id", "gesture", "binary_label"])
    y = full_df["binary_label"]

    # Keep only training features
    X = X[feature_names].fillna(X[feature_names].median())

    # Train-test split (same as main)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"✓ Data prepared: {len(X_train)} train, {len(X_test)} test samples")

    return binary_model, X_train, X_test, feature_names


def calibrate_lime():
    """Main calibration function."""
    print("=" * 80)
    print("🔧 LIME CALIBRATION: FINDING OPTIMAL KERNEL WIDTH")
    print("=" * 80)
    print()

    # Prepare data
    binary_model, X_train, X_test, feature_names = prepare_data_for_calibration()

    # Select 20 diverse samples from test set for calibration
    n_calib_samples = min(20, len(X_test))
    calib_indices = np.linspace(0, len(X_test) - 1, n_calib_samples, dtype=int)
    test_samples = X_test.iloc[calib_indices].values

    print(f"🧪 Testing calibration on {n_calib_samples} samples...")
    print()

    # Define widths to test (logarithmic scale)
    widths_to_test = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0]

    print(f"{'Kernel Width':<15} | {'Mean R²':<12} | {'Min R²':<12} | {'Status'}")
    print("-" * 65)

    best_width = None
    best_r2 = -1
    results = {}

    for width in widths_to_test:
        # Initialize LIME with this width
        explainer = lime_tabular.LimeTabularExplainer(
            training_data=X_train.values,
            feature_names=feature_names,
            class_names=['Non-BFRB', 'BFRB'],
            mode='classification',
            discretize_continuous=True,
            kernel_width=width,
            random_state=42
        )

        r2_scores = []

        for instance in test_samples:
            try:
                exp = explainer.explain_instance(
                    instance,
                    binary_model.predict_proba,
                    num_features=10,  # Just top 10 for speed
                    num_samples=1000  # Enough for stable R²
                )

                # LIME's internal score is the R² of the local linear model
                r2_scores.append(exp.score)
            except Exception as e:
                # If explanation fails, score is 0
                r2_scores.append(0.0)

        mean_r2 = np.mean(r2_scores)
        min_r2 = np.min(r2_scores)

        # Determine status
        if mean_r2 > 0.7:
            status = "✅ Excellent"
        elif mean_r2 > 0.6:
            status = "✅ Good"
        elif mean_r2 > 0.4:
            status = "⚠️  Acceptable"
        elif mean_r2 > 0.2:
            status = "⚠️  Poor"
        else:
            status = "❌ Bad"

        print(f"{width:<15.2f} | {mean_r2:<12.4f} | {min_r2:<12.4f} | {status}")

        results[width] = mean_r2

        if mean_r2 > best_r2:
            best_r2 = mean_r2
            best_width = width

    # Print results
    print("-" * 65)
    print()
    print("=" * 80)
    print(f"🏆 BEST KERNEL WIDTH: {best_width} (Mean R² = {best_r2:.4f})")
    print("=" * 80)
    print()

    # Interpretation and recommendations
    if best_r2 >= 0.7:
        print("✅ EXCELLENT! This width will give high-quality explanations.")
        print()
        print("📝 ACTION REQUIRED:")
        print(f"   Update your Config class:")
        print(f"   kernel_width: float = {best_width}  # (was 0.75)")
        print()
        print("   Then run your main LIME analysis.")

    elif best_r2 >= 0.5:
        print("✓ GOOD! This width will give usable explanations.")
        print()
        print("📝 ACTION REQUIRED:")
        print(f"   Update your Config class:")
        print(f"   kernel_width: float = {best_width}  # (was 0.75)")
        print()
        print("⚠️  Note: Results will be usable but mention R²={:.2f} in your paper.".format(best_r2))

    elif best_r2 >= 0.3:
        print("⚠️  ACCEPTABLE but not ideal.")
        print()
        print("📝 RECOMMENDED ACTIONS:")
        print(f"   1. Update kernel_width to {best_width} (best available)")
        print("   2. Try these alternatives:")
        print("      - Set discretize_continuous=False in LIME config")
        print("      - Test even larger widths: [100, 200, 500]")
        print("      - Your model may have complex non-linear interactions")

    else:
        print("❌ WARNING: Even the best width gives poor R²!")
        print()
        print("🔍 DIAGNOSIS:")
        print("   This suggests one of these issues:")
        print("   1. Model uses complex non-linear interactions LIME can't approximate")
        print("   2. Features are highly correlated (multicollinearity)")
        print("   3. Data needs different preprocessing")
        print()
        print("💡 TRY THESE FIXES:")
        print("   1. Set discretize_continuous=False")
        print("   2. Test much larger widths: [100, 200, 500, 1000]")
        print("   3. Consider using SHAP instead of LIME for this model")
        print("   4. Check feature correlations and remove redundant ones")

    print()
    print("=" * 80)

    # Save results to file for reference
    results_file = Path("lime_calibration_results.txt")
    with open(results_file, 'w') as f:
        f.write("LIME KERNEL WIDTH CALIBRATION RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Best Kernel Width: {best_width}\n")
        f.write(f"Best Mean R²: {best_r2:.4f}\n\n")
        f.write("All Results:\n")
        f.write("-" * 30 + "\n")
        for width, r2 in sorted(results.items()):
            f.write(f"Width {width:6.2f}: R² = {r2:.4f}\n")

    print(f"📄 Results saved to: {results_file}")
    print()


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    try:
        calibrate_lime()

        print("✅ Calibration complete!")
        print()
        print("Next steps:")
        print("1. Update kernel_width in your Config class (see above)")
        print("2. Run your main LIME analysis: python lime_consolidated_all_in_one.py")
        print()

    except FileNotFoundError as e:
        print()
        print("❌ ERROR: File not found!")
        print(f"   {e}")
        print()
        print("📝 Please update these paths at the top of this script:")
        print(f"   MODEL_DIR = {MODEL_DIR}")
        print(f"   DATA_FOLDER = {DATA_FOLDER}")
        print()

    except Exception as e:
        print()
        print("❌ ERROR occurred during calibration:")
        print(f"   {e}")
        print()
        import traceback

        traceback.print_exc()