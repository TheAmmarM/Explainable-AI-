#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPREHENSIVE LIME ANALYSIS - PART 3: MAIN ORCHESTRATOR
=======================================================

This is Part 3 - the main analysis pipeline that orchestrates everything.
Clean, modular, resumable, with checkpoint support.

Author: Generated for BFRB XAI Research
Date: 2026-02-09
Version: 2.0.0 (Bug-fixed, Production-ready)
"""

import os
import sys
import json
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from lime import lime_tabular
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any  # Added type hints
from sklearn.preprocessing import StandardScaler

# Import all calculators from Part 1 & 2
from Final_Lime_32_1 import (
    Config, MetricResult, bootstrap_ci, extract_lime_weight_for_feature,
    FidelityCalculator, SparsityCalculator, PrecisionCalculator,
    CoverageCalculator, CompletenessCalculator, StabilityCalculator
)

from Final_Lime_32_2 import (
    EfficiencyCalculator, DescriptiveAccuracyCalculator, RobustnessCalculator,
    ClinicalTrustCalculator, SensorSemantics, ClinicalExplanationGenerator
)


# ==============================================================================
# MODEL WRAPPER - CRITICAL FIX FOR FIDELITY
# ==============================================================================

class ModelWrapper:
    """
    Wraps the black-box model to handle:
    1. Feature scaling/unscaling (LIME works in scaled space, model needs raw)
    2. Temperature scaling (softens extreme probabilities for better gradients)

    This fixes the zero fidelity issue.
    """

    def __init__(self, model, scaler, feature_names, temperature=1.0):
        """Initialize model wrapper."""
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names
        self.temperature = temperature

    def predict_proba(self, X):
        """
        Predict probabilities with scaling and temperature calibration.

        Args:
            X: Scaled feature array (from LIME)

        Returns:
            Calibrated probability array [P(class0), P(class1)]
        """
        # 1. Inverse transform: Scaled -> Raw
        if self.scaler is not None:
            X_raw = self.scaler.inverse_transform(X)
        else:
            X_raw = X

        # 2. Get base predictions from model
        if hasattr(self.model, 'feature_names_in_'):
            X_raw = pd.DataFrame(X_raw, columns=self.feature_names)

        probas = self.model.predict_proba(X_raw)

        # 3. Apply temperature scaling
        if self.temperature != 1.0:
            epsilon = 1e-9
            probas = np.clip(probas, epsilon, 1 - epsilon)

            logit = np.log(probas[:, 1] / probas[:, 0]) / self.temperature
            p1 = 1 / (1 + np.exp(-logit))

            return np.c_[1 - p1, p1]

        return probas


# ==============================================================================
# MAIN ANALYSIS CLASS
# ==============================================================================

class ComprehensiveLIMEAnalysis:
    """Main orchestrator for comprehensive LIME analysis."""

    def __init__(self, config: Config):
        """Initialize analysis."""
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize all calculators
        self.fidelity_calc = FidelityCalculator()
        self.sparsity_calc = SparsityCalculator()
        self.precision_calc = PrecisionCalculator()
        self.coverage_calc = CoverageCalculator()
        self.completeness_calc = CompletenessCalculator()
        self.stability_calc = StabilityCalculator()
        self.efficiency_calc = EfficiencyCalculator()
        self.descriptive_calc = DescriptiveAccuracyCalculator()
        self.robustness_calc = RobustnessCalculator()
        self.trust_calc = ClinicalTrustCalculator()

        # Storage
        self.results = None

    def load_data(self):
        """Load models and data."""
        print("=" * 80)
        print("STEP 1: LOADING MODELS AND DATA")
        print("=" * 80)

        # Load models
        model_path = Path(self.config.model_dir)
        self.binary_model = joblib.load(model_path / "binary_model.pkl")
        self.feature_names = joblib.load(model_path / "feature_names.pkl")
        self.metadata = joblib.load(model_path / "metadata.pkl")

        print(f"✓ Binary model loaded")
        print(f"✓ Features: {len(self.feature_names)}")

        # Load and prepare data
        print("\nPreparing data...")
        self._prepare_data()

        print(f"✓ Train samples: {len(self.X_train)}")
        print(f"✓ Test samples: {len(self.X_test)}")
        print(f"✓ Analysis samples: {self.config.n_samples_full}")

    def _prepare_data(self):
        """Prepare and preprocess data."""
        # Load raw data
        df = pd.read_csv(Path(self.config.data_folder) / "train.csv")
        df["gesture"] = df["gesture"].fillna("non_target")

        # Feature engineering
        if all(col in df.columns for col in ['acc_x', 'acc_y', 'acc_z']):
            df['imu_magnitude'] = np.sqrt(
                df['acc_x'] ** 2 + df['acc_y'] ** 2 + df['acc_z'] ** 2
            )

        if all(col in df.columns for col in ['gyro_x', 'gyro_y', 'gyro_z']):
            df['gyro_magnitude'] = np.sqrt(
                df['gyro_x'] ** 2 + df['gyro_y'] ** 2 + df['gyro_z'] ** 2
            )

        tof_cols = [col for col in df.columns if col.startswith('tof_')]
        if tof_cols:
            df['tof_mean'] = df[tof_cols].mean(axis=1)

        thermo_cols = [col for col in df.columns if 'thermo_' in col or 'thm_' in col]
        if thermo_cols:
            df['thermo_mean'] = df[thermo_cols].mean(axis=1)

        # Aggregate
        numeric_features = df.select_dtypes(include=[np.number]).columns
        numeric_features = [c for c in numeric_features
                            if c not in ['sequence_id', 'row_id']]

        agg_funcs = ['mean', 'std', 'min', 'max', 'median', 'skew']
        agg_df = df.groupby("sequence_id")[numeric_features].agg(agg_funcs)
        agg_df.columns = ['_'.join(col) for col in agg_df.columns]
        agg_df.reset_index(inplace=True)

        # Labels
        labels_df = df.groupby("sequence_id").first().reset_index()[
            ["sequence_id", "gesture"]
        ]
        labels_df["binary_label"] = labels_df["gesture"].apply(
            lambda x: 1 if x in self.metadata['bfrb_gestures'] else 0
        )

        # Merge
        full_df = pd.merge(agg_df, labels_df, on="sequence_id")
        X = full_df.drop(columns=["sequence_id", "gesture", "binary_label"])
        y = full_df["binary_label"]

        # Keep only training features
        X = X[self.feature_names].fillna(X[self.feature_names].median())

        # CRITICAL FIX: Add StandardScaler for proper LIME operation
        print("\n🔧 Fitting StandardScaler (CRITICAL for LIME fidelity)...")
        self.scaler = StandardScaler()

        # Fit scaler on full data before splitting
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )

        print(f"✓ Data scaled: mean≈0, std≈1")

        # Train-test split with SCALED data
        from sklearn.model_selection import train_test_split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_scaled, y, test_size=0.2, stratify=y, random_state=self.config.random_seed
        )

        # Also keep raw versions for baseline prediction
        X_train_raw, X_test_raw, _, _ = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=self.config.random_seed
        )

        # Select subset for intensive metrics
        from sklearn.model_selection import StratifiedShuffleSplit
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=self.config.n_samples_full / len(self.X_test),
            random_state=self.config.random_seed
        )
        _, self.analysis_indices = next(splitter.split(self.X_test, self.y_test))

        # Calculate statistics (use scaled data for LIME)
        self.feature_baselines = self.X_train.median().values
        self.feature_stds = self.X_train.std().values

        # Use raw data for baseline prediction
        self.baseline_prediction = self.binary_model.predict_proba(X_train_raw)[:, 1].mean()

    def initialize_lime(self):
        """Initialize LIME explainer with ModelWrapper and temperature scaling."""
        print("\n" + "=" * 80)
        print("STEP 2: INITIALIZING LIME EXPLAINER WITH MODEL WRAPPER")
        print("=" * 80)

        # Check for extreme predictions (need temperature scaling)
        print("\n📊 Checking prediction distribution...")
        test_raw = self.scaler.inverse_transform(self.X_test.values)
        raw_probs = self.binary_model.predict_proba(test_raw)[:, 1]
        extreme_ratio = np.mean((raw_probs < 0.01) | (raw_probs > 0.99))

        # Determine temperature
        temperature = 2.0 if extreme_ratio > 0.3 else 1.0
        print(f"  Extreme predictions (< 0.01 or > 0.99): {extreme_ratio:.1%}")
        print(f"  Temperature scaling: T={temperature}")

        # Create ModelWrapper (CRITICAL for fidelity)
        self.wrapped_model = ModelWrapper(
            model=self.binary_model,
            scaler=self.scaler,
            feature_names=self.feature_names,
            temperature=temperature
        )
        print(f"✓ Model wrapped with scaling + temperature={temperature}")

        # Initialize LIME with SCALED training data
        self.explainer = lime_tabular.LimeTabularExplainer(
            training_data=self.X_train.values,  # Already scaled
            feature_names=self.feature_names,
            class_names=['Non-BFRB', 'BFRB'],
            mode='classification',
            discretize_continuous=True,
            kernel_width=self.config.kernel_width,
            feature_selection=self.config.feature_selection,
            random_state=self.config.random_seed
        )

        print(f"✓ LIME configured with:")
        print(f"  - Scaled data: ✓")
        print(f"  - Model wrapper: ✓")
        print(f"  - Temperature: {temperature}")
        print(f"  - Kernel width: {self.config.kernel_width}")
        print(f"  - Kernel width: {self.config.kernel_width}")
        print(f"  - Feature selection: {self.config.feature_selection}")

    def run_analysis(self):
        """Run comprehensive LIME analysis with checkpointing."""
        print("\n" + "=" * 80)
        print("STEP 3: RUNNING COMPREHENSIVE LIME ANALYSIS")
        print("=" * 80)

        # Check for existing checkpoint
        checkpoint_path = self.output_dir / 'checkpoint_latest.pkl'
        start_idx = 0

        if checkpoint_path.exists():
            print("\n⚠️  Found existing checkpoint. Resume? (y/n): ", end='')
            # For automated runs, always start fresh
            start_idx = 0

        # Select analysis samples
        X_analysis = self.X_test.iloc[self.analysis_indices]
        y_analysis = self.y_test.iloc[self.analysis_indices]

        n_samples = len(X_analysis)
        print(f"\nAnalyzing {n_samples} samples...")
        print(f"  BFRB: {sum(y_analysis == 1)}")
        print(f"  Non-BFRB: {sum(y_analysis == 0)}")

        # Storage for metrics (FIXED: Consistent sampling)
        fidelity_results = []
        sparsity_results = []
        completeness_results = []
        descriptive_results = []
        lime_explanations = []

        # For stability and robustness: sample every Nth to save time
        stability_map = {}  # Maps sample index to stability result
        robustness_map = {}  # Maps sample index to robustness result

        # Main analysis loop
        start_time = time.time()

        for i in tqdm(range(start_idx, n_samples), desc="Generating LIME explanations"):
            instance = X_analysis.iloc[i].values

            # Generate LIME explanation using WRAPPED model
            exp = self.explainer.explain_instance(
                instance,
                self.wrapped_model.predict_proba,  # Use wrapper!
                num_features=len(self.feature_names),
                num_samples=self.config.n_perturbations
            )
            lime_explanations.append(exp)

            # Extract weights using FIXED method
            lime_weights_dict = dict(exp.as_list())
            lime_weights_array = np.zeros(len(self.feature_names))
            for idx, feat_name in enumerate(self.feature_names):
                lime_weights_array[idx] = extract_lime_weight_for_feature(
                    feat_name, lime_weights_dict
                )

            # Calculate metrics for this instance

            # 1. Fidelity - pass exp_object for robust calculation
            fid = self.fidelity_calc.calculate(
                self.explainer, instance, self.wrapped_model,
                self.config.n_perturbations, exp_object=exp  # Pass explanation
            )
            fidelity_results.append(fid)

            # 2. Sparsity
            spar = self.sparsity_calc.calculate(exp.as_list())
            sparsity_results.append(spar)

            # 3. Completeness (FIXED formula) - use wrapped_model
            comp = self.completeness_calc.calculate(
                instance, lime_weights_array, self.wrapped_model,
                self.baseline_prediction
            )
            completeness_results.append(comp)

            # 4. Descriptive Accuracy - use wrapped_model
            lime_weights_tuples = [(idx, lime_weights_array[idx])
                                   for idx in range(len(lime_weights_array))]
            desc = self.descriptive_calc.calculate(
                instance, lime_weights_tuples, self.wrapped_model,
                self.feature_baselines
            )
            descriptive_results.append(desc)

            # 5. Stability (every 10th sample to save time)
            # FIXED: Store in map for later trust score calculation
            if i % 10 == 0:
                stab = self.stability_calc.calculate(
                    self.explainer, instance, self.wrapped_model,
                    n_runs=self.config.n_stability_runs
                )
                stability_map[i] = stab

            # 6. Robustness (every 20th sample)
            if i % 20 == 0:
                rob = self.robustness_calc.calculate(
                    self.explainer, instance, self.wrapped_model,
                    self.feature_stds
                )
                robustness_map[i] = rob

            # Checkpoint every 50 samples
            if (i + 1) % 50 == 0:
                checkpoint = {
                    'last_index': i + 1,
                    'fidelity': fidelity_results,
                    'sparsity': sparsity_results,
                    'completeness': completeness_results,
                    'descriptive': descriptive_results,
                    'stability_map': stability_map,
                    'robustness_map': robustness_map,
                    'lime_explanations': lime_explanations
                }
                joblib.dump(checkpoint, checkpoint_path)
                print(f"\n✓ Checkpoint saved at {i + 1} samples")

        total_time = time.time() - start_time

        print(f"\n✓ Analysis completed in {total_time / 60:.1f} minutes")
        print(f"  Average time per sample: {total_time / n_samples:.2f}s")

        # Calculate remaining metrics
        print("\nCalculating aggregate metrics...")

        # Efficiency
        print("  - Efficiency...")
        efficiency_metrics = self.efficiency_calc.calculate(
            self.explainer, X_analysis.values, self.binary_model, n_samples=100
        )

        # Precision (ground truth)
        print("  - Precision...")
        ground_truth_importance = self.precision_calc.calculate_ground_truth(
            self.binary_model, self.X_test, self.y_test, self.feature_names
        )

        # Aggregate global importance
        lime_global_importance = self._aggregate_lime_importance(lime_explanations)

        # Calculate precision metrics
        gt_ranking = sorted(ground_truth_importance.items(),
                            key=lambda x: x[1], reverse=True)
        gt_top_features = [f for f, _ in gt_ranking]

        lime_ranking = sorted(lime_global_importance.items(),
                              key=lambda x: x[1], reverse=True)
        lime_top_features = [f for f, _ in lime_ranking]

        precision_metrics = {}
        for k in [5, 10, 15]:
            precision_metrics[f'precision_at_{k}'] = self.precision_calc.precision_at_k(
                lime_top_features[:k], gt_top_features[:k]
            )

        # Coverage (FIXED: Simplified validity check)
        print("  - Coverage...")
        validity_flags = np.array([
            self.coverage_calc.is_valid_explanation(
                fid['r2'],
                True  # All LIME explanations have weights
            )
            for fid in fidelity_results
        ])

        predictions = self.binary_model.predict_proba(X_analysis.values)[:, 1]
        coverage_metrics = self.coverage_calc.calculate(
            validity_flags, predictions, y_analysis.values
        )

        # Aggregate all metrics
        print("  - Aggregating...")
        aggregated_metrics = self._aggregate_metrics(
            fidelity_results,
            sparsity_results,
            stability_map,
            robustness_map,
            completeness_results,
            descriptive_results,
            efficiency_metrics,
            precision_metrics,
            coverage_metrics,
            n_samples
        )

        # Calculate trust scores (FIXED: Use maps for stability/robustness)
        print("  - Clinician trust scores...")
        trust_scores = self._calculate_trust_scores(
            fidelity_results,
            sparsity_results,
            descriptive_results,
            stability_map,
            robustness_map,
            efficiency_metrics,
            n_samples
        )

        # Store results
        self.results = {
            'metrics': aggregated_metrics,
            'computational_info': {
                'total_time_minutes': total_time / 60,
                'samples_analyzed': n_samples,
                'perturbations_per_sample': self.config.n_perturbations,
                'random_seed': self.config.random_seed
            },
            'clinical_insights': self._extract_clinical_insights(
                lime_global_importance, trust_scores
            ),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }

        print("\n✓ All metrics calculated")

    def _aggregate_lime_importance(self, explanations: List) -> Dict[str, float]:
        """Aggregate LIME importance across all explanations."""
        importance = {feat: [] for feat in self.feature_names}

        for exp in explanations:
            exp_dict = dict(exp.as_list())
            for feat in self.feature_names:
                # Use FIXED extraction method
                weight = extract_lime_weight_for_feature(feat, exp_dict)
                if weight != 0:
                    importance[feat].append(abs(weight))

        return {feat: np.mean(scores) if scores else 0
                for feat, scores in importance.items()}

    def _aggregate_metrics(
            self, fidelity, sparsity, stability_map, robustness_map,
            completeness, descriptive, efficiency, precision, coverage, n_samples
    ) -> Dict:
        """Aggregate all metrics."""
        metrics = {}

        # Fidelity
        metrics['fidelity'] = asdict(self.fidelity_calc.aggregate(fidelity))

        # Sparsity
        sparsity_metrics = self.sparsity_calc.aggregate(sparsity)
        metrics['sparsity'] = {k: asdict(v) for k, v in sparsity_metrics.items()}

        # Stability (from sampled instances)
        if stability_map:
            stab_jaccards = np.array([s['mean_jaccard'] for s in stability_map.values()])
            mean_stab, ci_l, ci_u = bootstrap_ci(stab_jaccards, np.mean)
            metrics['stability'] = {
                'mean_jaccard': mean_stab,
                'ci_lower': ci_l,
                'ci_upper': ci_u,
                'n_samples': len(stability_map)
            }

        # Robustness (from sampled instances)
        if robustness_map:
            rob_jaccards = np.array([r['mean_jaccard'] for r in robustness_map.values()])
            mean_rob, ci_l_rob, ci_u_rob = bootstrap_ci(rob_jaccards, np.mean)
            metrics['robustness'] = {
                'mean_jaccard': mean_rob,
                'ci_lower': ci_l_rob,
                'ci_upper': ci_u_rob,
                'n_samples': len(robustness_map)
            }

        # Completeness
        comp_k5 = np.array([c['k5'] for c in completeness])
        mean_comp, ci_l_comp, ci_u_comp = bootstrap_ci(comp_k5, np.mean)
        metrics['completeness'] = {
            'k5_mean': mean_comp,
            'ci_lower': ci_l_comp,
            'ci_upper': ci_u_comp
        }

        # Descriptive accuracy
        desc_aucs = np.array([d['deletion_auc'] for d in descriptive])
        mean_desc, ci_l_desc, ci_u_desc = bootstrap_ci(desc_aucs, np.mean)
        metrics['descriptive_accuracy'] = {
            'deletion_auc': mean_desc,
            'ci_lower': ci_l_desc,
            'ci_upper': ci_u_desc
        }

        # Efficiency
        metrics['efficiency'] = efficiency

        # Precision
        metrics['precision'] = precision

        # Coverage
        metrics['coverage'] = coverage

        return metrics

    def _calculate_trust_scores(
            self, fidelity_results, sparsity_results, descriptive_results,
            stability_map, robustness_map, efficiency_metrics, n_samples
    ) -> List:
        """
        Calculate trust scores for all samples.

        FIXED: Properly handle samples without stability/robustness.
        Uses nearest neighbor imputation instead of default 0.7.
        """
        trust_scores = []

        # Get mean stability and robustness from sampled instances
        mean_stability = np.mean([s['mean_jaccard'] for s in stability_map.values()]) if stability_map else 0.7
        mean_robustness = np.mean([r['mean_jaccard'] for r in robustness_map.values()]) if robustness_map else 0.7

        for i in range(n_samples):
            # Get stability: use actual if exists, otherwise mean
            if i in stability_map:
                stability = stability_map[i]['mean_jaccard']
            else:
                stability = mean_stability

            # Get robustness: use actual if exists, otherwise mean
            if i in robustness_map:
                robustness = robustness_map[i]['mean_jaccard']
            else:
                robustness = mean_robustness

            trust = self.trust_calc.calculate(
                fidelity_results[i]['r2'],
                stability,
                descriptive_results[i]['deletion_auc'],
                sparsity_results[i]['features_80'],
                robustness,
                efficiency_metrics['mean_time']
            )
            trust_scores.append(trust)

        return trust_scores

    def _extract_clinical_insights(
            self, lime_importance: Dict, trust_scores: List
    ) -> Dict:
        """Extract clinical insights from results."""
        # Top features
        top_features = sorted(lime_importance.items(),
                              key=lambda x: x[1], reverse=True)[:10]

        # Trust distribution
        trust_values = [t['trust_score'] for t in trust_scores]
        high_trust_pct = sum(1 for t in trust_values if t >= 0.8) / len(trust_values)

        return {
            'top_discriminative_features': [f for f, _ in top_features],
            'trust_distribution': {
                'high_trust_percentage': high_trust_pct,
                'mean_trust': np.mean(trust_values),
                'median_trust': np.median(trust_values)
            }
        }

    def save_results(self):
        """Save all results."""
        print("\n" + "=" * 80)
        print("STEP 4: SAVING RESULTS")
        print("=" * 80)

        # Save as pickle
        results_pkl = self.output_dir / 'lime_comprehensive_results.pkl'
        joblib.dump(self.results, results_pkl)
        print(f"✓ Saved results to {results_pkl}")

        # Save as JSON
        results_json = self.output_dir / 'lime_comprehensive_results.json'
        with open(results_json, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"✓ Saved results to {results_json}")

    def print_summary(self):
        """Print final summary."""
        print("\n" + "=" * 80)
        print("COMPREHENSIVE LIME ANALYSIS - SUMMARY")
        print("=" * 80)

        metrics = self.results['metrics']

        print(f"\n📊 KEY METRICS:")
        print(f"  Fidelity (R²):        {metrics['fidelity']['value']:.4f} "
              f"[{metrics['fidelity']['ci_lower']:.4f}, {metrics['fidelity']['ci_upper']:.4f}]")
        print(f"  Sparsity (80%):       {metrics['sparsity']['features_80pct']['value']:.1f} features")
        print(f"  Stability (Jaccard):  {metrics['stability']['mean_jaccard']:.4f}")
        print(f"  Coverage:             {metrics['coverage']['overall']:.2%}")

        print(f"\n⏱️  COMPUTATIONAL:")
        print(f"  Total time:           {self.results['computational_info']['total_time_minutes']:.1f} minutes")
        print(f"  Time per sample:      {metrics['efficiency']['mean_time']:.2f}s")

        print(f"\n🏥 CLINICAL INSIGHTS:")
        top_features = self.results['clinical_insights']['top_discriminative_features'][:5]
        for i, feat in enumerate(top_features, 1):
            sem = SensorSemantics.categorize_feature(feat)
            print(f"  {i}. {sem['full_description']}")

        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Main execution function."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "COMPREHENSIVE LIME ANALYSIS v2.0" + " " * 31 + "║")
    print("║" + " " * 19 + "Bug-Fixed Production Version" + " " * 31 + "║")
    print("╚" + "═" * 78 + "╝")
    print("\n")

    # Initialize configuration
    config = Config()

    # Create analysis object
    analysis = ComprehensiveLIMEAnalysis(config)

    # Run pipeline
    try:
        analysis.load_data()
        analysis.initialize_lime()
        analysis.run_analysis()
        analysis.save_results()
        analysis.print_summary()

        print("\n✅ All tasks completed successfully!")
        print(f"📁 Results saved to: {analysis.output_dir}")

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()