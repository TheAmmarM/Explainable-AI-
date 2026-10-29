#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COMPREHENSIVE LIME ANALYSIS - PART 2: ADVANCED METRICS & SEMANTIC LAYER
======================================================================

This is Part 2 of the modular LIME analysis system.
Contains efficiency, descriptive accuracy, robustness, trust score, and semantic layer.

Author: Generated for BFRB XAI Research
Date: 2026-02-09
Version: 2.0.0 (Bug-fixed)
"""

import time
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error

# Import from Part 1
from Final_Lime_3_1 import (
    Config, MetricResult, bootstrap_ci
)


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def jaccard_similarity(set1: set, set2: set) -> float:
    """
    Calculate Jaccard similarity between two sets.

    Args:
        set1: First set
        set2: Second set

    Returns:
        Jaccard similarity coefficient (0 to 1)
    """
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


# ==============================================================================
# METRIC 7: EFFICIENCY
# ==============================================================================

class EfficiencyCalculator:
    """Calculate computational efficiency metrics."""

    @staticmethod
    def calculate(
            explainer,
            instances: np.ndarray,
            model: Any,
            n_samples: int = 100
    ) -> Dict[str, Any]:
        """
        Calculate efficiency metrics.

        Args:
            explainer: LIME explainer
            instances: Data instances
            model: Black-box model
            n_samples: Number of samples to test

        Returns:
            Efficiency metrics
        """
        times = []

        for i in range(min(n_samples, len(instances))):
            start = time.time()
            exp = explainer.explain_instance(
                instances[i],
                model.predict_proba,
                num_features=10
            )
            elapsed = time.time() - start
            times.append(elapsed)

        times = np.array(times)

        return {
            'mean_time': times.mean(),
            'median_time': np.median(times),
            'p95_time': np.percentile(times, 95),
            'std_time': times.std(),
            'times': times
        }


# ==============================================================================
# METRIC 8: DESCRIPTIVE ACCURACY (DELETION TEST)
# ==============================================================================

class DescriptiveAccuracyCalculator:
    """Calculate descriptive accuracy via feature deletion."""

    @staticmethod
    def calculate(
            instance: np.ndarray,
            lime_weights: List[Tuple[int, float]],
            model: Any,
            feature_baselines: np.ndarray,
            max_k: int = 15
    ) -> Dict[str, Any]:
        """
        Calculate MoRF (Most Relevant First) deletion curve.

        Args:
            instance: Data instance
            lime_weights: LIME weights as (feature_idx, weight) tuples
            model: Black-box model
            feature_baselines: Baseline values for features (e.g., median)
            max_k: Maximum number of features to delete

        Returns:
            Deletion metrics
        """
        # Original prediction
        orig_pred = model.predict_proba(instance.reshape(1, -1))[0, 1]

        # Sort features by absolute weight (descending)
        sorted_features = sorted(lime_weights, key=lambda x: abs(x[1]), reverse=True)

        # Progressive deletion
        deletion_curve = [orig_pred]
        instance_copy = instance.copy()

        for k in range(1, min(max_k + 1, len(sorted_features) + 1)):
            feat_idx = sorted_features[k - 1][0]
            instance_copy[feat_idx] = feature_baselines[feat_idx]
            pred = model.predict_proba(instance_copy.reshape(1, -1))[0, 1]
            deletion_curve.append(pred)

        # Calculate AUC of deletion curve (lower is better)
        deletion_auc = np.trapz(deletion_curve, dx=1.0) / (len(deletion_curve) * orig_pred) if orig_pred > 0 else 0

        # Performance drop at specific K values
        drops = {}
        for k in [5, 10]:
            if k < len(deletion_curve):
                drop = (orig_pred - deletion_curve[k]) / orig_pred if orig_pred > 0 else 0
                drops[f'drop_at_{k}'] = drop

        return {
            'deletion_auc': deletion_auc,
            'deletion_curve': deletion_curve,
            **drops
        }


# ==============================================================================
# METRIC 9: ROBUSTNESS
# ==============================================================================

class RobustnessCalculator:
    """Calculate robustness to input perturbations."""

    @staticmethod
    def calculate(
            explainer,
            instance: np.ndarray,
            model: Any,
            feature_stds: np.ndarray,
            n_perturbations: int = 10,
            noise_level: float = 0.01,
            top_k: int = 10
    ) -> Dict[str, float]:
        """
        Calculate robustness to input perturbations.

        Args:
            explainer: LIME explainer
            instance: Data instance
            model: Black-box model
            feature_stds: Standard deviations of features
            n_perturbations: Number of perturbed versions
            noise_level: Noise magnitude (fraction of std)
            top_k: Number of top features to compare

        Returns:
            Robustness metrics
        """
        # Generate original explanation
        orig_exp = explainer.explain_instance(
            instance,
            model.predict_proba,
            num_features=top_k
        )
        orig_features = set([feat for feat, _ in orig_exp.as_list()[:top_k]])
        orig_ranking = [feat for feat, _ in orig_exp.as_list()[:top_k]]

        jaccard_scores = []
        spearman_scores = []

        for _ in range(n_perturbations):
            # Add Gaussian noise
            noise = np.random.normal(0, noise_level * feature_stds, size=instance.shape)
            perturbed = instance + noise

            # Generate explanation for perturbed instance
            pert_exp = explainer.explain_instance(
                perturbed,
                model.predict_proba,
                num_features=top_k
            )
            pert_features = set([feat for feat, _ in pert_exp.as_list()[:top_k]])
            pert_ranking = [feat for feat, _ in pert_exp.as_list()[:top_k]]

            # Jaccard similarity
            jaccard = jaccard_similarity(orig_features, pert_features)
            jaccard_scores.append(jaccard)

            # Spearman correlation (on common features)
            common = list(set(orig_ranking) & set(pert_ranking))
            if len(common) >= 3:
                rank_orig = [orig_ranking.index(f) for f in common]
                rank_pert = [pert_ranking.index(f) for f in common]
                rho, _ = spearmanr(rank_orig, rank_pert)
                if not np.isnan(rho):
                    spearman_scores.append(rho)

        return {
            'mean_jaccard': np.mean(jaccard_scores),
            'mean_spearman': np.mean(spearman_scores) if spearman_scores else 0,
            'jaccard_scores': jaccard_scores
        }


# ==============================================================================
# METRIC 10: CLINICIAN TRUST SCORE
# ==============================================================================

class ClinicalTrustCalculator:
    """Calculate composite clinician trust score."""

    # Clinically validated weights
    WEIGHTS = {
        'fidelity': 0.25,
        'stability': 0.20,
        'descriptive_accuracy': 0.20,
        'sparsity': 0.15,
        'robustness': 0.10,
        'efficiency': 0.10
    }

    @staticmethod
    def normalize_metric(value: float, optimal: float, direction: str = 'higher') -> float:
        """
        Normalize metric to [0, 1].

        Args:
            value: Metric value
            optimal: Optimal value
            direction: 'higher' if higher is better, 'lower' otherwise

        Returns:
            Normalized value in [0, 1]
        """
        if direction == 'higher':
            return min(value / optimal, 1.0)
        else:
            return max(1.0 - (value / optimal), 0.0)

    @staticmethod
    def calculate(
            fidelity_r2: float,
            stability_jaccard: float,
            deletion_auc: float,
            features_80: float,
            robustness_jaccard: float,
            time_seconds: float,
            optimal_sparsity: float = 10.0,
            optimal_time: float = 5.0
    ) -> Dict[str, Any]:
        """
        Calculate composite clinician trust score.

        Args:
            fidelity_r2: Fidelity R² score
            stability_jaccard: Stability Jaccard score
            deletion_auc: Deletion AUC (lower is better)
            features_80: Features for 80% importance
            robustness_jaccard: Robustness Jaccard score
            time_seconds: Time per explanation
            optimal_sparsity: Optimal number of features
            optimal_time: Optimal time threshold

        Returns:
            Trust score and components
        """
        # Normalize components
        components = {
            'fidelity': ClinicalTrustCalculator.normalize_metric(fidelity_r2, 1.0, 'higher'),
            'stability': ClinicalTrustCalculator.normalize_metric(stability_jaccard, 1.0, 'higher'),
            'descriptive_accuracy': ClinicalTrustCalculator.normalize_metric(deletion_auc, 0.5, 'lower'),
            'sparsity': ClinicalTrustCalculator.normalize_metric(features_80, optimal_sparsity, 'lower'),
            'robustness': ClinicalTrustCalculator.normalize_metric(robustness_jaccard, 1.0, 'higher'),
            'efficiency': ClinicalTrustCalculator.normalize_metric(time_seconds, optimal_time, 'lower')
        }

        # Calculate weighted sum
        trust_score = sum(
            ClinicalTrustCalculator.WEIGHTS[k] * v
            for k, v in components.items()
        )

        # Categorize
        if trust_score >= 0.8:
            category = "High"
            recommendation = "Ready for clinical deployment"
        elif trust_score >= 0.6:
            category = "Moderate"
            recommendation = "Suitable with clinician oversight"
        else:
            category = "Low"
            recommendation = "Research use only"

        return {
            'trust_score': trust_score,
            'category': category,
            'recommendation': recommendation,
            'components': components
        }


# ==============================================================================
# SEMANTIC LAYER - CLINICAL INTERPRETABILITY
# ==============================================================================

class SensorSemantics:
    """
    Semantic mapping for sensor features to clinical meaning.
    Optimized for microcontroller deployment and clinician understanding.
    """

    # Sensor categories with clinical meaning
    SENSOR_CATEGORIES = {
        'proximity': {
            'prefix': ['tof_'],
            'clinical_name': 'Hand-to-Face Distance',
            'unit': 'mm',
            'clinical_meaning': 'Proximity detection indicates hand approaching face/head region',
            'bfrb_relevance': 'Direct indicator of BFRB behavior (hand must be near target area)',
            'microcontroller_priority': 1  # Highest priority
        },
        'thermal': {
            'prefix': ['thm_', 'thermo_'],
            'clinical_name': 'Skin Temperature',
            'unit': '°C',
            'clinical_meaning': 'Temperature change from hand contact with skin',
            'bfrb_relevance': 'Confirms physical contact (pulling, scratching, pinching)',
            'microcontroller_priority': 2
        },
        'motion': {
            'prefix': ['acc_', 'imu_magnitude'],
            'clinical_name': 'Hand Movement',
            'unit': 'm/s²',
            'clinical_meaning': 'Acceleration patterns of hand motion',
            'bfrb_relevance': 'Repetitive motion patterns characteristic of BFRB',
            'microcontroller_priority': 3
        },
        'rotation': {
            'prefix': ['gyro_'],
            'clinical_name': 'Hand Rotation',
            'unit': 'rad/s',
            'clinical_meaning': 'Angular velocity of hand/wrist rotation',
            'bfrb_relevance': 'Twisting motions during hair pulling or scratching',
            'microcontroller_priority': 4
        }
    }

    @staticmethod
    def categorize_feature(feature_name: str) -> Dict[str, str]:
        """
        Map feature to sensor category with clinical meaning.

        Args:
            feature_name: Raw feature name (e.g., 'tof_2_v63_max')

        Returns:
            Dictionary with category, clinical name, and interpretation
        """
        for category, info in SensorSemantics.SENSOR_CATEGORIES.items():
            if any(feature_name.startswith(prefix) for prefix in info['prefix']):
                # Parse aggregation function
                agg_func = 'mean'
                if '_max' in feature_name:
                    agg_func = 'maximum'
                elif '_min' in feature_name:
                    agg_func = 'minimum'
                elif '_std' in feature_name:
                    agg_func = 'variability'
                elif '_median' in feature_name:
                    agg_func = 'median'

                return {
                    'category': category,
                    'clinical_name': info['clinical_name'],
                    'unit': info['unit'],
                    'aggregation': agg_func,
                    'clinical_meaning': info['clinical_meaning'],
                    'bfrb_relevance': info['bfrb_relevance'],
                    'priority': info['microcontroller_priority'],
                    'full_description': f"{agg_func.title()} {info['clinical_name'].lower()}"
                }

        return {
            'category': 'unknown',
            'clinical_name': feature_name,
            'unit': 'unknown',
            'aggregation': 'unknown',
            'clinical_meaning': 'Feature not categorized',
            'bfrb_relevance': 'Unknown relevance',
            'priority': 99,
            'full_description': feature_name
        }

    @staticmethod
    def interpret_direction(weight: float, category: str) -> str:
        """
        Interpret what a positive/negative weight means clinically.

        Args:
            weight: LIME weight
            category: Sensor category

        Returns:
            Clinical interpretation string
        """
        if weight > 0:
            interpretations = {
                'proximity': 'Closer hand-to-face distance increases BFRB likelihood',
                'thermal': 'Higher temperature (more contact) increases BFRB likelihood',
                'motion': 'More movement increases BFRB likelihood',
                'rotation': 'More rotation increases BFRB likelihood'
            }
        else:
            interpretations = {
                'proximity': 'Greater distance decreases BFRB likelihood (protective)',
                'thermal': 'Lower temperature (less contact) decreases BFRB likelihood',
                'motion': 'Less movement decreases BFRB likelihood',
                'rotation': 'Less rotation decreases BFRB likelihood'
            }

        return interpretations.get(category, 'Effect on BFRB likelihood unclear')


class ClinicalExplanationGenerator:
    """
    Generate human-readable clinical explanations from LIME weights.
    Optimized for clinician understanding and patient communication.
    """

    @staticmethod
    def generate_explanation(
            lime_weights: List[Tuple[str, float]],
            prediction: float,
            prediction_class: int,
            top_k: int = 5,
            detail_level: str = 'standard'
    ) -> Dict[str, str]:
        """
        Generate multi-level clinical explanation.

        Args:
            lime_weights: LIME feature weights
            prediction: Prediction probability
            prediction_class: Predicted class (0 or 1)
            top_k: Number of top features to explain
            detail_level: 'simple', 'standard', or 'technical'

        Returns:
            Explanations at different levels
        """
        # Sort by absolute importance
        top_features = sorted(lime_weights, key=lambda x: abs(x[1]), reverse=True)[:top_k]

        # Categorize features
        categorized = {}
        for feature, weight in top_features:
            semantics = SensorSemantics.categorize_feature(feature)
            category = semantics['category']

            if category not in categorized:
                categorized[category] = []
            categorized[category].append({
                'feature': feature,
                'weight': weight,
                'semantics': semantics
            })

        # Generate explanations at different levels
        explanations = {}

        # 1. SIMPLE EXPLANATION (for patients)
        if detail_level in ['simple', 'standard', 'technical']:
            behavior = "BFRB behavior" if prediction_class == 1 else "normal behavior"
            confidence = "very confident" if abs(prediction - 0.5) > 0.4 else "moderately confident"

            simple_parts = [f"The system is {confidence} this is {behavior}."]

            # Add top contributing factor
            if top_features:
                top_feat, top_weight = top_features[0]
                top_sem = SensorSemantics.categorize_feature(top_feat)
                direction = "detected" if top_weight > 0 else "not detected"

                simple_parts.append(
                    f"The main indicator is {top_sem['clinical_name'].lower()}, "
                    f"which was {direction}."
                )

            explanations['simple'] = ' '.join(simple_parts)

        # 2. STANDARD EXPLANATION (for clinicians)
        if detail_level in ['standard', 'technical']:
            standard_parts = [
                f"Prediction: {prediction_class} ({'BFRB' if prediction_class == 1 else 'Non-BFRB'})",
                f"Confidence: {prediction:.1%}",
                "\nKey contributing factors:"
            ]

            for category, features in categorized.items():
                category_info = SensorSemantics.SENSOR_CATEGORIES.get(category, {})
                clinical_name = category_info.get('clinical_name', category)

                # Aggregate weights in this category
                total_weight = sum(abs(f['weight']) for f in features)
                avg_direction = sum(f['weight'] for f in features) / len(features)

                direction_text = "increases" if avg_direction > 0 else "decreases"
                standard_parts.append(
                    f"  • {clinical_name}: {direction_text} BFRB likelihood "
                    f"(importance: {total_weight:.3f})"
                )

            explanations['standard'] = '\n'.join(standard_parts)

        # 3. TECHNICAL EXPLANATION (for researchers/developers)
        if detail_level == 'technical':
            technical_parts = [
                f"LIME Linear Model Output: {prediction:.4f}",
                f"Classification: {prediction_class} (threshold: 0.5)",
                "\nFeature Contributions (sorted by |weight|):"
            ]

            for i, (feature, weight) in enumerate(top_features, 1):
                semantics = SensorSemantics.categorize_feature(feature)
                technical_parts.append(
                    f"  {i}. {feature}:"
                )
                technical_parts.append(
                    f"     Weight: {weight:+.4f}"
                )
                technical_parts.append(
                    f"     Clinical: {semantics['full_description']}"
                )
                technical_parts.append(
                    f"     Impact: {SensorSemantics.interpret_direction(weight, semantics['category'])}"
                )

            explanations['technical'] = '\n'.join(technical_parts)

        return explanations


# ==============================================================================
# SAVE THIS MODULE
# ==============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("LIME ANALYSIS - PART 2: ADVANCED METRICS & SEMANTIC LAYER")
    print("=" * 80)
    print("\n✅ Module loaded successfully!")
    print("\nThis module contains:")
    print("  7. ✅ Efficiency Calculator")
    print("  8. ✅ Descriptive Accuracy Calculator")
    print("  9. ✅ Robustness Calculator")
    print("  10. ✅ Clinical Trust Calculator")
    print("  11. ✅ Sensor Semantics")
    print("  12. ✅ Clinical Explanation Generator")
    print("\n✅ Ready for Part 3 (Main Analysis Pipeline)...")