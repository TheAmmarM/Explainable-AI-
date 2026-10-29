#!/usr/bin/env python3
"""
============================================
PHASE 3: LOCAL EXPLAINABILITY FOR XAI PROJECT
============================================
This script provides local (instance-level) explanations for credit risk predictions.
It answers: "Why was THIS specific customer rejected/approved?"

Author: XAI Project
Date: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class LocalExplainer:
    """Class to handle local explainability analysis for credit risk models."""

    def __init__(self):
        """Initialize the explainer with models and data."""
        self.models = {}
        self.data = {}
        self.feature_info = {}
        self.predictions = {}
        self.scaler = None

    def load_artifacts(self):
        """Load all saved models and data."""
        print("=" * 80)
        print("PHASE 3: LOCAL EXPLAINABILITY")
        print("Question: Why was THIS customer rejected/approved?")
        print("=" * 80)

        print("\nLoading models and data...")

        # Required files
        required_files = {
            'logistic_regression_model.pkl': 'logreg',
            'random_forest_model.pkl': 'rf',
            'xgboost_model.pkl': 'xgb',
            'X_train.pkl': 'X_train',
            'X_test.pkl': 'X_test',
            'y_train.pkl': 'y_train',
            'y_test.pkl': 'y_test',
            'feature_info.pkl': 'feature_info'
        }

        # Load each file
        for filename, key in required_files.items():
            if not Path(filename).exists():
                raise FileNotFoundError(
                    f"Required file '{filename}' not found. "
                    f"Please run Phase 1 and Phase 2 first."
                )

            obj = joblib.load(filename)

            if key in ['logreg', 'rf', 'xgb']:
                self.models[key] = obj
            elif key == 'feature_info':
                self.feature_info = obj
            else:
                self.data[key] = obj

        print("✓ All models and data loaded successfully")
        print(f"  - Training samples: {self.data['X_train'].shape[0]:,}")
        print(f"  - Test samples: {self.data['X_test'].shape[0]:,}")
        print(f"  - Number of features: {self.data['X_train'].shape[1]}")

        # Convert to numpy arrays for consistent indexing
        self._prepare_data()

    def _prepare_data(self):
        """Convert data to numpy arrays and prepare feature scaling."""
        # Convert to numpy
        for key in ['X_train', 'X_test', 'y_train', 'y_test']:
            if hasattr(self.data[key], 'values'):
                self.data[f'{key}_np'] = self.data[key].values
            else:
                self.data[f'{key}_np'] = np.array(self.data[key])

        # Get feature names
        self.feature_names = self.feature_info['all_features']

        # Prepare scaler for logistic regression
        from sklearn.preprocessing import StandardScaler

        numerical_features = self.feature_info.get('numerical_features', [])
        num_indices = [i for i, feat in enumerate(self.feature_names)
                       if feat in numerical_features]

        if num_indices:
            self.scaler = StandardScaler()
            X_train_scaled = self.data['X_train_np'].copy()
            X_train_scaled[:, num_indices] = self.scaler.fit_transform(
                self.data['X_train_np'][:, num_indices]
            )
            self.data['X_train_scaled'] = X_train_scaled

            X_test_scaled = self.data['X_test_np'].copy()
            X_test_scaled[:, num_indices] = self.scaler.transform(
                self.data['X_test_np'][:, num_indices]
            )
            self.data['X_test_scaled'] = X_test_scaled
        else:
            self.data['X_train_scaled'] = self.data['X_train_np']
            self.data['X_test_scaled'] = self.data['X_test_np']

        print(f"✓ Data prepared with {len(self.feature_names)} features")

    def generate_predictions(self):
        """Generate predictions from all models."""
        print("\n" + "=" * 80)
        print("GENERATING PREDICTIONS FROM ALL MODELS")
        print("=" * 80)

        X_test_np = self.data['X_test_np']
        X_test_scaled = self.data['X_test_scaled']

        # Logistic Regression (needs scaled data)
        print("  - Logistic Regression...")
        self.predictions['Logistic Regression'] = self.models['logreg'].predict_proba(
            X_test_scaled
        )[:, 1]

        # Random Forest
        print("  - Random Forest...")
        self.predictions['Random Forest'] = self.models['rf'].predict_proba(
            X_test_np
        )[:, 1]

        # XGBoost
        print("  - XGBoost...")
        self.predictions['XGBoost'] = self.models['xgb'].predict_proba(
            X_test_np
        )[:, 1]

        print("✓ All predictions generated")

    def select_interesting_cases(self, n_cases=5):
        """
        Select interesting cases for detailed analysis.

        Categories:
        1. High Risk - All models agree on high probability of default
        2. Model Disagreement - Models have very different predictions
        3. Borderline - Predictions near decision boundary
        4. High Confidence Correct - Models very confident and correct
        5. False Positive/Negative - Model mistakes
        """
        print("\n" + "=" * 80)
        print("SELECTING INTERESTING CASES FOR ANALYSIS")
        print("=" * 80)

        y_test_np = self.data['y_test_np']
        n_samples = len(y_test_np)

        case_categories = {
            'High Risk': [],
            'Model Disagreement': [],
            'Borderline': [],
            'High Confidence Correct': [],
            'False Positive': [],
            'False Negative': []
        }

        # Analyze each test sample
        for i in range(n_samples):
            preds = [pred[i] for pred in self.predictions.values()]
            avg_pred = np.mean(preds)
            max_pred = max(preds)
            min_pred = min(preds)
            actual = y_test_np[i]

            # High Risk: All models agree on high probability
            if all(p > 0.7 for p in preds):
                case_categories['High Risk'].append(i)

            # Model Disagreement: Large spread in predictions
            if max_pred - min_pred > 0.4:
                case_categories['Model Disagreement'].append(i)

            # Borderline: Average prediction near 0.5
            if 0.4 <= avg_pred <= 0.6:
                case_categories['Borderline'].append(i)

            # High Confidence Correct
            if (avg_pred > 0.8 and actual == 1) or (avg_pred < 0.2 and actual == 0):
                case_categories['High Confidence Correct'].append(i)

            # False Positive: Predicted bad but actually good
            if avg_pred > 0.5 and actual == 0:
                case_categories['False Positive'].append(i)

            # False Negative: Predicted good but actually bad
            if avg_pred < 0.5 and actual == 1:
                case_categories['False Negative'].append(i)

        # Print statistics
        print("\nCase Statistics:")
        print("-" * 60)
        for category, indices in case_categories.items():
            print(f"  {category:25}: {len(indices):4} cases")

        # Select diverse cases
        selected_cases = []
        selected_categories = []

        for category, indices in case_categories.items():
            if indices and len(selected_cases) < n_cases:
                # Select first case from each category
                selected_cases.append(indices[0])
                selected_categories.append(category)

        # Store for later use
        self.selected_cases = selected_cases
        self.selected_categories = selected_categories

        # Display selected cases
        self._display_selected_cases()

        return selected_cases, selected_categories

    def _display_selected_cases(self):
        """Display summary of selected cases."""
        print(f"\n✓ Selected {len(self.selected_cases)} cases for detailed analysis")
        print("\nSELECTED CASES SUMMARY:")
        print("=" * 90)
        print(f"{'Case':<6} {'Index':<8} {'Actual':<12} {'LogReg':<10} {'RF':<10} "
              f"{'XGB':<10} {'Category':<20}")
        print("=" * 90)

        for i, (idx, category) in enumerate(zip(self.selected_cases,
                                                self.selected_categories), 1):
            actual = "Bad Credit" if self.data['y_test_np'][idx] == 1 else "Good Credit"
            lr_pred = self.predictions['Logistic Regression'][idx]
            rf_pred = self.predictions['Random Forest'][idx]
            xgb_pred = self.predictions['XGBoost'][idx]

            print(f"{i:<6} {idx:<8} {actual:<12} {lr_pred:>8.1%}   "
                  f"{rf_pred:>8.1%}   {xgb_pred:>8.1%}   {category:<20}")

        print("=" * 90)

    def compute_feature_contributions(self, instance_idx):
        """
        Compute feature contributions for a specific instance.
        Uses a gradient-based approach for tree models.
        """
        instance = self.data['X_test_np'][instance_idx]
        X_train_np = self.data['X_train_np']

        # Get feature importances from XGBoost
        feature_importance = self.models['xgb'].feature_importances_

        contributions = []

        for i, feat_name in enumerate(self.feature_names):
            feat_value = instance[i]
            train_values = X_train_np[:, i]

            # Calculate statistics
            feat_mean = np.mean(train_values)
            feat_std = np.std(train_values)
            feat_min = np.min(train_values)
            feat_max = np.max(train_values)

            # Calculate z-score (how unusual this value is)
            if feat_std > 0:
                z_score = (feat_value - feat_mean) / feat_std
            else:
                z_score = 0

            # Calculate percentile
            percentile = np.sum(train_values <= feat_value) / len(train_values) * 100

            # Contribution score (importance × deviation)
            contribution_score = feature_importance[i] * abs(z_score)

            contributions.append({
                'feature': feat_name,
                'value': feat_value,
                'mean': feat_mean,
                'std': feat_std,
                'min': feat_min,
                'max': feat_max,
                'z_score': z_score,
                'percentile': percentile,
                'importance': feature_importance[i],
                'contribution': contribution_score
            })

        # Sort by contribution
        contributions.sort(key=lambda x: x['contribution'], reverse=True)

        return contributions

    def explain_instance(self, case_num, save_plot=True):
        """Generate comprehensive explanation for a specific case."""
        idx = self.selected_cases[case_num - 1]
        category = self.selected_categories[case_num - 1]

        print("\n" + "=" * 80)
        print(f"EXPLAINING CASE #{case_num}")
        print("=" * 80)

        # Get instance data
        instance = self.data['X_test_np'][idx]
        actual = self.data['y_test_np'][idx]

        # Get predictions
        xgb_pred = self.predictions['XGBoost'][idx]
        rf_pred = self.predictions['Random Forest'][idx]
        lr_pred = self.predictions['Logistic Regression'][idx]

        # Print header
        print(f"\nCase Information:")
        print(f"  Test Index: {idx}")
        print(f"  Category: {category}")
        print(f"  Actual Label: {'Bad Credit (1)' if actual == 1 else 'Good Credit (0)'}")
        print(f"\nModel Predictions (Probability of Bad Credit):")
        print(f"  Logistic Regression: {lr_pred:.1%}")
        print(f"  Random Forest:       {rf_pred:.1%}")
        print(f"  XGBoost:             {xgb_pred:.1%}")
        print(f"  Average:             {np.mean([lr_pred, rf_pred, xgb_pred]):.1%}")

        # Compute feature contributions
        contributions = self.compute_feature_contributions(idx)

        # Display top contributing features
        print(f"\n{'=' * 80}")
        print("TOP CONTRIBUTING FEATURES")
        print("=" * 80)
        print(f"{'Rank':<6} {'Feature':<25} {'Value':<12} {'Z-Score':<10} "
              f"{'Percentile':<12} {'Impact':<10}")
        print("-" * 80)

        for i, contrib in enumerate(contributions[:10], 1):
            z_score = contrib['z_score']
            percentile = contrib['percentile']

            # Determine impact level
            if abs(z_score) > 2:
                impact = "VERY HIGH"
            elif abs(z_score) > 1:
                impact = "HIGH"
            elif abs(z_score) > 0.5:
                impact = "MODERATE"
            else:
                impact = "LOW"

            print(f"{i:<6} {contrib['feature']:<25} {contrib['value']:<12.2f} "
                  f"{z_score:<10.2f} {percentile:<12.1f}% {impact:<10}")

        # Generate human-readable explanation
        self._generate_narrative_explanation(idx, contributions, xgb_pred)

        # Create visualization
        if save_plot:
            self._create_explanation_plot(case_num, idx, category, contributions,
                                          actual, xgb_pred)

        return contributions

    def _generate_narrative_explanation(self, idx, contributions, prediction):
        """Generate human-readable narrative explanation."""
        print(f"\n{'=' * 80}")
        print("HUMAN-READABLE EXPLANATION")
        print("=" * 80)

        print(f"\nThe model predicts a {prediction:.1%} probability of default.")

        # Determine decision
        if prediction > 0.7:
            decision = "REJECT"
            confidence = "HIGH"
        elif prediction > 0.5:
            decision = "REJECT"
            confidence = "MODERATE"
        elif prediction > 0.3:
            decision = "APPROVE"
            confidence = "MODERATE"
        else:
            decision = "APPROVE"
            confidence = "HIGH"

        print(f"Recommendation: {decision} (Confidence: {confidence})")

        # Explain top factors
        print("\nKey Decision Factors:")

        # Risk-increasing factors (positive z-scores for important features)
        risk_increasing = [c for c in contributions[:8] if c['z_score'] > 0.5]
        if risk_increasing:
            print("\n  Risk-INCREASING factors:")
            for i, contrib in enumerate(risk_increasing[:3], 1):
                print(f"    {i}. {contrib['feature']}: {contrib['value']:.2f}")
                print(f"       → {contrib['percentile']:.0f}th percentile "
                      f"(higher than {contrib['percentile']:.0f}% of applicants)")

        # Risk-decreasing factors (negative z-scores for important features)
        risk_decreasing = [c for c in contributions[:8] if c['z_score'] < -0.5]
        if risk_decreasing:
            print("\n  Risk-DECREASING factors:")
            for i, contrib in enumerate(risk_decreasing[:3], 1):
                print(f"    {i}. {contrib['feature']}: {contrib['value']:.2f}")
                print(f"       → {contrib['percentile']:.0f}th percentile "
                      f"(lower than {100 - contrib['percentile']:.0f}% of applicants)")

        # What-if scenarios
        if decision == "REJECT":
            print("\n  To improve approval chances:")
            for i, contrib in enumerate(risk_increasing[:2], 1):
                target = contrib['mean']
                print(f"    {i}. Reduce {contrib['feature']} from {contrib['value']:.2f} "
                      f"toward average of {target:.2f}")

        print("=" * 80)

    def _create_explanation_plot(self, case_num, idx, category, contributions,
                                 actual, prediction):
        """Create visualization of the explanation."""
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. Feature Contributions (Top 10)
        ax1 = fig.add_subplot(gs[0:2, 0])
        top_features = contributions[:10]
        feature_names = [c['feature'][:20] for c in top_features]
        contributions_vals = [c['contribution'] for c in top_features]
        z_scores = [c['z_score'] for c in top_features]

        colors = ['red' if z > 0 else 'green' for z in z_scores]
        bars = ax1.barh(range(len(feature_names)), contributions_vals, color=colors, alpha=0.6)
        ax1.set_yticks(range(len(feature_names)))
        ax1.set_yticklabels(feature_names)
        ax1.set_xlabel('Contribution Score', fontsize=11, fontweight='bold')
        ax1.set_title(f'Top 10 Contributing Features - Case #{case_num}',
                      fontsize=13, fontweight='bold')
        ax1.invert_yaxis()
        ax1.grid(axis='x', alpha=0.3)

        # 2. Prediction Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        models = ['LogReg', 'RF', 'XGB', 'Average']
        preds = [
            self.predictions['Logistic Regression'][idx],
            self.predictions['Random Forest'][idx],
            prediction,
            np.mean([self.predictions['Logistic Regression'][idx],
                     self.predictions['Random Forest'][idx], prediction])
        ]

        bars = ax2.bar(models, preds, color=['blue', 'orange', 'green', 'red'], alpha=0.7)
        ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Decision Threshold')
        ax2.set_ylabel('Probability of Bad Credit', fontsize=11, fontweight='bold')
        ax2.set_title('Model Predictions Comparison', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.legend()
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bar, pred in zip(bars, preds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height,
                     f'{pred:.1%}', ha='center', va='bottom', fontweight='bold')

        # 3. Case Summary
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.axis('off')

        summary_text = f"""
CASE SUMMARY

Index: {idx}
Category: {category}
Actual: {'Bad Credit' if actual == 1 else 'Good Credit'}
Predicted: {'Bad Credit' if prediction > 0.5 else 'Good Credit'}
Probability: {prediction:.1%}

Decision: {'REJECT' if prediction > 0.5 else 'APPROVE'}
Risk Level: {'HIGH' if prediction > 0.7 else 'MODERATE' if prediction > 0.5 else 'LOW'}

Top 3 Important Features:
1. {contributions[0]['feature']}: {contributions[0]['value']:.2f}
2. {contributions[1]['feature']}: {contributions[1]['value']:.2f}
3. {contributions[2]['feature']}: {contributions[2]['value']:.2f}
        """

        ax3.text(0.1, 0.5, summary_text, transform=ax3.transAxes,
                 fontsize=11, verticalalignment='center', family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

        # 4. Feature Value Distribution (for top feature)
        ax4 = fig.add_subplot(gs[2, :])
        top_feature_idx = self.feature_names.index(contributions[0]['feature'])
        train_values = self.data['X_train_np'][:, top_feature_idx]
        instance_value = contributions[0]['value']

        ax4.hist(train_values, bins=50, alpha=0.6, color='gray', edgecolor='black')
        ax4.axvline(instance_value, color='red', linestyle='--', linewidth=3,
                    label=f'This Case: {instance_value:.2f}')
        ax4.axvline(contributions[0]['mean'], color='blue', linestyle='--', linewidth=2,
                    label=f'Mean: {contributions[0]["mean"]:.2f}')
        ax4.set_xlabel('Value', fontsize=11, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
        ax4.set_title(f'Distribution of Top Feature: {contributions[0]["feature"]}',
                      fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)

        plt.suptitle(f'Local Explanation - Case #{case_num} (Index {idx})',
                     fontsize=15, fontweight='bold', y=0.995)

        # Save plot
        filename = f'local_explanation_case_{case_num}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\n✓ Visualization saved to '{filename}'")
        plt.close()

    def generate_lime_explanations(self):
        """Generate LIME explanations if available."""
        print("\n" + "=" * 80)
        print("GENERATING LIME EXPLANATIONS")
        print("=" * 80)

        try:
            from lime.lime_tabular import LimeTabularExplainer

            print("✓ LIME library found")

            # Create explainer
            explainer = LimeTabularExplainer(
                training_data=self.data['X_train_np'],
                feature_names=self.feature_names,
                class_names=['Good Credit', 'Bad Credit'],
                mode='classification',
                random_state=42,
                verbose=False
            )

            print("✓ LIME explainer created")

            # Generate explanations for each selected case
            for case_num, idx in enumerate(self.selected_cases, 1):
                print(f"\nGenerating LIME explanation for Case #{case_num}...")

                instance = self.data['X_test_np'][idx]

                # Generate explanation
                exp = explainer.explain_instance(
                    data_row=instance,
                    predict_fn=self.models['xgb'].predict_proba,
                    num_features=10,
                    top_labels=2  # Need both labels for proper explanation
                )

                # Get the available label (usually 1 for bad credit, but check first)
                available_labels = list(exp.available_labels())
                target_label = 1 if 1 in available_labels else available_labels[0]

                # Create LIME plot
                fig = exp.as_pyplot_figure(label=target_label)
                plt.title(f'LIME Explanation - Case #{case_num}',
                          fontsize=13, fontweight='bold')
                plt.tight_layout()

                filename = f'lime_explanation_case_{case_num}.png'
                plt.savefig(filename, dpi=300, bbox_inches='tight')
                print(f"  ✓ LIME plot saved to '{filename}'")
                plt.close()

                # Print LIME explanation
                print(f"\n  Top LIME Features for Case #{case_num}:")
                exp_list = exp.as_list(label=target_label)
                for i, (feature_desc, weight) in enumerate(exp_list[:5], 1):
                    direction = "↑ Risk" if weight > 0 else "↓ Risk"
                    print(f"    {i}. {feature_desc:40} → {weight:7.3f} ({direction})")

            return True

        except ImportError:
            print("⚠ LIME library not installed")
            print("  To install: pip install lime")
            print("  Skipping LIME explanations...")
            return False

    def create_summary_report(self):
        """Create summary report of all explanations."""
        print("\n" + "=" * 80)
        print("CREATING SUMMARY REPORT")
        print("=" * 80)

        reports = []

        for case_num, (idx, category) in enumerate(zip(self.selected_cases,
                                                       self.selected_categories), 1):
            # Get data
            actual = self.data['y_test_np'][idx]
            xgb_pred = self.predictions['XGBoost'][idx]

            # Compute contributions
            contributions = self.compute_feature_contributions(idx)

            # Create report entry
            report = {
                'case_number': case_num,
                'test_index': idx,
                'category': category,
                'actual_label': 'Bad' if actual == 1 else 'Good',
                'predicted_label': 'Bad' if xgb_pred > 0.5 else 'Good',
                'prediction_probability': xgb_pred,
                'correct_prediction': (xgb_pred > 0.5) == (actual == 1),
                'risk_level': 'HIGH' if xgb_pred > 0.7 else
                'MODERATE' if xgb_pred > 0.5 else 'LOW',
                'recommendation': 'REJECT' if xgb_pred > 0.5 else 'APPROVE',
                'logreg_prediction': self.predictions['Logistic Regression'][idx],
                'rf_prediction': self.predictions['Random Forest'][idx],
                'xgb_prediction': xgb_pred,
            }

            # Add top 5 features
            for i in range(5):
                if i < len(contributions):
                    report[f'top_feature_{i + 1}'] = contributions[i]['feature']
                    report[f'top_feature_{i + 1}_value'] = contributions[i]['value']
                    report[f'top_feature_{i + 1}_z_score'] = contributions[i]['z_score']
                else:
                    report[f'top_feature_{i + 1}'] = ''
                    report[f'top_feature_{i + 1}_value'] = 0
                    report[f'top_feature_{i + 1}_z_score'] = 0

            reports.append(report)

            # Print summary
            print(f"\nCase #{case_num} Summary:")
            print(f"  Index: {report['test_index']}")
            print(f"  Category: {report['category']}")
            print(f"  Actual/Predicted: {report['actual_label']}/{report['predicted_label']}")
            print(f"  Probability: {report['prediction_probability']:.1%}")
            print(f"  Risk Level: {report['risk_level']}")
            print(f"  Recommendation: {report['recommendation']}")

        # Save to CSV
        reports_df = pd.DataFrame(reports)
        filename = 'local_explanations_summary.csv'
        reports_df.to_csv(filename, index=False)
        print(f"\n✓ Summary report saved to '{filename}'")

        return reports_df

    def run_full_analysis(self, n_cases=5, use_lime=True):
        """Run complete local explainability analysis."""
        # Load data and models
        self.load_artifacts()

        # Generate predictions
        self.generate_predictions()

        # Select interesting cases
        self.select_interesting_cases(n_cases=n_cases)

        # Explain each case
        print("\n" + "=" * 80)
        print("GENERATING DETAILED EXPLANATIONS")
        print("=" * 80)

        for case_num in range(1, len(self.selected_cases) + 1):
            self.explain_instance(case_num, save_plot=True)

        # LIME explanations (if available and requested)
        if use_lime:
            self.generate_lime_explanations()

        # Create summary report
        summary_df = self.create_summary_report()

        # Final summary
        print("\n" + "=" * 80)
        print("PHASE 3 COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print(f"\n✓ Analyzed {len(self.selected_cases)} cases")
        print(f"✓ Generated {len(self.selected_cases)} detailed explanations")
        print(f"✓ Created {len(self.selected_cases)} visualization plots")
        print("✓ Created summary report")

        print("\nGenerated Files:")
        print("  - local_explanation_case_*.png (visualizations)")
        if use_lime:
            print("  - lime_explanation_case_*.png (LIME plots)")
        print("  - local_explanations_summary.csv (summary report)")

        return summary_df


def main():
    """Main execution function."""
    print("Starting Local Explainability Analysis...")
    print("This will analyze individual predictions to understand model decisions.\n")

    # Create explainer
    explainer = LocalExplainer()

    # Run full analysis
    try:
        summary_df = explainer.run_full_analysis(
            n_cases=5,  # Number of cases to analyze
            use_lime=True  # Try to use LIME if available
        )

        print("\n" + "=" * 80)
        print("Analysis complete! Check the generated files for detailed insights.")
        print("=" * 80)

    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure you have run the previous phases:")
        print("  1. Phase 1: Data preparation and model training")
        print("  2. Phase 2: Global explainability")
        print("\nThese phases should generate the required .pkl files.")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()