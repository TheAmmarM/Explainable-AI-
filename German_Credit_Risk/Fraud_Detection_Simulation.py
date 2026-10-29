#!/usr/bin/env python3
"""
============================================
REAL-TIME FRAUD DETECTION SIMULATION
============================================
This system simulates real-time transaction monitoring using the trained
credit risk model to detect fraudulent or high-risk transactions.

When a user attempts a transaction (withdrawal, purchase, loan request),
the system analyzes their profile and transaction details to determine if
the transaction should be allowed or blocked.

Author: XAI Project
Date: 2026
"""

import pandas as pd
import numpy as np
import joblib
import warnings
from datetime import datetime, timedelta
import time
import random
from colorama import init, Fore, Back, Style
import sys

warnings.filterwarnings('ignore')

# Initialize colorama for colored output
init(autoreset=True)


class FraudDetectionSystem:
    """Real-time fraud detection system for credit transactions."""

    def __init__(self):
        """Initialize the fraud detection system."""
        self.model = None
        self.feature_info = None
        self.feature_names = None
        self.label_encoders = None

        # Transaction limits and rules
        self.TRANSACTION_LIMITS = {
            'max_single_withdrawal': 5000,
            'max_daily_withdrawal': 10000,
            'max_weekly_withdrawal': 25000,
            'max_monthly_withdrawal': 50000,
            'max_loan_amount': 15000,
            'min_account_balance': 100,
            'max_transactions_per_hour': 5,
            'max_transactions_per_day': 20
        }

        # Risk thresholds
        self.RISK_THRESHOLDS = {
            'low': 0.3,  # < 30% risk = approve
            'medium': 0.5,  # 30-50% risk = review/warning
            'high': 0.7,  # 50-70% risk = block with override option
            'critical': 0.7  # > 70% risk = block completely
        }

        # User transaction history (simulated)
        self.user_history = {
            'today_withdrawals': 0,
            'today_transactions': 0,
            'week_withdrawals': 0,
            'month_withdrawals': 0,
            'last_transaction_time': None,
            'unusual_activity_count': 0
        }

    def load_model(self):
        """Load the trained XGBoost model and feature information."""
        print(Fore.CYAN + "=" * 80)
        print(Fore.CYAN + "INITIALIZING FRAUD DETECTION SYSTEM")
        print(Fore.CYAN + "=" * 80)

        try:
            # Load model
            print(Fore.YELLOW + "\nLoading AI model...")
            self.model = joblib.load('xgboost_model.pkl')
            print(Fore.GREEN + "✓ XGBoost model loaded")

            # Load feature info
            self.feature_info = joblib.load('feature_info.pkl')
            self.feature_names = self.feature_info['all_features']
            print(Fore.GREEN + f"✓ Feature information loaded ({len(self.feature_names)} features)")

            # Load label encoders
            self.label_encoders = joblib.load('label_encoders.pkl')
            print(Fore.GREEN + f"✓ Label encoders loaded")

            print(Fore.GREEN + "\n✓ System initialized successfully!")
            return True

        except FileNotFoundError as e:
            print(Fore.RED + f"\n✗ Error: Required files not found")
            print(Fore.RED + "Please ensure you have run Phase 1 (model training)")
            return False
        except Exception as e:
            print(Fore.RED + f"\n✗ Error loading model: {e}")
            return False

    def get_user_profile(self):
        """
        Get or create user profile.
        In a real system, this would fetch from database.
        For simulation, we'll use test data.
        """
        print(Fore.CYAN + "\n" + "=" * 80)
        print(Fore.CYAN + "USER AUTHENTICATION")
        print(Fore.CYAN + "=" * 80)

        # Load test data
        try:
            X_test = joblib.load('X_test.pkl')
            y_test = joblib.load('y_test.pkl')

            # Convert to DataFrame for easier handling
            if not isinstance(X_test, pd.DataFrame):
                X_test = pd.DataFrame(X_test, columns=self.feature_names)
            if not isinstance(y_test, pd.DataFrame) and not isinstance(y_test, pd.Series):
                y_test = pd.Series(y_test, name='Risk')

            # Select a random user from test set
            user_idx = random.randint(0, len(X_test) - 1)
            user_profile = X_test.iloc[user_idx].to_dict()
            actual_risk = y_test.iloc[user_idx] if hasattr(y_test, 'iloc') else y_test[user_idx]

            print(Fore.GREEN + f"✓ User authenticated (Profile ID: {user_idx})")

            return user_profile, user_idx, actual_risk, X_test, y_test

        except Exception as e:
            print(Fore.RED + f"✗ Error loading user data: {e}")
            return None, None, None, None, None

    def display_user_info(self, user_profile, user_idx, actual_risk):
        """Display user information."""
        print(Fore.CYAN + "\nUSER PROFILE:")
        print(Fore.CYAN + "-" * 80)

        # Extract key information
        checking_status = user_profile.get('checking_account_status', 0)
        credit_amount = user_profile.get('credit_amount', 0)
        age = user_profile.get('age_years', 0)
        employment = user_profile.get('employment_length', 0)
        existing_credits = user_profile.get('existing_credits', 0)

        print(f"  User ID: {Fore.YELLOW}{user_idx}")
        print(f"  Age: {Fore.YELLOW}{int(age)} years")
        print(f"  Employment Status: {Fore.YELLOW}{self._get_employment_desc(employment)}")
        print(f"  Checking Account: {Fore.YELLOW}{self._get_checking_desc(checking_status)}")
        print(f"  Existing Credit Amount: {Fore.YELLOW}€{int(credit_amount):,}")
        print(f"  Number of Existing Credits: {Fore.YELLOW}{int(existing_credits)}")
        print(
            f"  Actual Credit Risk: {Fore.RED if actual_risk == 1 else Fore.GREEN}{'BAD' if actual_risk == 1 else 'GOOD'}")

    def _get_employment_desc(self, code):
        """Get employment description from code."""
        descriptions = {
            0: "Unemployed",
            1: "< 1 year",
            2: "1-4 years",
            3: "4-7 years",
            4: "7+ years"
        }
        return descriptions.get(int(code), "Unknown")

    def _get_checking_desc(self, code):
        """Get checking account description from code."""
        descriptions = {
            0: "< 0 DM",
            1: "0 - 200 DM",
            2: ">= 200 DM",
            3: "No checking account"
        }
        return descriptions.get(int(code), "Unknown")

    def create_transaction(self, transaction_type='withdrawal'):
        """Create a transaction request."""
        print(Fore.CYAN + "\n" + "=" * 80)
        print(Fore.CYAN + "NEW TRANSACTION REQUEST")
        print(Fore.CYAN + "=" * 80)

        transaction = {
            'type': transaction_type,
            'timestamp': datetime.now(),
            'amount': 0,
            'purpose': None,
            'duration': None
        }

        print(f"\nTransaction Type: {Fore.YELLOW}{transaction_type.upper()}")

        # Get transaction amount
        if transaction_type == 'withdrawal':
            print(f"\nEnter withdrawal amount (Max: €{self.TRANSACTION_LIMITS['max_single_withdrawal']:,})")
            try:
                amount = float(input(Fore.YELLOW + "Amount (€): " + Fore.RESET))
                transaction['amount'] = amount
            except ValueError:
                print(Fore.RED + "Invalid amount. Using random value for simulation.")
                transaction['amount'] = random.choice([500, 1000, 2000, 3000, 5000, 7000, 10000])

        elif transaction_type == 'loan':
            print(f"\nEnter loan details:")
            try:
                amount = float(input(Fore.YELLOW + "Loan amount (€): " + Fore.RESET))
                duration = int(input(Fore.YELLOW + "Duration (months): " + Fore.RESET))
                transaction['amount'] = amount
                transaction['duration'] = duration
            except ValueError:
                print(Fore.RED + "Invalid input. Using random values for simulation.")
                transaction['amount'] = random.choice([1000, 3000, 5000, 8000, 10000])
                transaction['duration'] = random.choice([12, 24, 36, 48])

        print(Fore.GREEN + f"\n✓ Transaction created: €{transaction['amount']:,.2f}")

        return transaction

    def check_transaction_limits(self, transaction, user_profile):
        """Check if transaction violates hard limits."""
        violations = []
        amount = transaction['amount']
        trans_type = transaction['type']

        print(Fore.CYAN + "\n" + "=" * 80)
        print(Fore.CYAN + "CHECKING TRANSACTION LIMITS")
        print(Fore.CYAN + "=" * 80)

        # Check single transaction limit
        if trans_type == 'withdrawal':
            if amount > self.TRANSACTION_LIMITS['max_single_withdrawal']:
                violations.append({
                    'rule': 'Maximum Single Withdrawal',
                    'limit': self.TRANSACTION_LIMITS['max_single_withdrawal'],
                    'attempted': amount,
                    'severity': 'HIGH'
                })

        # Check daily limit
        if self.user_history['today_withdrawals'] + amount > self.TRANSACTION_LIMITS['max_daily_withdrawal']:
            violations.append({
                'rule': 'Daily Withdrawal Limit',
                'limit': self.TRANSACTION_LIMITS['max_daily_withdrawal'],
                'attempted': self.user_history['today_withdrawals'] + amount,
                'severity': 'HIGH'
            })

        # Check transaction frequency
        if self.user_history['today_transactions'] >= self.TRANSACTION_LIMITS['max_transactions_per_day']:
            violations.append({
                'rule': 'Maximum Transactions Per Day',
                'limit': self.TRANSACTION_LIMITS['max_transactions_per_day'],
                'attempted': self.user_history['today_transactions'] + 1,
                'severity': 'MEDIUM'
            })

        # Check for rapid successive transactions
        if self.user_history['last_transaction_time']:
            time_diff = (datetime.now() - self.user_history['last_transaction_time']).total_seconds()
            if time_diff < 60:  # Less than 1 minute
                violations.append({
                    'rule': 'Rapid Successive Transactions',
                    'limit': '1 minute interval',
                    'attempted': f'{time_diff:.0f} seconds',
                    'severity': 'CRITICAL'
                })

        # Check loan-specific limits
        if trans_type == 'loan':
            if amount > self.TRANSACTION_LIMITS['max_loan_amount']:
                violations.append({
                    'rule': 'Maximum Loan Amount',
                    'limit': self.TRANSACTION_LIMITS['max_loan_amount'],
                    'attempted': amount,
                    'severity': 'HIGH'
                })

        return violations

    def analyze_risk(self, user_profile, transaction):
        """Analyze risk using the AI model."""
        print(Fore.CYAN + "\n" + "=" * 80)
        print(Fore.CYAN + "AI RISK ANALYSIS IN PROGRESS...")
        print(Fore.CYAN + "=" * 80)

        # Simulate processing time
        for i in range(3):
            time.sleep(0.3)
            print(Fore.YELLOW + "." * (i + 1), end='\r')

        # Prepare feature vector
        feature_vector = pd.DataFrame([user_profile], columns=self.feature_names)

        # Get prediction
        risk_probability = self.model.predict_proba(feature_vector)[0, 1]
        risk_prediction = self.model.predict(feature_vector)[0]

        # Get feature importance for this prediction
        feature_contributions = self._calculate_feature_contributions(user_profile, feature_vector)

        print(Fore.GREEN + "\n✓ Risk analysis complete")

        return risk_probability, risk_prediction, feature_contributions

    def _calculate_feature_contributions(self, user_profile, feature_vector):
        """Calculate which features contribute most to the risk."""
        # Get feature importances
        feature_importance = self.model.feature_importances_

        # Calculate z-scores for unusual features
        X_train = joblib.load('X_train.pkl')
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train, columns=self.feature_names)

        contributions = []
        for i, feat_name in enumerate(self.feature_names):
            feat_value = feature_vector.iloc[0, i]
            feat_mean = X_train.iloc[:, i].mean()
            feat_std = X_train.iloc[:, i].std()

            if feat_std > 0:
                z_score = (feat_value - feat_mean) / feat_std
            else:
                z_score = 0

            contribution = feature_importance[i] * abs(z_score)

            contributions.append({
                'feature': feat_name,
                'value': feat_value,
                'z_score': z_score,
                'contribution': contribution
            })

        # Sort by contribution
        contributions.sort(key=lambda x: x['contribution'], reverse=True)

        return contributions[:5]  # Top 5

    def display_risk_analysis(self, risk_probability, risk_prediction, contributions):
        """Display risk analysis results."""
        print(Fore.CYAN + "\nRISK ANALYSIS RESULTS:")
        print(Fore.CYAN + "-" * 80)

        # Risk level
        if risk_probability < self.RISK_THRESHOLDS['low']:
            risk_level = "LOW"
            color = Fore.GREEN
        elif risk_probability < self.RISK_THRESHOLDS['medium']:
            risk_level = "MEDIUM"
            color = Fore.YELLOW
        elif risk_probability < self.RISK_THRESHOLDS['high']:
            risk_level = "HIGH"
            color = Fore.MAGENTA
        else:
            risk_level = "CRITICAL"
            color = Fore.RED

        print(f"\n  Risk Probability: {color}{risk_probability:.1%}")
        print(f"  Risk Level: {color}{risk_level}")
        print(f"  Risk Classification: {color}{'BAD CREDIT' if risk_prediction == 1 else 'GOOD CREDIT'}")

        # Top contributing factors
        print(Fore.CYAN + "\n  Top Risk Factors:")
        for i, contrib in enumerate(contributions, 1):
            z_score = contrib['z_score']
            impact = "↑ RISK" if z_score > 0 else "↓ RISK"
            impact_color = Fore.RED if z_score > 0 else Fore.GREEN

            print(f"    {i}. {contrib['feature']}: {contrib['value']:.2f}")
            print(f"       {impact_color}{impact} (Z-score: {z_score:.2f})")

    def make_decision(self, risk_probability, violations, transaction):
        """Make final decision on transaction."""
        print(Fore.CYAN + "\n" + "=" * 80)
        print(Fore.CYAN + "TRANSACTION DECISION")
        print(Fore.CYAN + "=" * 80)

        # Check for hard violations first
        if violations:
            return self._block_transaction_violations(violations, transaction)

        # Check risk level
        if risk_probability >= self.RISK_THRESHOLDS['critical']:
            return self._block_transaction_risk(risk_probability, transaction, "CRITICAL")
        elif risk_probability >= self.RISK_THRESHOLDS['high']:
            return self._block_transaction_risk(risk_probability, transaction, "HIGH")
        elif risk_probability >= self.RISK_THRESHOLDS['medium']:
            return self._warn_transaction(risk_probability, transaction)
        else:
            return self._approve_transaction(risk_probability, transaction)

    def _block_transaction_violations(self, violations, transaction):
        """Block transaction due to limit violations."""
        print(Fore.RED + "\n" + "=" * 80)
        print(Fore.RED + "⚠ TRANSACTION BLOCKED - LIMIT VIOLATIONS ⚠")
        print(Fore.RED + "=" * 80)

        print(Fore.RED + f"\nYour transaction of €{transaction['amount']:,.2f} has been BLOCKED.")
        print(Fore.RED + "\nViolated Rules:")

        for i, violation in enumerate(violations, 1):
            print(Fore.RED + f"\n  {i}. {violation['rule']}")
            print(Fore.YELLOW + f"     Limit: {violation['limit']}")
            print(Fore.YELLOW + f"     Your attempt: {violation['attempted']}")
            print(Fore.YELLOW + f"     Severity: {violation['severity']}")

        print(Fore.CYAN + "\n" + "=" * 80)
        print(Fore.CYAN + "WHAT YOU DID WRONG:")
        print(Fore.CYAN + "=" * 80)

        for violation in violations:
            if violation['rule'] == 'Maximum Single Withdrawal':
                print(Fore.YELLOW + f"\n❌ You attempted to withdraw €{violation['attempted']:,.2f}")
                print(Fore.YELLOW + f"   Maximum allowed per transaction: €{violation['limit']:,}")
                print(Fore.GREEN + f"\n✓ SOLUTION: Split into smaller withdrawals or contact support")

            elif violation['rule'] == 'Daily Withdrawal Limit':
                print(Fore.YELLOW + f"\n❌ Your total withdrawals today would be €{violation['attempted']:,.2f}")
                print(Fore.YELLOW + f"   Daily limit: €{violation['limit']:,}")
                print(Fore.GREEN + f"\n✓ SOLUTION: Wait until tomorrow or use a different payment method")

            elif violation['rule'] == 'Rapid Successive Transactions':
                print(Fore.YELLOW + f"\n❌ You made another transaction too quickly")
                print(Fore.YELLOW + f"   Time since last transaction: {violation['attempted']}")
                print(Fore.RED + f"   This is a common fraud pattern!")
                print(Fore.GREEN + f"\n✓ SOLUTION: Wait at least 1 minute between transactions")

            elif violation['rule'] == 'Maximum Transactions Per Day':
                print(Fore.YELLOW + f"\n❌ You've reached the maximum number of transactions today")
                print(Fore.YELLOW + f"   Limit: {violation['limit']} transactions per day")
                print(Fore.GREEN + f"\n✓ SOLUTION: Continue tomorrow or contact support for increased limits")

        print(Fore.RED + "\n" + "=" * 80)
        print(Fore.RED + "TRANSACTION DENIED - NO MONEY WITHDRAWN")
        print(Fore.RED + "=" * 80)

        return False

    def _block_transaction_risk(self, risk_probability, transaction, risk_level):
        """Block transaction due to high risk."""
        print(Fore.RED + "\n" + "=" * 80)
        print(Fore.RED + f"⚠ TRANSACTION BLOCKED - {risk_level} RISK ⚠")
        print(Fore.RED + "=" * 80)

        print(Fore.RED + f"\nYour transaction of €{transaction['amount']:,.2f} has been BLOCKED.")
        print(Fore.RED + f"Risk Level: {risk_probability:.1%}")

        print(Fore.CYAN + "\n" + "=" * 80)
        print(Fore.CYAN + "WHY WAS THIS BLOCKED?")
        print(Fore.CYAN + "=" * 80)

        print(Fore.YELLOW + "\nOur AI system detected a HIGH RISK of default based on:")
        print(Fore.YELLOW + "  • Your current credit profile")
        print(Fore.YELLOW + "  • Transaction amount and history")
        print(Fore.YELLOW + "  • Financial stability indicators")

        print(Fore.CYAN + "\n" + "=" * 80)
        print(Fore.CYAN + "WHAT YOU CAN DO:")
        print(Fore.CYAN + "=" * 80)

        print(Fore.GREEN + "\n1. Reduce the transaction amount")
        print(Fore.GREEN + "2. Improve your credit profile:")
        print(Fore.GREEN + "   • Maintain a positive checking account balance")
        print(Fore.GREEN + "   • Pay existing debts on time")
        print(Fore.GREEN + "   • Avoid multiple credit applications")
        print(Fore.GREEN + "3. Contact customer service for manual review")
        print(Fore.GREEN + "4. Provide additional collateral or guarantor")

        print(Fore.RED + "\n" + "=" * 80)
        print(Fore.RED + "TRANSACTION DENIED - NO MONEY WITHDRAWN")
        print(Fore.RED + "=" * 80)

        return False

    def _warn_transaction(self, risk_probability, transaction):
        """Warn user but allow transaction with confirmation."""
        print(Fore.YELLOW + "\n" + "=" * 80)
        print(Fore.YELLOW + "⚠ WARNING - MEDIUM RISK DETECTED ⚠")
        print(Fore.YELLOW + "=" * 80)

        print(Fore.YELLOW + f"\nTransaction: €{transaction['amount']:,.2f}")
        print(Fore.YELLOW + f"Risk Level: {risk_probability:.1%}")

        print(Fore.CYAN + "\n" + "=" * 80)
        print(Fore.CYAN + "RISK ASSESSMENT:")
        print(Fore.CYAN + "=" * 80)

        print(Fore.YELLOW + "\nOur AI system has detected moderate risk factors:")
        print(Fore.YELLOW + "  ⚠ This transaction may impact your credit standing")
        print(Fore.YELLOW + "  ⚠ Consider whether this is essential")
        print(Fore.YELLOW + "  ⚠ You may want to reduce the amount")

        print(Fore.CYAN + "\nDo you want to proceed? (yes/no): ", end='')

        try:
            confirm = input().lower().strip()
            if confirm in ['yes', 'y']:
                print(Fore.GREEN + "\n✓ Transaction approved with warning")
                self._update_transaction_history(transaction)
                return True
            else:
                print(Fore.YELLOW + "\n✓ Transaction cancelled by user")
                return False
        except KeyboardInterrupt:
            print(Fore.YELLOW + "\n\n✓ Transaction cancelled")
            return False

    def _approve_transaction(self, risk_probability, transaction):
        """Approve transaction."""
        print(Fore.GREEN + "\n" + "=" * 80)
        print(Fore.GREEN + "✓ TRANSACTION APPROVED ✓")
        print(Fore.GREEN + "=" * 80)

        print(Fore.GREEN + f"\nAmount: €{transaction['amount']:,.2f}")
        print(Fore.GREEN + f"Risk Level: {risk_probability:.1%} (LOW)")
        print(Fore.GREEN + f"Status: APPROVED")

        print(Fore.CYAN + "\nProcessing transaction...")
        time.sleep(1)

        print(Fore.GREEN + "\n✓ Transaction completed successfully!")
        print(Fore.GREEN + "✓ Money withdrawn from your account")

        self._update_transaction_history(transaction)

        return True

    def _update_transaction_history(self, transaction):
        """Update user transaction history."""
        self.user_history['today_withdrawals'] += transaction['amount']
        self.user_history['today_transactions'] += 1
        self.user_history['last_transaction_time'] = datetime.now()

    def run_simulation(self, interactive=True):
        """Run the fraud detection simulation."""
        # Load model
        if not self.load_model():
            return

        # Get user profile
        user_profile, user_idx, actual_risk, X_test, y_test = self.get_user_profile()
        if user_profile is None:
            return

        # Display user info
        self.display_user_info(user_profile, user_idx, actual_risk)

        print(Fore.CYAN + "\n" + "=" * 80)
        print(Fore.CYAN + "FRAUD DETECTION SYSTEM READY")
        print(Fore.CYAN + "=" * 80)

        if interactive:
            self._interactive_mode(user_profile)
        else:
            self._demo_mode(user_profile)

    def _interactive_mode(self, user_profile):
        """Interactive mode - user inputs transactions."""
        while True:
            print(Fore.CYAN + "\n" + "=" * 80)
            print(Fore.CYAN + "SELECT TRANSACTION TYPE:")
            print(Fore.CYAN + "=" * 80)
            print(Fore.YELLOW + "\n1. Withdrawal")
            print(Fore.YELLOW + "2. Loan Request")
            print(Fore.YELLOW + "3. Exit")

            try:
                choice = input(Fore.CYAN + "\nEnter choice (1-3): " + Fore.RESET).strip()

                if choice == '3':
                    print(Fore.GREEN + "\n✓ Thank you for using our fraud detection system!")
                    break

                elif choice == '1':
                    transaction = self.create_transaction('withdrawal')
                elif choice == '2':
                    transaction = self.create_transaction('loan')
                else:
                    print(Fore.RED + "Invalid choice. Please try again.")
                    continue

                # Check limits
                violations = self.check_transaction_limits(transaction, user_profile)

                # Analyze risk
                risk_prob, risk_pred, contributions = self.analyze_risk(user_profile, transaction)

                # Display analysis
                self.display_risk_analysis(risk_prob, risk_pred, contributions)

                # Make decision
                approved = self.make_decision(risk_prob, violations, transaction)

                print(Fore.CYAN + "\n" + "=" * 80)
                input(Fore.CYAN + "Press Enter to continue...")

            except KeyboardInterrupt:
                print(Fore.YELLOW + "\n\n✓ Exiting fraud detection system")
                break

    def _demo_mode(self, user_profile):
        """Demo mode - automated scenarios."""
        scenarios = [
            {'type': 'withdrawal', 'amount': 500},
            {'type': 'withdrawal', 'amount': 3000},
            {'type': 'withdrawal', 'amount': 8000},
            {'type': 'loan', 'amount': 5000, 'duration': 24},
            {'type': 'withdrawal', 'amount': 15000},
        ]

        for i, scenario in enumerate(scenarios, 1):
            print(Fore.CYAN + f"\n{'=' * 80}")
            print(Fore.CYAN + f"DEMO SCENARIO {i}/{len(scenarios)}")
            print(Fore.CYAN + f"{'=' * 80}")

            transaction = {
                'type': scenario['type'],
                'amount': scenario['amount'],
                'timestamp': datetime.now(),
                'duration': scenario.get('duration')
            }

            print(Fore.YELLOW + f"\nSimulating: {scenario['type']} of €{scenario['amount']:,.2f}")

            # Check limits
            violations = self.check_transaction_limits(transaction, user_profile)

            # Analyze risk
            risk_prob, risk_pred, contributions = self.analyze_risk(user_profile, transaction)

            # Display analysis
            self.display_risk_analysis(risk_prob, risk_pred, contributions)

            # Make decision
            self.make_decision(risk_prob, violations, transaction)

            time.sleep(2)


def main():
    """Main function."""
    print(Fore.CYAN + """
    ╔════════════════════════════════════════════════════════════════════════╗
    ║                                                                        ║
    ║           🔒 REAL-TIME FRAUD DETECTION SYSTEM 🔒                      ║
    ║                                                                        ║
    ║              Powered by AI & Machine Learning                          ║
    ║                                                                        ║
    ╚════════════════════════════════════════════════════════════════════════╝
    """)

    # Create system
    system = FraudDetectionSystem()

    # Ask for mode
    print(Fore.CYAN + "\nSelect mode:")
    print(Fore.YELLOW + "1. Interactive Mode (you input transactions)")
    print(Fore.YELLOW + "2. Demo Mode (automated scenarios)")

    try:
        choice = input(Fore.CYAN + "\nEnter choice (1-2): " + Fore.RESET).strip()

        if choice == '1':
            system.run_simulation(interactive=True)
        else:
            system.run_simulation(interactive=False)

    except KeyboardInterrupt:
        print(Fore.YELLOW + "\n\n✓ System shutdown")
    except Exception as e:
        print(Fore.RED + f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()