"""
MindX Maritime Compliance Engine
================================
A Python-based engine that benchmarks vessel performance against regulatory targets.
Implements IMO GHG Strategy 2023 compliance with market-based carbon trading.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle
import json
import os
from datetime import datetime


class ComplianceEngine:
    """
    Main compliance engine that translates IMO GHG regulations into mathematical models.

    Key Functions:
    1. Predicts CO2 emissions using ML
    2. Calculates GHG intensity for each vessel
    3. Sets regulatory targets (5% reduction from average)
    4. Implements compliance balance with carbon credit trading
    5. Detects operational anomalies
    """

    def __init__(self, data_path="dataset.csv"):
        """
        Initialize the compliance engine.

        Args:
            data_path (str): Path to dataset file
        """
        self.data_path = data_path
        self.df = None
        self.model = None
        self.label_encoders = {}
        self.target_intensity = None
        self.avg_intensity = None
        self.analysis_date = datetime.now().strftime("%Y-%m-%d")

        # Ship type to DWT (Deadweight Tonnage) mapping for GHG intensity calculation
        self.ship_type_capacity = {
            'Oil Service Boat': 5000,
            'Fishing Trawler': 200,
            'Container': 50000,
            'Tanker': 80000,
            'Bulk Carrier': 60000,
            'General Cargo': 30000,
            'RoRo': 15000,
            'Oil Tanker': 100000,
            'LNG Carrier': 65000,
            'Chemical Tanker': 25000,
            'Passenger': 5000,
            'Offshore': 8000,
            'Tug': 500
        }

    def load_data(self):
        """
        Load and preprocess the dataset.

        Returns:
            pandas.DataFrame: Preprocessed dataset
        """
        print(f"📁 Loading dataset from: {self.data_path}")

        # Handle different file formats
        if self.data_path.endswith('.csv'):
            self.df = pd.read_csv(self.data_path)
        elif self.data_path.endswith(('.xlsx', '.xls')):
            self.df = pd.read_excel(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")

        print(f"✅ Loaded {len(self.df)} rows, {len(self.df.columns)} columns")
        print(f"📋 Columns: {list(self.df.columns)}")

        # Convert numeric columns
        numeric_cols = ['distance', 'fuel_consumption', 'CO2_emissions', 'engine_efficiency']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')

        # Handle month column
        if 'month' in self.df.columns:
            self._process_month_column()

        # Fill missing values
        self.df = self.df.fillna(method='ffill').fillna(method='bfill')

        return self.df

    def _process_month_column(self):
        """Process month column to numeric format."""
        try:
            self.df['month'] = pd.to_numeric(self.df['month'], errors='coerce')
        except:
            month_map = {
                'january': 1, 'jan': 1, 'janvier': 1,
                'february': 2, 'feb': 2, 'février': 2,
                'march': 3, 'mar': 3, 'mars': 3,
                'april': 4, 'apr': 4, 'avril': 4,
                'may': 5, 'mai': 5,
                'june': 6, 'jun': 6, 'juin': 6,
                'july': 7, 'jul': 7, 'juillet': 7,
                'august': 8, 'aug': 8, 'août': 8,
                'september': 9, 'sep': 9, 'septembre': 9,
                'october': 10, 'oct': 10, 'octobre': 10,
                'november': 11, 'nov': 11, 'novembre': 11,
                'december': 12, 'dec': 12, 'décembre': 12
            }
            self.df['month'] = self.df['month'].astype(str).str.lower().map(month_map)

    def calculate_ghg_intensity(self):
        """
        Calculate GHG intensity following IMO guidelines.

        Formula: GHG Intensity = CO2 Emissions (kg) / [Distance (NM) × DWT (tonnes)]
        Units: kg CO2 per nautical mile-tonne

        Returns:
            pandas.DataFrame: Dataset with GHG intensity calculated
        """
        print("\n📊 CALCULATING GHG INTENSITY")
        print("=" * 40)
        print("Formula: GHG Intensity = CO2 Emissions (kg) / [Distance (NM) × DWT (tonnes)]")
        print("Units: kg CO2 per nautical mile-tonne")

        # Estimate DWT based on ship type
        if 'ship_type' in self.df.columns:
            self.df['estimated_dwt'] = self.df['ship_type'].map(
                lambda x: self.ship_type_capacity.get(x, 20000)
            )
        else:
            self.df['estimated_dwt'] = 20000  # Default 20,000 tonnes

        # Calculate GHG Intensity
        self.df['GHG_Intensity'] = self.df['CO2_emissions'] / (self.df['distance'] * self.df['estimated_dwt'])

        # Calculate fleet statistics
        self.avg_intensity = self.df['GHG_Intensity'].mean()
        self.target_intensity = self.avg_intensity * 0.95  # 5% reduction for 2026 target

        print(f"\n📈 FLEET GHG INTENSITY:")
        print(f"   Average GHG Intensity: {self.avg_intensity:.6f} kg CO2/NM-tonne")
        print(f"   2026 Target (5% reduction): {self.target_intensity:.6f} kg CO2/NM-tonne")
        print(f"   Required Reduction: {(self.avg_intensity - self.target_intensity):.6f} kg CO2/NM-tonne")

        return self.df

    def prepare_features(self, df):
        """
        Prepare features for machine learning model.

        Args:
            df (pandas.DataFrame): Input dataframe

        Returns:
            tuple: (X_features, y_target, feature_names)
        """
        features = []

        # Encode categorical features
        categorical_cols = ['ship_type', 'fuel_type', 'weather_conditions']
        for col in categorical_cols:
            if col in df.columns:
                le = LabelEncoder()
                encoded_col = f"{col}_encoded"
                df[encoded_col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
                features.append(encoded_col)

        # Add numerical features
        numerical_features = ['distance', 'fuel_consumption', 'engine_efficiency']
        for feat in numerical_features:
            if feat in df.columns:
                features.append(feat)

        # Add cyclical month features for seasonality
        if 'month' in df.columns and pd.api.types.is_numeric_dtype(df['month']):
            df['month'] = df['month'].clip(1, 12)
            df['month_sin'] = np.sin(2 * np.pi * (df['month'] - 1) / 12)
            df['month_cos'] = np.cos(2 * np.pi * (df['month'] - 1) / 12)
            features.extend(['month_sin', 'month_cos'])

        # Print feature details
        print(f"\n🎯 FEATURES FOR PREDICTION MODEL ({len(features)} total):")
        for i, feat in enumerate(features, 1):
            print(f"   {i:2d}. {feat}")

        # Prepare X and y
        X = df[features]
        y = df['CO2_emissions']

        return X, y, features

    def train_model(self):
        """
        Train Random Forest model for CO2 emission prediction.

        Returns:
            tuple: (vessel_compliance_df, metrics_dict)
        """
        print("\n" + "="*60)
        print("🤖 TRAINING CO2 EMISSION PREDICTION MODEL")
        print("="*60)

        # Load and preprocess data
        df = self.load_data()
        df = self.calculate_ghg_intensity()

        # Prepare features
        X, y, features = self.prepare_features(df)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"\n📊 DATA SPLIT:")
        print(f"   Training: {X_train.shape[0]} samples")
        print(f"   Testing:  {X_test.shape[0]} samples")

        # Train Random Forest model
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )

        self.model.fit(X_train, y_train)

        # Evaluate model
        y_pred = self.model.predict(X_test)
        metrics = self._calculate_model_metrics(y_test, y_pred, features)

        # Calculate compliance status
        vessel_compliance = self._calculate_compliance_status(df)

        return vessel_compliance, metrics

    def _calculate_model_metrics(self, y_test, y_pred, features):
        """Calculate and display model performance metrics."""
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print(f"\n📈 MODEL PERFORMANCE:")
        print(f"   R² Score: {r2:.4f} (1.0 is perfect)")
        print(f"   RMSE: {rmse:.2f} kg (lower is better)")
        print(f"   MAE: {mae:.2f} kg (lower is better)")

        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            print(f"\n🔝 TOP 5 FEATURE IMPORTANCES:")
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            for idx, row in feature_importance.head(5).iterrows():
                print(f"   {row['feature']}: {row['importance']:.4f}")

        return {
            'model_r2': float(r2),
            'model_rmse': float(rmse),
            'model_mae': float(mae),
            'avg_intensity': float(self.avg_intensity),
            'target_intensity': float(self.target_intensity)
        }

    def _calculate_compliance_status(self, df):
        """Calculate compliance status for each vessel."""
        # Determine compliance status
        df['Compliance_Status'] = np.where(
            df['GHG_Intensity'] <= self.target_intensity, 'Surplus', 'Deficit'
        )

        # Calculate deficit/surplus amounts
        df['Deficit_Amount'] = np.where(
            df['Compliance_Status'] == 'Deficit',
            df['GHG_Intensity'] - self.target_intensity,
            0
        )

        df['Surplus_Amount'] = np.where(
            df['Compliance_Status'] == 'Surplus',
            self.target_intensity - df['GHG_Intensity'],
            0
        )

        # Group by vessel
        vessel_compliance = df.groupby('ship_id').agg({
            'ship_type': 'first',
            'GHG_Intensity': 'mean',
            'Compliance_Status': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown',
            'Deficit_Amount': 'sum',
            'Surplus_Amount': 'sum',
            'CO2_emissions': 'sum',
            'distance': 'sum',
            'fuel_consumption': 'sum',
            'engine_efficiency': 'mean'
        }).reset_index()

        vessel_compliance.columns = [
            'Vessel_ID', 'Ship_Type', 'Avg_GHG_Intensity', 'Compliance_Status',
            'Total_Deficit', 'Total_Surplus', 'Total_CO2_kg', 'Total_Distance_NM',
            'Total_Fuel_MT', 'Avg_Engine_Efficiency'
        ]

        return vessel_compliance

    def calculate_compliance_balance(self, vessel_compliance):
        """
        Calculate compliance balance and simulate carbon credit trading.

        Args:
            vessel_compliance (pandas.DataFrame): Vessel compliance data

        Returns:
            tuple: (trades_list, compliance_metrics)
        """
        print("\n" + "="*60)
        print("💰 COMPLIANCE BALANCE & CARBON CREDIT TRADING")
        print("="*60)

        # Filter vessels
        surplus_vessels = vessel_compliance[vessel_compliance['Compliance_Status'] == 'Surplus']
        deficit_vessels = vessel_compliance[vessel_compliance['Compliance_Status'] == 'Deficit']

        # Calculate totals
        total_surplus = surplus_vessels['Total_Surplus'].sum()
        total_deficit = deficit_vessels['Total_Deficit'].sum()
        net_balance = total_surplus - total_deficit

        # Summary statistics
        surplus_count = len(surplus_vessels)
        deficit_count = len(deficit_vessels)
        total_vessels = len(vessel_compliance)
        compliance_rate = (surplus_count / total_vessels * 100) if total_vessels > 0 else 0

        print(f"\n📊 COMPLIANCE MARKET OVERVIEW:")
        print(f"   Total Vessels: {total_vessels}")
        print(f"   Surplus Vessels: {surplus_count} ({compliance_rate:.1f}% of fleet)")
        print(f"   Deficit Vessels: {deficit_count}")
        print(f"   Total Credits Available: {total_surplus:.6f}")
        print(f"   Total Credits Needed: {total_deficit:.6f}")
        print(f"   Net Balance: {net_balance:.6f}")
        print(f"   Market Status: {'SURPLUS' if net_balance >= 0 else 'DEFICIT'}")
        print(f"   Estimated Credit Price: $150-250/credit (based on EU ETS rates)")

        # Show top traders
        if len(surplus_vessels) > 0:
            top_surplus = surplus_vessels.sort_values('Total_Surplus', ascending=False).head(5)
            print(f"\n🏆 TOP 5 SURPLUS VESSELS (Can sell credits):")
            for idx, row in top_surplus.iterrows():
                print(f"   {row['Vessel_ID']} ({row['Ship_Type']}): {row['Total_Surplus']:.3f} credits")

        if len(deficit_vessels) > 0:
            top_deficit = deficit_vessels.sort_values('Total_Deficit', ascending=False).head(5)
            print(f"\n🚨 TOP 5 DEFICIT VESSELS (Must buy credits):")
            for idx, row in top_deficit.iterrows():
                print(f"   {row['Vessel_ID']} ({row['Ship_Type']}): {row['Total_Deficit']:.3f} credits")

        # Simulate trades
        trades = self._simulate_trades(surplus_vessels, deficit_vessels)

        # Compliance metrics
        compliance_metrics = {
            'total_vessels': total_vessels,
            'surplus_count': surplus_count,
            'deficit_count': deficit_count,
            'total_surplus': float(total_surplus),
            'total_deficit': float(total_deficit),
            'net_balance': float(net_balance),
            'compliance_rate': float(compliance_rate),
            'estimated_market_value': float(abs(net_balance) * 200),  # $200 per credit
            'fleet_compliant': net_balance >= 0
        }

        return trades, compliance_metrics

    def _simulate_trades(self, surplus_vessels, deficit_vessels):
        """Simulate carbon credit trades between vessels."""
        trades = []

        if len(surplus_vessels) == 0 or len(deficit_vessels) == 0:
            print("\n🤝 No trades possible (all vessels are in same category)")
            return trades

        print(f"\n🤝 POTENTIAL CARBON CREDIT TRADES:")

        # Create mutable copies
        surplus_copy = surplus_vessels.copy()
        deficit_copy = deficit_vessels.copy()

        # Match deficit vessels with surplus vessels
        for _, deficit_row in deficit_copy.iterrows():
            deficit_vessel = deficit_row['Vessel_ID']
            deficit_amount = deficit_row['Total_Deficit']

            if deficit_amount <= 0:
                continue

            # Find matching surplus vessels
            for _, surplus_row in surplus_copy.iterrows():
                surplus_vessel = surplus_row['Vessel_ID']
                surplus_amount = surplus_row['Total_Surplus']

                if surplus_amount <= 0 or deficit_amount <= 0:
                    continue

                # Calculate trade amount
                trade_amount = min(deficit_amount, surplus_amount)
                trade_value = trade_amount * 200  # $200 per credit

                trades.append({
                    'buyer': deficit_vessel,
                    'seller': surplus_vessel,
                    'credits': float(trade_amount),
                    'value_usd': float(trade_value),
                    'buyer_ship_type': deficit_row['Ship_Type'],
                    'seller_ship_type': surplus_row['Ship_Type']
                })

                # Update amounts
                deficit_amount -= trade_amount
                surplus_copy.loc[surplus_copy['Vessel_ID'] == surplus_vessel, 'Total_Surplus'] -= trade_amount

                if deficit_amount <= 0:
                    break

        # Display top 3 trades
        for trade in trades[:3]:
            print(f"   {trade['buyer']} → {trade['seller']}: {trade['credits']:.3f} credits (${trade['value_usd']:.2f})")

        if len(trades) == 0:
            print("   No feasible trades identified")

        return trades

    def explain_legal_mandate(self):
        """
        Explain how the mathematical model implements the IMO GHG regulations.
        """
        print("\n" + "="*60)
        print("⚖️ LEGAL MANDATE TRANSLATION: IMO GHG STRATEGY 2023")
        print("="*60)

        print(f"""
📜 REGULATORY FRAMEWORK:
   • International Maritime Organization (IMO) GHG Strategy
   • Target: Reduce carbon intensity by at least 40% by 2030
   • Interim Target: 5% reduction by 2026 (implemented here)

🔢 MATHEMATICAL IMPLEMENTATION:

STEP 1 - GHG INTENSITY CALCULATION:
   Formula: GHG Intensity = CO2 Emissions / [Distance × DWT]
   Current Fleet Average: {self.avg_intensity:.6f} kg CO2/NM-tonne
   Units: kg of CO2 per nautical mile of transport work

STEP 2 - REGULATORY TARGET SETTING:
   2026 Target = Current Average × (1 - 0.05)
               = {self.avg_intensity:.6f} × 0.95
               = {self.target_intensity:.6f} kg CO2/NM-tonne

STEP 3 - INDIVIDUAL VESSEL COMPLIANCE:
   Rule: Vessel GHG Intensity ≤ Target Intensity
   Surplus: Intensity < Target (can sell carbon credits)
   Deficit: Intensity > Target (must buy carbon credits)

STEP 4 - MARKET-BASED COMPLIANCE MECHANISM:
   • Carbon credits represent the right to emit 1 kg CO2/NM-tonne
   • Surplus vessels generate credits: Credit = Target - Actual
   • Deficit vessels need credits: Credit Needed = Actual - Target
   • Credits are tradable between vessels

STEP 5 - FLEET-WIDE COMPLIANCE:
   • Total Credits Available = Σ(Target - Actual) for surplus vessels
   • Total Credits Needed = Σ(Actual - Target) for deficit vessels
   • Fleet is compliant if: Available Credits ≥ Needed Credits

🛠️ COMPLIANCE OPTIONS FOR DEFICIT VESSELS:
   1. TECHNICAL MEASURES:
      • Improve engine efficiency
      • Install energy-saving devices
      • Use hull air lubrication

   2. OPERATIONAL MEASURES:
      • Speed optimization (slow steaming)
      • Weather routing optimization
      • Just-in-time port arrivals

   3. ALTERNATIVE FUELS:
      • Switch to LNG (20-25% CO2 reduction)
      • Use biofuels (up to 90% reduction)
      • Prepare for hydrogen/ammonia

   4. MARKET-BASED MEASURES:
      • Purchase carbon credits from surplus vessels
      • Invest in carbon offset projects
      • Join emissions trading schemes

📊 REGULATORY COMPLIANCE PATH:
   Year 2024: Baseline calculation (this analysis)
   Year 2026: 5% reduction target (implemented)
   Year 2030: 40% reduction target (future)
   Year 2050: Net-zero emissions (long-term goal)
""")

    def find_anomalies(self):
        """
        Detect operational anomalies for technical memo.

        Returns:
            list: Anomaly records
        """
        print("\n" + "="*60)
        print("🔍 TASK C: ANOMALY DETECTION")
        print("="*60)

        anomalies = []

        # Calculate fuel efficiency
        df = self.df.copy()
        df['fuel_efficiency'] = df['distance'] / df['fuel_consumption']

        # Statistical outlier detection
        Q1 = df['fuel_efficiency'].quantile(0.25)
        Q3 = df['fuel_efficiency'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        low_efficiency = df[df['fuel_efficiency'] < lower_bound]
        high_efficiency = df[df['fuel_efficiency'] > upper_bound]

        print(f"\n📊 FUEL EFFICIENCY ANALYSIS:")
        print(f"   Average: {df['fuel_efficiency'].mean():.4f} NM/MT")
        print(f"   Standard Deviation: {df['fuel_efficiency'].std():.4f}")
        print(f"   Low efficiency threshold (< {lower_bound:.4f}): {len(low_efficiency)} vessels")
        print(f"   High efficiency threshold (> {upper_bound:.4f}): {len(high_efficiency)} vessels")

        # Analyze anomalies
        if len(low_efficiency) > 0:
            print(f"\n🚨 LOW EFFICIENCY ANOMALIES (Potential Issues):")
            for idx, row in low_efficiency.head(3).iterrows():
                print(f"   • Vessel {row['ship_id']}: {row['fuel_efficiency']:.4f} NM/MT")
                print(f"     Possible causes: Hull fouling, engine issues, bad weather")

                anomalies.append({
                    'vessel_id': row['ship_id'],
                    'anomaly_type': 'Low Fuel Efficiency',
                    'metric_value': float(row['fuel_efficiency']),
                    'threshold': float(lower_bound),
                    'date': self.analysis_date,
                    'recommended_action': 'Schedule hull cleaning and engine maintenance'
                })

        # High CO2 per distance anomalies
        df['co2_per_distance'] = df['CO2_emissions'] / df['distance']
        co2_threshold = df['co2_per_distance'].quantile(0.95)
        high_co2 = df[df['co2_per_distance'] > co2_threshold]

        if len(high_co2) > 0:
            print(f"\n🔥 HIGH CO2 PER DISTANCE ANOMALIES (Top 5%):")
            for idx, row in high_co2.head(3).iterrows():
                print(f"   • Vessel {row['ship_id']}: {row['co2_per_distance']:.2f} kg/NM")
                print(f"     Fuel Type: {row['fuel_type']}")
                print(f"     Engine Efficiency: {row['engine_efficiency']}%")

                anomalies.append({
                    'vessel_id': row['ship_id'],
                    'anomaly_type': 'High CO2 per Distance',
                    'metric_value': float(row['co2_per_distance']),
                    'threshold': float(co2_threshold),
                    'date': self.analysis_date,
                    'recommended_action': 'Check engine calibration and consider fuel switch'
                })

        return anomalies

    def save_results(self, vessel_compliance, trades, metrics, anomalies):
        """
        Save all results to files.

        Args:
            vessel_compliance: Compliance data per vessel
            trades: Trading data
            metrics: Performance metrics
            anomalies: Anomaly data
        """
        # Create directories
        os.makedirs('backend/data', exist_ok=True)
        os.makedirs('backend/model', exist_ok=True)
        os.makedirs('backend/results', exist_ok=True)

        # Save model
        model_data = {
            'model': self.model,
            'label_encoders': self.label_encoders,
            'target_intensity': self.target_intensity,
            'avg_intensity': self.avg_intensity,
            'ship_type_capacity': self.ship_type_capacity,
            'training_date': self.analysis_date
        }

        with open('backend/model/compliance_model.pkl', 'wb') as f:
            pickle.dump(model_data, f)

        print(f"\n💾 Model saved to backend/model/compliance_model.pkl")

        # Save compliance data
        vessel_compliance.to_json('backend/data/compliance_data.json', orient='records', indent=2)
        print(f"✅ Compliance data saved to backend/data/compliance_data.json")

        # Save trading data
        if trades:
            with open('backend/data/trading_data.json', 'w') as f:
                json.dump(trades, f, indent=2)
            print(f"✅ Trading data saved to backend/data/trading_data.json")

        # Save anomalies
        if anomalies:
            with open('backend/data/anomalies.json', 'w') as f:
                json.dump(anomalies, f, indent=2)
            print(f"✅ Anomalies saved to backend/data/anomalies.json")

        # Save metrics
        all_metrics = {
            'model_performance': {
                'r2_score': metrics.get('model_r2', 0),
                'rmse': metrics.get('model_rmse', 0),
                'mae': metrics.get('model_mae', 0)
            },
            'compliance_metrics': metrics,
            'ghg_intensity': {
                'average': float(self.avg_intensity),
                'target_2026': float(self.target_intensity),
                'reduction_required': float(self.avg_intensity - self.target_intensity)
            },
            'analysis_date': self.analysis_date
        }

        with open('backend/data/model_metrics.json', 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"✅ Metrics saved to backend/data/model_metrics.json")

        # Save CSV for easy viewing
        vessel_compliance.to_csv('compliance_analysis.csv', index=False)
        print(f"📊 CSV report saved to compliance_analysis.csv")

    def generate_summary_report(self, compliance_metrics):
        """
        Generate final summary report.

        Args:
            compliance_metrics (dict): Compliance metrics
        """
        print("\n" + "="*60)
        print("📋 ANALYSIS COMPLETE - REGULATORY IMPLEMENTATION SUMMARY")
        print("="*60)

        print(f"""
✅ REGULATORY COMPLIANCE ENGINE IMPLEMENTED
   Date: {self.analysis_date}
   Regulation: IMO GHG Strategy 2023 (5% reduction by 2026)

📊 FLEET PERFORMANCE SUMMARY:
   • Total Vessels Analyzed: {compliance_metrics['total_vessels']}
   • Compliant Vessels: {compliance_metrics['surplus_count']} ({compliance_metrics['compliance_rate']:.1f}%)
   • Non-Compliant Vessels: {compliance_metrics['deficit_count']}
   • Fleet Compliance Status: {'COMPLIANT ✅' if compliance_metrics['fleet_compliant'] else 'NON-COMPLIANT ⚠️'}

💰 CARBON MARKET ANALYSIS:
   • Credits Available: {compliance_metrics['total_surplus']:.3f}
   • Credits Needed: {compliance_metrics['total_deficit']:.3f}
   • Net Balance: {compliance_metrics['net_balance']:.3f}
   • Estimated Market Value: ${compliance_metrics['estimated_market_value']:.2f}

🎯 GHG INTENSITY TARGETS:
   • Current Fleet Average: {self.avg_intensity:.6f} kg CO2/NM-tonne
   • 2026 Regulatory Target: {self.target_intensity:.6f} kg CO2/NM-tonne
   • Required Reduction: {(self.avg_intensity - self.target_intensity):.6f} kg CO2/NM-tonne ({((self.avg_intensity - self.target_intensity)/self.avg_intensity*100):.1f}%)

🤖 PREDICTIVE MODEL PERFORMANCE:
   • R² Score: {compliance_metrics.get('model_r2', 'N/A'):.4f}
   • Prediction Error (RMSE): {compliance_metrics.get('model_rmse', 'N/A'):.2f} kg CO2
   • Mean Absolute Error: {compliance_metrics.get('model_mae', 'N/A'):.2f} kg CO2

🚢 RECOMMENDED ACTIONS:
   1. For Deficit Vessels: Purchase credits or implement efficiency measures
   2. For Surplus Vessels: Monitor credit prices for optimal selling
   3. For Fleet Managers: Consider credit pooling strategies
   4. Technical Team: Address detected anomalies promptly

📈 NEXT STEPS:
   • Monthly compliance monitoring recommended
   • Consider 2030 target of 40% reduction
   • Explore alternative fuel options
   • Implement continuous improvement program
""")
        print("="*60)
        print(" All results saved to backend/ directory")
        print(" Detailed report: compliance_analysis.csv")
        print("="*60)


def main():
    """
    Main execution function for the Compliance Engine.
    """
    print("🚀" + "="*50)
    print("       MARITIME COMPLIANCE ENGINE v2.0")
    print("       IMO GHG Regulation Implementation")
    print("="*50 + "\n")

    # Find dataset
    dataset_path = None
    possible_paths = [
        "dataset.csv",
        "data/dataset.csv",
        "backend/data/dataset.csv",
        "data/raw/dataset.csv"
    ]

    for path in possible_paths:
        if os.path.exists(path):
            dataset_path = path
            print(f" Found dataset at: {path}")
            break

    if dataset_path is None:
        print(" No dataset found in standard locations.")
        print(" Searching for CSV files in current directory...")
        csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
        if csv_files:
            dataset_path = csv_files[0]
            print(f" Using: {dataset_path}")
        else:
            print(" ERROR: No CSV files found.")
            print(" Please place your dataset in the current directory as 'dataset.csv'")
            return

    print("\n" + "="*60)
    print("INITIALIZING COMPLIANCE ENGINE...")
    print("="*60)

    try:
        # Initialize engine
        engine = ComplianceEngine(data_path=dataset_path)

        # Step 1: Train model and calculate compliance
        print("\n STEP 1: Training predictive model...")
        vessel_compliance, model_metrics = engine.train_model()

        # Step 2: Calculate compliance balance and trading
        print("\n STEP 2: Calculating compliance balance...")
        trades, compliance_metrics = engine.calculate_compliance_balance(vessel_compliance)

        # Merge metrics
        all_metrics = {**model_metrics, **compliance_metrics}

        # Step 3: Explain legal mandate translation
        print("\n STEP 3: Explaining regulatory implementation...")
        engine.explain_legal_mandate()

        # Step 4: Detect anomalies
        print("\n STEP 4: Detecting operational anomalies...")
        anomalies = engine.find_anomalies()

        # Step 5: Save all results
        print("\n STEP 5: Saving results...")
        engine.save_results(vessel_compliance, trades, all_metrics, anomalies)

        # Step 6: Generate final report
        engine.generate_summary_report(all_metrics)

        print("\n COMPLIANCE ENGINE EXECUTION COMPLETE!")
        print("   All regulatory requirements implemented successfully.")

    except Exception as e:
        print(f"\n ERROR during execution: {e}")
        import traceback
        traceback.print_exc()
        print("\n TROUBLESHOOTING:")
        print("   1. Check dataset format (CSV with required columns)")
        print("   2. Ensure required columns: CO2_emissions, distance, fuel_consumption")
        print("   3. Check file permissions")
        print("   4. Contact support if issue persists")


if __name__ == "__main__":
    main()