import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

class AuraWatchAgent:
    """Monitoring Agent: Detects anomalies in transactions using Isolation Forest."""
    def __init__(self, model_path="aura_watch_model.pkl"):
        self.model_path = model_path
        self.model = IsolationForest(contamination=0.03, random_state=42)
        self.is_trained = False
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self.is_trained = True
            except:
                pass

    def _prepare_features(self, df):
        return abs(df['Amount']).fillna(0).values.reshape(-1, 1)

    def train(self, historical_df):
        expenses = historical_df[historical_df['Type'] == 'Expense'].copy()
        if len(expenses) < 10: return False
        X = self._prepare_features(expenses)
        self.model.fit(X)
        self.is_trained = True
        joblib.dump(self.model, self.model_path)
        return True

    def detect(self, transaction_dict):
        # Defensive: If 'Type' is missing, assume it's an Expense for checking
        t_type = transaction_dict.get('Type', 'Expense')
        if not self.is_trained or t_type != 'Expense': return False
        X = np.array([[abs(transaction_dict.get('Amount', 0))]])
        return self.model.predict(X)[0] == -1

class AuraBrainAgent:
    """Decision Agent: Provides forecasting."""
    def forecast_end_of_month_expenses(self, historical_df):
        # Defensive: If 'Type' missing, assume pre-filtered for Expense
        if 'Type' in historical_df.columns:
            expenses = historical_df[historical_df['Type'] == 'Expense'].copy()
        else:
            expenses = historical_df.copy()

        if expenses.empty: return 0
        
        expenses['Date'] = pd.to_datetime(expenses['Date'])
        expenses['YearMonth'] = expenses['Date'].dt.to_period('M')
        monthly_totals = expenses.groupby('YearMonth')['Amount'].sum().abs()
        return round(monthly_totals.iloc[-3:].mean(), 2) if len(monthly_totals) >= 1 else 0

class AuraWellnessAgent:
    """Calculates a Financial Wellness Score (0-100) preventing Debt Traps."""
    def calculate_score(self, historical_df, current_cash=5000):
        # Robust logic: handle missing 'Type'
        if 'Type' in historical_df.columns:
            expenses = historical_df[historical_df['Type'] == 'Expense'].copy()
            income = historical_df[historical_df['Type'] == 'Income'].copy()
        else:
            # If no 'Type', assume data is all expenses
            expenses = historical_df.copy()
            income = pd.DataFrame()

        total_exp = expenses['Amount'].sum() if not expenses.empty else 0
        
        # If no income data, use a benchmark (avg salary from generator)
        if income.empty or income['Amount'].sum() == 0:
            total_inc = 5500.0  # Default benchmark salary
        else:
            total_inc = income['Amount'].sum()
        
        if total_inc == 0: return 50
        
        savings_rate = (total_inc - total_exp) / total_inc
        # Score calculation: 50 base, + up to 50 for good savings rate (cap at 100)
        score = 50 + (savings_rate * 50)
        return min(max(int(score), 0), 100)
        return max(0, min(100, round(score)))
        
    def get_prescriptive_action(self, score):
        if score > 80: return ("Excellent", "Divert 15% of excess cash to high-yield investments.", "#10B981")
        elif score > 50: return ("Stable", "Watch discretionary spending to boost savings.", "#F59E0B")
        else: return ("At Risk", "Warning: High debt probability. Implement freeze on Non-Housing/Utility spending.", "#EF4444")

class AuraGreenAgent:
    """Analyzes Ecological Impact of transactions."""
    def analyze_total_footprint(self, historical_df):
        if 'Carbon_Score' not in historical_df.columns: return 0
        return round(historical_df['Carbon_Score'].sum(), 2)
