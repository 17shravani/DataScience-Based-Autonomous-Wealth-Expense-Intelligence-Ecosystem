import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# Added ESG & Carbon Metrics (kg CO2 per transaction avg or multiplier)
CATEGORIES = {
    "Housing": {"min": 1000, "max": 2500, "freq": "monthly", "carbon_multiplier": 0.5},
    "Utilities": {"min": 100, "max": 300, "freq": "monthly", "carbon_multiplier": 2.5},
    "Groceries": {"min": 50, "max": 150, "freq": "weekly", "carbon_multiplier": 1.2},
    "Transportation": {"min": 20, "max": 80, "freq": "daily", "carbon_multiplier": 5.0}, # High carbon
    "Dining": {"min": 30, "max": 100, "freq": "weekly", "carbon_multiplier": 1.5},
    "Entertainment": {"min": 20, "max": 200, "freq": "weekly", "carbon_multiplier": 0.3},
    "Healthcare": {"min": 50, "max": 500, "freq": "rare", "carbon_multiplier": 0.1},
    "Subscriptions": {"min": 9.99, "max": 59.99, "freq": "monthly", "carbon_multiplier": 0.1},
    "Shopping": {"min": 40, "max": 300, "freq": "weekly", "carbon_multiplier": 2.0} # Fast fashion high impact
}

def generate_historical_data(months=12):
    """Generate synthetic past transaction data with ESG & Wellness features."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30 * months)
    
    current_date = start_date
    transactions = []
    
    # Track subscriptions to simulate "creep"
    active_subs = {
        "Netflix": 15.99,
        "Adobe": 54.99,
        "Gym": 45.00
    }
    
    transaction_id = 1
    
    while current_date <= end_date:
        # First day of month - Salary and Subscriptions
        if current_date.day == 1:
            transactions.append({
                "ID": transaction_id,
                "Date": current_date,
                "Category": "Income",
                "Vendor": "Employer INC",
                "Amount": 5500.00,
                "Type": "Income",
                "Carbon_Score": 0.0,
                "Charity_Roundup": 0.0
            })
            transaction_id += 1
            
            # Subscriptions
            for sub, amount in active_subs.items():
                if sub == "Adobe" and current_date.month % 6 == 0:
                    active_subs[sub] += 2.00
                
                amount_spent = -active_subs[sub]
                roundup = np.ceil(abs(amount_spent)) - abs(amount_spent)
                
                transactions.append({
                    "ID": transaction_id,
                    "Date": current_date,
                    "Category": "Subscriptions",
                    "Vendor": sub,
                    "Amount": amount_spent,
                    "Type": "Expense",
                    "Carbon_Score": round(abs(amount_spent) * CATEGORIES["Subscriptions"]["carbon_multiplier"], 2),
                    "Charity_Roundup": round(roundup, 2)
                })
                transaction_id += 1
                
            # Rent/Housing
            transactions.append({
                "ID": transaction_id,
                "Date": current_date,
                "Category": "Housing",
                "Vendor": "Property Management",
                "Amount": -1800.00,
                "Type": "Expense",
                "Carbon_Score": round(1800.00 * CATEGORIES["Housing"]["carbon_multiplier"], 2),
                "Charity_Roundup": 0.0
            })
            transaction_id += 1

        # Daily randomness
        num_transactions_today = random.randint(0, 3)
        for _ in range(num_transactions_today):
            cat = random.choices(
                list(CATEGORIES.keys()), 
                weights=[0.05, 0.05, 0.3, 0.2, 0.2, 0.1, 0.05, 0.0, 0.05], 
                k=1
            )[0]
            
            if cat not in ["Housing", "Subscriptions", "Income"]:
                amount = round(random.uniform(CATEGORIES[cat]["min"], CATEGORIES[cat]["max"]), 2)
                
                if random.random() < 0.02 and cat == "Shopping":
                    amount *= random.uniform(3.0, 5.0)
                    
                amount = round(amount, 2)
                roundup = round(np.ceil(amount) - amount, 2)
                carbon = round(amount * CATEGORIES[cat]["carbon_multiplier"], 2)
                    
                transactions.append({
                    "ID": transaction_id,
                    "Date": current_date + timedelta(hours=random.randint(8, 22), minutes=random.randint(0, 59)),
                    "Category": cat,
                    "Vendor": f"Vendor_{cat}_{random.randint(1, 5)}",
                    "Amount": -amount,
                    "Type": "Expense",
                    "Carbon_Score": carbon,
                    "Charity_Roundup": roundup
                })
                transaction_id += 1
                
        current_date += timedelta(days=1)
        
    df = pd.DataFrame(transactions)
    df = df.sort_values(by="Date").reset_index(drop=True)
    return df

def generate_live_transaction():
    """Simulate a single live transaction stream with hyper-metrics."""
    cat = random.choices(
        list(CATEGORIES.keys()), 
        weights=[0.0, 0.0, 0.3, 0.2, 0.3, 0.1, 0.0, 0.0, 0.1], 
        k=1
    )[0]
    
    amount = round(random.uniform(CATEGORIES[cat]["min"], CATEGORIES[cat]["max"]), 2)
    
    is_anomaly = False
    if random.random() < 0.05:
        amount = round(amount * random.uniform(4.0, 8.0), 2)
        is_anomaly = True
        
    roundup = round(np.ceil(amount) - amount, 2)
    carbon = round(amount * CATEGORIES[cat]["carbon_multiplier"], 2)
        
    return {
        "ID": random.randint(100000, 999999), 
        "Date": pd.to_datetime(datetime.now()),
        "Category": cat,
        "Vendor": f"Live_Vendor_{cat}_{random.randint(1, 10)}",
        "Amount": -amount,
        "Type": "Expense",
        "Carbon_Score": carbon,
        "Charity_Roundup": roundup,
        "Is_Synthetic_Anomaly": is_anomaly
    }

if __name__ == "__main__":
    df = generate_historical_data(12)
    df.to_csv("historical_transactions.csv", index=False)
    print(f"Generated {len(df)} historical transactions with ESG factors.")
