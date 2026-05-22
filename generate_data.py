import pandas as pd
import numpy as np

np.random.seed(42)
N = 5000

categories = ['Food', 'Transport', 'Study', 'Entertainment', 'Other']
vendors    = ['Canteen', 'Pharmacy', 'Stationery', 'Snack_Bar', 'Bookshop']

# ── Generate raw features ─────────────────────────────────────────────────
data = {
    'transaction_amount':        np.random.randint(200, 5000, N),
    'category':                  np.random.choice(categories, N),
    'vendor_category':           np.random.choice(vendors, N),
    'daily_tx_count':            np.random.randint(1, 12, N),
    'wallet_balance_pct':        np.random.uniform(5, 100, N).round(1),
    'monthly_budget_used_pct':   np.random.uniform(10, 130, N).round(1),
    'category_budget_used_pct':  np.random.uniform(10, 150, N).round(1),
    'days_elapsed_in_month':     np.random.randint(1, 31, N),
    'spend_velocity':            np.random.uniform(500, 4000, N).round(0),
    'days_remaining_budget':     np.random.uniform(0, 30, N).round(1),
    'is_weekend':                np.random.randint(0, 2, N),
    'hour_of_day':               np.random.randint(6, 23, N),
}
df = pd.DataFrame(data)

# ── Label 1: expenditure_level ────────────────────────────────────────────
df['expenditure_level'] = pd.cut(
    df['monthly_budget_used_pct'],
    bins=[0, 50, 85, 200],
    labels=['Low', 'Medium', 'High']
)

# ── Label 2: duration_risk ────────────────────────────────────────────────
def duration_risk(row):
    if row['days_remaining_budget'] > 10: return 'Safe'
    if row['days_remaining_budget'] > 4:  return 'Warning'
    return 'Critical'

df['duration_risk'] = df.apply(duration_risk, axis=1)

# ── Label 3: behaviour_class ──────────────────────────────────────────────
def behaviour(row):
    if row['daily_tx_count'] >= 5 and row['transaction_amount'] < 800:
        return 'Leakage'
    if (row['monthly_budget_used_pct'] > 85 or
            row['category_budget_used_pct'] > 115):
        return 'Lifestyle'
    return 'Survival'

df['behaviour_class'] = df.apply(behaviour, axis=1)

# ── Save ──────────────────────────────────────────────────────────────────
df.to_csv('cafepay_training_data.csv', index=False)

print("Dataset created: cafepay_training_data.csv")
print(f"Total rows: {len(df)}")
print("\nbehaviour_class distribution:")
print(df['behaviour_class'].value_counts())
print("\nduration_risk distribution:")
print(df['duration_risk'].value_counts())
print("\nexpenditure_level distribution:")
print(df['expenditure_level'].value_counts())