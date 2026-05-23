import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

np.random.seed(42)
N = 5000
categories = ['Food', 'Transport', 'Study', 'Entertainment', 'Other']
vendors    = ['Canteen', 'Pharmacy', 'Stationery', 'Snack_Bar', 'Bookshop']

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

df['expenditure_level'] = pd.cut(
    df['monthly_budget_used_pct'],
    bins=[0, 50, 85, 200],
    labels=['Low', 'Medium', 'High']
)

def duration_risk(row):
    if row['days_remaining_budget'] > 10: return 'Safe'
    if row['days_remaining_budget'] > 4:  return 'Warning'
    return 'Critical'
df['duration_risk'] = df.apply(duration_risk, axis=1)

def behaviour(row):
    if row['daily_tx_count'] >= 5 and row['transaction_amount'] < 800:
        return 'Leakage'
    if (row['monthly_budget_used_pct'] > 85 or
            row['category_budget_used_pct'] > 115):
        return 'Lifestyle'
    return 'Survival'
df['behaviour_class'] = df.apply(behaviour, axis=1)

le_cat  = LabelEncoder()
le_vend = LabelEncoder()
df['category_enc']        = le_cat.fit_transform(df['category'])
df['vendor_category_enc'] = le_vend.fit_transform(df['vendor_category'])

FEATURES = [
    'transaction_amount', 'category_enc', 'vendor_category_enc',
    'daily_tx_count', 'wallet_balance_pct', 'monthly_budget_used_pct',
    'category_budget_used_pct', 'days_elapsed_in_month',
    'spend_velocity', 'days_remaining_budget', 'is_weekend', 'hour_of_day'
]
TARGETS = ['expenditure_level', 'duration_risk', 'behaviour_class']

X = df[FEATURES]
y = df[TARGETS]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = MultiOutputClassifier(
    DecisionTreeClassifier(max_depth=8, random_state=42)
)
model.fit(X_train, y_train)

joblib.dump(model, 'cafepay_model.pkl')
joblib.dump({'category': le_cat, 'vendor_category': le_vend}, 'encoder.pkl')

print("Model trained and saved.")