import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

print("Loading dataset...")
df = pd.read_csv('cafepay_training_data.csv')

# ── Encode text columns into numbers ──────────────────────────────────────
le_cat  = LabelEncoder()
le_vend = LabelEncoder()

df['category_enc']        = le_cat.fit_transform(df['category'])
df['vendor_category_enc'] = le_vend.fit_transform(df['vendor_category'])

# ── Define features and targets ───────────────────────────────────────────
FEATURES = [
    'transaction_amount',
    'category_enc',
    'vendor_category_enc',
    'daily_tx_count',
    'wallet_balance_pct',
    'monthly_budget_used_pct',
    'category_budget_used_pct',
    'days_elapsed_in_month',
    'spend_velocity',
    'days_remaining_budget',
    'is_weekend',
    'hour_of_day'
]

TARGETS = ['expenditure_level', 'duration_risk', 'behaviour_class']

X = df[FEATURES]
y = df[TARGETS]

# ── Split into training and test sets ─────────────────────────────────────
# 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y['behaviour_class']
)

print(f"Training rows: {len(X_train)}")
print(f"Testing rows:  {len(X_test)}")

# ── Build and train the model ─────────────────────────────────────────────
print("\nTraining Decision Tree model...")

base_tree = DecisionTreeClassifier(
    max_depth=8,            # how deep the tree can grow
    min_samples_split=20,   # minimum samples needed to split a node
    min_samples_leaf=10,    # minimum samples in each leaf
    class_weight='balanced',# handles class imbalance automatically
    random_state=42
)

# MultiOutputClassifier allows one model → three outputs simultaneously
model = MultiOutputClassifier(base_tree, n_jobs=-1)
model.fit(X_train, y_train)

print("Training complete.")

# ── Evaluate the model ────────────────────────────────────────────────────
print("\n" + "="*55)
print("MODEL EVALUATION RESULTS")
print("="*55)

y_pred = model.predict(X_test)

for i, target in enumerate(TARGETS):
    actual    = y_test.iloc[:, i]
    predicted = [p[i] for p in y_pred]
    acc       = accuracy_score(actual, predicted)

    print(f"\n--- {target.upper()} --- Accuracy: {acc*100:.1f}%")
    print(classification_report(actual, predicted))

# ── Save the model and encoders ───────────────────────────────────────────
print("\nSaving model files...")

joblib.dump(model,    'cafepay_model.pkl')
joblib.dump({
    'category':        le_cat,
    'vendor_category': le_vend
}, 'encoder.pkl')
joblib.dump(FEATURES, 'features.pkl')

print("Saved:")
print("  cafepay_model.pkl  — the trained ML model")
print("  encoder.pkl        — the text encoders")
print("  features.pkl       — the feature list")
print("\nDone. Model is ready to use.")