import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

df = pd.read_csv("ai4i2020.csv")

# Encode Type

encoder = LabelEncoder()
df["Type"] = encoder.fit_transform(df["Type"])
joblib.dump(encoder, "type_encoder.pkl")

# 1. Separate features and target

X = df.drop("Machine failure", axis=1)
y = df["Machine failure"]

# 2. Train-test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# 3. Create XGBoost model

model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.1,
    random_state=42
)

# 4. Train model

model.fit(X_train, y_train)

# 5. Evaluate model

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


joblib.dump(model, "predictive_maintenance_model.pkl")