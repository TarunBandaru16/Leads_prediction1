import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.pipeline import Pipeline

# Load data
df = pd.read_csv("data/current.csv")
X = df.drop(columns=["adjusted_total_usd"])
y = df["adjusted_total_usd"]

# Example model
model = XGBRegressor()
model.fit(X, y)

# Save retrained model
joblib.dump(model, "saved_models/retrained_model.pkl")
print("Retraining complete.")
