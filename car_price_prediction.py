import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================

# CAR PRICE PREDICTION USING MACHINE LEARNING

# ============================================================

# 1. Load Dataset

print("=" * 60)
print("CAR PRICE PREDICTION")
print("=" * 60)

df = pd.read_csv("car data.csv")

print("\nFirst 5 rows of the dataset:")
print(df.head())

# ============================================================

# 2. Explore Dataset

# ============================================================

print("\nDataset Shape:")
print(df.shape)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# ============================================================

# 3. Feature Engineering

# ============================================================

# Create car age from manufacturing year

current_year = 2026
df["Car_Age"] = current_year - df["Year"]

# Remove Year because Car_Age contains the required information

df = df.drop("Year", axis=1)

# Car_Name is removed because it contains many unique values

df = df.drop("Car_Name", axis=1)

# ============================================================

# 4. Separate Features and Target

# ============================================================

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

# ============================================================

# 5. Define Categorical Features

# ============================================================

categorical_features = [
"Fuel_Type",
"Selling_type",
"Transmission"
]

# ============================================================

# 6. Preprocessing

# ============================================================

preprocessor = ColumnTransformer(
transformers=[
(
"categorical",
OneHotEncoder(handle_unknown="ignore"),
categorical_features
)
],
remainder="passthrough"
)

# ============================================================

# 7. Create Machine Learning Model

# ============================================================

model = RandomForestRegressor(
n_estimators=200,
random_state=42
)

# ============================================================

# 8. Create Pipeline

# ============================================================

pipeline = Pipeline(
steps=[
("preprocessor", preprocessor),
("model", model)
]
)

# ============================================================

# 9. Split Dataset into Training and Testing Data

# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=0.20,
random_state=42
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# ============================================================

# 10. Train the Model

# ============================================================

print("\nTraining the Random Forest Regression model...")

pipeline.fit(X_train, y_train)

print("Model training completed!")

# ============================================================

# 11. Make Predictions

# ============================================================

y_pred = pipeline.predict(X_test)

# ============================================================

# 12. Evaluate the Model

# ============================================================

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")

# ============================================================

# 13. Actual vs Predicted Prices

# ============================================================

comparison = pd.DataFrame({
"Actual Price": y_test.values,
"Predicted Price": y_pred
})

print("\n" + "=" * 60)
print("ACTUAL VS PREDICTED PRICES")
print("=" * 60)

print(comparison.head(10))

# ============================================================

# 14. Visualization

# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
y_test,
y_pred,
alpha=0.7
)

# Perfect prediction reference line

plt.plot(
[y_test.min(), y_test.max()],
[y_test.min(), y_test.max()],
linestyle="--"
)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")

plt.tight_layout()
plt.show()

# ============================================================

# 15. Predict Price of a New Car

# ============================================================

new_car = pd.DataFrame({
"Present_Price": [8.5],
"Driven_kms": [25000],
"Fuel_Type": ["Petrol"],
"Selling_type": ["Dealer"],
"Transmission": ["Manual"],
"Owner": [0],
"Car_Age": [4]
})

predicted_price = pipeline.predict(new_car)

print("\n" + "=" * 60)
print("NEW CAR PRICE PREDICTION")
print("=" * 60)

print(
f"Predicted Selling Price: {predicted_price[0]:.2f}"
)

print("\nProgram completed successfully!")
