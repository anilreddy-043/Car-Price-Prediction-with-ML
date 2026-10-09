# 🚗 Car Price Prediction Using Machine Learning

## 📌 Project Overview

This project uses **Machine Learning** to predict the selling price of used cars based on different car-related features.

The project demonstrates how regression algorithms can be applied to a real-world problem such as estimating the market value of a vehicle.

The dataset contains information about cars including their present price, kilometers driven, fuel type, transmission, number of previous owners, and selling price.

---

## 🎯 Objectives

The main objectives of this project are:

* Collect and analyze car-related data.
* Perform data preprocessing and cleaning.
* Perform feature engineering.
* Convert categorical data into numerical form.
* Train a regression machine learning model.
* Predict the selling price of cars.
* Evaluate model performance using regression metrics.
* Visualize actual and predicted prices.
* Understand a real-world application of Machine Learning.

---

## 📂 Dataset

The project uses the `car data.csv` dataset.

### Dataset Features

| Feature         | Description                              |
| --------------- | ---------------------------------------- |
| `Car_Name`      | Name of the car                          |
| `Year`          | Manufacturing year                       |
| `Selling_Price` | Selling price of the car                 |
| `Present_Price` | Current/ex-showroom price                |
| `Driven_kms`    | Kilometers driven                        |
| `Fuel_Type`     | Fuel type such as Petrol, Diesel, or CNG |
| `Selling_type`  | Dealer or Individual                     |
| `Transmission`  | Manual or Automatic                      |
| `Owner`         | Number of previous owners                |

### Target Variable

The target variable is:

```text
Selling_Price
```

The model learns from the other car features and predicts this value.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **Matplotlib**

---

## 📦 Installation

Make sure Python is installed on your computer.

Install the required libraries using:

```bash
pip install pandas numpy scikit-learn matplotlib
```

---

## 📁 Project Structure

```text
Car-Price-Prediction/
│
├── car data.csv
├── car_price_prediction.py
├── README.md
└── requirements.txt
```

---

## 🔄 Machine Learning Workflow

The project follows these steps:

```text
Dataset
   ↓
Data Loading
   ↓
Data Exploration
   ↓
Data Preprocessing
   ↓
Feature Engineering
   ↓
Feature Selection
   ↓
Train-Test Split
   ↓
Categorical Encoding
   ↓
Model Training
   ↓
Price Prediction
   ↓
Model Evaluation
   ↓
Data Visualization
```

---

## 1. Import Libraries

The following Python libraries are used:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

---

## 2. Load the Dataset

The dataset is loaded using Pandas:

```python
df = pd.read_csv("car data.csv")
```

Display the first few records:

```python
print(df.head())
```

---

## 3. Explore the Dataset

Check the dimensions:

```python
print(df.shape)
```

Check column information:

```python
print(df.info())
```

Check statistical information:

```python
print(df.describe())
```

Check missing values:

```python
print(df.isnull().sum())
```

---

## 4. Feature Engineering

A new feature called `Car_Age` is created from the manufacturing year.

```python
df["Car_Age"] = 2026 - df["Year"]
```

The original `Year` column can then be removed:

```python
df = df.drop("Year", axis=1)
```

This helps the model understand how the age of the car affects its selling price.

---

## 5. Remove Unnecessary Features

`Car_Name` is removed in the basic model because there can be many unique car names.

```python
df = df.drop("Car_Name", axis=1)
```

---

## 6. Separate Features and Target

The input features are separated from the target:

```python
X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]
```

Where:

* `X` = input features
* `y` = target variable

---

## 7. Handle Categorical Features

Categorical features include:

```text
Fuel_Type
Selling_type
Transmission
```

These values are converted into numerical representations using **One-Hot Encoding**.

```python
categorical_features = [
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]
```

---

## 8. Train-Test Split

The dataset is divided into training and testing data.

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
```

### Split Ratio

* **80%** → Training data
* **20%** → Testing data

The training data is used to teach the model, while the testing data is used to evaluate its performance.

---

## 9. Machine Learning Model

This project uses the **Random Forest Regression** algorithm.

```python
model = RandomForestRegressor(
    n_estimators=200,
    random_state=42
)
```

Random Forest Regression combines multiple decision trees to produce a prediction.

---

## 10. Create the Machine Learning Pipeline

The preprocessing and model are combined into a single pipeline:

```python
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)
```

This ensures that the same preprocessing steps are applied consistently during training and prediction.

---

## 11. Train the Model

The model is trained using:

```python
pipeline.fit(X_train, y_train)
```

The model learns the relationship between the car features and selling price.

---

## 12. Make Predictions

Predictions are generated using the test data:

```python
y_pred = pipeline.predict(X_test)
```

The predicted values can then be compared with the actual selling prices.

---

## 13. Model Evaluation

The following evaluation metrics are used.

### Mean Absolute Error (MAE)

```python
mae = mean_absolute_error(y_test, y_pred)
```

MAE represents the average absolute difference between the actual and predicted prices.

---

### Mean Squared Error (MSE)

```python
mse = mean_squared_error(y_test, y_pred)
```

MSE gives greater importance to larger prediction errors.

---

### Root Mean Squared Error (RMSE)

```python
rmse = np.sqrt(mse)
```

RMSE represents the typical magnitude of prediction errors.

---

### R² Score

```python
r2 = r2_score(y_test, y_pred)
```

The R² score indicates how well the model explains the variation in the target variable.

A score closer to **1.0** generally indicates better predictive performance.

---

## 14. Visualization

Actual and predicted prices can be visualized using Matplotlib:

```python
plt.figure(figsize=(8, 6))

plt.scatter(y_test, y_pred)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle="--"
)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")

plt.show()
```

### Interpretation

If the predicted values are close to the diagonal line, the model is making relatively accurate predictions.

---

## 📊 Expected Output

The program displays:

```text
Model Performance
-------------------------
MAE : ...
MSE : ...
RMSE: ...
R2 Score: ...
```

It also displays a comparison between:

```text
Actual Price
Predicted Price
```

and generates an **Actual vs Predicted Car Prices** graph.

---

## 🔮 Predicting a New Car Price

A new car can be passed to the trained model:

```python
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

print("Predicted Car Price:", predicted_price[0])
```

The model will return an estimated selling price for the car.

---

## 💡 Real-World Applications

Car price prediction can be useful for:

* Used-car dealerships
* Online car marketplaces
* Car buyers and sellers
* Vehicle valuation systems
* Automotive companies
* Financial and insurance applications

A trained model can help estimate a reasonable market price based on the characteristics and condition of a vehicle.

---

## ⚠️ Limitations

The prediction depends on the quality and size of the dataset.

The current model does not consider some factors that can influence a real-world car price, such as:

* Vehicle condition
* Accident history
* Service history
* Location
* Number of previous accidents
* Tire condition
* Market demand
* Seasonal price changes

Therefore, the model should be considered an **estimated price prediction system**, not an exact valuation tool.

---

## 🚀 Future Improvements

The project can be improved by:

1. Collecting a larger dataset.
2. Including more car features.
3. Comparing multiple regression algorithms.
4. Performing hyperparameter tuning.
5. Using cross-validation.
6. Adding feature importance analysis.
7. Building a web interface using Flask or Streamlit.
8. Deploying the model as a web application.

Possible algorithms to compare include:

* Linear Regression
* Decision Tree Regression
* Random Forest Regression
* Gradient Boosting Regression
* XGBoost Regression

---

## 🏆 Conclusion

This project demonstrates how Machine Learning can be used to predict used-car selling prices.

The complete workflow includes:

**Data Collection → Data Preprocessing → Feature Engineering → Encoding → Model Training → Prediction → Evaluation → Visualization**

The project provides practical experience with **Pandas, NumPy, Scikit-learn, Matplotlib, regression algorithms, and real-world machine learning applications.**

---

## 👩‍💻 Author

**Car Price Prediction Machine Learning Project**

Developed using Python and Scikit-learn.
