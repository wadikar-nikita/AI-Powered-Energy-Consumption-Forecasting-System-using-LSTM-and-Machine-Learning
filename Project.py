# ==============================================
# AI-Powered Energy Consumption Forecasting System
# ==============================================

# STEP 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# ==============================================
# STEP 2: Generate Synthetic Dataset
# ==============================================

np.random.seed(42)

days = 365

dates = pd.date_range(start='2023-01-01', periods=days)

# Generate synthetic features
temperature = np.random.normal(30, 5, days)
humidity = np.random.normal(60, 10, days)
wind_speed = np.random.normal(10, 3, days)

# Energy consumption (target variable)
energy = (
    50 + (temperature * 1.5) + (humidity * 0.5) +
    np.random.normal(0, 5, days)
)

# Create DataFrame
df = pd.DataFrame({
    'date': dates,
    'temperature': temperature,
    'humidity': humidity,
    'wind_speed': wind_speed,
    'energy_consumption': energy
})

print("\nSample Data:\n", df.head())

# ==============================================
# STEP 3: Feature Engineering
# ==============================================

df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month

df = df.drop('date', axis=1)

# ==============================================
# STEP 4: Exploratory Data Analysis (EDA)
# ==============================================

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

# Plot Energy Consumption
plt.figure()
plt.plot(df['energy_consumption'])
plt.title("Energy Consumption Over Time")
plt.xlabel("Days")
plt.ylabel("Energy")
plt.show()

# ==============================================
# STEP 5: Prepare Data for Model
# ==============================================

X = df.drop('energy_consumption', axis=1)
y = df['energy_consumption']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ==============================================
# STEP 6: Train Random Forest Model
# ==============================================

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ==============================================
# STEP 7: Predictions
# ==============================================

y_pred = model.predict(X_test)

# ==============================================
# STEP 8: Model Evaluation
# ==============================================

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\nModel Performance:")
print("RMSE:", rmse)
print("R2 Score:", r2)

# ==============================================
# STEP 9: Visualization (Actual vs Predicted)
# ==============================================

plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Energy")
plt.ylabel("Predicted Energy")
plt.title("Actual vs Predicted Energy Consumption")
plt.show()

# Line Plot Comparison
plt.figure()
plt.plot(y_test.values, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title("Actual vs Predicted Comparison")
plt.show()

# ==============================================
# STEP 10: Conclusion
# ==============================================

print("\nProject Completed Successfully!")
