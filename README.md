# air-prediction
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Step 1: Load the dataset
# Replace 'air_quality_data.csv' with the path to your dataset
data = pd.read_csv('airquality.csv')

# Step 2: Preprocess the data
# Handle missing values
data = data.fillna(method='ffill')

# Split features and target variable
features = data[['temperature', 'humidity', 'traffic_density', 'weather_conditions']]  # Replace with relevant features
target = data['PM2.5']  # Replace with the target variable for air quality, e.g., PM2.5

# Normalize the features (optional)
features = (features - features.mean()) / features.std()

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Step 4: Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"R-squared: {r2}")

# Step 6: Make predictions
# Example prediction for a new data point
new_data_point = np.array([[25, 60, 120, 1]])  # Replace with new feature values
predicted_pm25 = model.predict(new_data_point)
print(f"Predicted PM2.5 level: {predicted_pm25[0]}")
