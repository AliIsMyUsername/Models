import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import re
from joblib import dump

# Load the dataset
file_path = r'E:/ML Models/PhiSampeld_allData_V2.csv'
data = pd.read_csv(file_path)



# Handle missing or invalid values
data = data.replace([np.inf, -np.inf], np.nan).dropna()

# Define input features and target
X = data[['R', 'C', 'HL', 'h_B', 'h', 'd', 'a_B', 'a', 'Frequency (GHz)', 'Theta', 'Phi']]
y = data['Abs(Dir.)']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize the input features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Regressor model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Generate predictions for a range of frequencies and angular values with fixed parameters
fixed_parameters = [4, 16, 1.8, 1.5, 0.6, 1.7, 0.4, 0.3]  # Replace with desired values
frequencies = np.arange(20.0, 30.01, 0.01)  # Generate frequencies from 20 to 30 GHz
theta_values = [0, 30, 60, 90]  # Example theta values
phi_values = [0, 90, 180, 270]  # Example phi values

# Create a DataFrame for predictions
input_data = pd.DataFrame([
    fixed_parameters + [freq, theta, phi]
    for freq in frequencies
    for theta in theta_values
    for phi in phi_values
], columns=['R', 'C', 'HL', 'h_B', 'h', 'd', 'a_B', 'a', 'Frequency (GHz)', 'Theta', 'Phi'])

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict Abs(Dir.) for the input data
predicted_abs_dir = model.predict(input_data_scaled)

# Combine inputs with predictions into a DataFrame
output_df = input_data.copy()
output_df['Abs(Dir.)'] = predicted_abs_dir

print("\nGenerated Predictions:")
print(output_df.head())

# Save to a CSV file if needed
output_df.to_csv('predicted_abs_dir.csv', index=False)
print("\nPredictions saved to 'predicted_abs_dir.csv'")

dump(model, 'Phi_random_forest_model.joblib')
print("Model saved to 'Phi_random_forest_model.joblib'")
