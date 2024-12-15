import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from joblib import dump

# Load the dataset
file_path = r'E:/ML Models/S11_allData_V2.csv'
data = pd.read_csv(file_path)

# Define input features and target
X = data[['R', 'C', 'HL', 'h_B', 'h', 'd', 'a_B', 'a', 'Frequency (GHz)']]
y = data['S Average']

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

# Generate predictions for a range of frequencies with fixed input parameters
fixed_parameters = [4, 16, 1.8, 1.5, 0.6, 1.7, 0.4, 0.3]  # Replace with desired values
frequencies = np.arange(20.0, 30.01, 0.01)  # Generate frequencies from 20 to 30 GHz

# Create a DataFrame for predictions
input_data = pd.DataFrame([
    fixed_parameters + [freq] for freq in frequencies
], columns=['R', 'C', 'HL', 'h_B', 'h', 'd', 'a_B', 'a', 'Frequency (GHz)'])

# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Predict S Average for the input data
predicted_s_average = model.predict(input_data_scaled)

# Combine frequencies with predictions into a DataFrame
output_df = input_data.copy()
output_df['S Average'] = predicted_s_average

print("\nGenerated Predictions:")
print(output_df.head())

# Save to a CSV file if needed
output_df.to_csv('predicted_s_average.csv', index=False)
print("\nPredictions saved to 'predicted_s_average.csv'")

dump(model, 'S11_random_forest_model.joblib')
print("Model saved to 'S11_random_forest_model.joblib'")

