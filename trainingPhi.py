import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from joblib import dump

# File path
file_path = r'E:/ML Models/Phi_allData.csv'

# Chunk size for processing data in parts
chunk_size = 1_000_000

# Data type specification for memory optimization
data_types = {
    'R': 'int32',  # R is an integer
    'C': 'int32',  # C is an integer
    'HL': 'float32',  # HL can stay as float if it's continuous
    'h_B': 'float32',
    'h': 'float32',
    'd': 'float32',
    'a_B': 'float32',
    'a': 'float32',
    'Frequency (GHz)': 'float32',  # Frequency as a float
    'Theta': 'int32',  # Angular parameters as integers
    'Phi': 'int32',
    'Abs(Dir.)': 'float32',  # Target variable
}

# Initialize model and scaler
model = RandomForestRegressor(n_estimators=100, random_state=42)
scaler = StandardScaler()

# Variables for tracking incremental training
data_processed = 0
first_chunk = True

# Process the dataset in chunks
for chunk in pd.read_csv(file_path, chunksize=chunk_size, dtype=data_types, low_memory=False):
    # Handle missing or invalid values
    chunk = chunk.replace([np.inf, -np.inf], np.nan).dropna()

    # Define input features and target
    X_chunk = chunk[['R', 'C', 'HL', 'h_B', 'h', 'd', 'a_B', 'a', 'Frequency (GHz)', 'Theta', 'Phi']]
    y_chunk = chunk['Abs(Dir.)']

    # Scale the input features
    if first_chunk:
        X_chunk_scaled = scaler.fit_transform(X_chunk)
        first_chunk = False
    else:
        X_chunk_scaled = scaler.transform(X_chunk)

    # Train the model incrementally
    model.fit(X_chunk_scaled, y_chunk)

    # Update the count of processed rows
    data_processed += len(chunk)
    print(f"Processed {data_processed} rows...")

# Evaluate the model on a separate test set
# Load a smaller test dataset or use a validation set
validation_data = pd.read_csv(file_path, nrows=100_000, dtype=data_types, low_memory=False)  # Adjust path or size as needed

# Handle missing or invalid values in the validation data
validation_data = validation_data.replace([np.inf, -np.inf], np.nan).dropna()
X_val = validation_data[['R', 'C', 'HL', 'h_B', 'h', 'd', 'a_B', 'a', 'Frequency (GHz)', 'Theta', 'Phi']]
y_val = validation_data['Abs(Dir.)']
X_val_scaled = scaler.transform(X_val)

# Make predictions and evaluate
y_pred = model.predict(X_val_scaled)
mse = mean_squared_error(y_val, y_pred)
r2 = r2_score(y_val, y_pred)

print("\nModel Evaluation:")
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

# Save the trained model and scaler for future use
dump(model, 'Phi_random_forest_model.joblib')
dump(scaler, 'scaler.pkl')
print("\nModel and scaler saved.")
