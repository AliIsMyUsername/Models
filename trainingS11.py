import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# File loader function
def fileLoader():
    file_path = r'E:/ML Models/S11_allData_V2.csv'
    df = pd.read_csv(file_path)
    return df

# Data cleaning function
def cleanData(df):
    cleaned_data = df.dropna(subset=['Frequency (GHz)', 'S Average'])  # Drop rows with missing target
    return cleaned_data

# Load and clean the data
dfCleaned = cleanData(fileLoader())

# Define features and target
X = dfCleaned[['R', 'C', ' HL', ' h_B', ' h', ' d', ' a_B', ' a', 'Frequency (GHz)']]  # Features
y = dfCleaned['S Average']  # Target

# Print dimensions of X and y
print(f"Features Shape: {X.shape}")
print(f"Target Shape: {y.shape}")

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Get model parameters
print("Model Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")
print(f"Intercept: {model.intercept_}")

# Predict on test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R² Score: {r2}")

# Visualization: Compare actual vs predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', lw=2)  # y=x line
plt.xlabel("Actual Values")
plt.ylabel("Predicted Values")
plt.title("Actual vs Predicted Values")
plt.show()

# Visualization: Single feature (Frequency (GHz)) vs predictions
plt.figure(figsize=(10, 6))
plt.scatter(X_test['Frequency (GHz)'], y_test, color='blue', label='Actual', alpha=0.6)
plt.scatter(X_test['Frequency (GHz)'], y_pred, color='red', label='Predicted', alpha=0.6)
plt.xlabel('Frequency (GHz)')
plt.ylabel('S Average')
plt.title('Frequency vs S Average')
plt.legend()
plt.show()
