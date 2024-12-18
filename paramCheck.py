import pandas as pd

# Load the dataset
# file_path = r'E:/ML Models/Phi_allData.csv'  # Replace with your file path
# df = pd.read_csv(file_path)

file_path = r'E:/ML Models/S11_allData_V2.csv'  # Replace with your file path
df1 = pd.read_csv(file_path)

def checking(df):
    # Define the formation you want to check
    target_values = {'R': 32, 'C': 34, 'HL': 1.9, 'h_B': 1.6, 'h': 0.7, 'd': 1.9, 'a_B': 0.6, 'a': 0.8}

    # Check if rows match the target values
    matches = df[(df['R'] == target_values['R']) &
                 (df['C'] == target_values['C']) &
                 (df['HL'] == target_values['HL']) &
                 (df['h_B'] == target_values['h_B']) &
                 (df['h'] == target_values['h']) &
                 (df['d'] == target_values['d']) &
                 (df['a_B'] == target_values['a_B']) &
                 (df['a'] == target_values['a'])]

    # Output the results
    if not matches.empty:
        print("Matching formation found:")
        # print(matches)
    else:
        print("No matching formation found.")

# checking(df)
checking(df1)
