import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = r'E:\Data_V3\Data\S11\04Rx16C/1.csv'
data = pd.read_csv(file_path)


# Extract frequency and S average data
frequency = data['Frequency (GHz)']
s_average = data['S Average ']

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(frequency, s_average, label='S11 vs Frequency', linewidth=2)
plt.xlabel('Frequency (GHz)')
plt.ylabel('S11')
plt.title('Frequency vs S11')
plt.grid(True)
plt.legend()
plt.show()


