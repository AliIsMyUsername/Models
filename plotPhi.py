import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plotPhi90():
    # Load the CSV file
    file_path = r'E:\Data_V3\Data\phi90\04Rx16C/1.csv'
    data = pd.read_csv(file_path)

    # Filter the data for f = 30 and Phi = 90
    filtered_data_90 = data[(data['Frequency (GHz)'] == '(f=30)') & (data['Phi'] == 90)]

    # Convert Theta to radians for plotting
    theta_phi_90 = filtered_data_90['Theta'] * (np.pi / 180)
    directivity_phi_90 = filtered_data_90['Abs(Dir.)']

    # Create the polar plot for Phi = 90
    plt.figure(figsize=(10, 10))

    ax = plt.subplot(111, polar=True)

    # Plot directivity for Phi = 90

    # ax.plot(theta_phi_90, directivity_phi_90, label='Phi = 90°', linewidth=2, color='red')

    ax.set_theta_zero_location("N")  # Set 0 degrees to the top

    ax.set_theta_direction(1)  # Clockwise direction

    # plt.title('Polar Plot of Directivity vs Theta (f = 30, Phi = 90°)', va='bottom')

    # plt.legend()

    # plt.show()
    # Mirror the Theta values for Phi = 270 around the center line
    filtered_data_270 = data[(data['Frequency (GHz)'] == '(f=30)') & (data['Phi'] == 270)]

    # Convert Theta to radians for plotting Phi = 270

    theta_phi_270 = filtered_data_270['Theta'] * (np.pi / 180) + theta_phi_90.max()

    directivity_phi_270 = filtered_data_270['Abs(Dir.)']

    theta_phi_270_mirrored = np.pi - theta_phi_270

    # Create the polar plot for Phi = 90 and mirrored Phi = 270

    # plt.figure(figsize=(10, 10))

    ax = plt.subplot(111, polar=True)

    # Plot directivity for Phi = 90
    ax.plot(theta_phi_90, directivity_phi_90, label='Phi = 90°', linewidth=2, color='red')

    # Plot directivity for mirrored Phi = 270
    ax.plot(theta_phi_270_mirrored, directivity_phi_270, label='Phi = 270° ', linewidth=2, color='red')
    ax.set_theta_zero_location("N")  # Set 0 degrees to the top
    ax.set_theta_direction(1)  # Clockwise direction
    plt.title('Polar Plot of Directivity vs Theta (f = 30, Phi = 90° )', va='bottom')
    plt.show()

def plotPhi180():
    # Load the CSV file
    file_path = r'E:\Data_V3\Data\phi180\04Rx16C/1.csv'
    data = pd.read_csv(file_path)

    # Filter the data for f = 30 and Phi = 90
    filtered_data_180 = data[(data['Frequency (GHz)'] == '(f=30)') & (data['Phi'] == 180)]

    # Convert Theta to radians for plotting
    theta_phi_180 = filtered_data_180['Theta'] * (np.pi / 180)
    directivity_phi_180 = filtered_data_180['Abs(Dir.)']

    # Create the polar plot for Phi = 90
    plt.figure(figsize=(10, 10))

    ax = plt.subplot(111, polar=True)

    # Plot directivity for Phi = 90

    # ax.plot(theta_phi_90, directivity_phi_90, label='Phi = 90°', linewidth=2, color='red')

    ax.set_theta_zero_location("N")  # Set 0 degrees to the top

    ax.set_theta_direction(1)  # Clockwise direction

    # plt.title('Polar Plot of Directivity vs Theta (f = 30, Phi = 90°)', va='bottom')

    # plt.legend()

    # plt.show()
    # Mirror the Theta values for Phi = 270 around the center line
    filtered_data_0 = data[(data['Frequency (GHz)'] == '(f=30)') & (data['Phi'] == 0)]

    # Convert Theta to radians for plotting Phi = 270

    theta_phi_0 = filtered_data_0['Theta'] * (np.pi / 180) + theta_phi_180.max()

    directivity_phi_0 = filtered_data_0['Abs(Dir.)']

    theta_phi_0_mirrored = np.pi - theta_phi_0

    # Create the polar plot for Phi = 90 and mirrored Phi = 270

    # plt.figure(figsize=(10, 10))

    ax = plt.subplot(111, polar=True)

    # Plot directivity for Phi = 90
    ax.plot(theta_phi_180, directivity_phi_180, label='Phi = 90°', linewidth=2, color='red')

    # Plot directivity for mirrored Phi = 270
    ax.plot(theta_phi_0_mirrored, directivity_phi_0, label='Phi = 270° ', linewidth=2, color='red')
    ax.set_theta_zero_location("N")  # Set 0 degrees to the top
    ax.set_theta_direction(1)  # Clockwise direction
    plt.title('Polar Plot of Directivity vs Theta (f = 30, Phi = 180° )', va='bottom')
    plt.show()


plotPhi90()
plotPhi180()