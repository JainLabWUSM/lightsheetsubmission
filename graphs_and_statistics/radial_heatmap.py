import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Prompt user for Excel file
excel_file_path = input("Enter the path to the Excel file: ")

# Load data from Excel file with headers
try:
    data = pd.read_excel(excel_file_path, header=None, names=['Azimuth', 'Polar'])
    print(data)
except Exception as e:
    print("Error reading Excel file:", e)
    exit()

print(data.head())

# Plot 1
# Plot 1
maxi = max(data['Azimuth']) - min(data['Azimuth'])
dists = []
for item in data['Azimuth']:
    temp_dist = 0
    dist_list = []
    for obj in data['Azimuth']:
        dist = abs(item - obj)
        dist = (maxi - dist)**2
        dist_list.append(dist)
        dist_list2 = []
    for i in range(5):
        max_val = max(dist_list)
        dist_list2.append(max_val)
        dist_list.remove(max_val)
    for distances in dist_list2:
        temp_dist += distances
    dists.append(temp_dist)

dists = [dist / max(dists) for dist in dists]
z = dists

print(min(dists))
print(max(dists))

# Create a polar subplot and capture the axis object
ax = plt.subplot(projection="polar")

# Adjust the azimuth to start from the top
ax.set_theta_zero_location('N')

# Scatter plot on polar axis
sc = ax.scatter(data['Azimuth'], data['Polar'], c=z, cmap='coolwarm')

# Add a colorbar with specific ticks and format
cbar = plt.colorbar(sc, ticks=[0, 0.5, 1], format='%.2f')

# Optionally, set the direction of increase (clockwise)
ax.set_theta_direction(-1)
ax.set_theta_zero_location('E')

# Set the title and labels
plt.title('Radial Heatmap of Polar Angles of Reference Gloms vs Cortical Depth')
# Note: Polar plots do not natively support the xlabel and ylabel methods as expected in Cartesian coordinates.
# For labeling radial and angular axes, consider using text annotations or adjusting the title to include this information.
plt.xlabel('Polar Angles (Degrees)')
plt.ylabel('FOV cortical Depth (Microns)', labelpad=30)
ax.set_rlabel_position(150)

# Rotate the polar plot by 180 degrees
#ax = plt.gca()
#ax.set_theta_direction(-1)

plt.show()

# Plot 2
dists = []
for item in data['Polar']:
    temp_dist = 0
    for obj in data['Polar']:
        dist = (item - obj)**2
        temp_dist += dist
    avg_dist = temp_dist / len(data['Polar'])
    dists.append(avg_dist)

z_max = max(dists)
z = [abs(z_max - item) for item in dists]

plt.subplot(projection="polar")
plt.scatter(data['Polar'], np.radians(data['Azimuth']), c=z, cmap='coolwarm')
plt.colorbar()
plt.title('Radial Heatmap of Polar Angles')
plt.xlabel('Polar Angles (Degrees)')

# Move the y-label to the left by an arbitrary distance (e.g., 40 units)
plt.ylabel('Azimuth Angles (Radians)', labelpad=30)

# Rotate the polar plot by 180 degrees
ax = plt.gca()
ax.set_theta_direction(-1)

plt.show()