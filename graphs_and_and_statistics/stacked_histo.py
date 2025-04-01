import pandas as pd
import matplotlib.pyplot as plt

def plot_overlapping_histogram(series1, series2, bin_width=1, xlabel="", ylabel="Frequency", legend_labels=None, title="Overlapping Histogram", x_limits=None, y_limits=None):
    combined_min = min(series1.min(), series2.min())
    combined_max = max(series1.max(), series2.max())
    
    bin_edges = [combined_min + i * bin_width for i in range(int((combined_max - combined_min) / bin_width) + 2)]

    plt.hist(series1, bins=bin_edges, alpha=0.5, label=legend_labels[0])
    plt.hist(series2, bins=bin_edges, alpha=0.5, label=legend_labels[1])

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)

    if x_limits:
        plt.xlim(x_limits[0], x_limits[1])

    if y_limits:
        plt.ylim(y_limits[0], y_limits[1])

    plt.show()

# Prompt user for Excel file
excel_file_path = input("Enter the path to the Excel file: ")

# Load data from Excel file
try:
    df = pd.read_excel(excel_file_path, header=None)  # Assuming there is no header in the Excel file
except Exception as e:
    print("Error reading Excel file:", e)
    exit()

# Extract data into separate DataFrames
df1 = df.iloc[:, 0]  # First column
df2 = df.iloc[:, 1]  # Second column

# Customize the parameters as needed
plot_overlapping_histogram(df1, df2, bin_width=0.00005, xlabel="Length of Nerve (um) per cubic micron within 10 um of glom surface", ylabel="Frequency", legend_labels=['Diabetic', 'Non-Diabetic'], title="Density of Nerves Near Gloms", x_limits=(df.min().min(), df.max().max()), y_limits=(0, 10))