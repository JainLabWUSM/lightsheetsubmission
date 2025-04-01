from nettracer3d import nettracer as n3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns

def excel_to_df(file_path, sheet_name=0):
    """
    Reads an Excel file and converts it to a pandas DataFrame.
    
    Parameters:
    file_path (str): Path to the Excel file
    sheet_name (str or int): Sheet to read. Default is 0 (first sheet)
    
    Returns:
    pandas.DataFrame: DataFrame containing the Excel data
    
    Example:
    >>> df = excel_to_df('data.xlsx')
    >>> df = excel_to_df('data.xlsx', sheet_name='Sheet2')
    """
    try:
        # Read the Excel file
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        
        return df

    except FileNotFoundError:
        raise FileNotFoundError(f"The file {file_path} was not found")
    except Exception as e:
        raise Exception(f"Error reading Excel file: {str(e)}")

data = excel_to_df('edge_node_quantity.xlsx')

node_list = data.iloc[:, 0].tolist()
skele = data.iloc[:, 1].to_list()
dens = data.iloc[:, 3].to_list()
degs = data.iloc[:, 4].to_list()

node_dict = {}
small_dict = {}
for i, node in enumerate(node_list):
    vals = [int(degs[i]), float(skele[i])]
    node_dict[int(node)] = vals
    small_dict[int(node)] = float(skele[i])

mothers = excel_to_df('pos_mothers.xlsx')
mothers = mothers.iloc[:, 0].to_list()

def plot_nodes_scatter(node_dict, mothers):
    """
    Creates a scatter plot of node values with different colors for mothers vs non-mothers,
    adds a linear trendline, and labels mother nodes.
    
    Parameters:
    node_dict (dict): Dictionary with node IDs as keys and [degree, skeleton] as values
    mothers (list): List of mother node IDs
    """
    # Separate data into x and y coordinates and colors
    x_vals = []
    y_vals = []
    colors = []
    
    for node_id, values in node_dict.items():
        x_vals.append(values[0])  # degrees
        y_vals.append(values[1])  # skeleton values
        # Determine color based on whether node is in mothers list
        colors.append('red' if node_id in mothers else 'blue')
    
    # Create the scatter plot
    plt.figure(figsize=(10, 6))
    
    # Plot points
    plt.scatter(x_vals, y_vals, c=colors, alpha=0.6)
    
    # Add labels for mother nodes
    for node_id, values in node_dict.items():
        if node_id in mothers:
            plt.annotate(str(node_id), 
                        (values[0], values[1]),
                        xytext=(5, 5),  # 5 points offset
                        textcoords='offset points',
                        fontsize=9,
                        color='red')
    
    # Calculate and plot trendline
    z = np.polyfit(x_vals, y_vals, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(x_vals), max(x_vals), 100)
    plt.plot(x_trend, p(x_trend), "k--", alpha=0.8)
    
    # Add trendline equation
    equation = f'y = {z[0]:.4f}x + {z[1]:.4f}'
    plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top')
    
    # Add labels and title
    plt.xlabel('Degrees')
    plt.ylabel('Skeletons')
    plt.title('Degrees vs Skeletons')
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=10, label='pos_mother'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=10, label='non-mother')
    ]
    plt.legend(handles=legend_elements, loc='best')
    
    # Show grid
    plt.grid(True, alpha=0.3)
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    
    # Show plot
    plt.show()

# Example usage:
plot_nodes_scatter(node_dict, mothers)
my_network = n3d.Network_3D()
my_network.load_network()
G = my_network.network

num_non = 0
degs_non = 0

num = 0
degs = 0

for node in node_list:
    if node in mothers:
        num += 1
        vals = node_dict[node]
        degs += vals[1]
    else:
        num_non += 1
        vals = node_dict[node]
        degs_non += vals[1]

print(f"Average mothers nerves: {degs/num}, average non-mother nerves: {degs_non/num_non}")


def create_side_by_side_violin_plots(data_dict, highlight_ids):
    """
    Create side-by-side violin plots comparing highlighted and non-highlighted points.
    
    Parameters:
    data_dict (dict): Dictionary with datapoint IDs as keys and float values
    highlight_ids (list): List of datapoint IDs to highlight
    
    Returns:
    None (displays the plot)
    """
    # Separate data into highlighted and non-highlighted
    highlighted_values = [value for key, value in data_dict.items() if key in highlight_ids]
    regular_values = [value for key, value in data_dict.items() if key not in highlight_ids]
    
    # Create DataFrame in format needed for seaborn
    highlighted_df = pd.DataFrame({
        'value': highlighted_values,
        'group': ['Pos_Mothers'] * len(highlighted_values)
    })
    
    regular_df = pd.DataFrame({
        'value': regular_values,
        'group': ['Regular'] * len(regular_values)
    })
    
    # Combine DataFrames
    df = pd.concat([regular_df, highlighted_df])
    
    # Create figure and axis
    plt.figure(figsize=(12, 6))
    
    # Create violin plots
    sns.violinplot(x='group', y='value', data=df, 
                  inner=None,  # Remove inner box plot
                  palette=['red', 'blue'])
    
    # Add individual points with jitter
    for idx, group in enumerate(['Regular', 'Pos_Mothers']):
        group_data = df[df['group'] == group]['value']
        
        # Create jitter for x-coordinates
        x_jitter = np.random.normal(idx, 0.03, size=len(group_data))
        
        # Plot points
        color = 'blue' if group == 'Regular' else 'red'
        plt.scatter(x_jitter, group_data, color=color, alpha=0.6)
        
        # Calculate and plot IQR statistics for each group
        if len(group_data) > 0:  # Only plot stats if group has data
            q1 = group_data.quantile(0.25)
            q3 = group_data.quantile(0.75)
            median = group_data.median()
            
            # Draw IQR box
            plt.hlines(y=[q1, q3], xmin=idx-0.2, xmax=idx+0.2, 
                      colors='black', linestyles='--', alpha=0.5)
            
            # Draw median line
            plt.hlines(y=median, xmin=idx-0.2, xmax=idx+0.2, 
                      colors='black', linewidth=2)
    
    # Customize plot
    plt.title('Comparison of Highlighted vs Regular Points')
    plt.xlabel('')
    plt.ylabel('Values')
    
    # Add sample size annotations
    plt.text(0, plt.ylim()[0], f'n={len(regular_values)}', 
             horizontalalignment='center', verticalalignment='top')
    plt.text(1, plt.ylim()[0], f'n={len(highlighted_values)}', 
             horizontalalignment='center', verticalalignment='top')
    
    plt.tight_layout()
    plt.show()

create_side_by_side_violin_plots(small_dict, mothers)