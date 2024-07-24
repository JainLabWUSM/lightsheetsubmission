import scipy.stats as stats
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
import powerlaw
from scipy.stats import f_oneway



def shapiro(alpha):

    # Path to your Excel file
    excel_file_path = input('data?: ')

    # Read the Excel file using Pandas to get a DataFrame
    df = pd.read_excel(excel_file_path, header = None, names = ['Data'])

    # Extract the single column as a NumPy array
    data = df['Data'].to_numpy()

    #for i in range(len(data)):
        #if data[i] < 2:
            #data[i] = data[i] + 6.28

    shapiro_test_statistic, shapiro_p_value = stats.shapiro(data)

    # Interpret the p-value
    alpha = alpha  # significance level
    print(shapiro_p_value)
    if shapiro_p_value > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')

def an(alpha):
    # Prompt the user for the name of the Excel file
    file_name = input("Please enter the name of the Excel file: ")
    # Load the Excel file into a DataFrame
    df = pd.read_excel(file_name)
    
    # Convert each column in the DataFrame to a 1D NumPy array
    arrays = [df[column].to_numpy() for column in df.columns]
    
    # Apply transformation: add 6.28 to data points under 2 for each array
    for i in range(len(arrays)):
        arrays[i] = np.where(arrays[i] < 2, arrays[i] + 6.28, arrays[i])
    
    # Perform ANOVA test on each array
    for i, array in enumerate(arrays):
        try:
            _, p_value = f_oneway(*arrays)
            print(f"Column {i+1}: p-value: {p_value}")
            
            if p_value < alpha:
                print("Null hypothesis rejected. There is a significant difference between groups.")
            else:
                print("Null hypothesis not rejected. No significant difference between groups.")
        except Exception as e:
            print(f"Error occurred while performing ANOVA test on array {i+1}: {e}")


def ks(alpha):

    test_data = np.random.normal(loc=0, scale=1, size=100)
    ks_statistic, ks_p_value = stats.kstest(test_data, 'norm', args=(test_data.mean(), test_data.std()))
    print(ks_statistic, ks_p_value)
    # Path to your Excel file
    excel_file_path = input('data?: ')

    # Read the Excel file using Pandas to get a DataFrame
    df = pd.read_excel(excel_file_path, header = None, names = ['Data'])

    # Extract the single column as a NumPy array
    data = df['Data'].to_numpy()
    mean_value = data.mean()
    print(f'Mean: {mean_value}')

    #for i in range(len(data)):
        #if data[i] < 2:
            #data[i] = data[i] + 6.28

    std_dev = np.std(data)
    print(f"std_dev is {std_dev}")

    alpha = alpha
    ks_statistic, ks_p_value = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
    print(ks_p_value)
    # Interpret the p-value
    if ks_p_value > alpha:
        print('Sample looks Gaussian (fail to reject H0)')
    else:
        print('Sample does not look Gaussian (reject H0)')


def tt(alpha):

    while True:
        t_var = input("Do a welch's t-test? (Y/N) (Student's t test will be used if N): ")
        if t_var == 'Y' or t_var == 'N':
            break


    # Path to your Excel file
    excel_file_path = input('data1?: ')
    excel_file_path_2 = input('data2?: ')

    # Read the Excel file using Pandas to get a DataFrame
    df = pd.read_excel(excel_file_path, header = None, names = ['Data'])

    # Extract the single column as a NumPy array
    data = df['Data'].to_numpy()

    #for i in range(len(data)): #Note this was only enabled for angles
        #if data[i] < 2:
            #data[i] = data[i] + 6.28


    # Read the Excel file using Pandas to get a DataFrame
    df2 = pd.read_excel(excel_file_path_2, header = None, names = ['Data'])

    # Extract the single column as a NumPy array
    data2 = df2['Data'].to_numpy()

    # Perform a two-sample t-test
    if t_var == 'Y':
        t_statistic, p_value = scipy.stats.ttest_ind(data, data2, equal_var = False)
    else:
        t_statistic, p_value = scipy.stats.ttest_ind(data, data2)
        

    # Output the results
    print("T-statistic:", t_statistic)
    print("P-value:", p_value)

    # Check if the result is statistically significant at a chosen significance level (e.g., 0.05)
    alpha = 0.05
    if p_value < alpha:
        print("Reject the null hypothesis. There is a significant difference between the groups.")
    else:
        print("Fail to reject the null hypothesis. There is not enough evidence to suggest a significant difference.")


def read_excel_to_lists(file_path):
    try:
        # Read the Excel file into a pandas DataFrame
        df = pd.read_excel(file_path, header=None)  # Specify header=None to ignore headers
        
        # Initialize a dictionary to hold the lists for each column
        column_lists = {}
        
        # Iterate through each column in the DataFrame
        for i, column in enumerate(df.columns):
            # Convert the column values to a list and store it in the dictionary with key i
            #column = [x for x in column if not np.isnan(x)]
            my_column = df[column].tolist()
            my_column = [x for x in my_column if not np.isnan(x)]
            column_lists[i] = my_column
        
        return column_lists
    except Exception as e:
        print("An error occurred:", e)
        return None

def vio():
    # Path to your Excel file
    excel_file_path = input('data?: ')

    column_lists = read_excel_to_lists(excel_file_path)

    # Extract the single column as a NumPy array
    data = column_lists[0]
    data1 = column_lists[1]
    data2 = column_lists[2]
    data3 = column_lists[3]
    data4 = column_lists[4]

    sort_outliers(data)
    sort_outliers(data1)
    sort_outliers(data2)
    sort_outliers(data3)
    sort_outliers(data4)

    # Create violin plots for the data without showing bars inside the violins
    violins = plt.violinplot([data, data1, data2, data3, data4], showmeans=False, showmedians=False, showextrema=False)

    # Customize the violins
    for violin in violins['bodies']:
        violin.set_alpha(0.5)  # Set transparency level for violins

    # Create boxplots without showing box and whiskers
    plt.boxplot([data, data1, data2, data3, data4], positions=[1, 2, 3, 4, 5], showfliers=False, showcaps=False, whiskerprops={'linewidth': 0})

    # Customize the plot
    plt.xticks([1, 2, 3, 4, 5], ['Neonate', 'Infant', 'Young Adult', 'Adult', 'Aged'])
    plt.ylabel('5x Glomerular Neuroinnervation Density Score')
    #plt.title('Violin Plot of Glomerular Neuroinnervation Density Score by Patient Age Group')

    # Show the plot
    plt.show()

def bw2():

    # Path to your Excel file
    excel_file_path = input('data?: ')


    column_lists = read_excel_to_lists(excel_file_path)

    # Extract the single column as a NumPy array
    #data = df['Data'].to_numpy()

    data = column_lists[0]
    print(data)
    data1 = column_lists[1]
    data2 = column_lists[2]
    data3 = column_lists[3]
    data4 = column_lists[4]


    # Create a box-and-whiskers plot for both sets of data
    plt.boxplot([data, data1, data2, data3, data4], labels=['Neonate', 'Infant', 'Young Adult', 'Adult', 'Aged'])

    # Customize the plot
    #plt.title('Volumes of Glomeruluses in Reference Samples by Patient Age Group')
    plt.ylabel('5x Glomerular Neuroinnervation Density Score')

    # Add legend
    #plt.legend(['*', '**'])

    # Show the plot
    plt.show()


def mw():
    # Path to your Excel file
    excel_file_path = input('data1?: ')
    excel_file_path_2 = input('data2?: ')

    # Read the Excel file using Pandas to get a DataFrame
    df = pd.read_excel(excel_file_path, header = None, names = ['Data'])

    # Extract the single column as a NumPy array
    data = df['Data'].to_numpy()

    # Read the Excel file using Pandas to get a DataFrame
    df2 = pd.read_excel(excel_file_path_2, header = None, names = ['Data'])

    # Extract the single column as a NumPy array
    data2 = df2['Data'].to_numpy()

    z_statistic, p_value = stats.ranksums(data, data2)

    print("Wilcoxon Rank-Sum statistic:", z_statistic)
    print("P-value:", p_value)

def ks_power():

    excel_file_path = input('data?: ')

    # Read the Excel file using Pandas to get a DataFrame
    df = pd.read_excel(excel_file_path, header = None, names = ['Data'])

    # Extract the single column as a NumPy array
    data = df['Data'].to_numpy()

    # Fit the power-law distribution to the data
    fit = powerlaw.Fit(data)

    # Perform the Kolmogorov-Smirnov test
    ks_statistic, p_value = fit.distribution_compare('power_law', 'exponential')

    # Print the test results
    print(f"KS Statistic: {ks_statistic}")
    print(f"P-Value: {p_value}")

    # Interpret the results
    if p_value > 0.05:
        print("The neigbhor distribution is likely power-law distributed.")
    else:
        print("The neighbor distribution is likely not power-law distributed.")

def calculate_bounds(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    upper_bound = q3 + (1.5 * iqr)
    lower_bound = q1 - (1.5 * iqr)
    return upper_bound, lower_bound

def sort_outliers(vals):
    upper_bound, lower_bound = calculate_bounds(vals)
    number = 0
    i = len(vals) - 1
    while i >= 0:
        if vals[i] > upper_bound or vals[i] < lower_bound:
            number += 1
        i -= 1
    print(number/len(vals))


print(f"Type 'ks' for Kolmogorov-Smirnov Test (Normality in for >50). Type 's' for Shapiro-Wilk Test (Normality for n<50). Type 'tt' for two-sample t.\nType bw2 for a two sample boxplot. Type p for ks test for power law. Type an for anova test. Type 'mw' for ranksums test. Type 'vio' to make violin plots")
Q = input("Please select test: ")

alpha = 0.05

if Q== "ks":
    ks(alpha)
elif Q == "s":
    shapiro(alpha)
elif Q == "tt":
    tt(alpha)
elif Q == "bw2":
    bw2()
elif Q == "p":
    ks_power()
elif Q == "an":
    an(alpha)
elif Q == "vio":
    vio()
elif Q == "mw":
    mw()

else:
    exit()


