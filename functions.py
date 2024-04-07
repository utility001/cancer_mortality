import pandas as  pd

def missing_data(data, show_all=True):
    """
    Calculate the number and percentage of missing values for each column in a DataFrame.
    
    Parameters:
        data (DataFrame): The input DataFrame.
        show_all (bool): Whether to show all columns regardless of missing values. Default is True.
    
    Returns:
        DataFrame: A DataFrame containing the number and percentage of missing values for each column.
                   If show_all is True, all columns are included. If show_all is False, only columns
                   with missing values are included.
    """
    # Calculate total missing values for each column
    total = data.isnull().sum()
    # Calculate percentage of missing values for each column
    percent = (data.isnull().sum() / data.isnull().count() * 100)
    # Concatenate total and percent missing values into a DataFrame
    tt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    # Get data types of each column
    types = []
    for col in data.columns:
        dtype = str(data[col].dtype)
        types.append(dtype)
    tt['datatype'] = types
    
    # Return either all columns or only columns with missing values based on show_all parameter
    if show_all:
        return tt
    else:
        return tt[tt['Total'] > 0]



def find_constant_columns(data):
    """
    Find columns in a DataFrame containing constant values.

    Parameters:
        data (DataFrame): The input DataFrame.

    Returns:
        list: A list containing the names of columns with constant values.
    """
    constant_columns = []
    for column in data.columns:
        # Get unique values in the column
        if data[column].nunique() == 1:
            constant_columns.append(column)
    return constant_columns



# Get highly imbalanced features
import pandas as pd

def get_highly_imbalanced_columns(df, threshold=0.9):
    """
    Get the names of categorical columns where any category constitutes more than a specified threshold.
    
    Parameters:
    - df: pandas DataFrame
        The input DataFrame containing categorical columns.
    - threshold: float (default=0.9)
        The threshold percentage (0 to 1) for identifying highly imbalanced categories.

    Returns:
    - list
        A list of column names (strings) representing highly imbalanced categorical columns.
    """
    highly_imbalanced_columns = []
    
    # Iterate over columns
    for col in df.columns:
        category_counts = df[col].value_counts(normalize=True)
        max_percentage = category_counts.max()
        if max_percentage > threshold:
            highly_imbalanced_columns.append(col)
    
    return highly_imbalanced_columns



def unique_values(data, max_colwidth=50):
    """
    Get unique values for each column in a DataFrame.

    Parameters:
        data (DataFrame): The input DataFrame.
        max_colwidth (int): Maximum width for displaying column values. Default is 50.

    Returns:
        DataFrame: A DataFrame containing the total count, number of unique values,
                   and unique values for each column.
    """
    # Set maximum column width for display
    pd.options.display.max_colwidth = max_colwidth
    
    # Count total values in each column
    total = data.count()
    
    # Create a DataFrame with total values
    tt = pd.DataFrame(total, columns=['Total'])
    
    # Initialize lists for unique values and their counts
    uniques = []
    values = []
    
    # Iterate over each column
    for col in data.columns:
        # Get unique values and their count for the column
        unique_values = data[col].unique()
        unique_count = data[col].nunique()
        
        # Append unique values and their count to respective lists
        values.append([unique_values])
        uniques.append(unique_count)
    
    # Add columns for unique values and their counts to the DataFrame
    tt['Uniques'] = uniques
    tt['Values'] = values
    
    # Sort DataFrame by number of unique values
    tt = tt.sort_values(by='Uniques', ascending=True)
    
    return tt


import matplotlib.pyplot as plt
import seaborn as sns

# Mini describe
def mini_describe(data, column_name):
    """
    Get Mini description of numeric data
    """
    desc = pd.DataFrame(data.describe().loc[:, column_name]).T
    desc["Range"] = desc["max"] - desc['min']
    desc['IQR'] = desc['75%'] - desc["25%"]
    return desc


from scipy import stats

def hist_box_qq(data, columns, second_plot='box', figsize=(12, 5)):
    """
    Plot histogram and boxplot(if second plot is 'box') or qqplot (if second plot is 'qq')
    """
    num_cols = len(columns)
    
    fig, axes = plt.subplots(num_cols, 2, figsize=(figsize[0] * 2, figsize[1] * num_cols))
    if num_cols == 1:
        axes = axes.reshape(1, -1)

    for i, column in enumerate(columns):
        ax_hist = axes[i, 0]
        ax_plot = axes[i, 1]

        # Plot histogram with KDE by default
        sns.histplot(data[column], bins='auto', color='blue', kde=True, ax=ax_hist)
        ax_hist.set_title(f'Histogram of {column}')

        # Plot boxplot if second_plot is 'box', otherwise plot Q-Q plot
        if second_plot == 'box':
            sns.boxplot(data[column], ax=ax_plot, orient="h", color='green')
            ax_plot.set_title(f'Boxplot of {column}')
        elif second_plot == 'qq':
            stats.probplot(data[column], dist="norm", plot=ax_plot)
            ax_plot.set_title(f'Q-Q plot of {column}')

    plt.tight_layout()
    plt.show()



from scipy.stats import normaltest

# Indentify all features that are normally and not normally distributed
def identify_normality(data, min_unique_values=10, alpha=0.05):
    """
    Returns two lists the first contains feature names of normal numeric columns  
    and the second contain feature names of non_normal columns
    """
    normal_features = []
    non_normal_features = []

    # Identify numeric features
    for column in data.select_dtypes(include='number').columns:
        # Check if column has less than min_unique_values unique values
        if len(data[column].unique()) < min_unique_values:
            continue
        
        # Perform D'Agostino's K-squared test for normality
        stat, p = normaltest(data[column])
        
        # Check if p-value is greater than alpha
        if p > alpha:
            normal_features.append(column)
        else:
            non_normal_features.append(column)
    
    return normal_features, non_normal_features


import numpy as np

# Check for columns that have z score outliers
def z_score_outlier_info(data, features, threshold=3):
    """
    It chcecks whether there aro outliers in a set of features passed to it
    Returna a dictionary and a list.
    The dictionary contains the feature and number of outliers found per feature
    The list contain a list of features that contan outliers
    """
    outlier_info = {}
    outlier_list = []
    
    # num_outliers = 0
    for feature_name in features:
        # Extract feature data
        feature_data = data[feature_name]
        
        # Calculate mean and standard deviation
        mean = np.mean(feature_data)
        std_dev = np.std(feature_data)
        
        # Calculate Z-scores
        z_scores = (feature_data - mean) / std_dev
        
        # Find outliers
        outliers = np.abs(z_scores) > threshold
        
        # If any outliers are found, add the feature to the dictionary
        num_outliers = np.sum(outliers)
        if num_outliers > 0:
            outlier_info[feature_name] = num_outliers
            outlier_list.append(feature_name)
    
    return outlier_list, outlier_info



def z_trim_cap_outliers(data, features, method="trim", threshold=3):
    """
    Return a dataframe
    Trim or cap the outliers
    method: str ['trim' to trim the outliers, 'cap' to cap the outliers]
    """
    # Make a copy of the original data
    processed_data = data.copy()
    
    for feature_name in features:
        # Extract feature data
        feature_data = processed_data[feature_name]
        
        # Calculate mean and sd and zscore
        mean = np.mean(feature_data)
        std_dev = np.std(feature_data)
        
        # Calculate the lower and upper bound
        upper_bound = mean + (threshold * std_dev)
        lower_bound = mean - (threshold * std_dev)

        if method == "trim":
            # Trim the outliers off
            processed_data = processed_data[(feature_data >= lower_bound) & (feature_data <= upper_bound)]

        elif method == "cap":
            # Cap outliers
            processed_data.loc[feature_data > upper_bound, feature_name] = upper_bound
            processed_data.loc[feature_data < lower_bound, feature_name] = lower_bound
    
    return processed_data




from scipy.stats import skew

# Identify skewed columns
def identify_skewed_columns(data, features, skew_threshold=0.5):
    """
    It identify skewed features from a list of features passed to it based on a threshold
    Returna a list and a dictionary
    The list is a list of those feaures
    The dictionary is information about those features
    """
    skewed_features = []
    skewed_info={}
    
    for feature in features:
        # Compute skewness of the feature
        feature_skewness = skew(data[feature])
        
        # Check if skewness exceeds the threshold
        if abs(feature_skewness) > skew_threshold:
            skewed_info[feature] = feature_skewness
            skewed_features.append(feature)
            
    skewed_info = pd.DataFrame(
        list(skewed_info.items()), 
        columns=["Features", "skew"]).set_index('Features')
    
    return skewed_features, skewed_info




import numpy as np
# Check for columns that have skewed outliers
def skewed_outliers_info(data, features, iqr_multiplier=1.5):
    """
    Get information about skewed outliers
    Returns
    """
    outlier_info = {}
    outlier_list = []
    
    for feature_name in features:
        # Extract feature data
        feature_data = data[feature_name]
        
        # Calculate IQR
        q25 = np.percentile(feature_data, 25)
        q75 = np.percentile(feature_data, 75)
        iqr = q75 - q25
        
        # Define outlier boundaries
        lower_bound = q25 - iqr_multiplier * iqr
        upper_bound = q75 + iqr_multiplier * iqr
        
        # Find outliers
        outliers = (feature_data < lower_bound) | (feature_data > upper_bound)
        
        # If any outliers are found, add the feature to the dictionary
        num_outliers = np.sum(outliers)
        if num_outliers > 0:
            outlier_info[feature_name] = num_outliers
            outlier_list.append(feature_name)

    outlier_info = pd.DataFrame(
        list(outlier_info.items()), 
        columns=["Features", "Number of outliers"]).set_index('Features')
    
    return outlier_list, outlier_info




def trim_cap_skewed_outliers(data, features, method='trim', iqr_multiplier=1.5):
    """
    Trims or cap skewed featuers that contain outliers
    Returns a dataframe
    """
    processed_data = data.copy()
    
    for feature_name in features:
        # Extract feature data
        feature_data = processed_data[feature_name]
        
        # Calculate IQR
        q25 = np.percentile(feature_data, 25)
        q75 = np.percentile(feature_data, 75)
        iqr = q75 - q25
        
        # Define outlier boundaries
        lower_bound = q25 - iqr_multiplier * iqr
        upper_bound = q75 + iqr_multiplier * iqr
        
        # Trimming outliers
        if method == 'trim':
            processed_data = processed_data[(feature_data >= lower_bound) & (feature_data <= upper_bound)]
        # Capping outliers
        elif method == 'cap':
            # Ensure dtype compatibility before assignment
            lower_bound = feature_data.dtype.type(lower_bound)
            upper_bound = feature_data.dtype.type(upper_bound)
            processed_data.loc[feature_data < lower_bound, feature_name] = lower_bound
            processed_data.loc[feature_data > upper_bound, feature_name] = upper_bound
    
    return processed_data




def assess_normality(y_true, y_pred):
    # Calculate residuals
    residuals = y_true - y_pred

    # Histogram with KDE
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    sns.histplot(residuals, bins='auto', kde=True, color='blue')
    plt.title('Histogram of Residuals')

    # Q-Q Plot
    plt.subplot(2, 2, 2)
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals')

    # Shapiro-Wilk Test
    _, shapiro_pvalue = stats.shapiro(residuals)
    print("Shapiro-Wilk Test p-value:", shapiro_pvalue)

    # Kolmogorov-Smirnov Test
    _, ks_pvalue = stats.kstest(residuals, 'norm')
    print("Kolmogorov-Smirnov Test p-value:", ks_pvalue)

    # Jarque-Bera Test
    jb_stat, jb_pvalue = stats.jarque_bera(residuals)
    print("Jarque-Bera Test p-value:", jb_pvalue)

    plt.tight_layout()
    plt.show()




def plot_residuals_vs_fitted(y_true, y_pred):
    residuals = y_true - y_pred
    fitted_values = y_pred
    plt.scatter(fitted_values, residuals, alpha=0.7)
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Fitted values')
    plt.axhline(y=0, color='r', linestyle='-')
    plt.show()





def scatter_plots(data, target):
    num_cols = len(data.columns)

    # Calculate the number of rows and columns for subplots
    num_rows = (num_cols - 1) // 3 + 1

    # Create subplots
    fig, axes = plt.subplots(num_rows, 3, figsize=(15, num_rows * 4))
    
    # Flatten axes to iterate over them easily
    axes = axes.flatten()

    # Iterate over each feature column
    for i, column in enumerate(data.columns):
        if column != target:
            ax = axes[i]

            # Plot scatter plot of feature vs target
            ax.scatter(data[column], data[target], alpha=0.5)
            ax.set_title(f'{column} vs {target}')
            ax.set_xlabel(column)
            ax.set_ylabel(target)

    # Remove empty subplots
    for j in range(i+1, num_rows*3):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()