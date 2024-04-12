from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

# This function splits the data
def split_data():
    # Import the dataset
    df = pd.read_csv("dataset/cancer_reg.csv",
                 encoding='unicode_escape')

    # Train test split
    train, test= train_test_split(df, test_size=0.2, random_state=42)

    return train, test


from scipy.stats import normaltest

# Indentify all features that are normally and not normally distributed
def identify_normality(data, min_unique_values=10, alpha=0.05):
    """
    Returns two lists.  
    The first contains feature names of normal numeric columns   
    The second contain feature names of non_normal columns

    Data: Dataframe
    min_unique values: The minimum number of unique values a feature must have in order to be considered continuous
    alpha: p value
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
            # Normal feature
            normal_features.append(column)
        else:
            # Not normal
            non_normal_features.append(column)
    
    return normal_features, non_normal_features



# Create a function to trim or cap outliers
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
            # Trim the outliers off i.e values greater than upper bound and less than lower bound are removed
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
    Returna a list and a dataframe
    The list is a list of those feaures
    The dataframe is information about those features
    """
    skewed_features = []
    skewed_info={}
    
    for feature in features:
        # Compute skewness of the feature
        feature_skewness = skew(data[feature])
        
        # Check if skewness exceeds the threshold
        if abs(feature_skewness) > skew_threshold:
            # Add it to the dictionary
            skewed_info[feature] = feature_skewness
            # And the list
            skewed_features.append(feature)

    # Crate a dataframe containing skew information
    skewed_info = pd.DataFrame(
        list(skewed_info.items()), 
        columns=["Features", "skew"]).set_index('Features')
    
    return skewed_features, skewed_info




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
            # Ensure dtype compatibility before assignment i.e convert from numpy float to normal float
            lower_bound = feature_data.dtype.type(lower_bound)
            upper_bound = feature_data.dtype.type(upper_bound)
            processed_data.loc[feature_data < lower_bound, feature_name] = lower_bound
            processed_data.loc[feature_data > upper_bound, feature_name] = upper_bound
    
    return processed_data





# Final outlier detection and capping
def train_outlier_capping(df):
    # Identify features that are normally and not normally distributed in the train data
    normal_features, non_normal_features = identify_normality(df.drop(columns=["TARGET_deathRate"]))
    
    # Make sure that the target is not selected for outlier detection
    assert 'TARGET_deathRate' not in normal_features
    assert 'TARGET_deathRate' not in non_normal_features

    # Cap the outliers in the train data
    train = z_trim_cap_outliers(df, normal_features, method='cap')

    # Of all teh non normal features, identify the extremely skewed ones with threshold as 1
    skew_feats, _ = identify_skewed_columns(df, non_normal_features, skew_threshold=1)
    # print(skew_feats)

    # Cap the outliers
    df = trim_cap_skewed_outliers(df, skew_feats, method="cap")

    return df



# This functino performs all categorical transformation
def cat_transform(df):

    df = df.copy()
    
    # Create a mapping for the binned inc column
    category_to_number = {
        "[22640, 34218.1]": 1,
        "(34218.1, 37413.8]": 2,
        "(37413.8, 40362.7]": 3,
        "(40362.7, 42724.4]": 4,
        "(42724.4, 45201]": 5,
        "(45201, 48021.6]": 6,
        "(48021.6, 51046.4]": 7,
        "(51046.4, 54545.6]": 8,
        "(54545.6, 61494.5]": 9,
        "(61494.5, 125635]": 10
    }

    # Apply the mapping
    df["binnedInc"] = df["binnedInc"].map(category_to_number)

    
    ## Split the geography column
    
    # Split and expand at ','
    df[["county", "state"]] = df["Geography"].str.split(",", expand=True)
    # Drop the expanded data
    df = df.drop(columns="Geography")
    # Strip
    df["county"] = df["county"].str.strip()
    df["state"] = df["state"].str.strip()

    # Drop the county column
    df = df.drop(columns=["county"])
    
    return df


