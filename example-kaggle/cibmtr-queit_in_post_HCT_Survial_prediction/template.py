# 1. load libs & install additional libs
# 2. conifgure the notebook
import numpy as np
import pandas
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filenames))

# %%capture
# # Install Lifelines functions from the WHL files.
#
# !pip install /kaggle/input/cibmtr-whl-files-for-installation/autograd-1.7.0-py3-none-any.whl
# !pip install /kaggle/input/cibmtr-whl-files-for-installation/autograd-gamma-0.5.0.tar.gz
# !pip install /kaggle/input/cibmtr-whl-files-for-installation/interface_meta-1.3.0-py3-none-any.whl
# !pip install /kaggle/input/cibmtr-whl-files-for-installation/formulaic-1.1.1-py3-none-any.whl
# !pip install /kaggle/input/cibmtr-whl-files-for-installation/lifelines-0.30.0-py3-none-any.whl

# Importing additional libraries
from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter # Import Kaplan Meier Function
# Importing additional libraries
from sklearn.preprocessing import LabelEncoder # label encoder function from sklearn
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_squared_log_error

# 3. Configure the Notebook
# %%time
# I like to disable my Notebook Warnings.
# import warnings
# warnings.filterwarnings('ignore')
#
# # Configure notebook display settings to only use 2 decimal places, tables look nicer.
# pd.options.display.float_format = '{:,.3f}'.format
# pd.set_option('display.max_columns', 15)
# pd.set_option('display.max_rows', 100)
#
# # Define some of the notebook parameters for future experiment replication.
SEED   = 548

# 4. loda information for the model
def load_csv_to_dataframe(file_path, ignore_fields=[]):
    """
    Load a CSV file into a pandas DataFrame, optionally ignoring specified fields.

    Parameters:
    file_path (str): The file path of the CSV file to be loaded.
    ignore_fields (list): A list of field names to be ignored when loading the CSV.

    Returns:
    pandas.DataFrame: A DataFrame containing the data from the CSV file, excluding the ignored fields.
    """
    # Read the CSV file from the given file path using pandas
    df = pd.read_csv(file_path)
    # Drop the fields that need to be ignored, if they exist in the DataFrame
    df = df.drop(columns=ignore_fields, errors='ignore')
    # Return the resulting DataFrame
    return df
# Example usage:
# df = load_csv_to_dataframe('data/sample.csv', ignore_fields=['column_to_ignore'])
# print(df.head())
# Load the competiion dataset
trn_input = '/kaggle/input/equity-post-HCT-survival-predictions/train.csv'
tst_input = '/kaggle/input/equity-post-HCT-survival-predictions/test.csv'
sub_input = '/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv'

trn_df = load_csv_to_dataframe(trn_input, ignore_fields=['id'])
tst_df = load_csv_to_dataframe(tst_input, ignore_fields=['id'])
sub_df = load_csv_to_dataframe(sub_input)
data_dict = '/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv'
ddt_df = load_csv_to_dataframe(data_dict)

# 5. data adjustments
# Beacause I'm one-hot encoding this characters break y feature naming.
# Define a reusable function for applying replacements
def replace_characters(df, replacements):
    return df.applymap(lambda x: multiple_replace(x, replacements) if isinstance(x, str) else x)
# Helper function to perform multiple replacements
def multiple_replace(text, replacements):
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text
# Define the replacements as a dictionary
replacements = {
    ',': '-',
    '>': 'GT',
    '<': 'LT',
    '>=': 'GE',
    '<=': 'LE'
}
# Apply the replacements to both training and testing DataFrames
#trn_df = replace_characters(trn_df, replacements)
#tst_df = replace_characters(tst_df, replacements)
# Data dictionary with information about the fields
ddt_df.head()

# 6. quick eda
def extensive_eda(df):
    """
    Perform exploratory data analysis (EDA) on the given DataFrame.
    Parameters:
    df (pandas.DataFrame): The DataFrame to analyze.
    Returns:
    None
    """
    from IPython.display import display
    # Display the DataFrame info
    print("Information about the DataFrame:")
    df_info = df.info()
    display(df_info)
    print(".....")
    print("\n")

    # Display the first few rows of data
    print("First few rows of the DataFrame:")
    display(df.head().T)
    print(".....")
    print("\n")

    # Display the number of duplicate values in each column
    print("Number of duplicate values in each column:")
    duplicate_counts = df.duplicated().sum()
    display(duplicate_counts)
    print(".....")
    print("\n")

    # Display the number of missing datapoints in each column
    print("Number of missing datapoints in each column:")
    missing_counts = df.isna().sum()
    display(missing_counts)
    print(".....")
    print("\n")

    # Display the number of outliers in each column using the IQR technique
    print("Number of outliers in each column (using IQR technique):")
    outliers = {}
    for column in df.select_dtypes(include=['number']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        outliers[column] = df[(df[column] < (Q1 - 1.5 * IQR)) | (df[column] > (Q3 + 1.5 * IQR))].shape[0]
    display(outliers)
    print(".....")
    print("\n")

    # Display basic statistics of the DataFrame
    print("Statistical summary of the DataFrame:")
    display(df.describe().T)
    print(".....")
    print("\n")

    # Display unique value count for each column
    print("Number of unique values in each column:")
    unique_counts = df.nunique()
    display(unique_counts)
    print(".....")
    print("\n")

    # Display column-wise summary in a table
    print("Column-wise summary:")
    summary_data = []
    for column in df.columns:
        column_summary = {
            "Column": column,
            "Data Type": df[column].dtype,
            "Missing Values": missing_counts[column],
            "Unique Values": unique_counts[column],
            "Outliers": outliers.get(column, 0) if df[column].dtype in ['int64', 'float64'] else "N/A",
            "Top 5 Values": df[column].value_counts().head().to_dict() if df[column].dtype == 'object' else "N/A"
        }
        summary_data.append(column_summary)
    summary_df = pd.DataFrame(summary_data)
    display(summary_df.style.set_properties(**{'text-align': 'left'}).set_table_styles(
        [dict(selector='th', props=[('text-align', 'left')])]))
    print(".....")
    print("\n")

    # Display correlation matrix for numerical features only
    print("Correlation matrix of numerical features:")
    numerical_df = df.select_dtypes(include=['number'])
    display(numerical_df.corr())
    print(".....")
    print("\n")

    # Display value counts for categorical columns in a table with widened format
    print("Value counts for categorical columns:")
    value_counts_data = []
    for column in df.select_dtypes(include=['object']).columns:
        value_counts = df[column].value_counts().head().to_dict()
        value_counts_data.append({"Column": column, "Top 5 Values": value_counts})
    value_counts_df = pd.DataFrame(value_counts_data)
    display(value_counts_df.style.set_properties(**{'text-align': 'left'}).set_table_styles(
        [dict(selector='th', props=[('text-align', 'left')])]))
    print(".....")
    print("\n")
# Example usage:
# df = load_csv_to_dataframe('data/sample.csv')
# perform_eda(df)
# Quick EDA ...
extensive_eda(trn_df)

# 7. single target creation from multiple targets
# Combining multiple targets into one
def transform_survival_probability(df, time_col='efs_time', event_col='efs'):
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    y = kmf.survival_function_at_times(df[time_col]).values
    return y
trn_df['target'] = transform_survival_probability(trn_df, time_col='efs_time', event_col='efs')

trn_st_df = trn_df.drop(columns = ['efs', 'efs_time', 'ID'])
tst_st_df = tst_df.drop(columns = ['ID'])
# %%time
# Display the first few rows of the dataframe after the target has been converted

trn_df['target'].head(10).T

###############################################################

# 8. feature engineering
# Creating some simple features
trn_st_df['age_gap'] = np.abs(trn_st_df['age_at_hct'] - trn_st_df['donor_age'])
tst_st_df['age_gap'] = np.abs(tst_st_df['age_at_hct'] - tst_st_df['donor_age'])
# Creating a feature to identify years sinse the first treatment completed in this datasets
trn_st_df['tech_progress'] = trn_st_df['year_hct'] - trn_st_df['year_hct'].min()
tst_st_df['tech_progress'] = tst_st_df['year_hct'] - tst_st_df['year_hct'].min()
# sex
def split_sex_match(df, column_name):
    """
    Splits the values of a column in a dataframe where each value is in the format "A-B"
    into two separate columns "column_name_one" and "column_name_two".

    Parameters:
    df (pd.DataFrame): The dataframe containing the column to split.
    column_name (str): The name of the column to split.

    Returns:
    pd.DataFrame: The dataframe with the new columns added.
    """
    # Split the column into two separate columns
    split_columns = df[column_name].str.split('-', expand=True)

    # Name the new columns
    df[f"{column_name}_one"] = split_columns[0]
    df[f"{column_name}_two"] = split_columns[1]
    return df
trn_st_df = split_sex_match(trn_st_df, 'sex_match')
tst_st_df = split_sex_match(tst_st_df, 'sex_match')
# missing value each column
import pandas as pd
import numpy as np
def count_nan_per_row(df, new_feature_name='missing_count'):
    """
    Adds a new column to the DataFrame that contains the count of NaN values in each row.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    new_feature_name (str): The name of the new feature/column to be created.

    Returns:
    pd.DataFrame: The DataFrame with the new feature added.
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input must be a pandas DataFrame")
    # Count NaN values per row
    df[new_feature_name] = df.isna().sum(axis=1)
    return df
# Add a feature to count missing values per row
trn_st_df = count_nan_per_row(trn_st_df, new_feature_name='missing_count')
tst_st_df = count_nan_per_row(tst_st_df, new_feature_name='missing_count')

# 9. data imputation, filling nans
from sklearn.impute import SimpleImputer
def impute_missing_values(train_df, test_df, target_column):
    """
    Impute missing values for categorical and numerical columns in the training and test DataFrames.

    Parameters:
    train_df (pandas.DataFrame): The training DataFrame with missing values.
    test_df (pandas.DataFrame): The testing DataFrame with missing values.
    target_column (str): The name of the target column.

    Returns:
    tuple: A tuple containing the training and testing DataFrames with imputed values.
    """
    # Create copies of the DataFrames to avoid modifying the originals
    train_imputed = train_df.copy()
    test_imputed = test_df.copy()

    # Separate categorical and numerical columns, excluding the target column
    categorical_columns = train_imputed.select_dtypes(include=['object']).columns.difference([target_column])
    numerical_columns = train_imputed.select_dtypes(include=['number']).columns.difference([target_column])

    # Impute missing values for categorical columns using the most frequent value
    # cat_imputer = SimpleImputer(strategy='most_frequent')
    cat_imputer = SimpleImputer(strategy='constant', fill_value='NAN')
    train_imputed[categorical_columns] = cat_imputer.fit_transform(train_imputed[categorical_columns])
    test_imputed[categorical_columns] = cat_imputer.transform(test_imputed[categorical_columns])

    # Impute missing values for numerical columns using the mean value
    # num_imputer = SimpleImputer(strategy='mean')
    num_imputer = SimpleImputer(strategy='constant', fill_value=-99)
    train_imputed[numerical_columns] = num_imputer.fit_transform(train_imputed[numerical_columns])
    test_imputed[numerical_columns] = num_imputer.transform(test_imputed[numerical_columns])

    return train_imputed, test_imputed
# Example usage:
# train_df = load_csv_to_dataframe('data/train.csv')
# test_df = load_csv_to_dataframe('data/test.csv')
# train_imputed, test_imputed = impute_missing_values(train_df, test_df, 'target')
# print(train_imputed.head())
# print(test_imputed.head())
# Utilize the imputation function...
train_imputed, test_imputed = impute_missing_values(trn_st_df, tst_st_df, 'target')

# 10. creating a list of categorical and numerical fields
# %%time
# Remove variables and create a list of features
remove_variables = ['id', 'efs', 'efs_time', 'target']
features = [feat for feat in trn_st_df if feat not in remove_variables]
categorical_features = [feat for feat in trn_st_df[features] if trn_st_df[feat].dtype == 'object']
numerical_features = [feat for feat in trn_st_df[features] if feat not in categorical_features]

# 11. k-means features
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
def create_kmeans_features(train_df, test_df, n_clusters=8, categorical_columns=None, numerical_columns=None, random_state=42):
    """
    Creates K-means features for train and test datasets.

    Parameters:
    train_df (pd.DataFrame): The training dataset.
    test_df (pd.DataFrame): The testing dataset.
    n_clusters (int): Number of clusters for K-means.
    categorical_columns (list): List of categorical column names.
    numerical_columns (list): List of numerical column names.
    random_state (int): Random state for K-means.

    Returns:
    pd.DataFrame, pd.DataFrame: Train and test datasets with K-means cluster features.
    """
    # Ensure input columns are provided
    if categorical_columns is None:
        categorical_columns = []
    if numerical_columns is None:
        numerical_columns = []

    # Combine categorical and numerical columns
    selected_columns = categorical_columns + numerical_columns

    # One-hot encode categorical variables
    train_encoded = pd.get_dummies(train_df[selected_columns], columns=categorical_columns, drop_first=True)
    test_encoded = pd.get_dummies(test_df[selected_columns], columns=categorical_columns, drop_first=True)

    # Align test dataset columns with train dataset columns (handling missing columns)
    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)

    # Fit K-means on the training dataset
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    train_clusters = kmeans.fit_predict(train_encoded)

    # Add cluster labels as a new feature in the training dataset
    train_df['kmeans_cluster'] = train_clusters

    # Predict clusters for the test dataset
    test_clusters = kmeans.predict(test_encoded)

    # Add cluster labels as a new feature in the testing dataset
    test_df['kmeans_cluster'] = test_clusters

    return train_df, test_df
# Example usage
# train, test = create_kmeans_features(train_df, test_df, n_clusters=5, categorical_columns=['cat_col1'], numerical_columns=['num_col1', 'num_col2'])
train_imputed, test_imputed = create_kmeans_features(train_imputed, test_imputed, n_clusters=5, categorical_columns=categorical_features, numerical_columns=numerical_features)

# 12. label encoding
def label_encode_datasets(train_df, test_df, categ_fields):
    """
    Label encode the categorical variables of the train and test DataFrames.

    Parameters:
    train_df (pandas.DataFrame): The training DataFrame.
    test_df (pandas.DataFrame): The testing DataFrame.

    Returns:
    tuple: A tuple containing the label encoded training and testing DataFrames.
    """
    # Create a copy of train and test dataframes to avoid modifying original dataframes
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    # Identify categorical columns
    # categorical_columns = test_encoded.select_dtypes(include=['object']).columns
    categorical_columns = categ_fields

    # Initialize label encoder
    le = LabelEncoder()

    # Apply label encoding to each categorical column
    for column in categorical_columns:
        print(f'Encoding: {column} ...')
        # Fit the label encoder on the train data
        le.fit(train_encoded[column])

        # Transform both train and test data using the same encoder
        train_encoded[column] = le.transform(train_encoded[column])
        if column in test_encoded.columns:
            # Handle cases where test set may have unseen labels by using fillna
            test_encoded[column] = test_encoded[column].map(
                lambda s: le.transform([s])[0] if s in le.classes_ else None)
            test_encoded[column].fillna(-1, inplace=True)
            test_encoded[column] = test_encoded[column].astype(int)

    return train_encoded, test_encoded
# Example usage:
# train_df = load_csv_to_dataframe('data/train.csv')
# test_df = load_csv_to_dataframe('data/test.csv')
# train_encoded, test_encoded = label_encode_datasets(train_df, test_df)
def one_hot_encode_datasets(train_df, test_df, categ_fields):
    """
    One-hot encode the categorical variables of the train and test DataFrames.

    Parameters:
    train_df (pandas.DataFrame): The training DataFrame.
    test_df (pandas.DataFrame): The testing DataFrame.

    Returns:
    tuple: A tuple containing the one-hot encoded training and testing DataFrames.
    """
    # Create a copy of train and test dataframes to avoid modifying original dataframes
    train_encoded = train_df.copy()
    test_encoded = test_df.copy()

    # Identify categorical columns
    categorical_columns = categ_fields

    # Combine train and test data to ensure consistent one-hot encoding across both datasets
    combined_df = pd.concat([train_encoded, test_encoded], axis=0, keys=["train", "test"])

    # Apply one-hot encoding
    combined_encoded = pd.get_dummies(combined_df, columns=categorical_columns, drop_first=True)

    # Split back into train and test dataframes
    train_encoded = combined_encoded.xs("train")
    test_encoded = combined_encoded.xs("test")

    return train_encoded, test_encoded
# Example usage:
# train_df = load_csv_to_dataframe('data/train.csv')
# test_df = load_csv_to_dataframe('data/test.csv')
# train_encoded, test_encoded = one_hot_encode_datasets(train_df, test_df, ['column1', 'column2'])
# Encoding the train and test datasets.
trn_encoded, tst_encoded = label_encode_datasets(train_imputed, test_imputed, categorical_features)
#trn_encoded, tst_encoded = one_hot_encode_datasets(train_imputed, test_imputed, categorical_features)

# 13. model training funcction for any model and target
# Function to train a model (e.g., CatBoost, XGBoost, LightGBM) with GPU support and store OOF predictions
def train_model(train_df, test_df, target_column, model_type="xgboost", param_file=None, n_splits=10):
    """
    Train a machine learning model using the provided training and test datasets with K-Fold cross-validation, utilizing GPU support where applicable.

    Parameters:
    train_df (pandas.DataFrame): The training DataFrame.
    test_df (pandas.DataFrame): The testing DataFrame.
    target_column (str): The name of the target column.
    model_type (str): The type of model to use ("xgboost", "catboost", "lgbm").
    param_file (dict): Dictionary of hyperparameters for the model.
    n_splits (int): The number of folds for cross-validation.

    Returns:
    tuple: A tuple containing the predictions on the test dataset, the OOF predictions, and the trained model.
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    from metric import score

    # Import model-specific libraries
    if model_type == "xgboost":
        from xgboost import XGBRegressor as Model
    elif model_type == "catboost":
        from catboost import CatBoostRegressor as Model
    elif model_type == "lgbm":
        from lightgbm import LGBMRegressor as Model
    else:
        raise ValueError("Unsupported model_type. Choose from 'xgboost', 'catboost', or 'lgbm'.")

    # Separate features and target from the training data
    X = train_df.drop(columns=[target_column])
    y = train_df[target_column]

    # Set default parameters if none are provided
    if param_file is None:
        if model_type == "xgboost":
            param_file = {
                'n_estimators': 2048,
                'learning_rate': 0.030,
                'max_depth': 3,
                'subsample': 0.80,
                'colsample_bytree': 0.50,
                'min_child_weight': 80,
                'tree_method': 'gpu_hist',
                'random_state': 42
            }
        elif model_type == "catboost":
            param_file = {
                'iterations': 1000,
                'learning_rate': 0.03,
                'depth': 6,
                'task_type': 'GPU',
                'random_seed': 42
            }
        elif model_type == "lgbm":
            param_file = {
                'n_estimators': 2048,
                'learning_rate': 0.030,
                'max_depth': -1,
                'subsample': 0.80,
                'colsample_bytree': 0.50,
                'min_child_weight': 80,
                'device': 'gpu',
                'random_state': 42
            }

    # Initialize the model with parameters from param_file
    model = Model(**param_file)

    # Initialize KFold cross-validation
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Lists to store cross-validation results
    mse_scores = []
    mae_scores = []
    r2_scores = []

    test_predictions = []
    oof_predictions = np.zeros(len(train_df))

    # Perform K-Fold cross-validation
    fold_number = 0
    for train_index, val_index in kf.split(X):
        fold_number += 1

        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions on the validation set
        y_val_pred = model.predict(X_val)
        oof_predictions[val_index] = y_val_pred

        # Calculate validation metrics
        mse_scores.append(mean_squared_error(y_val, y_val_pred))
        mae_scores.append(mean_absolute_error(y_val, y_val_pred))
        r2_scores.append(r2_score(y_val, y_val_pred))

        # Make predictions on the test dataset for each fold
        X_test = test_df.drop(columns=[target_column], errors='ignore')
        test_predictions.append(model.predict(X_test))

        # print(f'Fold {fold_number} MSE = {mse_scores[fold_number-1]}')

    # Calculate average metrics
    avg_mse = np.mean(mse_scores)
    avg_mae = np.mean(mae_scores)
    avg_r2 = np.mean(r2_scores)

    # Print the metrics in a readable format
    print("Model Performance Metrics (Cross-Validation):")
    print("..................")
    print(f"Average MSE: {avg_mse:.4f}")
    print(f"Average MAE: {avg_mae:.4f}")
    print(f"Average R2 Score: {avg_r2:.4f}")

    # Calculate the average predictions across all folds
    y_test_pred = np.mean(test_predictions, axis=0)

    return y_test_pred, oof_predictions, model
# Example usage:
# param_file = {
#     'n_estimators': 200,
#     'learning_rate': 0.05,
#     'max_depth': 8,
#     'tree_method': 'gpu_hist',  # Enable GPU support
#     'predictor': 'gpu_predictor'
# }
# train_df = load_csv_to_dataframe('data/train.csv')
# test_df = load_csv_to_dataframe('data/test.csv')
# predictions, oof_predictions, model = train_model(train_df, test_df, 'target', model_type='xgboost', param_file=param_file)
# print(predictions)
remove_features = ['kmeans_cluster','missing_count', 'sex_match_one', 'sex_match_two', 'age_gap', 'tech_progress']
trn_encoded = trn_encoded.drop(columns = remove_features)
tst_encoded = tst_encoded.drop(columns = remove_features)

# 14. Model Training Function for any Model and Target
# xgboost, lightgbm, catboost
# 14.1 xgboost
param_file = {
    'n_estimators': 2048,
    'learning_rate': 0.025,
    'max_depth': 3,
    'subsample': 0.80,
    'colsample_bytree': 0.50,
    'min_child_weight': 80,
    'enable_categorical': True,
    'random_state': SEED
    #'tree_method': 'gpu_hist',  # Enable GPU support
    #'predictor': 'gpu_predictor'
}
predictions_xgb, oof_predictions_xgb, model_xgb = train_model(trn_encoded, tst_encoded, 'target', model_type='xgboost', param_file=param_file)
from metric import score
y_true = trn_df[["ID","efs","efs_time","race_group"]].copy()
y_pred = trn_df[["ID"]].copy()
y_pred["prediction"] = oof_predictions_xgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"CV Score = {m}")

# 14.2 catboost
param_file = {
    'iterations': 512,
    'learning_rate': 0.1,
    'depth': 3,
    'random_seed': SEED,
    'silent': True
}
predictions_cb, oof_predictions_cb, model_cb = train_model(trn_encoded, tst_encoded, 'target', model_type='catboost', param_file=param_file)
from metric import score
y_true = trn_df[["ID","efs","efs_time","race_group"]].copy()
y_pred = trn_df[["ID"]].copy()
y_pred["prediction"] = oof_predictions_cb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"CV Score = {m}")

# 14.3 lgbm:
param_file = {
    'n_estimators': 4096,
    'learning_rate': 0.030,
    'max_depth': 3,
    'num_leaves': 31,
    'objective':'regression',
    'random_state': SEED,
    'verbose': -1
}
predictions_lgb, oof_predictions_lgb, model_lgb = train_model(trn_encoded, tst_encoded, 'target', model_type='lgbm', param_file=param_file)
y_true = trn_df[["ID","efs","efs_time","race_group"]].copy()
y_pred = trn_df[["ID"]].copy()
y_pred["prediction"] = oof_predictions_lgb
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"CV Score = {m}")

# 14.4. xgboost survival: cox
# Target modification for Survival Cox Analysis.
trn_encoded['target'] = trn_df['efs_time'].copy()
trn_encoded.loc[trn_df['efs'] == 0,'target'] *= -1
param_file = {
    'n_estimators': 2048,
    'learning_rate': 0.025,
    'max_depth': 3,
    'subsample': 0.80,
    'colsample_bytree': 0.50,
    'min_child_weight': 80,
    'enable_categorical': True,
    'objective':'survival:cox',
    'eval_metric':'cox-nloglik',
    'random_state': SEED
    #'tree_method': 'gpu_hist',  # Enable GPU support
    #'predictor': 'gpu_predictor'
}
predictions_xgb_cox, oof_predictions_xgb_cox, model_xgb_cox = train_model(trn_encoded, tst_encoded, 'target', model_type='xgboost', param_file=param_file)
# from metric import score
y_true = trn_df[["ID","efs","efs_time","race_group"]].copy()
y_pred = trn_df[["ID"]].copy()
y_pred["prediction"] = oof_predictions_xgb_cox
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"CV Score = {m}")

# 14.5 Catboost Survival: Cox -- Lossguide
param_file = {
    'grow_policy': 'Lossguide',
    'loss_function': 'Cox',
    'learning_rate': 0.03,
    'task_type': 'CPU',
    'num_trees': 2048,
    'subsample': 0.85,
    'reg_lambda': 8.0,
    'num_leaves': 32,
    'depth': 3,
    'random_state': SEED,
    'silent': True
}
predictions_cb_cox, oof_predictions_cb_cox, model_cb_cox = train_model(trn_encoded, tst_encoded, 'target', model_type='catboost', param_file=param_file)
# from metric import score
y_true = trn_df[["ID","efs","efs_time","race_group"]].copy()
y_pred = trn_df[["ID"]].copy()
y_pred["prediction"] = oof_predictions_cb_cox
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"CV Score = {m}")

# 14.6 Catboost Survival: Cox -- Depthwise
param_file = {
    'grow_policy': 'Depthwise',
    'loss_function': 'Cox',
    'learning_rate': 0.025,
    'task_type': 'CPU',
    'num_trees': 2048,
    'subsample': 0.85,
    'reg_lambda': 8.0,
    'depth': 3,
    'random_state': SEED,
    'silent': True
}
predictions_cbd_cox, oof_predictions_cbd_cox, model_cbd_cox = train_model(trn_encoded, tst_encoded, 'target', model_type='catboost', param_file=param_file)
# from metric import score
y_true = trn_df[["ID","efs","efs_time","race_group"]].copy()
y_pred = trn_df[["ID"]].copy()
y_pred["prediction"] = oof_predictions_cbd_cox
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"CV Score = {m}")

#14.7 Catboost Nelson-Aalen Target -- Depthwise
naf = NelsonAalenFitter()
naf.fit(durations=trn_df['efs_time'], event_observed=trn_df['efs'])
trn_encoded['target'] = -naf.cumulative_hazard_at_times(trn_df['efs_time']).values
param_file = {
    'grow_policy': 'Depthwise',
    'learning_rate': 0.025,
    'task_type': 'CPU',
    'num_trees': 2048,
    'subsample': 0.85,
    'reg_lambda': 8.0,
    'depth': 3,
    'random_state': SEED,
    'silent': True
}
predictions_cbd_na, oof_predictions_cbd_na, model_cbd_na = train_model(trn_encoded, tst_encoded, 'target', model_type='catboost', param_file=param_file)
# from metric import score
y_true = trn_df[["ID","efs","efs_time","race_group"]].copy()
y_pred = trn_df[["ID"]].copy()
y_pred["prediction"] = oof_predictions_cbd_na
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"CV Score = {m}")

# 14.8 Xgboost Nelson-Aalen Target
param_file = {
    'n_estimators': 2048,
    'learning_rate': 0.025,
    'max_depth': 4,
    'subsample': 0.80,
    'colsample_bytree': 0.50,
    'min_child_weight': 80,
    'enable_categorical': True,
    'random_state': SEED
    #'tree_method': 'gpu_hist',  # Enable GPU support
    #'predictor': 'gpu_predictor'
}
predictions_xgb_na, oof_predictions_xgb_na, model_xgb_na = train_model(trn_encoded, tst_encoded, 'target', model_type='xgboost', param_file=param_file)
# from metric import score
y_true = trn_df[["ID","efs","efs_time","race_group"]].copy()
y_pred = trn_df[["ID"]].copy()
y_pred["prediction"] = oof_predictions_xgb_na
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"CV Score = {m}")

#  14.9 LGBM Nelson-Aalen Target
param_file = {
    'n_estimators': 4096,
    'learning_rate': 0.030,
    'max_depth': 3,
    'num_leaves': 31,
    'objective':'regression',
    'random_state': SEED,
    'verbose': -1
}
predictions_lgb_na, oof_predictions_lgb_na, model_lgb_na = train_model(trn_encoded, tst_encoded, 'target', model_type='lgbm', param_file=param_file)
# from metric import score
y_true = trn_df[["ID","efs","efs_time","race_group"]].copy()
y_pred = trn_df[["ID"]].copy()
y_pred["prediction"] = oof_predictions_lgb_na
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"CV Score = {m}")

# 15. blending models(ensemble)
# Using Rank and Weigths to Ensemble Models
oof_preds = [
    oof_predictions_xgb,
    oof_predictions_cb,
    oof_predictions_lgb,
    oof_predictions_xgb_cox,
    oof_predictions_cb_cox,
    oof_predictions_cbd_cox,
    oof_predictions_xgb_na,
    oof_predictions_cbd_na,
    oof_predictions_lgb_na,
]
weights = [1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]
from scipy.stats import rankdata
ranked_preds = np.array([rankdata(p) for p in oof_preds])
ensemble_preds = np.sum([w * p for w, p in zip(weights, ranked_preds)], axis=0)
# from metric import score
y_true = trn_df[["ID","efs","efs_time","race_group"]].copy()
y_pred = trn_df[["ID"]].copy()
y_pred["prediction"] = ensemble_preds
m = score(y_true.copy(), y_pred.copy(), "ID")
print(f"CV Score = {m}")

# 15.2 Using LR to Ensemble Models
#     # Use Linear Regression to optimize blending weights
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
ranked_preds = np.array([rankdata(p) for p in oof_preds])
oof_matrix = np.column_stack(ranked_preds)
meta_model = XGBRegressor()
#meta_model = LinearRegression()
X_train, X_val, y_train, y_val = train_test_split(oof_matrix, trn_df['target'], test_size=0.9, random_state=42)
meta_model.fit(X_train, y_train)
# Blend the predictions using the trained meta-model
blended_val_predictions = meta_model.predict(oof_matrix)
# Print blending coefficients
# print("Blending Coefficients:", meta_model.coef_)
# from metric import score
y_true = trn_df[["ID","efs","efs_time","race_group"]].copy()
X_train, X_val, y_train, y_val = train_test_split(oof_matrix, y_true, test_size=0.9, random_state=42)
y_pred = y_val[["ID"]].copy()
blended_val_predictions = meta_model.predict(X_val)
y_pred["prediction"] = blended_val_predictions
m = score(y_val.copy(), y_pred.copy(), "ID")
print(f"CV Score = {m}")

# 16. creating submissions
from IPython import display
preds = [
    predictions_xgb,
    predictions_cb,
    predictions_lgb,
    predictions_xgb_cox,
    predictions_cb_cox,
    predictions_cbd_cox,
    predictions_xgb_na,
    predictions_cbd_na,
    predictions_lgb_na,
]
weights = [1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9, 1/9]
ranked_preds = np.array([rankdata(p) for p in preds])
ensemble_preds = np.sum([w * p for w, p in zip(weights, ranked_preds)], axis=0)
sub_df['prediction'] = ensemble_preds
sub_df.to_csv('submission.csv', index=False)
display(sub_df.head())
