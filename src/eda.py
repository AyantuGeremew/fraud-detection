import pandas as pd
import numpy as np
import ipaddress
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE

# --------------------------------------------------
# Data Cleaning
# ---------------------------------------------------

def handle_missing_values(df, missing_threshold=0.5):
    """
    Handle missing values in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    missing_threshold : float, default=0.5
        Drop columns with more than this proportion of missing values.

    Returns
    -------
    pandas.DataFrame
        Cleaned DataFrame.
    """

    df_clean = df.copy()

    # 1. Drop columns with excessive missing values
    missing_ratio = df_clean.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > missing_threshold].index

    if len(cols_to_drop) > 0:
        print(f"Dropping columns: {list(cols_to_drop)}")
        df_clean.drop(columns=cols_to_drop, inplace=True)

    # 2. Identify numerical and categorical columns
    numerical_cols = df_clean.select_dtypes(include=["number"]).columns
    categorical_cols = df_clean.select_dtypes(include=["object", "category"]).columns

    # 3. Impute numerical features with median
    if len(numerical_cols) > 0:
        num_imputer = SimpleImputer(strategy="median")
        df_clean[numerical_cols] = num_imputer.fit_transform(
            df_clean[numerical_cols]
        )

    # 4. Impute categorical features with mode
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        df_clean[categorical_cols] = cat_imputer.fit_transform(
            df_clean[categorical_cols]
        )

    print("\nMissing values after treatment:")
    print(df_clean.isnull().sum().sum())

    return df_clean

def remove_duplicates(df):
    """
    Remove duplicate rows from a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.

    Returns
    -------
    pandas.DataFrame
        DataFrame with duplicates removed.
    """
    
    df_clean = df.copy()

    # Count duplicate rows
    n_duplicates = df_clean.duplicated().sum()
    print(f"Number of duplicate rows: {n_duplicates}")

    # Remove duplicates
    df_clean = df_clean.drop_duplicates()

    print(f"Shape after removing duplicates: {df_clean.shape}")

    return df_clean

def correct_data_types(df, dtype_mapping):
    """
    Correct column data types in a DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    dtype_mapping : dict
        Dictionary of column names and target data types.

    Returns
    -------
    pandas.DataFrame
        DataFrame with corrected data types.
    """

    df_clean = df.copy()

    for col, dtype in dtype_mapping.items():

        if col not in df_clean.columns:
            print(f"Column '{col}' not found.")
            continue

        try:
            if dtype == "datetime":
                df_clean[col] = pd.to_datetime(
                    df_clean[col],
                    errors="coerce"
                )

            elif dtype == "category":
                df_clean[col] = df_clean[col].astype("category")

            elif dtype == "string":
                df_clean[col] = df_clean[col].astype("string")

            elif dtype in ["int", "float"]:
                df_clean[col] = pd.to_numeric(
                    df_clean[col],
                    errors="coerce"
                )

                if dtype == "int":
                    df_clean[col] = df_clean[col].astype("Int64")

            print(f"✓ {col} converted to {dtype}")

        except Exception as e:
            print(f"✗ Error converting {col}: {e}")

    return df_clean

def auto_correct_data_types(df):
    """
    Automatically infer better data types.
    """

    df_clean = df.copy()

    df_clean = df_clean.convert_dtypes()

    print("Updated Data Types:")
    print(df_clean.dtypes)

    return df_clean

# --------------------------------------------------
# Exploratory Data Analysis
# ---------------------------------------------------

def univariate_analysis(df, columns=None):
    """
    Perform univariate analysis on selected columns.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    columns : list, optional
        Columns to analyze. If None, all columns are analyzed.

    Returns
    -------
    None
    """

    if columns is None:
        columns = df.columns

    for col in columns:

        print("\n" + "=" * 60)
        print(f"Variable: {col}")
        print("=" * 60)

        # Numerical Variables
        if pd.api.types.is_numeric_dtype(df[col]):

            print(df[col].describe())

            plt.figure(figsize=(8, 4))

            sns.histplot(
                df[col].dropna(),
                kde=True,
                bins=30
            )

            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.show()

        # Categorical Variables
        else:

            print(df[col].value_counts(dropna=False))

            plt.figure(figsize=(10, 4))

            df[col].value_counts().plot(
                kind="bar"
            )

            plt.title(f"Distribution of {col}")
            plt.xlabel(col)
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

def plot_numeric_distributions(df, numeric_cols):
    """
    Plot distributions of numerical variables.
    """

    for col in numeric_cols:

        plt.figure(figsize=(8, 4))

        sns.histplot(
            df[col].dropna(),
            kde=True,
            bins=30
        )

        plt.title(f"Distribution of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")

        plt.show()

def bivariate_analysis(df, target):
    """
    Analyze relationships between features and a target variable.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    target : str
        Target variable name.

    Returns
    -------
    None
    """

    if target not in df.columns:
        raise ValueError(f"'{target}' not found in DataFrame.")

    for col in df.columns:

        if col == target:
            continue

        print("\n" + "=" * 60)
        print(f"{col} vs {target}")
        print("=" * 60)

        # Numerical Feature vs Numerical Target
        if pd.api.types.is_numeric_dtype(df[col]):

            correlation = df[[col, target]].corr().iloc[0, 1]
            print(f"Correlation: {correlation:.4f}")

            plt.figure(figsize=(8, 5))
            sns.scatterplot(
                data=df,
                x=col,
                y=target,
                alpha=0.6
            )

            plt.title(f"{col} vs {target}")
            plt.show()

        # Categorical Feature vs Numerical Target
        else:

            group_means = (
                df.groupby(col)[target]
                .mean()
                .sort_values(ascending=False)
            )

            print("Average Target by Category:")
            print(group_means)

            plt.figure(figsize=(10, 5))

            sns.boxplot(
                data=df,
                x=col,
                y=target
            )

            plt.title(f"{col} vs {target}")
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.show()

def feature_target_correlation(df, target):
    """
    Calculate correlations between numerical features
    and a numerical target.
    """

    numeric_df = df.select_dtypes(include="number")

    correlations = (
        numeric_df.corr()[target]
        .drop(target)
        .sort_values(key=abs, ascending=False)
        .reset_index()
    )

    correlations.columns = ["Feature", "Correlation"]

    return correlations

def quantify_class_imbalance(df, target):
    """
    Quantify class imbalance for a classification target.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    target : str
        Target column.

    Returns
    -------
    pandas.DataFrame
        Summary of class distribution.
    """

    if target not in df.columns:
        raise ValueError(f"'{target}' not found in DataFrame.")

    # Class counts
    counts = df[target].value_counts()

    # Class percentages
    percentages = round(
        df[target].value_counts(normalize=True) * 100,
        2
    )

    # Summary table
    summary = pd.DataFrame({
        "Count": counts,
        "Percentage (%)": percentages
    })

    # Imbalance ratio
    imbalance_ratio = counts.max() / counts.min()

    print(f"\nClass Distribution for '{target}'")
    print("-" * 40)
    print(summary)

    print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1")

    # Visualization
    plt.figure(figsize=(6, 4))

    sns.countplot(
        data=df,
        x=target
    )

    plt.title(f"Class Distribution of {target}")
    plt.ylabel("Count")

    plt.show()

    return summary

# --------------------------------------------------
# Geolocation Integration
# ---------------------------------------------------

def convert_ip_to_integer(df, ip_column):
    """
    Convert IPv4 addresses to integer format.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame.
    ip_column : str
        Column containing IP addresses.

    Returns
    -------
    pandas.DataFrame
        DataFrame with converted IP addresses.
    """

    df_copy = df.copy()

    def ip_to_int(ip):
        try:
            return int(ipaddress.ip_address(ip))
        except (ValueError, TypeError):
            return pd.NA

    df_copy[f"{ip_column}_int"] = df_copy[ip_column].apply(ip_to_int)

    print(f"Converted '{ip_column}' to '{ip_column}_int'")

    return df_copy

def merge_ip_country(fraud_df, ip_country_df):
    """
    Merge Fraud_Data with IpAddress_to_Country using
    range-based IP lookup.

    Parameters
    ----------
    fraud_df : pandas.DataFrame
        Fraud transaction dataset.
    ip_country_df : pandas.DataFrame
        IP-to-country mapping dataset.

    Returns
    -------
    pandas.DataFrame
        Fraud dataset enriched with country information.
    """

    # Create copies
    fraud = fraud_df.copy()
    ip_map = ip_country_df.copy()

    # Ensure numeric types
    fraud["ip_address"] = pd.to_numeric(
        fraud["ip_address"],
        errors="coerce"
    )

    ip_map["lower_bound_ip_address"] = pd.to_numeric(
        ip_map["lower_bound_ip_address"],
        errors="coerce"
    )

    ip_map["upper_bound_ip_address"] = pd.to_numeric(
        ip_map["upper_bound_ip_address"],
        errors="coerce"
    )

    # Sort for merge_asof
    fraud = fraud.sort_values("ip_address")
    ip_map = ip_map.sort_values("lower_bound_ip_address")

    # Merge on nearest lower bound
    merged = pd.merge_asof(
        fraud,
        ip_map,
        left_on="ip_address",
        right_on="lower_bound_ip_address",
        direction="backward"
    )

    # Keep only rows where IP falls within range
    merged = merged[
        merged["ip_address"] <= merged["upper_bound_ip_address"]
    ]

    print(f"Fraud records: {len(fraud_df):,}")
    print(f"Matched records: {len(merged):,}")

    return merged

def prepare_ip_data(fraud_df, ip_country_df):
    """
    Prepare datasets for IP range lookup.

    Parameters
    ----------
    fraud_df : pd.DataFrame
        Fraud dataset containing ip_address.
    ip_country_df : pd.DataFrame
        IP-to-country dataset containing IP ranges.

    Returns
    -------
    tuple
        Prepared fraud_df and ip_country_df.
    """

    fraud = fraud_df.copy()
    ip_map = ip_country_df.copy()

    # Convert IP columns to numeric
    fraud["ip_address"] = pd.to_numeric(
        fraud["ip_address"],
        errors="coerce"
    )

    ip_map["lower_bound_ip_address"] = pd.to_numeric(
        ip_map["lower_bound_ip_address"],
        errors="coerce"
    )

    ip_map["upper_bound_ip_address"] = pd.to_numeric(
        ip_map["upper_bound_ip_address"],
        errors="coerce"
    )

    # Remove rows with invalid IP values
    fraud = fraud.dropna(subset=["ip_address"])

    ip_map = ip_map.dropna(
        subset=[
            "lower_bound_ip_address",
            "upper_bound_ip_address"
        ]
    )

    # Sort data for merge_asof
    fraud = fraud.sort_values("ip_address")
    ip_map = ip_map.sort_values("lower_bound_ip_address")

    return fraud, ip_map


def merge_fraud_with_country(fraud_df, ip_country_df):
    """
    Merge Fraud_Data with IpAddress_to_Country using
    IP range lookup.

    Parameters
    ----------
    fraud_df : pd.DataFrame
    ip_country_df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Fraud data enriched with country information.
    """

    fraud, ip_map = prepare_ip_data(
        fraud_df,
        ip_country_df
    )

    # Match each IP with nearest lower bound
    merged = pd.merge_asof(
        fraud,
        ip_map,
        left_on="ip_address",
        right_on="lower_bound_ip_address",
        direction="backward"
    )

    # Keep only valid range matches
    merged = merged[
        merged["ip_address"]
        <= merged["upper_bound_ip_address"]
    ]

    print(f"Original fraud records : {len(fraud_df):,}")
    print(f"Matched records        : {len(merged):,}")
    print(
        f"Match rate             : "
        f"{len(merged)/len(fraud_df)*100:.2f}%"
    )

    return merged

def country_summary(df):
    """
    Display top countries after merge.
    """

    return (
        df["country"]
        .value_counts()
        .head(10)
        .reset_index()
        .rename(
            columns={
                "index": "country",
                "country": "count"
            }
        )
    )

def analyze_fraud_by_country(
    df,
    country_col="country",
    fraud_col="class",
    top_n=15
):
    """
    Analyze fraud patterns by country.

    Parameters
    ----------
    df : pd.DataFrame
        Fraud dataset containing country and fraud label.
    country_col : str
        Country column name.
    fraud_col : str
        Fraud target column (0=legitimate, 1=fraud).
    top_n : int
        Number of countries to display.

    Returns
    -------
    pd.DataFrame
        Country-level fraud summary.
    """

    # Country summary
    country_summary = (
        df.groupby(country_col)
        .agg(
            Total_Transactions=(fraud_col, "count"),
            Fraud_Transactions=(fraud_col, "sum")
        )
        .reset_index()
    )

    # Fraud rate
    country_summary["Fraud_Rate (%)"] = (
        country_summary["Fraud_Transactions"]
        / country_summary["Total_Transactions"]
        * 100
    ).round(2)

    # Sort by fraud rate
    country_summary = country_summary.sort_values(
        by="Fraud_Rate (%)",
        ascending=False
    )

    print("\nTop Countries by Fraud Rate")
    print(country_summary.head(top_n))

    # Visualization
    plt.figure(figsize=(12, 6))

    sns.barplot(
        data=country_summary.head(top_n),
        x="Fraud_Rate (%)",
        y=country_col
    )

    plt.title(f"Top {top_n} Countries by Fraud Rate")
    plt.xlabel("Fraud Rate (%)")
    plt.ylabel("Country")
    plt.tight_layout()
    plt.show()

    return country_summary

def plot_country_fraud_volume(
    df,
    country_col="country",
    fraud_col="class",
    top_n=10
):
    """
    Compare total and fraudulent transactions by country.
    """

    summary = (
        df.groupby(country_col)
        .agg(
            Total_Transactions=(fraud_col, "count"),
            Fraud_Transactions=(fraud_col, "sum")
        )
        .sort_values(
            by="Fraud_Transactions",
            ascending=False
        )
        .head(top_n)
        .reset_index()
    )

    summary.plot(
        x=country_col,
        y=[
            "Total_Transactions",
            "Fraud_Transactions"
        ],
        kind="bar",
        figsize=(12, 6)
    )

    plt.title(
        f"Top {top_n} Countries by Fraud Volume"
    )

    plt.ylabel("Transactions")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    return summary

def identify_high_risk_countries(
    df,
    country_col="country",
    fraud_col="class",
    min_transactions=100
):
    """
    Identify countries with high fraud rates.

    Parameters
    ----------
    min_transactions : int
        Ignore countries with fewer transactions.
    """

    summary = (
        df.groupby(country_col)
        .agg(
            Total_Transactions=(fraud_col, "count"),
            Fraud_Transactions=(fraud_col, "sum")
        )
        .reset_index()
    )

    summary = summary[
        summary["Total_Transactions"] >= min_transactions
    ]

    summary["Fraud_Rate (%)"] = (
        summary["Fraud_Transactions"]
        / summary["Total_Transactions"]
        * 100
    )

    summary = summary.sort_values(
        "Fraud_Rate (%)",
        ascending=False
    )

    return summary

# --------------------------------------------------
# Feature Engineering 
# ---------------------------------------------------

def _prepare_data(df, user_col, time_col):
    """
    Helper function to clean and sort data
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([user_col, time_col])
    return df


def create_frequency_velocity_features(
    df,
    user_col="user_id",
    time_col="timestamp",
    amount_col="amount",
    windows=("1D", "7D", "30D")
):
    """
    Create transaction frequency and velocity features per user
    over multiple rolling time windows.
    """

    df = _prepare_data(df, user_col, time_col)

    results = []

    for window in windows:
        tmp = df.copy()
        tmp = tmp.set_index(time_col)

        # Frequency (count of transactions in window)
        freq = (
            tmp.groupby(user_col)[amount_col]
            .rolling(window)
            .count()
            .reset_index()
            .rename(columns={amount_col: f"txn_count_{window}"})
        )

        # Velocity (sum of transaction amounts in window)
        velocity = (
            tmp.groupby(user_col)[amount_col]
            .rolling(window)
            .sum()
            .reset_index()
            .rename(columns={amount_col: f"txn_velocity_{window}"})
        )

        merged = freq.merge(
            velocity,
            on=[user_col, time_col],
            how="left"
        )

        results.append(merged)

    # Merge all windows together
    final = results[0]
    for r in results[1:]:
        final = final.merge(r, on=[user_col, time_col], how="outer")

    return df.merge(final, on=[user_col, time_col], how="left")

def _prepare_time_data(df, time_col):
    """
    Helper function to ensure datetime format and validate column existence.
    """
    df = df.copy()

    # ✅ check if column exists
    if time_col not in df.columns:
        raise KeyError(
            f"Column '{time_col}' not found in dataframe. "
            f"Available columns: {list(df.columns)}"
        )

    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    return df


def create_time_features(
    df,
    time_col="purchase_time",
    add_hour=True,
    add_day_of_week=True
):
    """
    Create time-based features:
    - hour_of_day
    - day_of_week
    """

    df = _prepare_time_data(df, time_col)

    if add_hour:
        df["hour_of_day"] = df[time_col].dt.hour

    if add_day_of_week:
        df["day_of_week"] = df[time_col].dt.dayofweek

    return df

def _prepare_time_columns(df, signup_col, purchase_col):
    """
    Ensure datetime format for time-based calculations
    """
    df = df.copy()
    df[signup_col] = pd.to_datetime(df[signup_col])
    df[purchase_col] = pd.to_datetime(df[purchase_col])
    return df


def create_time_since_signup_feature(
    df,
    signup_col="signup_time",
    purchase_col="purchase_time",
    unit="days"
):
    """
    Create time_since_signup feature:
    duration between signup and purchase time.

    Parameters:
    - df : pandas.DataFrame
    - signup_col : str
    - purchase_col : str
    - unit : str ('seconds', 'minutes', 'hours', 'days')

    Returns:
    - DataFrame with new feature column
    """

    df = _prepare_time_columns(df, signup_col, purchase_col)

    # Calculate time difference
    time_diff = df[purchase_col] - df[signup_col]

    # Convert based on unit
    if unit == "seconds":
        df["time_since_signup"] = time_diff.dt.total_seconds()

    elif unit == "minutes":
        df["time_since_signup"] = time_diff.dt.total_seconds() / 60

    elif unit == "hours":
        df["time_since_signup"] = time_diff.dt.total_seconds() / 3600

    elif unit == "days":
        df["time_since_signup"] = time_diff.dt.days

    else:
        raise ValueError("Unsupported unit. Use: seconds, minutes, hours, days")

    return df

# --------------------------------------------------
# Data Transformation
# ---------------------------------------------------

def scale_numerical_features(
    df,
    columns=None,
    method="standard"
):
    """
    Scale numerical features using StandardScaler or MinMaxScaler.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : list, optional
        Columns to scale. If None, all numeric columns are used.
    method : str
        "standard" -> StandardScaler (mean=0, std=1)
        "minmax"   -> MinMaxScaler (0 to 1)

    Returns
    -------
    pd.DataFrame, scaler object
        Scaled DataFrame and fitted scaler.
    """

    data = df.copy()

    # Select numeric columns if not provided
    if columns is None:
        columns = data.select_dtypes(include="number").columns.tolist()

    # Choose scaler
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be 'standard' or 'minmax'")

    # Fit and transform
    data[columns] = scaler.fit_transform(data[columns])

    print(f"Scaled columns using {method} scaler:")
    print(columns)

    return data, scaler

def one_hot_encode_features(
    df,
    columns=None,
    drop_first=False
):
    """
    One-hot encode categorical features in a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    columns : list, optional
        List of categorical columns to encode.
        If None, all object/category columns are used.
    drop_first : bool
        Whether to drop the first category (to avoid multicollinearity).

    Returns
    -------
    pd.DataFrame
        DataFrame with one-hot encoded features.
    """

    data = df.copy()

    # Auto-detect categorical columns if not provided
    if columns is None:
        columns = data.select_dtypes(
            include=["object", "category"]
        ).columns.tolist()

    print(f"Encoding columns: {columns}")

    # One-hot encoding
    data = pd.get_dummies(
        data,
        columns=columns,
        drop_first=drop_first
    )

    return data

# --------------------------------------------------
# Handle Class Imbalance
# ---------------------------------------------------

def apply_smote_to_training_set(X_train, y_train, random_state=42, k_neighbors=5):
    """
    SMOTE-safe pipeline:
    - keep numeric only
    - impute missing values
    - apply SMOTE
    """

    # 1. Keep only numeric columns
    X_train = X_train.select_dtypes(include=["int64", "float64"])

    # 2. Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X_train_imputed = imputer.fit_transform(X_train)

    # 3. Apply SMOTE
    smote = SMOTE(
        random_state=random_state,
        k_neighbors=k_neighbors
    )

    X_resampled, y_resampled = smote.fit_resample(X_train_imputed, y_train)

    print("Original class distribution:")
    print(pd.Series(y_train).value_counts())

    print("\nResampled class distribution:")
    print(pd.Series(y_resampled).value_counts())

    return X_resampled, y_resampled