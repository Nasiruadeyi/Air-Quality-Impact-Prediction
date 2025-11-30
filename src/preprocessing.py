import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_data(filepath="../AirQualityData.csv"):
    """
    Loads the air quality dataset stored in the repo root directory.
    """
    df = pd.read_csv(filepath)
    return df


def preprocess_data(df):
    """
    Cleans and preprocesses the dataset:
    - Standardizes column names
    - Handles missing values
    - Encodes categorical data
    - Splits into train and test sets
    """

    # Clean column names
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    # Handle missing values
    df = df.fillna(df.median(numeric_only=True))

    # Encode categorical features
    categorical_cols = df.select_dtypes(include=["object"]).columns
    if len(categorical_cols) > 0:
        encoder = LabelEncoder()
        for col in categorical_cols:
            df[col] = encoder.fit_transform(df[col].astype(str))

    # Define target
    if "healthimpactscore" not in df.columns:
        raise ValueError("Target column 'HealthImpactScore' not found in dataset.")

    X = df.drop("healthimpactscore", axis=1)
    y = df["healthimpactscore"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test, df

