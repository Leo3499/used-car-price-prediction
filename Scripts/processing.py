import pandas as pd
import numpy as np

def preprocess(df, drop_id=True, drop_year=True, add_log_price=True):
    df = df.copy()

    # Drop ID column
    if drop_id and 'ID' in df.columns:
        df.drop(columns=['ID'], inplace=True)

    # Create car_age from year
    if 'year' in df.columns:
        df['car_age'] = 2025 - df['year']
        if drop_year:
            df.drop(columns=['year'], inplace=True)

    # Drop rows with missing values (can adjust later)
    df.dropna(inplace=True)

    # === Encoding === #

    # 1. Frequency encoding for 'model'
    if 'model' in df.columns:
        freq_encoding = df['model'].value_counts().to_dict()
        df['model_freq'] = df['model'].map(freq_encoding)
        df.drop(columns=['model'], inplace=True)

    # 2. One-hot encoding for low-cardinality categorical columns
    one_hot_cols = ['brand', 'transmission', 'fuelType']
    existing_cols = [col for col in one_hot_cols if col in df.columns]

    df = pd.get_dummies(df, columns=existing_cols, drop_first=True)

        # Cap extreme price values to reduce outlier impact (especially for log transform)
    if 'price' in df.columns:
        lower = df['price'].quantile(0.01)
        upper = df['price'].quantile(0.99)
        df['price'] = df['price'].clip(lower, upper)

    # 3. Optional: add log-transformed target for linear/NN models
    if add_log_price and 'price' in df.columns:
        df['log_price'] = np.log1p(df['price'])

    return df