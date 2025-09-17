import numpy as np
import pandas as pd

def split_dataframe(df, train_ratio=0.7, val_ratio=0.15, verbose=True):
    """
    Splits a DataFrame into train, validation, and test sets based on ratios.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The full DataFrame to split.
    train_ratio : float
        Proportion of data to use for training.
    val_ratio : float
        Proportion of data to use for validation.
    verbose : bool
        If True, print shape and distribution information.

    Returns:
    --------
    df_train, df_val, df_test : pd.DataFrame
        The split dataframes.
    """
    assert train_ratio + val_ratio < 1.0, "Train + Val ratio must be less than 1."

    n = len(df)
    train_size = int(n * train_ratio)
    val_size = int(n * val_ratio)

    df_train = df.iloc[:train_size].copy()
    df_val = df.iloc[train_size:train_size + val_size].copy()
    df_test = df.iloc[train_size + val_size:].copy()

    if verbose:
        print(f"   [INFO] Train set shape: {df_train.shape}")
        print(f"   [INFO] Validation set shape: {df_val.shape}")
        print(f"   [INFO] Test set shape: {df_test.shape}")

    return df_train, df_val, df_test

import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler
import json

def normalize_splits(
    df_train, df_val, df_test,
    tail_column='volume',
    feature_range=(-1, 1),
    metadata_output_path='normalization_metadata.json'
):
    """
    Normalize train/val/test DataFrames with robust scaling for a heavy-tailed column
    and MinMax scaling for others, using scalers fit only on training data.

    Parameters:
    - df_train, df_val, df_test: DataFrames to normalize.
    - tail_column: Name of the column with heavy-tailed distribution (use RobustScaler).
    - feature_range: Tuple for MinMax scaling range for other columns.
    - metadata_output_path: File path to save JSON metadata of scaling parameters.

    Returns:
    - df_train_scaled, df_val_scaled, df_test_scaled: Scaled DataFrames
    - metadata: Dictionary of normalization parameters
    """
    # Make copies
    df_train = df_train.copy()
    df_val = df_val.copy()
    df_test = df_test.copy()

    # Separate columns
    other_columns = [col for col in df_train.columns if col != tail_column]

    # Initialize scalers
    robust_scaler = RobustScaler()
    minmax_scaler = MinMaxScaler(feature_range=feature_range)

    # Fit on training set only
    df_train[tail_column] = robust_scaler.fit_transform(df_train[[tail_column]])
    df_val[tail_column] = robust_scaler.transform(df_val[[tail_column]])
    df_test[tail_column] = robust_scaler.transform(df_test[[tail_column]])

    df_train[other_columns] = minmax_scaler.fit_transform(df_train[other_columns])
    df_val[other_columns] = minmax_scaler.transform(df_val[other_columns])
    df_test[other_columns] = minmax_scaler.transform(df_test[other_columns])

    # Build metadata
    metadata = {
        "scaling_type": {
            tail_column: "robust",
            "other_features": "minmax"
        },
        "feature_range": feature_range,
        "features": {}
    }

    metadata["features"][tail_column] = {
        "scaler": "robust",
        "center": float(robust_scaler.center_[0]),
        "scale": float(robust_scaler.scale_[0])
    }

    for idx, col in enumerate(other_columns):
        metadata["features"][col] = {
            "scaler": "minmax",
            "min": float(minmax_scaler.data_min_[idx]),
            "max": float(minmax_scaler.data_max_[idx])
        }

    # Save metadata
    with open(metadata_output_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    return df_train, df_val, df_test, metadata



########################## Feature Functions ##########################

def compute_SMA(df, window=20):
    """
    Compute the Simple Moving Average (SMA) for a given DataFrame.

    Note:
        - The first (window - 1) rows will contain NaN values for the SMA because 
          there is not enough data to compute a rolling window.
    """
    # Compute the Simple Moving Average (SMA)
    df['SMA'] = df['close'].rolling(window=window, min_periods=1).mean()

    return df

def compute_MACD(df, fast=12, slow=26, signal=9):
    """
    Compute the Moving Average Convergence Divergence (MACD) indicator.

    This function calculates the MACD line, signal line, and histogram
    and appends the results to the given DataFrame.

    Parameters:
    -----------
    df : pandas.DataFrame
        A DataFrame containing a 'close' column, which represents the closing prices 
        of a financial instrument.
    fast : int, optional
        The short-term EMA period for MACD calculation. Default is 12.
    slow : int, optional
        The long-term EMA period for MACD calculation. Default is 26.
    signal : int, optional
        The signal line EMA period for MACD calculation. Default is 9.
            
    """
    # Compute MACD using pandas-ta. This appends MACD columns to the DataFrame.
    #df.ta.macd(close='close', fast=fast, slow=slow, signal=signal, append=True)

        # Compute the fast and slow exponential moving averages with min_periods=1
    fast_ema = df['close'].ewm(span=fast, min_periods=1, adjust=False).mean()
    slow_ema = df['close'].ewm(span=slow, min_periods=1, adjust=False).mean()

    # Compute MACD as the difference between the fast and slow EMA
    df['MACD'] = fast_ema - slow_ema

    # Compute the signal line using the MACD, again with min_periods=1
    df['MACD_signal'] = df['MACD'].ewm(span=signal, min_periods=1, adjust=False).mean()

    # Calculate the MACD histogram
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']

    return df

def compute_bollinger_bands(df, window=20, num_std=2, fill_method='bfill'):
    """
    Compute Bollinger Bands for a given DataFrame.

    This function calculates the upper and lower Bollinger Bands for the 'close' column
    in the provided DataFrame. Bollinger Bands are a technical analysis tool that help
    identify periods of high or low volatility. The bands are calculated using a simple
    moving average (SMA) and the rolling standard deviation of the 'close' prices.

    Note:
        - The first (window - 1) rows will contain NaN values for the Bollinger Bands because
          there is not enough data to compute a rolling window.
    """
    """
    Compute Bollinger Bands for the 'close' column in the provided DataFrame.

    Parameters:
    - df: DataFrame with a 'close' price column.
    - window: Rolling window size (default 20).
    - num_std: Number of standard deviations for the bands (default 2).
    - fill_method: How to fill initial NaNs: 'bfill' (default) or 'ffill'.

    Returns:
    - DataFrame with added 'Bollinger_Upper' and 'Bollinger_Lower' columns.
    """
    df = df.copy()

    # Rolling statistics
    sma = df['close'].rolling(window=window, min_periods=1).mean()
    rolling_std = df['close'].rolling(window=window, min_periods=1).std()

    # Bands
    df['Bollinger_Upper'] = sma + (rolling_std * num_std)
    df['Bollinger_Lower'] = sma - (rolling_std * num_std)

    # Fill initial NaNs
    if fill_method == 'bfill':
        df['Bollinger_Upper'] = df['Bollinger_Upper'].bfill()
        df['Bollinger_Lower'] = df['Bollinger_Lower'].bfill()
    elif fill_method == 'ffill':
        df['Bollinger_Upper'] = df['Bollinger_Upper'].ffill()
        df['Bollinger_Lower'] = df['Bollinger_Lower'].ffill()
    else:
        raise ValueError("fill_method must be 'bfill' or 'ffill'")

    return df


def compute_stochastic_oscillator(df,  k_period=14, d_period=3):
    """
    Calculate the Stochastic Oscillator (%K and %D).

    The %K value is calculated using:
        %K = (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    over the specified lookback period (k_period). The %D value is a simple moving 
    average of the %K values over the last d_period periods.

    Returns:
        tuple: Two lists containing %K and %D values respectively.
            For indices where there is insufficient data, the values will be None.
    """
    highs = df['high']
    lows = df['low']
    closes = df['close']
    
    # Ensure all input lists are of the same length.
    if not (len(highs) == len(lows) == len(closes)):
        raise ValueError("Highs, lows, and closes must have the same length.")
        
    k_values = []
        
    # Calculate %K for each data point where enough data is available.
    for i in range(len(closes)):
        if i < k_period - 1:
            k_values.append(None)
        else:
            period_high = max(highs[i - k_period + 1:i + 1])
            period_low = min(lows[i - k_period + 1:i + 1])
            # Avoid division by zero.
            if period_high - period_low == 0:
                k = 0
            else:
                k = (closes[i] - period_low) / (period_high - period_low) * 100
            k_values.append(k)
        
    # Calculate %D as the simple moving average of %K values.
    d_values = [None] * len(k_values)
    for i in range(len(k_values)):
        # We need at least d_period values of %K to compute %D.
        if i < k_period - 1 + d_period - 1:
            d_values[i] = None
        else:
            valid_k = [k for k in k_values[i - d_period + 1:i + 1] if k is not None]
            d_values[i] = sum(valid_k) / len(valid_k) if valid_k else None

    df['k_values'] = k_values
    df['d_values'] = d_values

    # Backfill each column where you have missing values
    df['k_values'] = df['k_values'].bfill()
    df['d_values'] = df['d_values'].bfill()
    return df

import pandas as pd

def add_returns(df: pd.DataFrame,
                price_col: str = 'close',
                return_col: str = 'return_pct',
                drop_na: bool = True
               ):
    """
    Compute simple percent‐change returns from one row to the next.

    Parameters
    ----------
    df : (pd.DataFrame) Input DataFrame, must contain `price_col`.
    price_col : (str) Name of the column holding prices.
    return_col : (str) Name to give the new returns column.
    drop_na : (bool) Whether to drop the first row (which will have NaN return).

    Returns
    -------
    (pd.DataFrame) A new DataFrame with the added `return_col`.
    """
    df2 = df.copy()
    # percent change: (P_t - P_{t-1}) / P_{t-1}
    df2[return_col] = df2[price_col].pct_change()
    if drop_na:
        df2 = df2.iloc[1:].reset_index(drop=True)
    return df2

import pandas as pd

def process_timestamp_column(df, timestamp_col='timestamp', use_day_of_week=True):
    """
    Processes the timestamp column in a DataFrame.
    
    If `use_day_of_week` is True, replaces the timestamp column with the day of the week (0=Monday, 6=Sunday).
    Otherwise, drops the timestamp column.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with a timestamp column.
    - timestamp_col (str): Name of the timestamp column. Default is 'timestamp'.
    - use_day_of_week (bool): Whether to convert timestamps to day of week. If False, timestamp is dropped.

    Returns:
    - pd.DataFrame: Updated DataFrame with timestamp processed.
    """
    df = df.copy()

    # Ensure datetime type
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])

    # Ensure sorted by timestamp
    if not df[timestamp_col].is_monotonic_increasing:
        print(f"   [INFO] Timestamps are NOT sorted. Sorting now...")
        df = df.sort_values(by=timestamp_col, ascending=True)

    # Process timestamp
    if use_day_of_week:
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df = df.drop(columns=[timestamp_col])
    else:
        df = df.drop(columns=[timestamp_col])

    return df


########################## More functions ##########################

def clip_outliers_percentile(df, lower_pct=0.01, upper_pct=0.99, verbose=False):
    """
    Clips outlier values in each column of a DataFrame based on percentiles.

    Values below the specified lower percentile are set to the value at that percentile,
    and values above the upper percentile are set to the value at that percentile.
    This operation preserves the shape of the DataFrame and helps mitigate the influence
    of extreme outliers during normalization or model training.

    Parameters
    ----------
    df : The input DataFrame containing numerical features to clip.
    lower_pct : The lower percentile threshold (between 0 and 1) below which values will be clipped.
    upper_pct : The upper percentile threshold (between 0 and 1) above which values will be clipped.
    
    """
    df_clipped = df.copy()
    for col in df_clipped.columns:
        lower = df_clipped[col].quantile(lower_pct)
        upper = df_clipped[col].quantile(upper_pct)
        df_clipped[col] = df_clipped[col].clip(lower, upper)

    if verbose:
        print(f'   Data Clipping between {100*lower_pct:.2f}% and {100*upper_pct:.2f}%')

    return df_clipped


import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import pandas as pd
import matplotlib.patheffects as pe

def plot_normalization_qc(
    df_raw,
    df_norm,
    features: list,
    pdf_path: str = 'normalization_qc.pdf',
    bins: int = 50,
    use_kde: bool = True
):
    """
    Generate a PDF with one page per feature showing side-by-side histograms
    (and optionally KDE curves) of the raw vs. normalized distributions,
    annotated with mean and standard deviation.

    Parameters
    ----------
    df_raw : pd.DataFrame
        DataFrame containing raw, unnormalized feature values.
    df_norm : pd.DataFrame
        DataFrame containing normalized feature values.
    features : list of str
        List of feature names to include in the PDF.
    pdf_path : str, optional
        Output path for the generated PDF file.
    bins : int, optional
        Number of histogram bins to use.
    use_kde : bool, optional
        Whether to overlay KDE curve on each histogram. Default is False.
    """
    with PdfPages(pdf_path) as pdf:
        for feat in features:
            raw_data = df_raw[feat].dropna()
            norm_data = df_norm[feat].dropna()

            # Compute stats
            raw_mean, raw_std = raw_data.mean(), raw_data.std()
            norm_mean, norm_std = norm_data.mean(), norm_data.std()

            raw_xlim = (raw_data.min(), raw_data.max())
            norm_xlim = (norm_data.min(), norm_data.max())

            fig, axes = plt.subplots(1, 2, figsize=(12, 4), tight_layout=True)

            # Raw plot
            if use_kde:
                #sns.histplot(raw_data, bins=bins, kde=True, ax=axes[0], color='skyblue')

                sns.histplot(raw_data, bins=bins, kde=True, ax=axes[0], color="skyblue", edgecolor="black", linewidth=1.0)
                kde_line = axes[0].lines[-1]
                kde_line.set_path_effects([
                    pe.Stroke(linewidth=3.6, foreground="black"),  # thick black behind
                    pe.Normal()                                 # then the original skyblue line
                ])
            else:
                axes[0].hist(raw_data, bins=bins, alpha=0.7, color='skyblue', edgecolor="black")
            axes[0].axvline(raw_mean, color='red', linestyle='--', linewidth=1)
            axes[0].set_title(f"{feat} (Raw)")
            axes[0].set_xlim(raw_xlim)
            axes[0].legend([f"Mean={raw_mean:.2f}, Std={raw_std:.2f}"])
            axes[0].set_xlabel(feat)
            axes[0].set_ylabel("Count")

            # Normalized plot
            if use_kde:
                sns.histplot(norm_data, bins=bins, kde=True, ax=axes[1], color="skyblue", edgecolor="black", linewidth=1.0)
                kde_line = axes[1].lines[-1]
                kde_line.set_path_effects([
                    pe.Stroke(linewidth=3.6, foreground="black"),  # thick black behind
                    pe.Normal()                                 # then the original skyblue line
                ])
            else:
                axes[1].hist(norm_data, bins=bins, alpha=0.7, color='skyblue', linewidth=0.7, edgecolor="black")
            axes[1].axvline(norm_mean, color='red', linestyle='--', linewidth=1)
            axes[1].set_title(f"{feat} (Normalized)")
            axes[1].set_xlim(norm_xlim)
            axes[1].legend([f"Mean={norm_mean:.2f}, Std={norm_std:.2f}"])
            axes[1].set_xlabel(feat)
            axes[1].set_ylabel("Count")

            # Suptitle
            fig.suptitle(f"Feature: {feat}", fontsize=14)
            pdf.savefig(fig)
            plt.close(fig)




