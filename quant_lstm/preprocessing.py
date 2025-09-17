import numpy as np
import pandas as pd

def _make_labels(close, lookahead=5, threshold=0.002, direction="UP"):
    """
    Convert a close-price series into binary 0/1 labels for either an UP or DOWN classifier.

    Parameters
    ----------
    close : pd.Series
        Series of closing prices, indexed by time.
    lookahead : int, default 5
        Number of periods in the future over which to compute the return.
    threshold : float, default 0.002
        Minimum percentage change (in decimal form, e.g. 0.002 ≅ 0.2%) 
        to consider a move “significant.”
    direction : {"UP", "DOWN"}, default "UP"
        - "UP":   label = 1 whenever forward return >= +threshold, else 0.
        - "DOWN": label = 1 whenever forward return <= -threshold, else 0.

    Returns
    -------
    np.ndarray (dtype=int8)
        1D array of length `len(close)`-`lookahead`, with 1 for “event” and 0 otherwise. 
        The last `lookahead` entries will always be 0 since there's no forward return to compute.
    """
    if direction not in ("UP", "DOWN"):
        raise ValueError("`direction` must be either 'UP' or 'DOWN'.")

    # Compute forward returns: (close[t + lookahead] / close[t]) - 1
    fwd_ret = close.pct_change(periods=lookahead).shift(-lookahead).to_numpy()

    # Initialize all labels to 0
    labels = np.zeros_like(fwd_ret, dtype=np.int8)

    if direction == "UP":
        # Label = 1 where return ≥ +threshold
        labels[fwd_ret >= threshold] = 1
    else:  # direction == "DOWN"
        # Label = 1 where return ≤ -threshold
        labels[fwd_ret <= -threshold] = 1

    return labels

def _check_nan_inf(arr, name):
    print(f"   {name}: NaNs -> {np.isnan(arr).any()}   Infs -> {np.isinf(arr).any()}")

def create_sequences(df, feature_cols, window_size=50,
                             lookahead=5, threshold=0.002, direction='UP', verbose=False):
    """
    Builds rolling sequences from any set of feature columns **and**
    auto-creates 0/1/2 future-move labels from 'close'.

    Parameters
    ----------
    df : pd.DataFrame (chronologically sorted)
    feature_cols : list[str]
        Columns to feed the model (can include 'close' plus indicators, volume, …).
    window_size : int
        Length of each sliding window.
    lookahead : int
        How many steps ahead to evaluate the future move for labelling.
    threshold : float
        ± percent threshold that defines “up”, “down”, “flat”.
    direction: str
        Used for making labels.
        - "UP":   label = 1 whenever forward return >= +threshold, else 0.
        - "DOWN": label = 1 whenever forward return <= -threshold, else 0.

    Returns
    -------
    X_seq : ndarray, shape (n_sequences, window_size, n_features)
    y_seq : ndarray, shape (n_sequences,)        # integer labels 0/1/2
    """
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain a 'close' column for labelling.")

    # -------- build labels --------
    y = _make_labels(df['close'], lookahead, threshold, direction=direction)

    # we must drop the last `lookahead` rows: their future return is NaN
    valid_length  = len(df) - lookahead
    feature_vals  = df[feature_cols].iloc[:valid_length].to_numpy()
    labels_trim   = y[:valid_length]

    n_samples, n_features = feature_vals.shape
    n_sequences = n_samples - window_size + 1
    if n_sequences <= 0:
        raise ValueError("Not enough rows for the chosen window_size.")

    # -------- roll into sequences --------
    X_seq = np.lib.stride_tricks.sliding_window_view(
                feature_vals, window_shape=(window_size, n_features)
            ).squeeze(axis=1)          # shape (n_sequences, window_size, n_features)
    y_seq = labels_trim[window_size-1:]  # align with last candle in each window

    # --- Verbose diagnostics ---
    if verbose:
        print(f"   [INFO] Created {X_seq.shape[0]} sequences of shape {X_seq.shape[1:]}")
        print(f"   [INFO] Label distribution: {np.bincount(y_seq)} (0s and 1s)")
        print(f"   [INFO] Checking for NaNs or infs:")
        _check_nan_inf(X_seq, "X_seq")
        _check_nan_inf(y_seq, "y_seq")
        print()

    return X_seq, y_seq

