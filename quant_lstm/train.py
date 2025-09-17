import numpy as np
from sklearn.utils.class_weight import compute_class_weight

def compute_class_weights(y, override=None, verbose=False):
    """
    Computes balanced class weights or uses a provided override.
    
    Parameters:
    - y: 1D array-like, the training labels
    - override: Optional dict of manual class weights
    - verbose: Whether to print the weight summary
    
    Returns:
    - class_weight: dict mapping class index to weight
    """
    if override is not None:
        if verbose:
            print("   [INFO] Using manually specified class weights:")
            for k, v in override.items():
                print(f"       {k}: {v:.4f}")
            print()
        return override

    y = np.asarray(y).squeeze()
    class_ids = np.unique(y)
    class_wt = compute_class_weight(class_weight='balanced', classes=class_ids, y=y)
    class_weight = dict(zip(class_ids, class_wt))

    if verbose:
        print("   [INFO] Computed class weights:")
        for key in sorted(class_weight):
            print(f"       {key}: {class_weight[key]:.4f}")
        print()
    
    return class_weight

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import tensorflow as tf

def _train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, early_stop_patience, class_weight, save_path, verbose=False):
    """
    Fit a compiled Keras model on time-series data.

    Parameters
    ----------
    model : tf.keras.Model
    X_train, y_train : np.ndarray
    X_val, y_val     : np.ndarray
    params : dict
        Must contain 'epochs' and 'batch_size'.
    early_stop_patience : int, default 8
        Epochs with no val_loss improvement before stopping.
    class_weight : dict or None
        Optional per-class weight mapping.
    save_path : str
        basename used to store best weights (ModelCheckpoint).
    verbose : int {0,1,2}

    Returns
    -------
    history : tf.keras.callbacks.History
    best_val_loss : float
    """
    
    if verbose:
        verbose_int = 1
    else:
        verbose_int = 0

    # ── callback that logs lr each epoch ────────────────────────────
    class LrLogger(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            # .learning_rate may be a tf.Variable or a schedule; call .numpy()
            # (or tf.keras.backend.get_value) to turn it into a Python scalar
            logs['lr'] = tf.keras.backend.get_value(
                            self.model.optimizer.learning_rate)
        
    callbacks = [
        EarlyStopping(patience=early_stop_patience, restore_best_weights=True),
        ReduceLROnPlateau(patience=4, factor=0.5, verbose=verbose_int),
        ModelCheckpoint(save_path, save_best_only=True, monitor="val_loss"),
        LrLogger()
    ]

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_val, y_val),
        shuffle=False,  # Important for time series data
        callbacks=callbacks,
        class_weight=class_weight,
        verbose=verbose_int
    )

    return history

def run_training(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    epochs,
    batch_size,
    early_stop_patience,
    class_weight,
    save_path,
    verbose=False
):
    """
    Wrapper that trains the model and logs timing and progress info.
    
    Returns:
        model: the trained model
        history: the training history object
    """
    import time

    if verbose:
        print("\n" + "="*40)
        print("[STEP] Starting model training")
        print("="*40 + "\n")

    start_time = time.time()

    history = _train_model(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        early_stop_patience=early_stop_patience,
        class_weight=class_weight,
        save_path=save_path,
        verbose=verbose
    )

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n[INFO] Training completed in {elapsed:.2f} seconds\n")

    return model, history
