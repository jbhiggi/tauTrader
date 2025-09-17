from pathlib import Path
import tensorflow as tf
from quant_lstm.model import create_binary_categorical_model

def load_models(config, basename, ROOT, verbose=0):
    """
    Load both the final saved model and the best-weights-restored model.

    Parameters
    ----------
    config : dict
        Configuration dictionary used to build the model.
    basename : str
        Base name of the run (used for file paths).
    ROOT : Path
        Root directory of the project.
    verbose : int, optional
        Verbosity level passed to model creation.

    Returns
    -------
    final_model : tf.keras.Model
        The fully saved final model.
    best_model : tf.keras.Model
        The model reconstructed from config and loaded with best weights.
    """

    # Paths
    final_model_path = ROOT / "results" / basename / f"{basename}_final_model.keras"
    best_weights_path = ROOT / "results" / basename / f"{basename}_best_weights.keras"

    # 1. Load final saved model directly
    final_model = tf.keras.models.load_model(final_model_path)

    # 2. Recreate architecture and load best weights
    best_model = create_binary_categorical_model(
        timesteps=config['data']['window_size'],
        features=len(config['data']['feature_cols']),
        units=config['model']['units'],
        dropout_rate=config['model']['dropout_rate'],
        learning_rate=config['model']['learning_rate'],
        l2_lambda=config['model']['l2_lambda'],
        verbose=verbose
    )
    best_model.load_weights(best_weights_path)

    return final_model, best_model

import json
from pathlib import Path

def load_linear_fit_coeffs(save_path_json):
    save_path_json = Path(save_path_json)
    with open(save_path_json, "r") as f:
        coeffs = json.load(f)
    return coeffs

def map_tau_val_to_test(tau_val, coeffs):
    a, b = coeffs["a"], coeffs["b"]
    return a * tau_val + b