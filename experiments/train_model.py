import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from quant_lstm.data_loader import load_data, prepare_results_dirs
from quant_lstm.preprocessing import create_sequences
from quant_lstm.train import compute_class_weights, run_training
from quant_lstm.model import create_binary_categorical_model
from quant_lstm.evaluate import (
    get_prediction_outputs,
    generate_diagnostics_figures,
    generate_training_report
)
import quant_lstm.feature_engineering as fe

import yaml
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import random

def main(config_path="config.yaml"):

    ##########################################################################

    # ________________________________________________________________________
    # _________________ Load Config File _____________________________________

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    verbose = config.get("general", {}).get("verbose", False)
    print("⚙️ Config file loaded.")

    # ________________________________________________________________________
    # _________________ Global Variables _____________________________________

    # --- Set random seed ---
    seed = config["general"].get("seed", 42)  # default to 42 if missing
    random.seed(seed) # Apply seed everywhere

    # --- Paths / run info ---
    basename = config["run"]["basename"]

    # --- Data files ---
    raw_data_path = config["data"]["raw_data_path"]
    raw_data_filename = config["data"]["raw_data_filename"]

    # --- Preprocessing ---
    clipping_lower_pct = config["preprocessing"]["clipping"]["lower_pct"]
    clipping_upper_pct = config["preprocessing"]["clipping"]["upper_pct"]
    normalization_feature_range = tuple(config["preprocessing"]["normalization"]["feature_range"])

    # --- Splits ---
    train_ratio = config["split"]["train"]
    val_ratio = config["split"]["val"]

    # --- General ---
    verbose = config.get("general", {}).get("verbose", False)

    # --- Results dirs ---
    ROOT = Path(__file__).resolve().parents[1]

    dirs = prepare_results_dirs(ROOT, basename)
    print("📁 Results will be saved in:", dirs["results"])
    print("📁 Figures will be saved in:", dirs["figures"])
    print("📁 Modified data will be saved in:", dirs["modified_data"])

    # ________________________________________________________________________
    # _________________ Load Data ____________________________________________

    OHLC_df_path = ROOT / raw_data_path / raw_data_filename
    df = load_data(OHLC_df_path)
    print(f"📂 OHLC data loaded. Shape: {df.shape}")

    # ________________________________________________________________________
    # _________________ Compute Functions ____________________________________

    sma_window = 20
    fast = 12
    slow = 26
    signal = 9
    num_std = 2
    k_period = 14
    d_period = 3

    df = fe.compute_SMA(df, window=sma_window)
    df = fe.compute_MACD(df, fast=fast, slow=slow, signal=signal)
    df = fe.compute_bollinger_bands(df, window=sma_window, num_std=num_std)
    df = fe.compute_stochastic_oscillator(df,  k_period=k_period, d_period=d_period)
    df = fe.add_returns(df, price_col='close', return_col='return_pct')
    df = fe.process_timestamp_column(df, timestamp_col='timestamp', use_day_of_week=True)
    print("🧮 Technical indicators (SMA, MACD, Bollinger, etc.) computed.")

    # ________________________________________________________________________
    # _________________ Clip Data ____________________________________________

    df = fe.clip_outliers_percentile(df, lower_pct=clipping_lower_pct, upper_pct=clipping_upper_pct)
    print("✂️ Outliers clipped.")

    # ________________________________________________________________________
    # _________________ Save Clipped Data ____________________________________

    save_path_clipped_data = dirs["modified_data"] / f"{basename}_data_features_clipped.csv"
    df.to_csv(save_path_clipped_data, index=False)
    print(f"💾 Clipped feature data saved to {save_path_clipped_data}")

    # ________________________________________________________________________
    # _________________ Split the Data _______________________________________

    df_train, df_val, df_test = fe.split_dataframe(df, train_ratio=train_ratio, val_ratio=val_ratio, verbose=verbose)
    print("🔀 Dataset split into train / val / test.")

    # ________________________________________________________________________
    # _________________ Normalize Data _______________________________________

    save_path_norm_meta = dirs["modified_data"] / f"{basename}_normalization_meta_data.json"
    df_train_scaled, df_val_scaled, df_test_scaled, norm_meta = fe.normalize_splits(
        df_train,
        df_val,
        df_test,
        tail_column="volume",
        feature_range=normalization_feature_range,
        metadata_output_path=save_path_norm_meta
    )
    print(f"📊 Data normalized. Metadata saved to {save_path_norm_meta}")

    # ________________________________________________________________________
    # _________________ Check Distributions __________________________________

    print("📊 Creating feature distribution plots...")

    features = list(df_train_scaled.columns)

    # --- Distribution QC Plots ---
    save_path_train_fig = dirs["figures"] / f"{basename}_feature_norm_comparison_train.pdf"
    fe.plot_normalization_qc(
        df_raw=df_train,
        df_norm=df_train_scaled,
        features=features,
        pdf_path=save_path_train_fig,
        bins=50
    )

    save_path_val_fig = dirs["figures"] / f"{basename}_feature_norm_comparison_val.pdf"
    fe.plot_normalization_qc(
        df_raw=df_val,
        df_norm=df_val_scaled,
        features=features,
        pdf_path=save_path_val_fig,
        bins=50
    )

    save_path_test_fig = dirs["figures"] / f"{basename}_feature_norm_comparison_test.pdf"
    fe.plot_normalization_qc(
        df_raw=df_test,
        df_norm=df_test_scaled,
        features=features,
        pdf_path=save_path_test_fig,
        bins=50
    )

    print(f"✅ Distribution plots saved to {dirs['figures']}")

    # ________________________________________________________________________
    # _________________ Save Normalized Data _________________________________

    save_path_train = dirs["modified_data"] / f"{basename}_data_features_clipped_normalized_train.csv"
    save_path_val   = dirs["modified_data"] / f"{basename}_data_features_clipped_normalized_val.csv"
    save_path_test  = dirs["modified_data"] / f"{basename}_data_features_clipped_normalized_test.csv"

    df_train_scaled.to_csv(save_path_train, index=False)
    df_val_scaled.to_csv(save_path_val, index=False)
    df_test_scaled.to_csv(save_path_test, index=False)

    print(f"💾 Normalized train/val/test data saved to {dirs['modified_data']}")
    print("✨ Feature engineering finished!")

    ##########################################################################

    # ________________________________________________________________________
    # _________________ Create Sequences _____________________________________

    X_train, y_train = create_sequences(
        df=df_train_scaled,
        feature_cols=config['data']['feature_cols'],
        window_size=config['data']['window_size'],
        lookahead=config['data']['lookahead'],
        threshold=config['data']['threshold'],
        direction=config['data']['direction'],
        verbose=verbose
    )

    X_val, y_val = create_sequences(
        df=df_val_scaled,
        feature_cols=config['data']['feature_cols'],
        window_size=config['data']['window_size'],
        lookahead=config['data']['lookahead'],
        threshold=config['data']['threshold'],
        direction=config['data']['direction'],
        verbose=verbose
    )

    X_test, y_test = create_sequences(
        df=df_test_scaled,
        feature_cols=config['data']['feature_cols'],
        window_size=config['data']['window_size'],
        lookahead=config['data']['lookahead'],
        threshold=config['data']['threshold'],
        direction=config['data']['direction'],
        verbose=verbose
    )
    print("🧩 Training/validation/test sequences created.")

    # ________________________________________________________________________
    # _________________ Calculate Class Weights ______________________________

    manual_weights = config.get("training", {}).get("class_weight_override")
    class_weight = compute_class_weights(y_train, override=manual_weights, verbose=verbose)
    print("⚖️ Class weights calculated.")

    # ________________________________________________________________________
    # _________________ Create the Model _____________________________________

    model = create_binary_categorical_model(timesteps=config['data']['window_size'],
                                            features=len(config['data']['feature_cols']), 
                                            units=config['model']['units'], 
                                            dropout_rate=config['model']['dropout_rate'],
                                            learning_rate=config['model']['learning_rate'],
                                            l2_lambda=config['model']['l2_lambda'],
                                            verbose=verbose)
    print("🏗️ Model architecture created.")

    # ________________________________________________________________________
    # _________________ Train the Model ______________________________________

    print("🚀 Model training started...")
    save_path_model_best_weights = ROOT / "results" / basename / f"{basename}_best_weights.keras"
    model, history = run_training(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=config['training']["epochs"],
        batch_size=config['training']["batch_size"],
        early_stop_patience=config['training']["early_stop_patience"],
        class_weight=class_weight,
        save_path=save_path_model_best_weights,
        verbose=verbose
    )
    print("✅ Model training completed.")

    final_model_path = ROOT / "results" / basename / f"{basename}_final_model.keras"
    model.save(final_model_path)
    print(f"💾 Final model saved → {final_model_path}")

    # Evaluate the model (validation)
    y_prob, y_pred, precision, recall, thresholds = get_prediction_outputs(
    model, X_val, y_val, default_threshold=0.5)

    # Evaluate the model (test)
    y_prob_test, y_pred_test, precision_test, recall_test, thresholds_test = get_prediction_outputs(
    model, X_test, y_test, default_threshold=0.5)

    diag_pdf_path = dirs["figures"] / f"{basename}_combined_diagnostics.pdf"
    generate_diagnostics_figures(
        y_val, y_prob, y_pred,
        y_test, y_prob_test, y_pred_test,
        precision, recall, thresholds,
        precision_test, recall_test, thresholds_test,
        history,
        output_path=diag_pdf_path
    )
    print(f"📈 Diagnostic figures generated → {diag_pdf_path}")

    # --- Training Report ---
    report_path = dirs["results"] / f"{basename}_training_report.json"
    report = generate_training_report(
        y_val, y_prob, y_pred,
        y_test, y_prob_test, y_pred_test,
        val_loss=history.history['val_loss'],
        target_recall_1=0.3,
        target_recall_2=0.6,
        verbose=True,
        save_path=report_path
    )
    print(f"📝 Training report generated → {report_path}")
    print("🎉 Full pipeline run completed successfully!")


if __name__ == "__main__":
    config_name = "config.yaml"
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        if not config_name.endswith(".yaml"):
            config_name += ".yaml"

    ROOT = Path(__file__).resolve().parents[1]
    config_path = ROOT / "quant_lstm" / "configs" / config_name

    main(config_path)
