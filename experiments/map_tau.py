import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import sys
from pathlib import Path
import yaml
import json
from pathlib import Path
import pandas as pd
import numpy as np
import random

import warnings
warnings.filterwarnings(
    "ignore",
    message="Skipping variable loading for optimizer",
    category=UserWarning
)

from quant_lstm.map_tau_utils import load_models, load_linear_fit_coeffs, map_tau_val_to_test
from quant_lstm.preprocessing import create_sequences
from quant_lstm.evaluate import get_prediction_outputs
from quant_lstm.figures import plot_pr_curve_with_thresholds

if __name__ == "__main__":
    config_name = "config.yaml"
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"

    try:
        ROOT = Path(__file__).resolve().parents[1]
    except NameError:
        ROOT = Path.cwd().resolve().parents[0]

    config_path = ROOT / "quant_lstm" / "configs" / config_name

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    verbose = config.get("general", {}).get("verbose", False)
    print("⚙️ Config file loaded.")

    # --- Set random seed ---
    seed = config["general"].get("seed", 42)  # default to 42 if missing
    random.seed(seed) # Apply seed everywhere

    basename = config["run"]["basename"]
    tau_model_choice = config["tau_mapping"]["model"]
    thr_point = config["tau_mapping"]["thr_point"] # Threshold point to select desired validation set PR-curve point
    prec_point = config["tau_mapping"]["prec_point"] # Precision value to select desired validation set PR-curve point
    rec_point = config["tau_mapping"]["rec_point"] # Recall value to select desired validation set PR-curve point

    tau_val = config["tau_mapping"]["tau_validation_mapping_value"] # Value chosen from the PR curve to map validation set to test set

    # ________________________________________________________________________
    # _________________ Load Data ____________________________________________

    final_model, best_model = load_models(config, basename, ROOT, verbose=0)

    if tau_model_choice == "best_model":
        model = best_model
        print("🏆 Using best model for tau mapping")
    elif tau_model_choice == "final_model":
        model = final_model
        print("📦 Using final model for tau mapping")
    else:
        raise ValueError(f"❌ Unknown tau_mapping.model: {tau_model_choice}")

    save_path_train = ROOT / "results" / basename / "modified_data" / f"{basename}_data_features_clipped_normalized_train.csv"
    save_path_val   = ROOT / "results" / basename / "modified_data" / f"{basename}_data_features_clipped_normalized_val.csv"
    save_path_test  = ROOT / "results" / basename / "modified_data" / f"{basename}_data_features_clipped_normalized_test.csv"

    df_train_scaled = pd.read_csv(save_path_train)
    df_val_scaled   = pd.read_csv(save_path_val)
    df_test_scaled  = pd.read_csv(save_path_test)

    print("📂 Data loaded")

    # ________________________________________________________________________
    # _________________ Create Sequences _____________________________________


    X_train, y_train = create_sequences(
        df=df_train_scaled,
        feature_cols=config['data']['feature_cols'],
        window_size=config['data']['window_size'],
        lookahead=config['data']['lookahead'],
        threshold=config['data']['threshold'],
        direction=config['data']['direction'],
        verbose=False
    )

    X_val, y_val = create_sequences(
        df=df_val_scaled,
        feature_cols=config['data']['feature_cols'],
        window_size=config['data']['window_size'],
        lookahead=config['data']['lookahead'],
        threshold=config['data']['threshold'],
        direction=config['data']['direction'],
        verbose=False
    )

    X_test, y_test = create_sequences(
        df=df_test_scaled,
        feature_cols=config['data']['feature_cols'],
        window_size=config['data']['window_size'],
        lookahead=config['data']['lookahead'],
        threshold=config['data']['threshold'],
        direction=config['data']['direction'],
        verbose=False
    )
    print("🔗 Sequences created")

    # ________________________________________________________________________
    # _________________ Peform tau mapping ___________________________________

    print("📐 Tau mapping") 

    # Evaluate the model (validation)
    y_prob, y_pred, precision, recall, thresholds = get_prediction_outputs(
        model, X_val, y_val, default_threshold=0.5)

    # Evaluate the model (test)
    y_prob_test, y_pred_test, precision_test, recall_test, thresholds_test = get_prediction_outputs(
        model, X_test, y_test, default_threshold=0.5)

    # Create PR curve figure for validation set
    pr_curve_fig_val = plot_pr_curve_with_thresholds(
            prec=precision,
            rec=recall,
            thresholds=thresholds,
            thr_point=thr_point,
            prec_point=prec_point,
            rec_point=rec_point,
            base_rate=np.mean(y_val),
            class_label="Validation",
            show=False,
            return_fig=True
        )
    save_path_pr_curve_val_tau_mapping = ROOT / "master_figures" / f"{basename}_pr_curve_val_tau_mapping.png"
    pr_curve_fig_val.savefig(save_path_pr_curve_val_tau_mapping, dpi=300, bbox_inches="tight")
    save_path_pr_curve_val_tau_mapping = ROOT / "results" / basename / "figures" / f"{basename}_pr_curve_val_tau_mapping.pdf"
    pr_curve_fig_val.savefig(save_path_pr_curve_val_tau_mapping, dpi=300, bbox_inches="tight")

    # Load coefficients
    coeffs = load_linear_fit_coeffs(ROOT / "results" / basename / f"{basename}_linear_fit.json")

    # Map a single tau
    tau_test = map_tau_val_to_test(tau_val, coeffs)
    print(f"τ_val = {tau_val:.3f} → τ_test = {tau_test:.3f}")

    # Create PR curve figure for test set with mapping result
    pr_curve_fig_test = plot_pr_curve_with_thresholds(
            prec=precision_test,
            rec=recall_test,
            thresholds=thresholds_test,
            thr_point=tau_test,
            prec_point=None,
            rec_point=None,
            base_rate=np.mean(y_test),
            class_label="Test",
            show=False,
            return_fig=True
        )
    save_path_pr_curve_test_tau_mapping = ROOT / "master_figures" / f"{basename}_pr_curve_test_tau_mapping.png"
    pr_curve_fig_test.savefig(save_path_pr_curve_test_tau_mapping, dpi=300, bbox_inches="tight")
    save_path_pr_curve_test_tau_mapping = ROOT / "results" / basename / "figures" / f"{basename}_pr_curve_test_tau_mapping.pdf"
    pr_curve_fig_test.savefig(save_path_pr_curve_test_tau_mapping, dpi=300, bbox_inches="tight")

    # ________________________________________________________________________
    # _________ Calculate final performance without mapping __________________

    # Find improvement with niave tau (not using mapping) for comparison
    idx_naive = np.argmin(np.abs(thresholds_test - tau_val)) # Find the index of the closest threshold
    prec_at_tau_naive = precision_test[idx_naive + 1] # Precision and recall corresponding to this threshold
    rec_at_tau_naive  = recall_test[idx_naive + 1]
    print(f"📊 [WITHOUT MAPPING] At threshold {tau_val:.3f}: precision={prec_at_tau_naive:.3f}, recall={rec_at_tau_naive:.3f}")
    print(f"📊 [WITHOUT MAPPING] Model achieves a perforance relative to baseline of {100*(prec_at_tau_naive-np.mean(y_test)):.2f}%")

    # Create PR curve figure for test set WITHOUT mapping result
    pr_curve_fig_test_naive = plot_pr_curve_with_thresholds(
            prec=precision_test,
            rec=recall_test,
            thresholds=thresholds_test,
            thr_point=tau_val,
            prec_point=None,
            rec_point=None,
            base_rate=np.mean(y_test),
            class_label="Test",
            show=False,
            return_fig=True
        )
    save_path_pr_curve_test_tau_naive = ROOT / "master_figures" / f"{basename}_pr_curve_test_tau_naive.png"
    pr_curve_fig_test_naive.savefig(save_path_pr_curve_test_tau_naive, dpi=300, bbox_inches="tight")
    save_path_pr_curve_test_tau_naive = ROOT / "results" / basename / "figures" / f"{basename}_pr_curve_test_tau_naive.pdf"
    pr_curve_fig_test_naive.savefig(save_path_pr_curve_test_tau_naive, dpi=300, bbox_inches="tight")

    # ________________________________________________________________________
    # _________________ Calculate final performance __________________________

    # Find the index of the closest threshold
    idx = np.argmin(np.abs(thresholds_test - tau_test))

    # Precision and recall corresponding to this threshold
    prec_at_tau = precision_test[idx + 1]   # note: +1 shift
    rec_at_tau  = recall_test[idx + 1]

    print(f"📊 At threshold {tau_test:.3f}: precision={prec_at_tau:.3f}, recall={rec_at_tau:.3f}")
    print(f"🎯 Baseline precision on the test set: {np.mean(y_test):.3f}\n")

    if prec_at_tau > np.mean(y_test):
        print("============================================")
        print(f"✅  Model outperforms baseline by {100*(prec_at_tau-np.mean(y_test)):.2f}% ✅")
        print("============================================")
    else:
        print("=========================================")
        print("⚠️  Model did not outperform baseline ⚠️")
        print("=========================================")

