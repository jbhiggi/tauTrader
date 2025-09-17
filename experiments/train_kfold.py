import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import yaml
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold
import random

from quant_lstm.data_loader import prepare_results_dirs
from quant_lstm.preprocessing import create_sequences
from quant_lstm.train import compute_class_weights, run_training
from quant_lstm.model import create_binary_categorical_model
from quant_lstm.figures import plot_pr_curve_with_thresholds
import quant_lstm.feature_engineering as fe
from matplotlib.backends.backend_pdf import PdfPages
from quant_lstm.cv_evaluate import (validate_metric,
                             evaluate_thresholds,
                             get_pr_point_at_threshold,
                             combine_figures_side_by_side)

def main(config_path="config.yaml"):


    ##########################################################################

    # ________________________________________________________________________
    # _________________ Load Config File _____________________________________

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    #verbose = config.get("general", {}).get("verbose", False)
    verbose = False
    print("⚙️ Config file loaded.")

    # --- Set random seed ---
    seed = config["general"].get("seed", 42)  # default to 42 if missing
    random.seed(seed) # Apply seed everywhere
    
    # --- Paths / run info ---
    basename = config["run"]["basename"]

    # --- Results dirs ---
    ROOT = Path(__file__).resolve().parents[1]

    dirs = prepare_results_dirs(ROOT, basename)
    print("📁 Results will be saved in:", dirs["results"])
    print("📁 Figures will be saved in:", dirs["figures"])
    print("📁 Modified data will be saved in:", dirs["modified_data"])

    # ________________________________________________________________________
    # _________________ Global Variables _____________________________________

    n_splits = 10
    thresholds_array = np.linspace(0, 1, 101)
    recall_threshold = 0.4
    normalization_feature_range = (-1,1)
    metric = 'precision_gain_over_base'

    # --- Paths / run info ---
    basename = config["run"]["basename"]

    # ________________________________________________________________________
    # _________________ Load Data ____________________________________________

    clipped_df_path = ROOT / 'results' / basename / 'modified_data' / f'{basename}_data_features_clipped.csv'
    df = pd.read_csv(clipped_df_path)
    print(f"📂 Clipped OHLC data loaded. Shape: {df.shape}")

    ##############################################################

    validate_metric(metric)

    # ________________________________________________________________________
    # _________________ Initialize K-fold ____________________________________

    val_thresholds = []
    test_thresholds = []
    val_scores = []
    test_scores = []
    fig_list = []

    print("🚀 K-fold Cross Validation started...")

    kf = KFold(n_splits=n_splits, shuffle=False)
    splits = kf.split(df)

    # Wrap with tqdm if not verbose
    if not verbose:
        splits = tqdm(splits, total=n_splits, desc="K-Fold CV")

    for fold_idx, (train_val_idx, test_idx) in enumerate(splits):
        
        if verbose:
            print(f"\n🧪 Fold {fold_idx+1}/{n_splits}")
        
        # ________________________________________________________________________
        # _________________ Split the Data _______________________________________

        df_train_val = df.iloc[train_val_idx].copy()
        df_test = df.iloc[test_idx].copy()

        val_size = int(len(df_train_val) * 0.2)
        df_val = df_train_val.iloc[-val_size:].copy()
        df_train = df_train_val.iloc[:-val_size].copy()
        
        # ________________________________________________________________________
        # _________________ Normalize Data _______________________________________

        save_path_norm_meta = dirs["modified_data"] / f"{basename}_normalization_meta_data_k_fold_{fold_idx}_of_{n_splits}.json"
        df_train_scaled, df_val_scaled, df_test_scaled, norm_meta = fe.normalize_splits(
            df_train, df_val, df_test,
            tail_column='volume',
            feature_range=normalization_feature_range,
            metadata_output_path=save_path_norm_meta
        )

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

        # ________________________________________________________________________
        # _________________ Calculate Class Weights ______________________________

        manual_weights = config.get("training", {}).get("class_weight_override")
        class_weight = compute_class_weights(y_train, override=manual_weights, verbose=verbose)

        # ________________________________________________________________________
        # _________________ Create the Model _____________________________________

        model = create_binary_categorical_model(timesteps=config['data']['window_size'],
                                                features=len(config['data']['feature_cols']), 
                                                units=config['model']['units'], 
                                                dropout_rate=config['model']['dropout_rate'],
                                                learning_rate=config['model']['learning_rate'],
                                                l2_lambda=config['model']['l2_lambda'],
                                                verbose=verbose)
        
        # ________________________________________________________________________
        # _________________ Train the Model ______________________________________

        save_path_model_best_weights = ROOT / "results" / basename / f"{basename}_best_weights_k_fold_{fold_idx}_of_{n_splits}.keras"
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

        # ________________________________________________________________________
        # _________________ Find best thresholds _________________________________

        # VAL EVALUATION
        y_val_prob = model.predict(X_val, verbose=0).ravel()
        y_val_true = y_val
        best_val_thresh, best_val_score = evaluate_thresholds(
            y_true=y_val_true,
            y_prob=y_val_prob,
            metric='precision_gain_over_base',
            thresholds=thresholds_array,
            recall_threshold=recall_threshold
        )

        # TEST EVALUATION
        y_test_prob = model.predict(X_test, verbose=0).ravel()
        y_test_true = y_test
        best_test_thresh, best_test_score = evaluate_thresholds(
            y_true=y_test_true,
            y_prob=y_test_prob,
            metric='precision_gain_over_base',
            thresholds=thresholds_array,
            recall_threshold=recall_threshold
        )

        # APPEND THE RESULTS
        val_thresholds.append(best_val_thresh)
        test_thresholds.append(best_test_thresh)
        val_scores.append(best_val_score)
        test_scores.append(best_test_score)

        # ________________________________________________________________________
        # _________________ Create Diagnostic Plot _______________________________

        # --- Validation PR curve ---
        prec_val, rec_val, pr_thresholds_val, prec_point_val, rec_point_val = get_pr_point_at_threshold(
            y_true=y_val,
            y_prob=y_val_prob,
            best_threshold=best_val_thresh
        )

        fig_val = plot_pr_curve_with_thresholds(
            prec_val, rec_val, pr_thresholds_val,
            base_rate=np.mean(y_val),
            class_label="Validation",
            thr_point=best_val_thresh,
            prec_point=None, # prec_point_val
            rec_point=None, # rec_point_val
            show=False,
            return_fig=True
        )

        # --- Test PR curve ---
        prec_test, rec_test, pr_thresholds_test, prec_point_test, rec_point_test = get_pr_point_at_threshold(
            y_true=y_test,
            y_prob=y_test_prob,
            best_threshold=best_test_thresh
        )

        fig_test = plot_pr_curve_with_thresholds(
            prec_test, rec_test, pr_thresholds_test,
            base_rate=np.mean(y_test),
            class_label="Test",
            thr_point=best_test_thresh,
            prec_point=None, # prec_point_test
            rec_point=None, # rec_point_test
            show=False,
            return_fig=True
        )

        fig_combined = combine_figures_side_by_side(fig_val, fig_test)
        fig_list.append(fig_combined)
        plt.close(fig_combined)

    print("🔄 K-fold Cross-Validation completed.")

    # ----- Save all to PDF -----
    save_path_k_fold_diagnostics = ROOT / "results" / basename / "figures" / f"{basename}_k_fold_diagnostics.pdf"
    with PdfPages(save_path_k_fold_diagnostics) as pdf:
        for fig in fig_list:
            if fig is not None:
                pdf.savefig(fig)

    print(f"🧾 Saved diagnostics to: {save_path_k_fold_diagnostics}")

    from quant_lstm.cv_output import save_linear_fit, save_linear_fit_plot
    # JSON with slope/intercept goes into results/basename/
    save_path_json = ROOT / "results" / basename / f"{basename}_linear_fit.json"
    coeffs = save_linear_fit(val_thresholds, test_thresholds, save_path_json)

    # Figure goes into results/basename/figures/
    save_path_fig = ROOT / "master_figures" / f"{basename}_val_to_test_fit.png"
    fig_path = save_linear_fit_plot(val_thresholds, test_thresholds, save_path_fig)
    save_path_fig = ROOT / "results" / basename / "figures" / f"{basename}_val_to_test_fit.pdf"
    fig_path = save_linear_fit_plot(val_thresholds, test_thresholds, save_path_fig)

    # Pipeline handles logging
    print(f"📊 Linear fit coefficients saved → {save_path_json}")
    print(f"   🔹 a = {coeffs['a']:.4f}, b = {coeffs['b']:.4f}")
    print(f"🖼️ Figure saved → {fig_path}")


if __name__ == "__main__":
    config_name = "config.yaml"
    if len(sys.argv) > 1:
        config_name = sys.argv[1]
        if not config_name.endswith(".yaml"):
            config_name += ".yaml"

    ROOT = Path(__file__).resolve().parents[1]
    config_path = ROOT / "quant_lstm" / "configs" / config_name

    main(config_path)
    print(f"✅ Finished pipeline run with config: {config_name}")
