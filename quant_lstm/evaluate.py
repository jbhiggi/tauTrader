from sklearn.metrics import precision_recall_curve

def get_prediction_outputs(model, X, y_true, default_threshold=0.5):
    """
    Computes predicted probabilities, optimal thresholds, and binary predictions.
    
    Returns:
        y_prob: predicted probabilities
        y_pred: binary predictions at default_threshold
        precision, recall, thresholds: from precision_recall_curve
    """
    y_prob = model.predict(X).ravel()
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    y_pred = (y_prob >= default_threshold).astype(int)

    return y_prob, y_pred, precision, recall, thresholds


import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import quant_lstm.figures as f

def generate_diagnostics_figures(
    y_val, y_prob, y_pred,
    y_test, y_prob_test, y_pred_test,
    precision, recall, thresholds,
    precision_test, recall_test, thresholds_test,
    history,
    output_path="outputs/diagnostics.pdf"
):
    """
    Generates a multi-page PDF of diagnostic plots for validation and test sets.

    Parameters
    ----------
    y_val, y_prob, y_pred : Validation ground truth, predicted probs, predicted labels
    y_test, y_prob_test, y_pred_test : Test ground truth, predicted probs, predicted labels
    precision, recall, thresholds : PR curve components (validation)
    precision_test, recall_test, thresholds_test : PR curve components (test)
    history : Keras training history object
    output_path : str
        Path to save the diagnostic PDF
    """
    figures = []

    # ----- Validation Diagnostics -----
    fig1 = f.plot_probability_histogram(y_prob, bins=30, show=False, return_fig=True)
    figures.append(fig1)

    fig2, _ = f.plot_score_histograms(
        y_true_val=y_val, scores_val=y_prob,
        y_true_test=None, scores_test=None,
        n_bins=30, show=False, return_figs=True
    )
    figures.append(fig2)

    fig3 = f.plot_confusion_matrix(
        y_val, y_pred,
        normalize=False,
        show=False,
        return_fig=True,
        title="Confusion Matrix (Val)"
    )
    figures.append(fig3)

    fig4 = f.plot_learning_curves(
        history.history['loss'],
        history.history['val_loss'],
        show=False,
        return_fig=True
    )
    figures.append(fig4)

    fig5 = f.plot_accuracy_curve(
        history.history['accuracy'],
        show=False,
        return_fig=True,
        title="Validation Accuracy per Epoch"
    )
    figures.append(fig5)

    fig6 = f.plot_lr_schedule(
        history.history['lr'],
        show=False,
        return_fig=True
    )
    figures.append(fig6)

    fig7 = f.plot_pr_curve_with_thresholds(
        prec=precision,
        rec=recall,
        thresholds=thresholds,
        base_rate=np.mean(y_val),
        class_label="Validation",
        show=False,
        return_fig=True
    )
    figures.append(fig7)

    # ----- Test Diagnostics -----
    fig8 = f.plot_confusion_matrix(
        y_test, y_pred_test,
        normalize=False,
        show=False,
        return_fig=True,
        title="Confusion Matrix (Test)"
    )
    figures.append(fig8)

    fig9, fig10 = f.plot_score_histograms(
        y_true_val=y_val, scores_val=y_prob,
        y_true_test=y_test, scores_test=y_prob_test,
        n_bins=30, show=False, return_figs=True
    )
    figures.append(fig9)
    figures.append(fig10)

    fig11 = f.plot_pr_curve_with_thresholds(
        prec=precision_test,
        rec=recall_test,
        thresholds=thresholds_test,
        base_rate=np.mean(y_test),
        class_label="Test",
        show=False,
        return_fig=True
    )
    figures.append(fig11)

    # ----- Save all to PDF -----
    with PdfPages(output_path) as pdf:
        for fig in figures:
            if fig is not None:
                pdf.savefig(fig)

    print(f"[INFO] Saved diagnostics to: {output_path}")


from sklearn.metrics import precision_recall_curve
import numpy as np

def _threshold_max_precision_at_recall(y_true, y_scores, target_recall):
    """
    Return the threshold t that yields the maximum precision among 
    all points on the PR curve where recall >= target_recall.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    prec = precision[:-1]
    rec  = recall[:-1]
    th   = thresholds

    # Find all indices where recall >= target_recall
    valid_idx = np.where(rec >= target_recall)[0]
    if len(valid_idx) == 0:
        # No threshold can give recall >= target; return the best we can
        best_idx = np.argmax(rec)  # index of maximum recall
    else:
        # Among those indices, pick the one with maximum precision
        best_idx = valid_idx[np.argmax(prec[valid_idx])]

    return float(th[best_idx]), float(prec[best_idx]), float(rec[best_idx])

import numpy as np

def _best_f1_threshold(y_true, y_scores, eps=1e-8):
    """
    Find the threshold that maximizes the F1 score, given arrays of thresholds, precision, and recall.
    Assumes the first entry in `prec`/`rec` should be skipped (e.g., prec[0] = base rate, rec[0] = 1.0),
    and that `thr` has already been aligned to match the sliced arrays.

    Parameters
    ----------
    thr : np.ndarray
        1D array of threshold values (length = len(prec) - 1).
    prec : np.ndarray
        1D array of precisions (length ≥ len(thr) + 1).
    rec : np.ndarray
        1D array of recalls (length ≥ len(thr) + 1).
    eps : float, optional
        Small constant to avoid division by zero when computing F1 (default: 1e-8).

    Returns
    -------
    best_threshold : float
        The threshold that yields the highest F1 score.
    best_prec : float
        Precision at the best threshold.
    best_rec : float
        Recall at the best threshold.
    best_f1 : float
        The maximum F1 score.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    prec = precision[:-1]
    rec  = recall[:-1]
    thr   = thresholds

    # Skip the first entry so that thr[i] corresponds to (prec[i+1], rec[i+1])
    prec_sliced = prec[1:]
    rec_sliced  = rec[1:]

    # Compute F1 scores
    f1_scores = 2 * (prec_sliced * rec_sliced) / (prec_sliced + rec_sliced + eps)

    # Identify the index of the maximum F1
    best_idx = np.argmax(f1_scores)

    best_threshold = thr[best_idx]
    best_prec      = prec_sliced[best_idx]
    best_rec       = rec_sliced[best_idx]
    best_f1        = f1_scores[best_idx]

    return float(best_threshold), float(best_prec), float(best_rec), float(best_f1)

import json
from sklearn.metrics import classification_report
import numpy as np

def generate_training_report(
    y_val, y_prob_val, y_pred_val,
    y_test, y_prob_test, y_pred_test,
    val_loss,
    target_recall_1=0.3,
    target_recall_2=0.6,
    verbose=True,
    save_path=None
):
    """
    Generate training and evaluation summary report including:
    - Best epoch
    - Base rate
    - Classification reports (val and test)
    - Threshold diagnostics (val and test)

    Parameters
    ----------
    y_val, y_prob_val, y_pred_val : np.array
        Validation set ground truth, predicted probabilities, and predictions.
    y_test, y_prob_test, y_pred_test : np.array
        Test set ground truth, predicted probabilities, and predictions.
    val_loss : list or np.array
        Validation loss values per epoch.
    target_precision : float
        Precision threshold for evaluation.
    target_recall : float
        Recall threshold for evaluation.
    verbose : bool
        Whether to print the report.
    save_path : str or None
        Optional path to save the report as a JSON file.

    Returns
    -------
    report_dict : dict
        Dictionary with all computed results.
    """
    # Epoch and base rate
    best_epoch = int(np.argmin(val_loss)) + 1
    best_val_loss = val_loss[best_epoch - 1]
    base_rate_val = float(np.mean(y_val))
    base_rate_test = float(np.mean(y_test))

    # Classification reports
    report_val = classification_report(y_val, y_pred_val, digits=4, output_dict=True)
    report_test = classification_report(y_test, y_pred_test, digits=4, output_dict=True)

    # Threshold diagnostics – validation
    best_thr, best_p, best_r, best_f1 = _best_f1_threshold(y_val, y_prob_val)
    thr_mp_v1, p_mp_v1, r_mp_v1 = _threshold_max_precision_at_recall(y_val, y_prob_val, target_recall_1)
    thr_mp_v2, p_mp_v2, r_mp_v2 = _threshold_max_precision_at_recall(y_val, y_prob_val, target_recall_2)

    # Threshold diagnostics – test
    best_thr_t, best_p_t, best_r_t, best_f1_t = _best_f1_threshold(y_test, y_prob_test)
    thr_mp_t1, p_mp_t1, r_mp_t1 = _threshold_max_precision_at_recall(y_test, y_prob_test, target_recall_1)
    thr_mp_t2, p_mp_t2, r_mp_t2 = _threshold_max_precision_at_recall(y_test, y_prob_test, target_recall_2)

    # Package everything
    report = {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "base_rate": {
            "val": base_rate_val,
            "test": base_rate_test
        },
        "classification_report": {
            "val": report_val,
            "test": report_test
        },
        "threshold_diagnostics": {
            "val": {
                "best_f1": {"threshold": best_thr, "precision": best_p, "recall": best_r, "f1": best_f1},
                "max_precision_at_recall_1": {"threshold": thr_mp_v1, "precision": p_mp_v1, "recall": r_mp_v1},
                "max_precision_at_recall_2": {"threshold": thr_mp_v2, "precision": p_mp_v2, "recall": r_mp_v2}
            },
            "test": {
                "best_f1": {"threshold": best_thr_t, "precision": best_p_t, "recall": best_r_t, "f1": best_f1_t},
                "max_precision_at_recall_1": {"threshold": thr_mp_t1, "precision": p_mp_t1, "recall": r_mp_t1},
                "max_precision_at_recall_2": {"threshold": thr_mp_t2, "precision": p_mp_t2, "recall": r_mp_t2}
            }
        }
    }

    if verbose:
        print(f"\n🧪 Best Validation Loss: {best_val_loss:.6f} at Epoch {best_epoch}")

        print("\n📊 Classification Report (Validation):")
        print(classification_report(y_val, y_pred_val, digits=4))

        print("\n📊 Classification Report (Test):")
        print(classification_report(y_test, y_pred_test, digits=4))

        print(f"\n📉 Base Rate (Val):   {base_rate_val:.4f}")
        print(f"📉 Base Rate (Test):    {base_rate_test:.4f}")

        print()

        print(f"\n🏅 Best F1 Threshold (Val):                  {best_thr:.4f} → P={best_p:.4f}, R={best_r:.4f}, F1={best_f1:.4f}")
        print(f"🎯 Max Precision @ R≥{target_recall_1} (Val):                      {thr_mp_v1:.4f} → P={p_mp_v1:.4f}, R={r_mp_v1:.4f}")
        print(f"🎯 Max Precision @ R≥{target_recall_2} (Val):                      {thr_mp_v2:.4f} → P={p_mp_v2:.4f}, R={r_mp_v2:.4f}")
        print()
        print(f"🏅 Best F1 Threshold (Test):           {best_thr_t:.4f} → P={best_p_t:.4f}, R={best_r_t:.4f}, F1={best_f1_t:.4f}")
        print(f"🎯 Max Precision @ R≥{target_recall_1} (Test):                     {thr_mp_t1:.4f} → P={p_mp_t1:.4f}, R={r_mp_t1:.4f}")
        print(f"🎯 Max Precision @ R≥{target_recall_2} (Test):                     {thr_mp_t2:.4f} → P={p_mp_t2:.4f}, R={r_mp_t2:.4f}")

    if save_path:
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        if verbose:
            print(f"\n✅ Report saved to: {save_path}")

    return report





