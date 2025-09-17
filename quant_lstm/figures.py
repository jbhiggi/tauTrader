import matplotlib.pyplot as plt
import seaborn as sns

def plot_probability_histogram(scores,
                                bins=30,
                                figsize=(6, 4),
                                title="Histogram of Predicted Probabilities",
                                xlabel="Predicted Probability",
                                ylabel="Count",
                                show=False,
                                return_fig=True):
    """
    Plots histogram of predicted probabilities with seaborn darkgrid style
    and black outline around the figure.
    """
    # Apply seaborn darkgrid style
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=figsize)

    # Histogram with dark edges
    ax.hist(scores, bins=bins, edgecolor='black', linewidth=1.2, alpha=0.7, color="tab:blue")

    # Labels and titles
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)

    # Add gridlines (controlled by seaborn style)
    ax.grid(True, linestyle="--", alpha=0.6)

    # ---- Add black outline around the entire plot ----
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig:
        return fig


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_score_histograms(y_true_val, scores_val, 
                          y_true_test=None, scores_test=None, 
                          n_bins=30,
                          show=True,
                          return_figs=False):
    """
    Plot score histograms for positives vs. negatives on validation and optionally test sets,
    styled with seaborn darkgrid, black edges on bars, and black spines.
    """
    # Seaborn darkgrid style
    sns.set_style("darkgrid")

    y_val = np.asarray(y_true_val)
    s_val = np.asarray(scores_val)

    # Compute bin range
    if y_true_test is not None and scores_test is not None:
        y_test = np.asarray(y_true_test)
        s_test = np.asarray(scores_test)
        global_min = min(s_val.min(), s_test.min())
        global_max = max(s_val.max(), s_test.max())
    else:
        y_test = None
        s_test = None
        global_min = s_val.min()
        global_max = s_val.max()

    bin_edges = np.linspace(global_min, global_max, n_bins + 1)

    # ---------------- Validation Plot ----------------
    pos_scores_val = s_val[y_val == 1]
    neg_scores_val = s_val[y_val == 0]

    fig_val, ax_val = plt.subplots(figsize=(8, 5))
    ax_val.hist(neg_scores_val, bins=bin_edges, alpha=0.6, color='tab:blue',
                edgecolor="black", linewidth=1.1, label="Validation: Negatives")
    ax_val.hist(pos_scores_val, bins=bin_edges, alpha=0.6, color='C1',
                edgecolor="black", linewidth=1.1, label="Validation: Positives")
    ax_val.set_title("Score Histogram on Validation Set", fontsize=13, fontweight="bold")
    ax_val.set_xlabel("Model Score", fontsize=11)
    ax_val.set_ylabel("Count", fontsize=11)
    ax_val.legend(loc="upper left")
    ax_val.grid(axis="y", linestyle="--", alpha=0.6)

    # Add black spines (full outline)
    for spine in ax_val.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    if show:
        plt.show()
    else:
        plt.close(fig_val)

    # ---------------- Test Plot (optional) ----------------
    if y_test is not None and s_test is not None:
        pos_scores_test = s_test[y_test == 1]
        neg_scores_test = s_test[y_test == 0]

        fig_test, ax_test = plt.subplots(figsize=(8, 5))
        ax_test.hist(neg_scores_test, bins=bin_edges, alpha=0.6, color='tab:blue',
                     edgecolor="black", linewidth=1.1, label="Test: Negatives")
        ax_test.hist(pos_scores_test, bins=bin_edges, alpha=0.6, color='C1',
                     edgecolor="black", linewidth=1.1, label="Test: Positives")
        ax_test.set_title("Score Histogram on Test Set", fontsize=13, fontweight="bold")
        ax_test.set_xlabel("Model Score", fontsize=11)
        ax_test.set_ylabel("Count", fontsize=11)
        ax_test.legend(loc="upper left")
        ax_test.grid(axis="y", linestyle="--", alpha=0.6)

        for spine in ax_test.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.2)

        if show:
            plt.show()
        else:
            plt.close(fig_test)
    else:
        fig_test = None

    if return_figs:
        return fig_val, fig_test

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred,
                          normalize=False,
                          class_names=["Negative", "Positive"],
                          title="Confusion Matrix",
                          cmap="Blues",
                          figsize=(6, 5),
                          show=True,
                          return_fig=False):
    """
    Plots a confusion matrix with seaborn darkgrid style and black spines.
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    # Apply seaborn darkgrid style
    sns.set_style("dark")

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=9)

    # Labels and title
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title=title
    )

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotate each cell
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    fontsize=11,
                    color="white" if cm[i, j] > thresh else "black")

    # Add black spines (outline)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    # Add black spines (outline) to colorbar as well
    for spine in cbar.ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    fig.tight_layout()

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig:
        return fig


import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curves(loss, val_loss,
                         show=True,
                         return_fig=False,
                         figsize=(6, 4),
                         title="Learning Curve"):
    """
    Plot training vs validation loss with seaborn darkgrid style
    and black outline around the figure.
    """
    # Apply seaborn darkgrid style
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=figsize)

    # Plot curves with thicker lines for clarity
    ax.plot(loss, label='Train Loss', linewidth=2, color="C0")
    ax.plot(val_loss, label='Val Loss', linewidth=2, color="C1")

    # Labels and title
    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Loss', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    # Black spines around the figure
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig:
        return fig

def plot_accuracy_curve(accuracy,
                        show=True,
                        return_fig=False,
                        figsize=(6, 4),
                        title="Accuracy Over Epochs"):
    """
    Plot accuracy over epochs with seaborn darkgrid style,
    baseline line, and black spines.
    """
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(accuracy, label='Accuracy', linewidth=2, color="C0")
    ax.axhline(y=0.5, ls='--', color='black', label='Baseline (0.5)')

    ax.set_ylim(
        bottom=max(0.8 * min(accuracy), 0),
        top=max(0.55, 1.1 * max(accuracy))
    )

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.6)

    # Black spines
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig:
        return fig

def plot_lr_schedule(lr_history,
                     show=True,
                     return_fig=False,
                     figsize=(6, 4),
                     title="Learning Rate Schedule"):
    """
    Plot learning rate schedule with seaborn darkgrid style
    and black spines.
    """
    sns.set_style("darkgrid")

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(lr_history, linewidth=2, color="C0")

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Learning Rate', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)

    # Black spines
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig:
        return fig

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_pr_curve_with_thresholds(prec, rec, thresholds, base_rate,
                                  class_label="Class 1",
                                  thr_point=None, prec_point=None, rec_point=None,
                                  figsize=(6, 5),
                                  show=True,
                                  return_fig=False):
    """
    Plot a precision-recall curve, coloring points by threshold.
    Styled with seaborn darkgrid, black spines, and framed colorbar.

    Parameters
    ----------
    prec, rec : array-like
        Precision and recall from sklearn.metrics.precision_recall_curve.
        Note: points associated with `thresholds[i]` are (rec[i+1], prec[i+1]).
    thresholds : array-like
        Threshold values (length = len(prec)-1).
    base_rate : float
        Class prevalence for reference line.
    class_label : str
        Used in plot title.
    thr_point, prec_point, rec_point : float, optional
        Pass exactly one of these to highlight a point. The other two will be
        inferred by nearest match on the curve. Priority: thr_point > rec_point > prec_point.
    figsize : tuple
        Figure size.
    show : bool
        Whether to call plt.show().
    return_fig : bool
        Whether to return the matplotlib figure object.
    """
    prec = np.asarray(prec, dtype=float)
    rec  = np.asarray(rec, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    # --- Infer highlight point (priority: thr > rec > prec) ---
    hl_thr = hl_rec = hl_prec = None

    if thr_point is not None:
        i = int(np.argmin(np.abs(thresholds - thr_point)))
        hl_thr = thresholds[i]
        hl_rec = rec[i+1]
        hl_prec = prec[i+1]
    elif rec_point is not None:
        k = int(np.argmin(np.abs(rec - rec_point)))
        hl_rec = rec[k]
        hl_prec = prec[k]
        hl_thr = thresholds[k-1] if k >= 1 else None
    elif prec_point is not None:
        k = int(np.argmin(np.abs(prec - prec_point)))
        hl_rec = rec[k]
        hl_prec = prec[k]
        hl_thr = thresholds[k-1] if k >= 1 else None

    # --- Plot ---
    sns.set_style("darkgrid")
    fig, ax = plt.subplots(figsize=figsize)

    # 1) Base PR curve
    ax.plot(rec, prec, color='gray', linestyle='-', linewidth=1.5,
            label='PR curve', zorder=0)

    # 2) Threshold-colored scatter points (no edges)
    scatter = ax.scatter(
        rec[1:], prec[1:], c=thresholds, cmap='viridis', s=35,
        label='Threshold'
    )

    # 3) Baseline precision line
    ax.axhline(base_rate, linestyle='--', color='red',
               label=f'Baseline = {base_rate:.2f}')

    # 4) Highlight point if requested
    if hl_rec is not None and hl_prec is not None:
        label_bits = [f"P={hl_prec:.3f}", f"R={hl_rec:.3f}"]
        if hl_thr is not None:
            label_bits.append(f"t={hl_thr:.3f}")
        ax.scatter(
            hl_rec, hl_prec, marker='o', color='red', s=70, zorder=2,
            label=' / '.join(label_bits)
        )

    # 5) Labels, grid, limits
    ax.set_xlabel('Recall', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title(f'Precision-Recall Curve — {class_label}',
                 fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc='lower left', fontsize='small')

    # 6) Colorbar with black spines
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label('Threshold (probability)')
    for spine in cbar.ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    # 7) Black spines on main plot
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    if show:
        plt.show()
    else:
        plt.close(fig)

    if return_fig:
        return fig

