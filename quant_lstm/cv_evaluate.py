def validate_metric(metric):
    allowed_metrics = {
        'precision',
        'recall',
        'f1',
        'accuracy',
        'precision_gain_over_base'
    }
    if metric not in allowed_metrics:
        raise ValueError(
            f"Invalid metric '{metric}'. Must be one of: {sorted(allowed_metrics)}"
        )
    
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    precision_recall_curve, roc_auc_score
)

def _compute_metric(y_true, y_prob=None, metric='precision_gain_over_base', threshold=None, recall_threshold=0.4):
    """
    Compute an evaluation metric for binary classification.
    
    Parameters:
        y_true (np.array): Ground truth labels (0 or 1).
        y_prob (np.array): Probabilities (required for most metrics).
        metric (str): Metric name.
        threshold (float): Threshold to convert probs to binary (if needed).
        recall_threshold (float): Minimum recall for some metrics (e.g. precision_gain_over_base).
    
    Returns:
        float: Computed metric value.
    """
    if metric == 'precision_gain_over_base':
        if y_prob is None:
            raise ValueError("y_prob is required for 'precision_gain_over_base'")
        base_rate = float(np.mean(y_true))
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        valid = recall >= recall_threshold
        #print(valid)
        #print(np.sum(valid))
        if not np.any(valid):
            print('returning 0.0')
            return 0.0
        max_precision = np.max(precision[valid])
        #print(f'max precision: {max_precision}')
        return max_precision - base_rate

    if threshold is None:
        raise ValueError(f"Threshold is required for metric '{metric}'")

    # Convert to binary predictions
    y_pred = (y_prob >= threshold).astype(int)

    if metric == 'precision':
        return precision_score(y_true, y_pred, zero_division=0)
    elif metric == 'recall':
        return recall_score(y_true, y_pred, zero_division=0)
    elif metric == 'f1':
        return f1_score(y_true, y_pred, zero_division=0)
    elif metric == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif metric == 'roc_auc':
        return roc_auc_score(y_true, y_prob)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

import numpy as np
from sklearn.metrics import precision_recall_curve

def evaluate_thresholds(y_true,
                        y_prob,
                        metric,
                        thresholds=np.linspace(0, 1, 101),
                        recall_threshold=0.4):
    """
    Sweep over thresholds to find the one that maximizes a chosen metric,
    **with a special branch for 'precision_gain_over_base'** which is itself
    defined by a PR-curve sweep.
    """
    # Special case: precision_gain_over_base
    if metric == 'precision_gain_over_base':
        # 1) compute full PR curve
        prec, rec, pr_thresh = precision_recall_curve(y_true, y_prob)
        base_rate = np.mean(y_true)
        # 2) find all points with recall ≥ recall_threshold
        valid_idx = np.where(rec >= recall_threshold)[0]
        if valid_idx.size == 0:
            return 0.0, None
        # 3) pick the idx among those with maximum precision
        best_pos = valid_idx[np.argmax(prec[valid_idx])]
        best_score = prec[best_pos] - base_rate

        if best_pos == 0:
            # the first precision point has no corresponding threshold
            best_threshold = 0.0
        else:
            best_threshold = pr_thresh[best_pos - 1]
        return best_threshold, best_score

    # Otherwise, do the simple sweep over user-supplied thresholds:
    best_score = -np.inf
    best_threshold = None
    for t in thresholds:
        try:
            score = _compute_metric(
                y_true=y_true,
                y_prob=y_prob,
                threshold=t,
                metric=metric,
                recall_threshold=recall_threshold
            )
        except ValueError:
            continue

        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold, best_score

from sklearn.metrics import precision_recall_curve
import numpy as np

def get_pr_point_at_threshold(y_true, y_prob, best_threshold):
    """
    Computes the precision-recall curve and finds the precision and recall
    at the point closest to the given threshold.
    
    Parameters:
        y_true (np.array): True binary labels.
        y_prob (np.array): Predicted probabilities.
        best_threshold (float): Threshold of interest.
        
    Returns:
        prec (np.array): Precision values for PR curve.
        rec (np.array): Recall values for PR curve.
        pr_thresholds (np.array): Thresholds used in PR curve.
        prec_point (float): Precision at closest threshold.
        rec_point (float): Recall at closest threshold.
    """
    prec, rec, pr_thresholds = precision_recall_curve(y_true, y_prob)
    
    # Find the closest threshold index
    best_idx = np.argmin(np.abs(pr_thresholds - best_threshold))
    
    # Offset by +1 because prec/rec are 1 element longer than pr_thresholds
    prec_point = prec[best_idx + 1]
    rec_point = rec[best_idx + 1]
    
    return prec, rec, pr_thresholds, prec_point, rec_point

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Rectangle

def combine_figures_side_by_side(fig1,
                                 fig2,
                                 titles=("Validation PR Curve", "Test PR Curve"),
                                 share_cbar=True):
    """
    Combine two PR-curve figures side-by-side, preserving styling:
      1) PR curves (lines)
      2) threshold-colored scatter
      3) red highlight dot(s) on top
      4) seaborn darkgrid style, black spines, outer border
    """
    sns.set_style("darkgrid")

    fig_combined, axes = plt.subplots(1, 2, figsize=(12, 5))
    first_scatter = None

    for ax_src, ax_dst, title in zip([fig1.axes[0], fig2.axes[0]], axes, titles):
        # 1) Copy PR curve lines
        for line in ax_src.get_lines():
            ax_dst.plot(
                line.get_xdata(),
                line.get_ydata(),
                linestyle=line.get_linestyle(),
                label=line.get_label(),
                color=line.get_color(),
                zorder=0
            )

        # Separate red-dot collections
        red_colls, other_colls = [], []
        for coll in ax_src.collections:
            face = coll.get_facecolor()
            rgba = face[0] if face.ndim > 1 else face
            if np.allclose(rgba[:3], [1, 0, 0], atol=1e-2):
                red_colls.append(coll)
            else:
                other_colls.append(coll)

        # 2) Threshold-colored scatter
        for coll in other_colls:
            offsets = coll.get_offsets()
            xs, ys = offsets[:, 0], offsets[:, 1]
            sizes = coll.get_sizes()
            arr = coll.get_array()
            if arr is not None:
                sc = ax_dst.scatter(
                    xs, ys,
                    c=arr,
                    cmap=coll.get_cmap(),
                    norm=coll.norm,
                    s=sizes,
                    label=coll.get_label(),
                    zorder=1
                )
                if first_scatter is None:
                    first_scatter = sc
            else:
                ax_dst.scatter(xs, ys, color=coll.get_facecolor(), s=sizes,
                               label=coll.get_label(), zorder=1)

        # 3) Red highlight dot(s)
        for coll in red_colls:
            offsets = coll.get_offsets()
            xs, ys = offsets[:, 0], offsets[:, 1]
            sizes = coll.get_sizes()
            ax_dst.scatter(xs, ys,
                           marker=coll.get_paths()[0],
                           color=coll.get_facecolor()[0],
                           s=sizes,
                           label=coll.get_label(),
                           zorder=2)

        # Formatting
        ax_dst.set_title(title, fontsize=13, fontweight="bold")
        ax_dst.set_xlabel("Recall", fontsize=11)
        ax_dst.set_ylabel("Precision", fontsize=11)
        ax_dst.set_xlim(0, 1)
        ax_dst.set_ylim(0, 1)
        ax_dst.grid(True, linestyle="--", alpha=0.6)

        # Legend
        handles, labels = ax_src.get_legend_handles_labels()
        if handles:
            ax_dst.legend(handles, labels, loc='lower left', fontsize='small')

        # Black spines
        for spine in ax_dst.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.2)

    plt.tight_layout()

    # Shared colorbar
    if share_cbar and first_scatter is not None:
        cbar = fig_combined.colorbar(first_scatter, ax=axes.ravel().tolist())
        cbar.set_label("Threshold (probability)")
        # Black outline for colorbar spines
        for spine in cbar.ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.2)

    # Outer figure border
    rect = Rectangle((0, 0), 1, 1, transform=fig_combined.transFigure,
                     fill=False, linewidth=1.2, edgecolor="black")
    rect.set_clip_on(False)
    fig_combined.add_artist(rect)

    return fig_combined
