import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def save_linear_fit(val_thresholds, test_thresholds, outpath):
    """
    Fit a linear relation (y = a*x + b) between validation and test thresholds,
    and save the coefficients as JSON at `outpath`.

    Returns
    -------
    dict : {"a": slope, "b": intercept}
    """
    X_val = np.array(val_thresholds).reshape(-1, 1)
    y_test = np.array(test_thresholds)

    model = LinearRegression().fit(X_val, y_test)
    a = float(model.coef_[0])
    b = float(model.intercept_)

    with open(outpath, "w") as f:
        json.dump({"a": a, "b": b}, f, indent=4)

    return {"a": a, "b": b}


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

def save_linear_fit_plot(val_thresholds, test_thresholds, outpath):
    """
    Create and save a regression plot with error bands at `outpath`.
    Returns the saved path.
    """
    # Style
    sns.set_style("darkgrid")

    X_val  = np.array(val_thresholds).reshape(-1, 1)
    y_test = np.array(test_thresholds)

    model = LinearRegression().fit(X_val, y_test)

    # Fit line domain
    val_range = np.linspace(min(val_thresholds) - 0.01,
                            max(val_thresholds) + 0.01, 200).reshape(-1, 1)
    y_pred = model.predict(val_range)

    # Residuals and standard error of prediction band
    y_fit = model.predict(X_val)
    residuals = y_test - y_fit
    mse = np.mean(residuals ** 2)

    n = len(X_val)
    x_mean = np.mean(X_val)
    x_var  = np.sum((X_val - x_mean) ** 2)
    se = np.sqrt(mse * (1/n + (val_range - x_mean) ** 2 / x_var)).ravel()

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(val_thresholds, test_thresholds,
               label='Observed (val, test)', s=45, color="C0")
    ax.plot(val_range, y_pred, label='Fitted Line',
            color="red", linestyle='--', linewidth=2)
    ax.fill_between(val_range.ravel(), y_pred - se, y_pred + se,
                    color='red', alpha=0.18, label='±1σ Error Band')

    # Labels & cosmetics
    ax.set_title('Mapping Between Validation and Test Thresholds',
                 fontsize=13, fontweight="bold")
    ax.set_xlabel('Validation Threshold', fontsize=11)
    ax.set_ylabel('Test Threshold', fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()

    # Black spines (frame)
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)

    return outpath
