# Extracted from:  https://scikit-learn.org/stable/auto_examples/applications/plot_prediction_latency.html#sphx-glr-auto-examples-applications-plot-prediction-latency-py

import time
import numpy as np
import matplotlib.pyplot as plt

def atomic_benchmark_estimator(estimator, X_test, verbose=False):
    """Measure runtime prediction of each instance."""
    n_instances = X_test.shape[0]
    runtimes = np.zeros(n_instances, dtype=float)
    for i in range(n_instances):
        instance = X_test[[i], :]
        start = time.time()
        estimator.predict(instance)
        runtimes[i] = time.time() - start
    if verbose:
        print(
            "atomic_benchmark runtimes:",
            min(runtimes),
            np.percentile(runtimes, 50),
            max(runtimes),
        )
    return runtimes

def boxplot_runtimes(runtimes, pred_type, configuration):
    """
    Plot a new `Figure` with boxplots of prediction runtimes.

    Parameters
    ----------
    runtimes : list of `np.array` of latencies in micro-seconds
    cls_names : list of estimator class names that generated the runtimes
    pred_type : 'bulk' or 'atomic'

    """

    fig, ax1 = plt.subplots(figsize=(10, 6))
    bp = plt.boxplot(
        runtimes,
    )

    cls_infos = [
        "%s\n(%d %s)"
        % (
            estimator_conf["name"],
            estimator_conf["complexity_computer"](estimator_conf["instance"]),
            estimator_conf["complexity_label"],
        )
        for estimator_conf in configuration["estimators"]
    ]
    plt.setp(ax1, xticklabels=cls_infos)
    plt.setp(bp["boxes"], color="black")
    plt.setp(bp["whiskers"], color="black")
    plt.setp(bp["fliers"], color="red", marker="+")

    ax1.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)

    ax1.set_axisbelow(True)
    ax1.set_title(
        "Prediction Time per Instance - %s, %d feats."
        % (pred_type.capitalize(), configuration["n_features"])
    )
    ax1.set_ylabel("Prediction Time (us)")

    plt.show()