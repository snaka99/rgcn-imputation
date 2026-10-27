# Metrics of the paper.
# MAE, RMSE, R2: on the original scale.
# WD (Wasserstein distance) and JSD (Jensen-Shannon divergence): on log1p values, so big producers do not dominate.

import numpy as np
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon


def mae(y, p):
    return float(np.mean(np.abs(y - p)))


def rmse(y, p):
    return float(np.sqrt(np.mean((y - p) ** 2)))


def r2(y, p):
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def wd(y, p):
    return float(wasserstein_distance(np.log1p(y), np.log1p(p)))


def jsd(y, p, bins=30):
    # both histograms on the same bins; base 2 -> value between 0 and 1
    a, b = np.log1p(y), np.log1p(p)
    edges = np.histogram_bin_edges(np.concatenate([a, b]), bins=bins)
    h1, _ = np.histogram(a, edges)
    h2, _ = np.histogram(b, edges)
    return float(jensenshannon(h1 + 1e-12, h2 + 1e-12, base=2) ** 2)


def all_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return {"MAE": mae(y, p), "RMSE": rmse(y, p), "R2": r2(y, p), "WD": wd(y, p), "JSD": jsd(y, p)}
