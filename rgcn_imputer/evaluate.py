# Leak-free evaluation, as in the paper:
#   random_masking : hide 20 / 50 / 90 % of the KNOWN values, impute them, compare with the truth
#   backtest       : hide every known value after a year (e.g. 2021), impute them (look-back window)
#   baselines      : interpolation inside each series, kNN and Iterative Imputer (Bayesian ridge), on the same hidden values

import warnings

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401  (needed to import IterativeImputer)
from sklearn.impute import KNNImputer, IterativeImputer

from .imputer import RGCNImputer
from .metrics import all_metrics


def _hide(df, value_col, rows):
    d = df.copy()
    d.loc[rows, value_col] = np.nan
    return d


def random_masking(df, imputer_kwargs, rates=(0.2, 0.5, 0.9), seed=0, baselines=True):
    df = df.reset_index(drop=True)
    value_col = imputer_kwargs["value_col"]
    known = np.flatnonzero(df[value_col].notna().to_numpy())
    rng = np.random.default_rng(seed)
    results = []
    for rate in rates:
        rows = rng.choice(known, size=int(round(rate * len(known))), replace=False)
        hidden = _hide(df, value_col, rows)
        truth = df.loc[rows, value_col].to_numpy(float)
        out = RGCNImputer(**imputer_kwargs).fit_transform(hidden)
        results.append({"setting": f"random {int(rate * 100)}%", "model": "R-GCN",
                        **all_metrics(truth, out.loc[rows, "value_imputed"].to_numpy())})
        if baselines:
            for name, pred in _baselines(hidden, imputer_kwargs, rows).items():
                results.append({"setting": f"random {int(rate * 100)}%", "model": name, **all_metrics(truth, pred)})
    return pd.DataFrame(results)


def backtest(df, imputer_kwargs, last_train_year=2021, baselines=True):
    df = df.reset_index(drop=True)
    value_col, year_col = imputer_kwargs["value_col"], imputer_kwargs["year_col"]
    rows = np.flatnonzero((df[year_col] > last_train_year).to_numpy() & df[value_col].notna().to_numpy())
    hidden = _hide(df, value_col, rows)
    out = RGCNImputer(**imputer_kwargs).fit_transform(hidden)
    preds = {"R-GCN": out.loc[rows, "value_imputed"].to_numpy()}
    if baselines:
        preds.update(_baselines(hidden, imputer_kwargs, rows))
    results = []
    years = df.loc[rows, year_col].to_numpy()
    truth = df.loc[rows, value_col].to_numpy(float)
    for name, pred in preds.items():
        for y in sorted(set(years)) + ["average"]:
            sel = years == y if y != "average" else np.ones(len(rows), bool)
            if y == "average":
                m = pd.DataFrame([all_metrics(truth[years == yy], pred[years == yy]) for yy in sorted(set(years))]).mean().to_dict()
            else:
                m = all_metrics(truth[sel], pred[sel])
            results.append({"year": y, "model": name, **m})
    return pd.DataFrame(results)


def _baselines(hidden, kw, rows):
    # wide table: one row per record (+ status), one column per year, log1p values
    keys = kw["record_cols"] + ([kw["status_col"]] if kw.get("status_col") else [])
    d = hidden.copy()
    d["_key"] = d[keys].astype(str).agg("|".join, axis=1)
    wide = d.pivot_table(index="_key", columns=kw["year_col"], values=kw["value_col"], aggfunc="first", dropna=False)
    wide = wide.reindex(d["_key"].unique())
    logw = np.log1p(wide)
    preds = {}
    # simple reference: straight line (in log) between the known years of the same series, nearest value at the ends
    line = logw.interpolate(axis=1, limit_direction="both")
    line = line.apply(lambda r: r.fillna(np.nanmean(logw.to_numpy())), axis=1)    # series without any value
    sub = d.loc[rows]
    preds["Interpolation"] = np.maximum(np.expm1(np.array([line.loc[k, y] for k, y in zip(sub["_key"], sub[kw["year_col"]])])), 0)
    for name, imp in [("kNN", KNNImputer(n_neighbors=5)),
                      ("Iterative Imputer", IterativeImputer(max_iter=20, random_state=0))]:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")      # sklearn warns about empty years and convergence
            arr = imp.fit_transform(logw.to_numpy())
        # keep the result inside the range of the known values (Iterative Imputer can explode at 90 % missing)
        arr = np.clip(arr, np.nanmin(logw.to_numpy()), np.nanmax(logw.to_numpy()))
        filled = pd.DataFrame(arr, index=logw.index,
                              columns=logw.columns[~logw.isna().all().to_numpy()])
        # years that are empty in every record cannot be imputed by these methods -> row mean
        filled = filled.reindex(columns=logw.columns)
        filled = filled.apply(lambda r: r.fillna(r.mean()), axis=1)
        sub = d.loc[rows]
        preds[name] = np.maximum(np.expm1(np.array([filled.loc[k, y] for k, y in zip(sub["_key"], sub[kw["year_col"]])])), 0)
    return preds
