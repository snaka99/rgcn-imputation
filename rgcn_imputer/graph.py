# Build the multi-relational graph from a table.
# One node = one row of the table (one record and one year).
# Two nodes get an edge of relation r if they share the value of column r.
# Extra relations:
#   "same_year"   : records of the same year (cross-sectional link)
#   "status_pair" : same record and year, different status (e.g. production vs inventory)
#   "temporal"    : same record, neighbouring years (|year difference| <= time_window)

import numpy as np
import torch


def _pairs_same_value(values, years, year_window):
    # all pairs (i, j), i != j, with the same value
    # if year_window is a number, only pairs with |year_i - year_j| <= year_window
    src, dst = [], []
    groups = {}
    for i, v in enumerate(values):
        groups.setdefault(v, []).append(i)
    for idx in groups.values():
        idx = np.array(idx)
        if len(idx) < 2:
            continue
        a, b = np.meshgrid(idx, idx, indexing="ij")
        a, b = a.ravel(), b.ravel()
        keep = a != b
        if year_window is not None:
            keep &= np.abs(years[a] - years[b]) <= year_window
        src.append(a[keep])
        dst.append(b[keep])
    if not src:
        return np.zeros(0, int), np.zeros(0, int)
    return np.concatenate(src), np.concatenate(dst)


def build_graph(df, relation_cols, year_col, record_cols, status_col=None,
                year_window=1, time_window=2, use_same_year=True):
    """Returns a list of relations. Each relation is a dict with
    name, src, dst (numpy arrays) and edge_index (2 x E torch tensor, the PyTorch Geometric format)."""
    n = len(df)
    years = df[year_col].to_numpy(int)

    # a key for "the same record" (same series), e.g. country + company + product
    record = df[record_cols].astype(str).agg("|".join, axis=1).to_numpy()

    relations = []

    # 1. one relation per categorical column (country, product, ore type, ...)
    for col in relation_cols:
        src, dst = _pairs_same_value(df[col].astype(str).to_numpy(), years, year_window)
        relations.append({"name": col, "src": src, "dst": dst})

    # 2. same year
    if use_same_year:
        src, dst = _pairs_same_value(years, years, None)
        relations.append({"name": "same_year", "src": src, "dst": dst})

    # 3. status pair: same record and year, different status
    if status_col is not None:
        key = np.array([f"{r}|{y}" for r, y in zip(record, years)])
        src, dst = _pairs_same_value(key, years, None)
        relations.append({"name": "status_pair", "src": src, "dst": dst})
        # in the temporal relation below the status is part of the record
        record = np.array([f"{r}|{s}" for r, s in zip(record, df[status_col].astype(str))])

    # 4. temporal: other years of the same record
    src, dst = _pairs_same_value(record, years, time_window)
    relations.append({"name": "temporal", "src": src, "dst": dst})

    # drop empty relations and make the edge_index of each relation
    out = []
    for r in relations:
        if len(r["src"]) == 0:
            continue
        # PyTorch Geometric sends messages from edge_index[0] (source) to edge_index[1] (target)
        r["edge_index"] = torch.tensor(np.vstack([r["dst"], r["src"]]), dtype=torch.long)
        out.append(r)
    return out
