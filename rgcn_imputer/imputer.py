# RGCNImputer: fill the missing values of a long table (one row per record and year).
#
#   imp = RGCNImputer(value_col="production", year_col="year",
#                     relation_cols=["country", "product", "ore_type"],
#                     record_cols=["country", "company", "product"], status_col="status")
#   out = imp.fit_transform(df)       # out has value_imputed, value_std, imputed
#
# How it works (see the paper for the details):
#   1. log1p of the value and z-score inside each cohort (cohort = record_cols)
#   2. graph: one node per row, relations from build_graph()
#   3. training = masked reconstruction: every epoch some known values are hidden from the input
#      and the model must predict them (composite loss). 10 % of the known values are kept for early stopping.
#   4. prediction with Monte-Carlo dropout: many forward passes -> mean = imputed value, std = uncertainty

import numpy as np
import pandas as pd
import torch

from .graph import build_graph
from .model import RGCN
from .losses import composite_loss


class RGCNImputer:
    def __init__(self, value_col, year_col, relation_cols, record_cols, status_col=None,
                 hidden=256, dropout=0.4, lr=1e-2, weight_decay=5e-3, epochs=100, patience=30,
                 hide=0.3, val_share=0.1, mc_samples=50, year_window=1, time_window=2,
                 loss_weights=(1.0, 1.0, 0.5, 0.1), seed=42, verbose=False):
        self.value_col, self.year_col = value_col, year_col
        self.relation_cols, self.record_cols, self.status_col = list(relation_cols), list(record_cols), status_col
        self.hidden, self.dropout, self.lr, self.weight_decay = hidden, dropout, lr, weight_decay
        self.epochs, self.patience, self.hide, self.val_share = epochs, patience, hide, val_share
        self.mc_samples, self.year_window, self.time_window = mc_samples, year_window, time_window
        self.loss_weights, self.seed, self.verbose = loss_weights, seed, verbose

    # ---------------------------------------------------------------- data preparation
    def _cohort_stats(self, df, log_value, known):
        # mean and std of log1p(value) per cohort; cohorts with too few values use the global numbers
        cohort = df[self.record_cols].astype(str).agg("|".join, axis=1)
        g_mean = log_value[known].mean()
        g_std = log_value[known].std() if known.sum() > 1 else 1.0
        mean = np.full(len(df), g_mean)
        std = np.full(len(df), g_std)
        for c in cohort.unique():
            rows = (cohort == c).to_numpy()
            vals = log_value[rows & known]
            if len(vals) >= 1:
                mean[rows] = vals.mean()
            if len(vals) >= 3:
                std[rows] = max(vals.std(), 0.1 * g_std)
        return mean, std

    def _features(self, df):
        # one-hot of the categorical columns + the year scaled to 0..1
        cols = self.relation_cols + ([self.status_col] if self.status_col else [])
        onehot = pd.get_dummies(df[cols].astype(str)).to_numpy(float)
        y = df[self.year_col].to_numpy(float)
        year = (y - y.min()) / max(y.max() - y.min(), 1.0)
        return np.column_stack([onehot, year])

    def _series(self, df):
        # row numbers of every time series, sorted by year
        keys = self.record_cols + ([self.status_col] if self.status_col else [])
        key = df[keys].astype(str).agg("|".join, axis=1)
        years = df[self.year_col].to_numpy(float)
        groups = [np.array(sorted(idx, key=lambda i: years[i])) for idx in df.groupby(key).indices.values()]
        return groups, years

    def _context(self, z, visible):
        # for every row: the previous and the next VISIBLE value of its own series,
        # how far away they are (in years), and the straight line between them (interpolation)
        n = len(z)
        prev_z, next_z = np.zeros(n), np.zeros(n)
        has_prev, has_next = np.zeros(n), np.zeros(n)
        d_prev, d_next = np.zeros(n), np.zeros(n)
        line = np.zeros(n)
        for idx in self.groups_:
            yrs = self.years_[idx]
            vis = idx[visible[idx]]
            if len(vis) == 0:
                continue
            vy = self.years_[vis]
            for i, yr in zip(idx, yrs):
                before = vis[vy < yr]
                after = vis[vy > yr]
                if len(before):
                    j = before[-1]
                    prev_z[i], has_prev[i], d_prev[i] = z[j], 1, yr - self.years_[j]
                if len(after):
                    j = after[0]
                    next_z[i], has_next[i], d_next[i] = z[j], 1, self.years_[j] - yr
                if has_prev[i] and has_next[i]:
                    w = d_prev[i] / (d_prev[i] + d_next[i])
                    line[i] = (1 - w) * prev_z[i] + w * next_z[i]
                else:
                    line[i] = prev_z[i] if has_prev[i] else next_z[i]
        return np.column_stack([prev_z, next_z, has_prev, has_next, np.log1p(d_prev), np.log1p(d_next), line])

    def _inputs(self, base, z, visible):
        # inputs = categorical one-hot + year, the own value (0 if hidden) + flag "visible",
        # and the series context (previous / next visible value of the same series)
        zv = np.where(visible, z, 0.0)
        ctx = self._context(z, visible)
        x = torch.tensor(np.column_stack([base, zv, visible.astype(float), ctx]), dtype=torch.float32)
        line = torch.tensor(ctx[:, -1], dtype=torch.float32)
        return x, line

    def _predict(self, model, base, z, visible, adjs):
        # the model learns a CORRECTION to the straight line between the neighbouring known values:
        # if the graph has nothing to add, the result is the interpolation inside the series
        x, line = self._inputs(base, z, visible)
        return line + model(x, adjs)

    # ---------------------------------------------------------------- training
    def fit_transform(self, df):
        rng = np.random.default_rng(self.seed)
        torch.manual_seed(self.seed)
        df = df.reset_index(drop=True).copy()

        value = df[self.value_col].to_numpy(float)
        known = ~np.isnan(value)
        log_value = np.log1p(np.where(known, np.maximum(value, 0), 0))
        mean, std = self._cohort_stats(df, log_value, known)
        z = np.where(known, (log_value - mean) / std, 0.0)

        relations = build_graph(df, self.relation_cols, self.year_col, self.record_cols, self.status_col,
                                self.year_window, self.time_window)
        adjs = [r["edge_index"] for r in relations]
        # edges for the smoothness loss: the temporal relation (same record, near years)
        temporal = [r for r in relations if r["name"] == "temporal"]
        src = torch.tensor(temporal[0]["src"] if temporal else [], dtype=torch.long)
        dst = torch.tensor(temporal[0]["dst"] if temporal else [], dtype=torch.long)

        base = self._features(df)
        self.groups_, self.years_ = self._series(df)
        target = torch.tensor(z, dtype=torch.float32)
        mean_t, std_t = torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)

        # split the known values: train / validation (validation is never visible to the model)
        known_idx = np.flatnonzero(known)
        rng.shuffle(known_idx)
        n_val = max(1, int(round(self.val_share * len(known_idx))))
        val_idx, train_idx = known_idx[:n_val], known_idx[n_val:]

        model = RGCN(base.shape[1] + 9, len(adjs), self.hidden, self.dropout)
        opt = torch.optim.Adam(model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=10)

        # the validation loss before training (= interpolation only) is the first "best":
        # if training never beats it, we keep the untrained model
        vis = np.zeros(len(df), bool)
        vis[train_idx] = True
        vmask = torch.zeros(len(df), dtype=torch.bool)
        vmask[val_idx] = True
        model.eval()
        with torch.no_grad():
            pred = self._predict(model, base, z, vis, adjs)
            best = composite_loss(pred, target, vmask, mean_t, std_t, src, dst, self.loss_weights).item()
        best_state, wait = {k: v.clone() for k, v in model.state_dict().items()}, 0

        for epoch in range(self.epochs):
            # hide a random part of the training values from the input
            hidden_now = train_idx[rng.random(len(train_idx)) < self.hide]
            visible = np.zeros(len(df), bool)
            visible[train_idx] = True
            visible[hidden_now] = False
            mask = torch.zeros(len(df), dtype=torch.bool)
            mask[hidden_now] = True
            if mask.sum() == 0:
                continue

            model.train()
            opt.zero_grad()
            pred = self._predict(model, base, z, visible, adjs)
            loss = composite_loss(pred, target, mask, mean_t, std_t, src, dst, self.loss_weights)
            loss.backward()
            opt.step()

            # validation: all training values visible, validation values hidden
            model.eval()
            with torch.no_grad():
                pred = self._predict(model, base, z, vis, adjs)
                val_loss = composite_loss(pred, target, vmask, mean_t, std_t, src, dst, self.loss_weights).item()
            sched.step(val_loss)
            if self.verbose and epoch % 10 == 0:
                print(f"epoch {epoch:3d}  train {loss.item():.4f}  val {val_loss:.4f}")

            # early stopping
            if val_loss < best - 1e-6:
                best, best_state, wait = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.model_, self.relations_ = model, [r["name"] for r in relations]

        # ------------------------------------------------------------ prediction with MC dropout
        # now every known value is visible
        # imputed value: one pass without dropout; uncertainty: Monte-Carlo dropout (many passes with dropout on)
        lo, hi = log_value[known].min() - 1, log_value[known].max() + 1      # stay near the range of the data
        model.eval()
        with torch.no_grad():
            z_hat = self._predict(model, base, z, known, adjs).numpy()
        model.train()
        with torch.no_grad():
            draws = np.stack([self._predict(model, base, z, known, adjs).numpy() for _ in range(self.mc_samples)])
        best_value = np.expm1(np.clip(z_hat * std + mean, lo, hi))
        draws = np.expm1(np.clip(draws * std + mean, lo, hi))

        df["value_imputed"] = np.where(known, value, np.maximum(best_value, 0))
        df["value_std"] = np.where(known, 0.0, draws.std(0))
        df["imputed"] = ~known
        return df

    def relation_weights(self):
        # softmax gates per layer, as a small table
        w = self.model_.relation_weights()
        return pd.DataFrame({"relation": self.relations_, "layer1": w[0], "layer2": w[1]})
