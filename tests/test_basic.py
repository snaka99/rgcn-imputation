# Small checks: run with  python -m pytest tests
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from rgcn_imputer import RGCNImputer
from make_example_data import make_data

SETTINGS = dict(value_col="production", year_col="year", relation_cols=["country", "product", "ore_type"],
                record_cols=["country", "company", "product"], status_col="status", epochs=20, mc_samples=5)


def test_fills_every_gap_and_keeps_known_values():
    df = make_data(seed=3)
    out = RGCNImputer(**SETTINGS).fit_transform(df)
    assert out["value_imputed"].notna().all()                       # no gap left
    known = df["production"].notna().to_numpy()
    assert np.allclose(out.loc[known, "value_imputed"], df.loc[known, "production"])   # known values unchanged
    assert (out["value_imputed"] >= 0).all()
    assert (out.loc[known, "value_std"] == 0).all()


def test_same_seed_same_result():
    df = make_data(seed=3)
    a = RGCNImputer(**SETTINGS).fit_transform(df)["value_imputed"]
    b = RGCNImputer(**SETTINGS).fit_transform(df)["value_imputed"]
    assert np.allclose(a, b)


def test_works_without_status_column():
    df = make_data(seed=3)
    df = df[df.status == "production"].drop(columns="status")
    s = dict(SETTINGS, status_col=None)
    out = RGCNImputer(**s).fit_transform(df)
    assert out["value_imputed"].notna().all()
