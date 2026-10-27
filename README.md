# RGCN-imputation

Fill missing values in production time series with a multi-relational graph network (R-GCN).

This is the code of the paper *Multi-Relational GCN Imputation for Missing Rare Earth Elements (REE) Production Time Series* (IEEE EAIS 2026). It is written to be general: you can use it on any long table where each row is one record in one year (for example country, company, product, year, value) and some values are missing.

## How it works

1. Every row of the table is a node of a graph.
2. Nodes are linked by several relations: same value in a column you choose (e.g. country, product, ore type), same year, same record with a different status (e.g. production vs. inventory), and the neighbouring years of the same record.
3. Two R-GCN layers. Each relation has its own weight matrix, and the relations are mixed with learned softmax gates (so you can see which relation the model uses most).
4. Values are log-transformed and standardised inside each cohort (one cohort = one record), so large producers do not dominate.
5. Every node also gets the previous and the next known value of its own series. The network predicts a correction to the straight line between them, and it starts at zero: an untrained model gives plain interpolation, and training only keeps a correction if it lowers the validation loss.
6. Loss = Huber + median pinball + MAE on the original scale + graph smoothness.
7. The imputed value is one pass without dropout. Monte-Carlo dropout (many passes with dropout on) gives the uncertainty (std).

## Install

```bash
git clone https://github.com/snaka99/rgcn-imputation.git
cd rgcn-imputation
pip install -r requirements.txt
```

## Use

Your data must be a long table, one row per record and year, with NaN where the value is missing. Values must be ≥ 0 (negative values are used as 0 by the model, but known values are never changed):

| country | company | product | ore_type | status | year | production |
|---|---|---|---|---|---|---|
| A | company_A0 | oxide | monazite | production | 2010 | 1520.3 |
| A | company_A0 | oxide | monazite | production | 2011 | NaN |

```python
from rgcn_imputer import RGCNImputer

imp = RGCNImputer(
    value_col="production",
    year_col="year",
    relation_cols=["country", "product", "ore_type"],   # one graph relation per column
    record_cols=["country", "company", "product"],      # what one time series is
    status_col="status",                                # or None
)
out = imp.fit_transform(df)
# out["value_imputed"]  known values unchanged, gaps filled
# out["value_std"]      uncertainty of the filled values (0 for known values)
# out["imputed"]        True where the value was filled
print(imp.relation_weights())                           # learned softmax gates
```

The model is built with PyTorch Geometric, as in the paper. The default settings are the ones of the paper (hidden 256, dropout 0.4, Adam lr 1e-2, weight decay 5e-3, up to 100 epochs, early stopping after 30, learning rate reduced on plateau).

## Evaluation

```python
from rgcn_imputer import random_masking, backtest

settings = dict(value_col="production", year_col="year", relation_cols=["country", "product", "ore_type"],
                record_cols=["country", "company", "product"], status_col="status")
random_masking(df, settings)                        # hide 20 / 50 / 90 % of the known values
backtest(df, settings, last_train_year=2021)        # train on <= 2021, predict 2022-2024
```

Both compare R-GCN with interpolation inside each series, kNN and Iterative Imputer and report MAE, RMSE, R² (original scale) and Wasserstein distance and Jensen-Shannon divergence (on log values).

A full run on made-up data:

```bash
python examples/run_example.py
```

The example data are random numbers with the same shape as the data of the paper. The real data of the paper (MaaSiveTwin project) are not included in this repository.

## When it helps

The graph helps when many records move together (same country, same product, same year) and a series has long gaps. For a few short series with small gaps, a straight line between the known years is hard to beat; always look at the "Interpolation" row of the evaluation before you use the R-GCN values.

## Differences from the paper

This is a simple version that is easy to read and change:

- The model predicts a correction to the interpolation inside the series (see "How it works", point 5).
- The input of a node also contains its own standardised value when it is known. During training a random part of the known values is hidden and the model has to rebuild them, so it never sees the value it is asked for.
- Links of the column relations (country, product, ...) are only made between years at most `year_window` (default 1) apart, to keep the graph small. Set `year_window=None` to link all years.
- The MAE term is divided by the mean of the true values, so it has the same size as the other loss terms.

So the numbers you get will not be exactly the ones in the paper.

## Tests

```bash
python -m pytest tests
```

## Citation

If you use this code, please cite the paper:

```bibtex
@inproceedings{naka2026rgcn,
  title     = {Multi-Relational {GCN} Imputation for Missing Rare Earth Elements ({REE}) Production Time Series},
  author    = {Naka, Stelina and Antonakakis, Marios and Livanos, Giorgos and Darma, Susana and Liao, Che-Hsien and Zervakis, Michail},
  booktitle = {2026 IEEE International Conference on Evolving and Adaptive Intelligent Systems (EAIS)},
  address   = {Pisa, Italy},
  year      = {2026},
  publisher = {IEEE}
}
```

GitHub also shows a "Cite this repository" button from `CITATION.cff`.

## Acknowledgment

This work is supported by the EU/Swiss project MaaSiveTwin (Manufacturing As A Service And Supply Chain Predictive Twin For Critical Raw Materials), EU grant no. 101138334, Swiss State Secretariat for Education, Research and Innovation (SERI) contract no. 24.00047.

## License

MIT, see `LICENSE`.
