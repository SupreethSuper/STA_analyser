# Random Forest Delay-Classification Baseline

## Purpose

This experiment adds a Random Forest baseline for pre-routing net-delay and
cell-delay classification. It uses the existing `data/8_rat` graph dataset and
the same feature extractors and fixed circuit split as the K-means experiments.

The older `TimingPredict/train_stat_rf.py` cannot be used for this checkout
because it depends on the unavailable `data/4_netstat` dataset. The working
baseline is implemented in `random_forest_baseline.py`.

## Method

- Model: `sklearn.ensemble.RandomForestClassifier`
- Classes: five equal-frequency delay bins, C1-C5
- Class value: `max(L_rise, L_fall)` for each net or cell arc
- Class boundaries: calculated only from the 14 training circuits
- Split: the existing fixed 14-training/7-test circuit split
- Model selection: three-fold design-grouped cross-validation on training
  circuits only
- Selection metric: mean per-design macro F1
- Training sampling: fixed, design-balanced sample of at most 40,000 arcs per
  training circuit
- Final estimators: 100 trees
- Bootstrap fraction: 0.8 per tree
- Seeds: 0, 1, 2, 3, 4, 5, 6
- Test-set use: evaluation only; no parameter tuning uses test data

### Features

- Net delay: the K-means `full` feature set (geometry, fanout, HPWL, star
  length, capacitance and pin-role features)
- Cell delay: the K-means `lut_sta_ctx` feature set (Liberty lookup-table
  values evaluated at estimated slew/load, plus load and cell/net context)

## Full-run commands

```powershell
python .\random_forest_baseline.py --task net --seeds 0 1 2 3 4 5 6 --jobs 4
python .\random_forest_baseline.py --task cell --seeds 0 1 2 3 4 5 6 --jobs 4
```

The small end-to-end smoke test is:

```powershell
python .\random_forest_baseline.py --quick --task net --seeds 42 --jobs 4
python .\random_forest_baseline.py --quick --task cell --seeds 42 --jobs 4
```

## Results

Scores below are pooled over all seven held-out circuits and reported as the
mean +/- sample standard deviation across seven Random Forest seeds.

| Task | Accuracy | Macro F1 | Weighted F1 | Mean fit time |
|---|---:|---:|---:|---:|
| Net delay | 82.1935% +/- 0.0139% | **82.2765% +/- 0.0139%** | 82.1976% +/- 0.0138% | 21.57 s |
| Cell delay | 89.0328% +/- 0.0180% | **88.7994% +/- 0.0192%** | 89.0364% +/- 0.0186% | 216.65 s |

The repository's `k_means_info.md` reports K-means F1 values of 81% for net
delay and 88% for cell delay. Relative to those rounded values, Random Forest
is higher by about 1.28 percentage points for net delay and 0.80 percentage
points for cell delay. This comparison should be labelled provisional until it
is confirmed whether the K-means numbers refer to macro F1 or weighted F1.

## Generated outputs

Detailed CSV, JSON and PNG artifacts are written under:

```text
results/random_forest_baseline/net_delay/
results/random_forest_baseline/cell_delay/
```

The `results/` directory is intentionally ignored by Git. It contains detailed
per-seed/per-design metrics, confusion matrices, feature importance, selected
hyperparameters, class boundaries and plots. It can be regenerated using the
commands above.
