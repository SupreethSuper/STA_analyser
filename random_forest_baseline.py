"""Random Forest baseline for pre-routing net- and cell-delay classes.

This script deliberately reuses the feature extractors and the fixed circuit
split already used by the K-means scripts.  It predicts one of five delay
classes, where the class boundaries are quintiles computed from the 14
training designs only.  The seven held-out designs are never used for tuning.

The model-selection score is mean per-design macro F1 from three-fold,
design-grouped cross-validation.  The final configuration is rerun with
multiple random seeds and reported as mean +/- standard deviation.

Examples
--------
Small end-to-end check (recommended first command):

    python random_forest_baseline.py --quick --task net --seeds 42

Full experiment required for the report:

    python random_forest_baseline.py --task both --seeds 0 1 2 3 4 5 6
"""

import argparse
import csv
import gc
import importlib.util
import json
import sys
import time
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

import paths


N_CLASSES = 5
CLASS_IDS = np.arange(N_CLASSES)
CLASS_LABELS = ["C1", "C2", "C3", "C4", "C5"]
CV_FOLDS = 3
SAMPLING_SEED = 42
DEFAULT_SEEDS = (0, 1, 2, 3, 4, 5, 6)

# These are intentionally modest: model selection happens on the training
# designs only, while still being practical on a student laptop.
FULL_CONFIGS = [
    {"max_depth": 12, "min_samples_leaf": 10, "max_features": "sqrt"},
    {"max_depth": 20, "min_samples_leaf": 5, "max_features": "sqrt"},
    {"max_depth": 20, "min_samples_leaf": 5, "max_features": 1.0},
    {"max_depth": None, "min_samples_leaf": 20, "max_features": "sqrt"},
]
QUICK_CONFIGS = FULL_CONFIGS[:2]


def _load_module(name, file_path):
    """Load one of the existing feature-extraction scripts by file path."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load feature extractor from {}".format(file_path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_csv(path, rows):
    if not rows:
        return
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path):
    """Read a small generated CSV if it exists."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def _metrics_from_confusion(cm):
    """Return accuracy, macro F1, weighted F1, and per-class metrics."""
    cm = np.asarray(cm, dtype=np.int64)
    true_support = cm.sum(axis=1)
    predicted_support = cm.sum(axis=0)
    true_positive = np.diag(cm).astype(np.float64)
    precision = np.divide(
        true_positive,
        predicted_support,
        out=np.zeros(N_CLASSES, dtype=np.float64),
        where=predicted_support != 0,
    )
    recall = np.divide(
        true_positive,
        true_support,
        out=np.zeros(N_CLASSES, dtype=np.float64),
        where=true_support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros(N_CLASSES, dtype=np.float64),
        where=(precision + recall) != 0,
    )
    count = int(cm.sum())
    return {
        "n_samples": count,
        "accuracy": float(true_positive.sum() / count) if count else 0.0,
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=true_support)) if count else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": true_support,
    }


def _validated_design(task, engine, design, feature_names):
    """Load one design and return its feature matrix and scalar class delay."""
    block = engine.load_block(task, design)
    X = np.column_stack([block["cols"][name] for name in feature_names])
    delay_ns = np.asarray(task.class_value(block["y"]), dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != len(feature_names):
        raise ValueError("Unexpected feature matrix shape for {}".format(design))
    if len(X) != len(delay_ns):
        raise ValueError("Feature/label length mismatch for {}".format(design))
    if not np.isfinite(X).all() or not np.isfinite(delay_ns).all():
        raise ValueError("Non-finite values found for {}".format(design))
    return X.astype(np.float32, copy=False), delay_ns


def _sample_indices(n_samples, cap, design):
    """Fixed design-balanced sample; identical for every forest seed."""
    if cap is None or n_samples <= cap:
        return np.arange(n_samples)
    seed = (SAMPLING_SEED + zlib.crc32(design.encode("utf-8"))) % (2 ** 32)
    return np.sort(np.random.default_rng(seed).choice(n_samples, cap, replace=False))


def _load_training(task, engine, designs, feature_names, sample_cap):
    """Keep a bounded sample per design, but derive class edges from all rows."""
    sampled = {}
    all_delays = []
    full_sizes = {}
    for design in designs:
        started = time.time()
        X, delay_ns = _validated_design(task, engine, design, feature_names)
        full_sizes[design] = len(delay_ns)
        all_delays.append(delay_ns)
        selected = _sample_indices(len(delay_ns), sample_cap, design)
        sampled[design] = (X[selected], delay_ns[selected])
        print(
            "  {:15} full={:>9,} train_sample={:>7,}  {:.1f}s".format(
                design, len(delay_ns), len(selected), time.time() - started
            ),
            flush=True,
        )
        del X, delay_ns, selected
    class_edges = np.quantile(
        np.concatenate(all_delays), np.linspace(0, 1, N_CLASSES + 1)[1:-1]
    )
    return sampled, full_sizes, class_edges.astype(np.float64)


def _classes(delay_ns, class_edges):
    return np.digitize(delay_ns, class_edges).astype(np.int8)


def _make_forest(config, seed, n_estimators, max_samples, jobs):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="gini",
        bootstrap=True,
        max_samples=max_samples,
        n_jobs=jobs,
        random_state=seed,
        class_weight=None,
        **config,
    )


def _fold_map(designs, full_sizes):
    by_size = sorted(designs, key=lambda design: -full_sizes[design])
    return {design: index % CV_FOLDS for index, design in enumerate(by_size)}


def _stack_designs(data, designs, class_edges):
    X = np.concatenate([data[design][0] for design in designs], axis=0)
    y = np.concatenate(
        [_classes(data[design][1], class_edges) for design in designs], axis=0
    )
    return X, y


def _tune_forest(data, full_sizes, designs, class_edges, configs,
                 n_estimators, max_samples, jobs):
    """Select settings by design-grouped CV, never by held-out test score."""
    fold_for = _fold_map(designs, full_sizes)
    rows = []
    best_config = None
    best_score = -np.inf
    for config_id, config in enumerate(configs):
        design_scores = []
        started = time.time()
        for fold in range(CV_FOLDS):
            fit_designs = [d for d in designs if fold_for[d] != fold]
            validation_designs = [d for d in designs if fold_for[d] == fold]
            X_fit, y_fit = _stack_designs(data, fit_designs, class_edges)
            model = _make_forest(
                config, SAMPLING_SEED, n_estimators, max_samples, jobs
            )
            model.fit(X_fit, y_fit)
            del X_fit, y_fit
            for design in validation_designs:
                X_valid, delay_valid = data[design]
                y_valid = _classes(delay_valid, class_edges)
                prediction = model.predict(X_valid)
                cm = confusion_matrix(y_valid, prediction, labels=CLASS_IDS)
                score = _metrics_from_confusion(cm)["macro_f1"]
                design_scores.append(score)
                rows.append(
                    {
                        "config_id": config_id,
                        "fold": fold,
                        "design": design,
                        "macro_f1": score,
                        "params": json.dumps(config, sort_keys=True),
                    }
                )
            del model
            gc.collect()
        score = float(np.mean(design_scores))
        print(
            "CV config {} macro F1={:.4f} time={:.1f}s {}".format(
                config_id, score, time.time() - started, config
            ),
            flush=True,
        )
        if score > best_score:
            best_score = score
            best_config = dict(config)
    return best_config, best_score, rows


def _metric_row(task_name, seed, design, cm, feature_set, config,
                cv_score, fit_seconds, predict_seconds):
    metrics = _metrics_from_confusion(cm)
    row = {
        "task": task_name,
        "seed": seed,
        "design": design,
        "n_samples": metrics["n_samples"],
        "accuracy": metrics["accuracy"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "feature_set": feature_set,
        "cv_macro_f1": cv_score,
        "fit_seconds": fit_seconds,
        "predict_seconds": predict_seconds,
        "params": json.dumps(config, sort_keys=True),
    }
    for index, label in enumerate(CLASS_LABELS):
        row["{}_precision".format(label)] = metrics["precision"][index]
        row["{}_recall".format(label)] = metrics["recall"][index]
        row["{}_f1".format(label)] = metrics["f1"][index]
        row["{}_support".format(label)] = int(metrics["support"][index])
    return row


def _evaluate_seed(task_name, task, engine, model, seed, test_designs,
                   feature_names, feature_set, class_edges, config,
                   cv_score, fit_seconds):
    rows = []
    confusion_rows = []
    pooled_cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    total_predict_seconds = 0.0
    for design in test_designs:
        X_test, delay_ns = _validated_design(task, engine, design, feature_names)
        y_true = _classes(delay_ns, class_edges)
        started = time.time()
        y_pred = model.predict(X_test).astype(np.int8)
        predict_seconds = time.time() - started
        total_predict_seconds += predict_seconds
        cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
        pooled_cm += cm
        row = _metric_row(
            task_name, seed, design, cm, feature_set, config, cv_score,
            fit_seconds, predict_seconds,
        )
        rows.append(row)
        print(
            "  seed {:>2} {:15} n={:>9,} accuracy={:.4f} macro_f1={:.4f}".format(
                seed, design, row["n_samples"], row["accuracy"], row["macro_f1"]
            ),
            flush=True,
        )
        del X_test, delay_ns, y_true, y_pred

    pooled = _metric_row(
        task_name, seed, "POOLED", pooled_cm, feature_set, config, cv_score,
        fit_seconds, total_predict_seconds,
    )
    rows.append(pooled)
    for true_index, true_label in enumerate(CLASS_LABELS):
        for predicted_index, predicted_label in enumerate(CLASS_LABELS):
            confusion_rows.append(
                {
                    "task": task_name,
                    "seed": seed,
                    "true_class": true_label,
                    "predicted_class": predicted_label,
                    "count": int(pooled_cm[true_index, predicted_index]),
                }
            )
    print(
        "  seed {:>2} POOLED          n={:>9,} accuracy={:.4f} macro_f1={:.4f}".format(
            seed, pooled["n_samples"], pooled["accuracy"], pooled["macro_f1"]
        ),
        flush=True,
    )
    return rows, confusion_rows, pooled_cm


def _save_plots(output_dir, task_title, pooled_rows, confusion_matrices,
                importance_rows):
    seeds = [row["seed"] for row in pooled_rows]
    macro = [row["macro_f1"] for row in pooled_rows]
    weighted = [row["weighted_f1"] for row in pooled_rows]
    accuracy = [row["accuracy"] for row in pooled_rows]
    x = np.arange(len(seeds))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(x - width, accuracy, width, label="accuracy")
    ax.bar(x, macro, width, label="macro F1")
    ax.bar(x + width, weighted, width, label="weighted F1")
    ax.set(
        xticks=x,
        xticklabels=seeds,
        ylim=(0, 1),
        xlabel="random seed",
        ylabel="score",
        title="{} Random Forest held-out scores".format(task_title),
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "seed_scores.png", dpi=220)
    plt.close(fig)

    mean_cm = np.mean(confusion_matrices, axis=0)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(mean_cm, cmap="Blues")
    fig.colorbar(image, ax=ax, label="mean samples across seeds")
    ax.set_xticks(CLASS_IDS, CLASS_LABELS)
    ax.set_yticks(CLASS_IDS, CLASS_LABELS)
    ax.set(
        xlabel="predicted class",
        ylabel="true class",
        title="{} pooled confusion matrix".format(task_title),
    )
    threshold = mean_cm.max() / 2 if mean_cm.size else 0
    for row in range(N_CLASSES):
        for column in range(N_CLASSES):
            ax.text(
                column,
                row,
                "{:.0f}".format(mean_cm[row, column]),
                ha="center",
                va="center",
                color="white" if mean_cm[row, column] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=220)
    plt.close(fig)

    by_feature = {}
    for row in importance_rows:
        by_feature.setdefault(row["feature"], []).append(row["importance"])
    ordered = sorted(by_feature, key=lambda name: np.mean(by_feature[name]))[-20:]
    means = [np.mean(by_feature[name]) for name in ordered]
    stds = [np.std(by_feature[name], ddof=1) if len(by_feature[name]) > 1 else 0.0
            for name in ordered]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.32 * len(ordered))))
    ax.barh(ordered, means, xerr=stds, color="#3987e5")
    ax.set(
        xlabel="mean impurity-based importance",
        title="{} Random Forest feature importance".format(task_title),
    )
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=220)
    plt.close(fig)


def _summary_rows(task_name, pooled_rows):
    rows = []
    for metric in ("accuracy", "macro_f1", "weighted_f1", "fit_seconds",
                   "predict_seconds"):
        values = np.array([row[metric] for row in pooled_rows], dtype=np.float64)
        rows.append(
            {
                "task": task_name,
                "metric": metric,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "n_seeds": len(values),
            }
        )
    return rows


def run_task(task_name, task, engine, feature_set, train_designs, test_designs,
             output_root, configs, sample_cap, tune_estimators,
             final_estimators, max_samples, jobs, seeds, quick):
    feature_names = list(task.featsets[feature_set])
    output_dir = output_root / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== {} ===".format(task.title), flush=True)
    print("Loading {} training designs".format(len(train_designs)), flush=True)
    train, full_sizes, class_edges = _load_training(
        task, engine, train_designs, feature_names, sample_cap
    )
    print("Training-derived class boundaries (ns): {}".format(class_edges.tolist()))
    best_config, cv_score, cv_rows = _tune_forest(
        train, full_sizes, train_designs, class_edges, configs,
        tune_estimators, max_samples, jobs,
    )
    _write_csv(output_dir / "cross_validation.csv", cv_rows)
    print(
        "Selected by training-only CV: macro F1={:.4f} {}".format(
            cv_score, best_config
        ),
        flush=True,
    )

    X_train, y_train = _stack_designs(train, train_designs, class_edges)
    del train
    gc.collect()
    metric_rows = []
    confusion_rows = []
    confusion_matrices = []
    importance_rows = []
    pooled_rows = []

    for seed in seeds:
        model = _make_forest(
            best_config, seed, final_estimators, max_samples, jobs
        )
        started = time.time()
        model.fit(X_train, y_train)
        fit_seconds = time.time() - started
        print(
            "seed {} fitted {:,} samples in {:.1f}s".format(
                seed, len(y_train), fit_seconds
            ),
            flush=True,
        )
        rows, cms, pooled_cm = _evaluate_seed(
            task_name, task, engine, model, seed, test_designs, feature_names,
            feature_set, class_edges, best_config, cv_score, fit_seconds,
        )
        metric_rows.extend(rows)
        confusion_rows.extend(cms)
        confusion_matrices.append(pooled_cm)
        pooled_rows.append(rows[-1])
        for feature, importance in zip(feature_names, model.feature_importances_):
            importance_rows.append(
                {
                    "task": task_name,
                    "seed": seed,
                    "feature": feature,
                    "importance": float(importance),
                }
            )
        del model
        gc.collect()

    _write_csv(output_dir / "metrics.csv", metric_rows)
    _write_csv(output_dir / "confusion_matrix.csv", confusion_rows)
    _write_csv(output_dir / "feature_importance.csv", importance_rows)
    summary_rows = _summary_rows(task_name, pooled_rows)
    _write_csv(output_dir / "summary.csv", summary_rows)
    _save_plots(
        output_dir, task.title, pooled_rows, confusion_matrices, importance_rows
    )

    metadata = {
        "task": task.title,
        "quick": quick,
        "feature_set": feature_set,
        "features": feature_names,
        "train_designs": train_designs,
        "test_designs": test_designs,
        "full_train_samples": int(sum(full_sizes.values())),
        "fit_train_samples": int(len(y_train)),
        "sample_cap_per_design": sample_cap,
        "class_edges_ns": class_edges.tolist(),
        "selection_rule": "highest mean per-design macro F1 from 3-fold design-grouped CV on training designs",
        "cv_macro_f1": cv_score,
        "selected_config": best_config,
        "tune_estimators": tune_estimators,
        "final_estimators": final_estimators,
        "max_samples_per_tree": max_samples,
        "seeds": seeds,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    del X_train, y_train
    gc.collect()

    macro = next(row for row in summary_rows if row["metric"] == "macro_f1")
    print(
        "{} complete: pooled macro F1 = {:.4f} +/- {:.4f}; results in {}".format(
            task.title, macro["mean"], macro["std"], output_dir
        ),
        flush=True,
    )
    return summary_rows


def main():
    parser = argparse.ArgumentParser(
        description="Random Forest delay-classification baseline"
    )
    parser.add_argument("--task", choices=("net", "cell", "both"), default="both")
    parser.add_argument("--quick", action="store_true", help="small 5/2-design smoke test")
    parser.add_argument("--seeds", nargs="+", type=int, help="model random seeds")
    parser.add_argument("--sample-cap", type=int, help="maximum training rows per design")
    parser.add_argument("--tune-estimators", type=int, help="trees per CV forest")
    parser.add_argument("--n-estimators", type=int, help="trees per final forest")
    parser.add_argument("--max-samples", type=float, default=0.8,
                        help="bootstrap fraction per tree (default 0.8)")
    parser.add_argument("--jobs", type=int, default=-1,
                        help="parallel CPU jobs; -1 uses all logical cores")
    parser.add_argument("--out", help="results root directory")
    args = parser.parse_args()
    if not 0 < args.max_samples <= 1:
        parser.error("--max-samples must be in (0, 1]")

    net_delay_file, cell_delay_file = paths.paths().dependency_paths()
    net_engine = _load_module("random_forest_net_kmeans", net_delay_file)
    cell_module = _load_module("random_forest_cell_kmeans", cell_delay_file)

    if args.quick:
        train_designs = list(net_engine.QUICK_TRAIN)
        test_designs = list(net_engine.QUICK_TEST)
        configs = QUICK_CONFIGS
        sample_cap = args.sample_cap or 5000
        tune_estimators = args.tune_estimators or 10
        final_estimators = args.n_estimators or 20
        seeds = args.seeds if args.seeds is not None else [42]
    else:
        train_designs = sorted(net_engine.train_data_keys)
        test_designs = sorted(net_engine.test_data_keys)
        configs = FULL_CONFIGS
        sample_cap = args.sample_cap or 40000
        tune_estimators = args.tune_estimators or 30
        final_estimators = args.n_estimators or 100
        seeds = args.seeds if args.seeds is not None else list(DEFAULT_SEEDS)
        if len(train_designs) != 14 or len(test_designs) != 7:
            raise ValueError("Expected the fixed 14/7 split across 21 designs")

    result_dir = Path(paths.paths().dataset_files()[3])
    output_root = (
        Path(args.out)
        if args.out
        else result_dir / ("random_forest_baseline_quick" if args.quick
                           else "random_forest_baseline")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    all_summary = []
    if args.task in ("net", "both"):
        all_summary.extend(
            run_task(
                "net_delay", net_engine.NetDelayTask(), net_engine, "full",
                train_designs, test_designs, output_root, configs, sample_cap,
                tune_estimators, final_estimators, args.max_samples, args.jobs,
                seeds, args.quick,
            )
        )
    if args.task in ("cell", "both"):
        all_summary.extend(
            run_task(
                "cell_delay", cell_module.CellDelayTask(), cell_module.km,
                "lut_sta_ctx", train_designs, test_designs, output_root,
                configs, sample_cap, tune_estimators, final_estimators,
                args.max_samples, args.jobs, seeds, args.quick,
            )
        )
    # Tasks are often run separately on a laptop.  Rebuild the root summary
    # from both task folders so a later cell-only run does not hide the
    # earlier net-delay summary (or vice versa).
    combined_summary = []
    for folder in ("net_delay", "cell_delay"):
        combined_summary.extend(_read_csv(output_root / folder / "summary.csv"))
    _write_csv(output_root / "summary.csv", combined_summary or all_summary)
    print("\nAll requested tasks completed. Results: {}".format(output_root))


if __name__ == "__main__":
    main()
