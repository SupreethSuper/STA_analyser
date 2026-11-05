"""Train decision trees for the existing net- and cell-delay features.

The K-means modules provide feature extraction and their shared fixed design
split. This script uses the extracted features directly for classification:
five delay classes are defined by training-only quantiles, tree settings are
selected with design-grouped cross-validation, and the final models are
evaluated on the seven held-out designs. Results for each task are written
under results/net_delay and results/cell_delay.
"""

import csv
import importlib.util
import json
import sys
import zlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree

import paths


# Class IDs are zero-based for scikit-learn; the corresponding display names
# are one-based to match the C1-C5 convention used in the project.
N_CLASSES = 5
CLASS_IDS = np.arange(N_CLASSES)
CLASS_LABELS = ["C1", "C2", "C3", "C4", "C5"]
CV_FOLDS = 3
# Bound cross-validation cost on very large designs. This does not limit the
# final training fit or held-out evaluation, which use every sample.
CV_SAMPLE_CAP = 40000
SEED = 42

# Include the template's parameters as a candidate and compare alternatives
# using only training-design folds, never the held-out test designs.
TREE_CONFIGS = [
    {"criterion": criterion, "max_depth": depth, "min_samples_split": split,
     "min_samples_leaf": leaf, "random_state": SEED}
    for criterion in ("gini", "entropy")
    for depth, split, leaf in (
        (6, 20, 10),
        (10, 20, 10),
        (16, 50, 25),
        (None, 100, 50),
    )
]


def _load_module(name, file_path):
    """Load an extractor from its configured path without requiring a package."""
    spec = importlib.util.spec_from_file_location(name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError("Cannot load feature extractor from {}".format(file_path))
    module = importlib.util.module_from_spec(spec)
    # Register first so module-level imports/dataclasses can resolve the module
    # while its source executes.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_design(task, engine, design, feature_names):
    """Get one design's ordered feature matrix and the delay used for classes."""
    # load_block owns the extractor's feature cache; the requested feature order
    # is the order declared by that task's feature-set definition.
    block = engine.load_block(task, design)
    X = np.column_stack([block["cols"][name] for name in feature_names])
    # The task chooses the class-defining delay statistic (currently the worse
    # late-corner rise/fall delay), keeping labels aligned with K-means.
    delay_ns = np.asarray(task.class_value(block["y"]), dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != len(feature_names):
        raise ValueError("Unexpected feature matrix shape for {}".format(design))
    if len(X) != len(delay_ns):
        raise ValueError("Feature/label length mismatch for {}".format(design))
    if not np.isfinite(X).all():
        raise ValueError("Non-finite feature values found for {}".format(design))
    if not np.isfinite(delay_ns).all():
        raise ValueError("Non-finite delay labels found for {}".format(design))
    return X.astype(np.float32, copy=False), delay_ns


def _sample_for_cv(X, y, design):
    """Deterministically cap a design's CV rows while retaining full test data."""
    if len(y) <= CV_SAMPLE_CAP:
        return X, y
    # CRC32 avoids Python's process-randomized hash, so each design gets the
    # same sample subset across independent runs.
    seed = SEED + zlib.crc32(design.encode("utf-8"))
    indices = np.sort(
        np.random.default_rng(seed).choice(len(y), CV_SAMPLE_CAP, replace=False)
    )
    return X[indices], y[indices]


def _tune_tree(train, class_edges):
    """Select tree hyperparameters using three folds grouped by design.

    Every class boundary is computed before this function from training
    designs only. Folds also contain whole designs, so samples from one design
    cannot appear in both a fold's fit and validation sets.
    """
    cv_data = {}
    for design, (X, delay_ns) in train.items():
        y = np.digitize(delay_ns, class_edges).astype(np.int8)
        cv_data[design] = _sample_for_cv(X, y, design)

    # Interleave large designs across folds to reduce imbalance in sample
    # counts while preserving design-level separation.
    designs_by_size = sorted(train, key=lambda design: -len(train[design][1]))
    fold_for_design = {
        design: index % CV_FOLDS for index, design in enumerate(designs_by_size)
    }

    best_config = None
    best_score = -np.inf
    for config in TREE_CONFIGS:
        design_scores = []
        for fold in range(CV_FOLDS):
            fit_designs = [
                design for design in train if fold_for_design[design] != fold
            ]
            validation_designs = [
                design for design in train if fold_for_design[design] == fold
            ]
            X_fit = np.concatenate([cv_data[d][0] for d in fit_designs], axis=0)
            y_fit = np.concatenate([cv_data[d][1] for d in fit_designs], axis=0)
            tree = DecisionTreeClassifier(**config)
            tree.fit(X_fit, y_fit)
            for design in validation_designs:
                X_valid, y_valid = cv_data[design]
                prediction = tree.predict(X_valid)
                cm = confusion_matrix(y_valid, prediction, labels=CLASS_IDS)
                # Score each design equally, rather than allowing a large
                # design to dominate the hyperparameter choice by row count.
                design_scores.append(_metrics_from_confusion(cm)["macro_f1"])

        score = float(np.mean(design_scores))
        print("CV macro F1 {:.4f}: {}".format(score, config))
        if score > best_score:
            best_score = score
            best_config = config

    return best_config, best_score


def _metrics_from_confusion(cm):
    """Calculate fixed-five-class metrics directly from a confusion matrix.

    Explicitly retaining all five classes gives stable metric columns even
    when a particular design contains no examples of one or more classes.
    """
    cm = np.asarray(cm, dtype=np.int64)
    true_support = cm.sum(axis=1)
    predicted_support = cm.sum(axis=0)
    true_positive = np.diag(cm).astype(np.float64)
    precision = np.divide(
        true_positive, predicted_support,
        out=np.zeros(N_CLASSES, dtype=np.float64), where=predicted_support != 0
    )
    recall = np.divide(
        true_positive, true_support,
        out=np.zeros(N_CLASSES, dtype=np.float64), where=true_support != 0
    )
    f1 = np.divide(
        2 * precision * recall, precision + recall,
        out=np.zeros(N_CLASSES, dtype=np.float64),
        where=(precision + recall) != 0
    )
    count = int(cm.sum())
    return {
        "n_samples": count,
        "accuracy": float(true_positive.sum() / count) if count else 0.0,
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.average(f1, weights=true_support))
        if count else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": true_support,
    }


def evaluate(y_true, y_pred):
    """Return classification metrics and the fixed five-class confusion matrix.

    Metrics for a design or for all held-out designs pooled together can be
    calculated through this same path.
    """
    cm = confusion_matrix(y_true, y_pred, labels=CLASS_IDS)
    return _metrics_from_confusion(cm), cm


def _metrics_row(design, cm, params, cv_score, feature_set, train_designs,
                 class_edges):
    """Flatten aggregate and per-class scores into one CSV-compatible row."""
    result = _metrics_from_confusion(cm)
    row = {
        "design": design,
        "n_samples": result["n_samples"],
        "accuracy": result["accuracy"],
        "macro_f1": result["macro_f1"],
        "weighted_f1": result["weighted_f1"],
        "feature_set": feature_set,
        "cv_macro_f1": cv_score,
        "tree_params": json.dumps(params, sort_keys=True),
        "training_designs": ";".join(train_designs),
    }
    for index, label in enumerate(CLASS_LABELS):
        row["{}_precision".format(label)] = result["precision"][index]
        row["{}_recall".format(label)] = result["recall"][index]
        row["{}_f1".format(label)] = result["f1"][index]
        row["{}_support".format(label)] = int(result["support"][index])
        # The last class is unbounded above and therefore has no upper edge.
        row["edge_{}_upper_ns".format(label)] = (
            float(class_edges[index]) if index < len(class_edges) else ""
        )
    for true_index, true_label in enumerate(CLASS_LABELS):
        for predicted_index, predicted_label in enumerate(CLASS_LABELS):
            row["cm_{}_{}".format(true_label, predicted_label)] = int(
                cm[true_index, predicted_index]
            )
    return row


def _write_metrics(path, rows):
    """Write per-design rows followed by the pooled row."""
    fieldnames = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_predictions(path, design, delay_ns, true_classes, predicted_classes,
                       probabilities, model_classes):
    """Append one held-out design's per-sample labels and class probabilities.

    A decision tree only returns probability columns for classes observed
    during fitting. Expand those columns into the fixed C1-C5 order so the CSV
    schema is identical for every task and model.
    """
    fields = [
        "design", "sample_index", "delay_ns", "true_class", "true_class_name",
        "predicted_class", "predicted_class_name", "correct",
    ] + ["prob_{}".format(label) for label in CLASS_LABELS]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        if write_header:
            writer.writeheader()
        class_probabilities = np.zeros((len(delay_ns), N_CLASSES), dtype=np.float32)
        for column, class_id in enumerate(model_classes):
            class_probabilities[:, int(class_id)] = probabilities[:, column]
        for index, (delay, actual, predicted) in enumerate(
            zip(delay_ns, true_classes, predicted_classes)
        ):
            row = {
                "design": design,
                "sample_index": index,
                "delay_ns": float(delay),
                "true_class": int(actual),
                "true_class_name": CLASS_LABELS[int(actual)],
                "predicted_class": int(predicted),
                "predicted_class_name": CLASS_LABELS[int(predicted)],
                "correct": int(actual == predicted),
            }
            row.update({
                "prob_{}".format(label): float(class_probabilities[index, class_id])
                for class_id, label in enumerate(CLASS_LABELS)
            })
            writer.writerow(row)


def _save_visualizations(output_dir, tree, feature_names, pooled_cm, design_rows):
    """Save tree, pooled confusion, per-design score, and feature plots."""
    # Limit the displayed tree depth for readability; the fitted estimator
    # itself remains at the CV-selected maximum depth.
    fig, ax = plt.subplots(figsize=(24, 12))
    plot_tree(
        tree,
        filled=True,
        rounded=True,
        feature_names=feature_names,
        class_names=[CLASS_LABELS[int(class_id)] for class_id in tree.classes_],
        max_depth=3,
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(output_dir / "decision_tree.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    # Counts in this matrix are pooled over all seven held-out designs.
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(pooled_cm, cmap="Blues")
    fig.colorbar(image, ax=ax, label="samples")
    ax.set_xticks(CLASS_IDS, CLASS_LABELS)
    ax.set_yticks(CLASS_IDS, CLASS_LABELS)
    ax.set(xlabel="predicted class", ylabel="true class",
           title="Pooled held-out designs")
    threshold = pooled_cm.max() / 2 if pooled_cm.size else 0
    for row in range(N_CLASSES):
        for column in range(N_CLASSES):
            ax.text(
                column, row, str(int(pooled_cm[row, column])),
                ha="center", va="center",
                color="white" if pooled_cm[row, column] > threshold else "black",
            )
    fig.tight_layout()
    fig.savefig(output_dir / "confusion_matrix.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    # Display each held-out design's macro-F1 separately.
    ordered = sorted(design_rows, key=lambda row: row["design"])
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(
        [row["design"] for row in ordered],
        [row["macro_f1"] for row in ordered],
        color="#3987e5",
    )
    ax.set(ylim=(0, 1), ylabel="Macro F1", title="Held-out design performance")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output_dir / "design_macro_f1.png", dpi=250, bbox_inches="tight")
    plt.close(fig)

    # Show up to the twenty most informative inputs, in ascending order for
    # a readable horizontal bar chart.
    order = np.argsort(tree.feature_importances_)[-20:]
    fig, ax = plt.subplots(figsize=(8, max(4, 0.3 * len(order))))
    ax.barh(
        [feature_names[index] for index in order],
        tree.feature_importances_[order],
        color="#3987e5",
    )
    ax.set(xlabel="impurity-based importance", title="Decision-tree feature importance")
    fig.tight_layout()
    fig.savefig(output_dir / "feature_importance.png", dpi=250, bbox_inches="tight")
    plt.close(fig)


def run_task(folder_name, task, engine, feature_set, train_designs,
             test_designs, result_dir):
    """Fit, evaluate, and save all outputs for one delay-prediction task."""
    feature_names = list(task.featsets[feature_set])
    output_dir = result_dir / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n{}: loading {} training designs".format(task.title, len(train_designs)))
    train = {
        design: _load_design(task, engine, design, feature_names)
        for design in train_designs
    }
    train_delays = np.concatenate([train[design][1] for design in train_designs])
    # Five equal-frequency bins are derived exclusively from the concatenated
    # training-design delays and reused unchanged for CV, fit, and test.
    class_edges = np.quantile(
        train_delays, np.linspace(0, 1, N_CLASSES + 1)[1:-1]
    )
    print("Training-derived class boundaries (ns):", class_edges.tolist())

    best_params, cv_score = _tune_tree(train, class_edges)
    print("Selected tree settings (design-grouped CV macro F1 {:.4f}): {}".format(
        cv_score, best_params
    ))

    # The bins use four internal cut points and produce class IDs 0 through 4.
    y_train = np.concatenate([
        np.digitize(train[design][1], class_edges).astype(np.int8)
        for design in train_designs
    ])
    X_train = np.concatenate([train[design][0] for design in train_designs], axis=0)
    del train
    tree = DecisionTreeClassifier(**best_params)
    tree.fit(X_train, y_train)
    del X_train, y_train

    predictions_path = output_dir / "pooled_predictions.csv"
    # Do not accidentally append this run's predictions to an older run.
    if predictions_path.exists():
        predictions_path.unlink()
    metric_rows = []
    design_metric_rows = []
    pooled_cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)

    print("{}: evaluating {} held-out designs".format(task.title, len(test_designs)))
    for design in test_designs:
        X_test, delay_ns = _load_design(task, engine, design, feature_names)
        y_true = np.digitize(delay_ns, class_edges).astype(np.int8)
        y_pred = tree.predict(X_test).astype(np.int8)
        probabilities = tree.predict_proba(X_test)
        metrics, cm = evaluate(y_true, y_pred)
        pooled_cm += cm
        metric_row = _metrics_row(
            design, cm, best_params, cv_score, feature_set, train_designs, class_edges
        )
        metric_rows.append(metric_row)
        design_metric_rows.append(metric_row)
        _write_predictions(
            predictions_path, design, delay_ns, y_true, y_pred, probabilities,
            tree.classes_,
        )
        print("  {}: n={} accuracy={:.4f} macro_f1={:.4f}".format(
            design, metrics["n_samples"], metrics["accuracy"], metrics["macro_f1"]
        ))
        del X_test, delay_ns, y_true, y_pred, probabilities

    # Summing confusion matrices pools the individual predictions without
    # weighting designs equally; the resulting summary is sample-weighted.
    metric_rows.append(_metrics_row(
        "POOLED", pooled_cm, best_params, cv_score, feature_set,
        train_designs, class_edges,
    ))
    _write_metrics(output_dir / "metrics.csv", metric_rows)
    _save_visualizations(output_dir, tree, feature_names, pooled_cm, design_metric_rows)
    pooled_metrics = _metrics_from_confusion(pooled_cm)
    print("  POOLED: n={} accuracy={:.4f} macro_f1={:.4f}".format(
        pooled_metrics["n_samples"], pooled_metrics["accuracy"],
        pooled_metrics["macro_f1"],
    ))


def main():
    """Load the project extractors, obtain their split, and run both tasks."""
    net_delay_file, cell_delay_file = paths.paths().dependency_paths()
    net_engine = _load_module("decision_tree_net_kmeans", net_delay_file)
    cell_module = _load_module("decision_tree_cell_kmeans", cell_delay_file)

    train_designs = sorted(net_engine.train_data_keys)
    test_designs = sorted(net_engine.test_data_keys)
    # Fail clearly if the extractor's canonical split changes unexpectedly.
    if len(train_designs) != 14 or len(test_designs) != 7:
        raise ValueError("Expected the existing 14/7 split across 21 designs")

    result_dir = Path(paths.paths().dataset_files()[3])
    result_dir.mkdir(parents=True, exist_ok=True)
    run_task(
        "net_delay", net_engine.NetDelayTask(), net_engine, "full",
        train_designs, test_designs, result_dir,
    )
    run_task(
        "cell_delay", cell_module.CellDelayTask(), cell_module.km, "lut_sta_ctx",
        train_designs, test_designs, result_dir,
    )


if __name__ == "__main__":
    main()
