"""K-means for pre-routing NET DELAY prediction on the TimingPredict graphs.

Built from library code only: scikit-learn for the model (KMeans /
MiniBatchKMeans, StandardScaler, NearestNeighbors, KNeighborsRegressor, Ridge)
and the scores (r2_score, confusion_matrix, f1_score, ...), numpy for the
array work.  Two scripts share it:

    k_means_numpy_variation.py               net delay  (this file)
    k_means_Cell_delay_numpy_variation.py    cell delay (imports this file)

The task.  One sample per net arc (driver pin -> sink pin) of data/8_rat; the
label is the arc's wire delay 4-vector [E_rise, E_fall, L_rise, L_fall] and
it is scored, like train_gnn.py:test_netdelay, in the
n_net_delays_log = log(1e-4 + d) + 7.6 space.  Features are pre-route only:
placement geometry of the arc and its net, and Liberty pin capacitances.

K-means as a predictor.  The features are clustered, then a head turns the
clustering into a prediction:
    mean              label mean of the nearest centroid (traditional k-means)
    soft_m{m}         KNeighborsRegressor(m, weights='distance') over the centroids
    ridge_l{lam}      one Ridge per cluster, shrunk toward a global Ridge fit
    softridge_m_l     the m nearest clusters' ridge models, inverse-distance blended
    vote              majority delay class of the nearest centroid (classifier)

The search, all of it written as data to results/<script>/:
    screen    every feature set x feature transform, at one k
    k_sweep   the two best (feature set, transform) pairs over k
    engine    random init, restarts, mini-batch, design-balanced sampling
Each configuration is validated by 3-fold leave-designs-out CV on the 14
training designs, then refit on all 14 and scored on the 7 test designs.  The
variant with the best validation R^2 is refit while tracking train and test
error per iteration; it gets the plots, confusion matrices and cluster tables.

Classes (confusion matrix, F1): 5 equal-frequency bins of the late-corner
worst delay max(L_rise, L_fall), bin edges from the training designs.  A
regression head's prediction is binned the same way.

Run with the project venv; the per-design feature cache needs torch/dgl once,
after that only numpy, sklearn and matplotlib:

    ..\\TimingPredict\\.venv312\\Scripts\\python.exe k_means_numpy_variation.py            # full search
    ..\\TimingPredict\\.venv312\\Scripts\\python.exe k_means_numpy_variation.py --quick    # 7 small designs, small grid
"""
import argparse
import csv
import functools
import json
import os
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans, kmeans_plusplus
from sklearn.linear_model import Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support, r2_score)
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.preprocessing import StandardScaler

# ---- paths: relative to this file's folder (STA_analyser/) ----
dataset_path = r"../TimingPredict/data/8_rat"    # <design>.graph.bin
cache_path   = r"../TimingPredict/cache/kmeans"  # feature cache *.npz
tee_path     = r"../TimingPredict"               # folder holding tee.py
result_path  = r"./results"                      # run outputs

# anchor them to this file, so the launch folder doesn't matter
script_dir   = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(script_dir, dataset_path)
cache_path   = os.path.join(script_dir, cache_path)
tee_path     = os.path.join(script_dir, tee_path)
result_path  = os.path.join(script_dir, result_path)

sys.path.append(tee_path)   # so `import tee` finds TimingPredict/tee.py
import tee  # noqa: E402

log = functools.partial(print, flush=True)

# ---------------------------------------------------------------- the split
# the fixed split of data_graph.py (14 train / 7 test).  Re-derived rather
# than imported: importing data_graph loads all 21 graphs at once (several
# GB) and topologically sorts each one.
random.seed(8026728)
available_data = 'blabla usb_cdc_core BM64 jpeg_encoder salsa20 usbf_device aes128 wbqspiflash aes192 cic_decimator xtea aes256 des spm y_huff aes_cipher picorv32a synth_ram zipdiv genericfir usb'.split()
train_data_keys = random.sample(available_data, 14)
test_data_keys = [k for k in available_data if k not in train_data_keys]

# --quick: the 7-design subset of checkpoints/smoke1 (5 train / 2 test)
QUICK_TRAIN = ['cic_decimator', 'usb', 'usb_cdc_core', 'wbqspiflash', 'zipdiv']
QUICK_TEST = ['spm', 'xtea']

SEED = 0
N_FOLDS = 3
N_CLASSES = 5
BALANCED_CAP = 40000            # samples kept per design under sampling='balanced'
SOFT_MS = (3, 8)
RIDGE_LAMS = (0.01, 0.1, 1, 10, 100)
SOFTRIDGE = ((3, 1), (3, 10), (8, 10))
M_MAX = max(SOFT_MS + tuple(m for m, _ in SOFTRIDGE))
HEADS = (['mean'] + ['soft_m{}'.format(m) for m in SOFT_MS]
         + ['ridge_l{:g}'.format(l) for l in RIDGE_LAMS]
         + ['softridge_m{}_l{:g}'.format(m, l) for m, l in SOFTRIDGE])
CFG_KEYS = ('featset', 'transform', 'k', 'init', 'n_init', 'algorithm', 'sampling')
BASE = dict(init='k-means++', n_init=1, algorithm='lloyd', sampling='all')

# ====================================================== the sklearn engine
ENGINE = 'sklearn'


class SkKMeans:
    """KMeans / MiniBatchKMeans behind the interface Runner drives: fit(X,
    callback), centers_, inertia_, n_iter_, best_run_.

    Without a callback this is a plain KMeans / MiniBatchKMeans fit.  With
    one (the tracked refit of the selected variant), Lloyd is stepped one
    iteration at a time -- KMeans(init=C, n_init=1, max_iter=1) from the
    previous centres, which walks the same path as one uninterrupted run --
    and mini-batch one epoch of partial_fit at a time.  callback(run,
    iteration, centres) runs after the initialisation (iteration 0) and
    after every step."""

    def __init__(self, k, init='k-means++', n_init=1, algorithm='lloyd', max_iter=100,
                 tol=1e-4, batch_size=4096, max_epochs=30, seed=SEED):
        self.k, self.init, self.n_init, self.algorithm = k, init, n_init, algorithm
        self.max_iter, self.tol, self.seed = max_iter, tol, seed
        self.batch_size, self.max_epochs = batch_size, max_epochs

    def fit(self, X, callback=None):
        if callback is None:
            if self.algorithm == 'lloyd':
                est = KMeans(self.k, init=self.init, n_init=self.n_init, max_iter=self.max_iter,
                             tol=self.tol, random_state=self.seed)
            else:
                est = MiniBatchKMeans(self.k, init=self.init, n_init=self.n_init,
                                      batch_size=self.batch_size, max_iter=self.max_epochs,
                                      random_state=self.seed)
            est.fit(X)
            self.centers_ = est.cluster_centers_.astype(np.float32)
            self.inertia_, self.n_iter_, self.best_run_ = float(est.inertia_), int(est.n_iter_), 0
            return self
        # sklearn's stopping rule: summed squared centre shift <= tol * mean feature variance
        tol = self.tol * float(X.var(0).mean())
        best = None
        for run in range(self.n_init):
            rs = self.seed + run
            if self.init == 'k-means++':
                C = kmeans_plusplus(X, self.k, random_state=rs)[0]
            else:
                C = X[np.random.default_rng(rs).choice(len(X), self.k, replace=False)]
            callback(run, 0, C)
            if self.algorithm == 'lloyd':
                for it in range(1, self.max_iter + 1):
                    est = KMeans(self.k, init=C, n_init=1, max_iter=1, random_state=rs).fit(X)
                    shift = float(((est.cluster_centers_ - C) ** 2).sum())
                    C = est.cluster_centers_
                    callback(run, it, C)
                    if shift <= tol:
                        break
            else:
                est = MiniBatchKMeans(self.k, init=C, n_init=1, batch_size=self.batch_size,
                                      random_state=rs)
                rng = np.random.default_rng(rs)
                for it in range(1, self.max_epochs + 1):
                    perm = rng.permutation(len(X))
                    for lo in range(0, len(X), self.batch_size):
                        est.partial_fit(X[perm[lo:lo + self.batch_size]])
                    shift = float(((est.cluster_centers_ - C) ** 2).sum())
                    C = est.cluster_centers_.copy()
                    callback(run, it, C)
                    if shift <= tol:
                        break
            inertia = -est.score(X)
            if best is None or inertia < best[1]:
                best = (C.astype(np.float32), inertia, it, run)
        self.centers_, self.inertia_, self.n_iter_, self.best_run_ = best
        return self


def neighbors(C, X, m):
    """The m nearest centres of every row of X, nearest first, as (index,
    squared distance), both (n, m)."""
    dist, idx = NearestNeighbors(n_neighbors=min(m, len(C))).fit(C).kneighbors(X)
    return idx, (dist ** 2).astype(np.float32)


def _aug(X):
    return np.hstack([X, np.ones((len(X), 1), X.dtype)])


def cluster_means(lab, k, Y):
    """Per-cluster label mean; an empty cluster gets the global mean."""
    cnt = np.bincount(lab, minlength=k)
    mu = np.stack([np.bincount(lab, weights=Y[:, c], minlength=k)
                   for c in range(Y.shape[1])], 1) / np.maximum(cnt, 1)[:, None]
    mu[cnt == 0] = Y.mean(0)
    return mu.astype(np.float32)


def blend_linear(X, W, idx, wts):
    """sum_q wts[:, q] * ([x 1] . W[idx[:, q]]), in row blocks."""
    out = np.zeros((len(X), W.shape[2]), dtype=np.float32)
    step = 1 << 16
    for lo in range(0, len(X), step):
        xa = _aug(X[lo:lo + step])
        for q in range(idx.shape[1]):
            out[lo:lo + step] += wts[lo:lo + step, q:q + 1] * np.einsum(
                'np,npt->nt', xa, W[idx[lo:lo + step, q]])
    return out


def idw(d2):
    """Inverse-distance weights over the nearest centres, as
    KNeighborsRegressor's weights='distance', for the softridge blend."""
    w = 1.0 / (np.sqrt(d2) + 1e-6)
    return w / w.sum(1, keepdims=True)


def head_m(name):
    """How many nearest centres a head reads."""
    return int(name.split('_m')[1].split('_')[0]) if name.startswith('soft') else 1


def head_lam(name):
    return float(name.split('_l')[1]) if '_l' in name else None


class Heads:
    """Every regression head on one clustering, in one target space.

    mean            the nearest centroid's cluster label mean
    soft_m{m}       KNeighborsRegressor(m, weights='distance') fitted on
                    (centroids, cluster label means)
    ridge / softridge
                    one Ridge per cluster on [X, 1] with no separate
                    intercept, fitted to the residual of a global Ridge -- so
                    w_c = argmin |[X_c 1] w - y_c|^2 + lam |w - w_g|^2, shrunk
                    toward the global fit w_g; softridge blends the m nearest
                    clusters' models by inverse distance."""

    def __init__(self, C, X, nn, Y, only=None):
        self.names = [h for h in HEADS if only is None or h in only]
        k = len(C)
        lab = nn[0][:, 0]
        self.mu = cluster_means(lab, k, Y)
        self.knn = {m: KNeighborsRegressor(n_neighbors=min(m, k), weights='distance').fit(C, self.mu)
                    for m in SOFT_MS if 'soft_m{}'.format(m) in self.names}
        lams = sorted({head_lam(h) for h in self.names} - {None})
        self.W = {}
        if lams:
            Xa = _aug(X).astype(np.float64)
            p = Xa.shape[1]
            glob = Ridge(alpha=1e-6 * (Xa ** 2).sum() / p, fit_intercept=False).fit(Xa, Y)
            wg = glob.coef_.T
            R = Y - Xa @ wg
            order = np.argsort(lab, kind='stable')
            bounds = np.concatenate([[0], np.cumsum(np.bincount(lab, minlength=k))])
            for lam in lams:
                W = np.repeat(wg[None], k, 0)
                for c in range(k):
                    sel = order[bounds[c]:bounds[c + 1]]
                    if len(sel):
                        W[c] += Ridge(alpha=lam, fit_intercept=False, solver='cholesky').fit(
                            Xa[sel], R[sel]).coef_.T
                self.W[lam] = W.astype(np.float32)

    def predict(self, X, nn):
        idx, d2 = nn
        out = {}
        for h in self.names:
            m = head_m(h)
            if h == 'mean':
                out[h] = self.mu[idx[:, 0]]
            elif h.startswith('soft_'):
                out[h] = self.knn[m].predict(X).astype(np.float32)
            else:
                wts = np.ones((len(X), 1), np.float32) if m == 1 else idw(d2[:, :m])
                out[h] = blend_linear(X, self.W[head_lam(h)], idx[:, :m], wts)
        return out


# ========================================================== the net task
def load_graph(design):
    import dgl      # torch/dgl are only needed to (re)build the feature cache
    return dgl.load_graphs(os.path.join(dataset_path, '{}.graph.bin'.format(design)))[0][0]


def net_geometry(n, s, d, x, y, man):
    """Per driver pin: HPWL of its net (bounding box of driver + sinks), star
    wirelength (sum of driver->sink manhattan distances) and the longest one."""
    lo_x, hi_x, lo_y, hi_y = x.copy(), x.copy(), y.copy(), y.copy()
    np.minimum.at(lo_x, s, x[d])
    np.maximum.at(hi_x, s, x[d])
    np.minimum.at(lo_y, s, y[d])
    np.maximum.at(hi_y, s, y[d])
    far = np.zeros(n)
    np.maximum.at(far, s, man)
    return (hi_x - lo_x) + (hi_y - lo_y), np.bincount(s, weights=man, minlength=n), far


def netlog(y):
    return np.log(1e-4 + y) + 7.6      # data_graph.py's n_net_delays_log


def netlog_inv(t):
    return np.exp(t - 7.6) - 1e-4


class NetDelayTask:
    key = 'net'
    title = 'net delay'
    version = 1
    space = 'log(1e-4 + d/ns) + 7.6'
    featsets = {
        'geom': ['adx', 'ady', 'man'],
        'net': ['adx', 'ady', 'man', 'fanout', 'hpwl', 'star', 'man_frac'],
        'full': ['adx', 'ady', 'man', 'fanout', 'hpwl', 'star', 'man_frac',
                 'sink_cap', 'net_cap', 'sink_po', 'drv_pi'],
    }
    scale = {}                          # distances are um and caps fF already
    nolog = {'man_frac', 'sink_po', 'drv_pi'}
    targets = {'log': (netlog, netlog_inv)}
    clip_zero = True                    # every net delay in the dataset is positive
    allpin = True
    xy = ('man', 'fanout')
    xy_label = ('driver-sink manhattan distance (um)', 'driver fanout (sinks)')

    def eval(self, y_ns):
        return netlog(y_ns)

    def class_value(self, y_ns):
        return y_ns[:, 2:4].max(1)

    def build(self, design):
        """Net-arc features of one design, read from its data/8_rat graph."""
        g = load_graph(design)
        s, d = (t.numpy().astype(np.int64) for t in g.edges(etype='net_out'))
        nf = g.ndata['nf'].numpy()
        ef = g.edges['net_out'].data['ef'].numpy()
        y = g.ndata['n_net_delays'].numpy()[d]
        n = g.num_nodes()
        del g
        assert len(np.unique(d)) == len(d), 'a sink pin with two drivers'
        x, yy = nf[:, 3], nf[:, 5]          # distance to die left / bottom = position
        adx, ady = np.abs(ef[:, 0]), np.abs(ef[:, 1])
        man = adx + ady
        hpwl, star, far = net_geometry(n, s, d, x, yy, man)
        sink_cap = nf[d, 6:10].mean(1) * 1000          # pF -> fF
        return dict(
            adx=adx, ady=ady, man=man,
            fanout=np.bincount(s, minlength=n)[s], hpwl=hpwl[s], star=star[s],
            man_frac=man / np.maximum(far[s], 1e-9),
            sink_cap=sink_cap, net_cap=np.bincount(s, weights=sink_cap, minlength=n)[s],
            sink_po=nf[d, 0], drv_pi=nf[s, 0],
            y=y, n_nodes=np.array(n))

    def columns(self, raw):
        cols = {k: v.astype(np.float32) for k, v in raw.items() if k not in ('y', 'n_nodes')}
        return cols

    def allpin_r2(self, block, tf, pf):
        """R^2 over every pin, as train_gnn.py:test_netdelay computes it: the
        pins no net arc lands on (drivers, primary inputs) carry the label of a
        zero delay, which their is_fanout bit predicts exactly."""
        n_other = 4 * (block['n_nodes'] - len(tf) // 4)
        c0 = netlog(0.0)
        n = len(tf) + n_other
        mean = (tf.sum(dtype=np.float64) + n_other * c0) / n
        sst = ((tf - mean) ** 2).sum(dtype=np.float64) + n_other * (c0 - mean) ** 2
        return 1 - ((tf - pf) ** 2).sum(dtype=np.float64) / sst


def load_block(task, design):
    """One design's columns and labels, through a per-design .npz cache."""
    path = os.path.join(cache_path, '{}.{}.v{}.npz'.format(design, task.key, task.version))
    if os.path.exists(path):
        with np.load(path) as z:
            raw = {k: z[k] for k in z.files}
    else:
        t = time.time()
        raw = task.build(design)
        os.makedirs(cache_path, exist_ok=True)
        np.savez(path, **raw)
        log('  cached {:15} {:>8} samples  {:.1f}s'.format(design, len(raw['y']), time.time() - t))
    return {'cols': task.columns(raw), 'y': raw['y'].astype(np.float32),
            'n_nodes': int(raw.get('n_nodes', 0))}


# ============================================================== scoring
CLS = np.arange(N_CLASSES)


def confusion(t, p):
    return confusion_matrix(t, p, labels=CLS)


def f1_from_cm(cm):
    """(accuracy, macro F1, weighted F1, (precision, recall, f1) per class)
    of a confusion matrix.  sklearn.metrics scores its cells as (true,
    predicted) samples weighted by their counts, so per-design matrices can
    be summed and scored once; empty cells are left out, so the macro mean is
    over the classes present in truth or prediction, as f1_score averages."""
    t, p = np.divmod(np.arange(N_CLASSES ** 2), N_CLASSES)
    w = np.asarray(cm).ravel()
    t, p, w = t[w > 0], p[w > 0], w[w > 0]
    return (accuracy_score(t, p, sample_weight=w),
            f1_score(t, p, average='macro', sample_weight=w, zero_division=0),
            f1_score(t, p, average='weighted', sample_weight=w, zero_division=0),
            precision_recall_fscore_support(t, p, labels=CLS, sample_weight=w, zero_division=0)[:3])


def design_scores(task, block, y_ns, p_ns, c_true, c_pred):
    """Scores of one head on one design; p_ns is None for the vote classifier."""
    cm = confusion(c_true, c_pred)
    acc, f1m, f1w, _ = f1_from_cm(cm)
    rec = {'n': len(c_true), 'acc': acc, 'f1_macro': f1m, 'f1_weighted': f1w}
    if p_ns is None:
        return rec, cm, None
    tf = task.eval(y_ns).reshape(-1).astype(np.float64)
    pf = task.eval(p_ns).reshape(-1).astype(np.float64)
    rec['r2'] = r2_score(tf, pf)
    # train_gnn.py's argument order, kept upstream to reproduce the paper (issue #7)
    rec['r2_paper'] = r2_score(pf, tf)
    rec['mse'] = float(((tf - pf) ** 2).mean())
    rec['mae_ns'] = float(np.abs(p_ns - y_ns).mean())
    if task.allpin:
        rec['r2_allpin'] = task.allpin_r2(block, tf, pf)
    sums = np.array([len(tf), ((tf - pf) ** 2).sum(), tf.sum(), (tf ** 2).sum(),
                     np.abs(p_ns - y_ns).sum(dtype=np.float64), p_ns.size])
    return rec, cm, sums


class Score:
    """One head on one design set: per-design records plus pooled sums."""

    def __init__(self):
        self.per, self.cm, self.sums = [], np.zeros((N_CLASSES, N_CLASSES), np.int64), None

    def add(self, design, rec, cm, sums):
        self.per.append(dict(rec, design=design))
        self.cm = self.cm + cm
        if sums is not None:
            self.sums = sums if self.sums is None else self.sums + sums

    def merge(self, other):
        for r in other.per:
            self.add(r['design'], {k: v for k, v in r.items() if k != 'design'}, 0, None)
        self.cm = self.cm + other.cm
        if other.sums is not None:
            self.sums = other.sums if self.sums is None else self.sums + other.sums

    def summary(self):
        acc, f1m, f1w, _ = f1_from_cm(self.cm)
        out = {'acc': acc, 'f1_macro': f1m, 'f1_weighted': f1w}
        if self.sums is not None:
            n, sse, s, ss, ae, na = self.sums
            out['r2'] = float(np.mean([r['r2'] for r in self.per]))
            out['r2_pooled'] = 1 - sse / (ss - s * s / n)
            out['r2_paper'] = float(np.mean([r['r2_paper'] for r in self.per]))
            out['mse'] = sse / n
            out['mae_ns'] = ae / na
            if 'r2_allpin' in self.per[0]:
                out['r2_allpin'] = float(np.mean([r['r2_allpin'] for r in self.per]))
        return out


def majority(lab, k, cls):
    """Majority class of every cluster (an empty one gets the overall majority)."""
    t = np.bincount(lab * N_CLASSES + cls, minlength=k * N_CLASSES).reshape(k, N_CLASSES)
    v = t.argmax(1)
    v[t.sum(1) == 0] = np.bincount(cls, minlength=N_CLASSES).argmax()
    return v


def split_head(name):
    """'ridge_l10[log]' -> ('ridge_l10', 'log'); a bare name -> (name, None)."""
    if name.endswith(']'):
        base, sp = name[:-1].split('[')
        return base, sp
    return name, None


def write_csv(path, rows):
    keys = []
    for r in rows:
        keys += [k for k in r if k not in keys]

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return '' if not np.isfinite(v) else '{:.6g}'.format(v)
        return v

    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(v) for k, v in r.items()})


class Prep:
    """Feature transform + StandardScaler.  'z' z-scores the raw columns;
    'logz' first takes log1p(scale * x) of every non-negative heavy-tailed
    column."""

    def __init__(self, task, featset, transform):
        self.cols = task.featsets[featset]
        self.log = np.array([transform == 'logz' and c not in task.nolog for c in self.cols])
        self.mult = np.array([task.scale.get(c, 1.0) for c in self.cols], np.float32)
        self.sc = StandardScaler()

    def _pre(self, X):
        X = np.array(X, dtype=np.float32)
        if self.log.any():
            X[:, self.log] = np.log1p(np.maximum(X[:, self.log] * self.mult[self.log], 0))
        return X

    def fit(self, X):
        self.sc.fit(self._pre(X))
        return self

    def __call__(self, X):
        return np.asarray(self.sc.transform(self._pre(X)), dtype=np.float32)

    def inverse(self, Z):
        """Scaled points (centroids) back to the raw feature units."""
        X = np.asarray(self.sc.inverse_transform(Z), dtype=np.float64)
        X[:, self.log] = np.expm1(X[:, self.log]) / self.mult[self.log]
        return X


def cfg_name(cfg):
    return '{featset}/{transform}/k{k}/{init}/n{n_init}/{algorithm}/{sampling}'.format(**cfg)


def class_names(edges):
    e = np.concatenate([[0], np.asarray(edges) * 1000, [np.inf]])     # ns -> ps
    return ['C{} {}'.format(i + 1, '<{:.3g}ps'.format(e[1]) if i == 0 else
                            '>={:.3g}ps'.format(e[i]) if i == N_CLASSES - 1 else
                            '{:.3g}-{:.3g}ps'.format(e[i], e[i + 1]))
            for i in range(N_CLASSES)]


def read_csv(path):
    """A CSV written by write_csv, with numbers turned back into numbers."""
    def conv(v):
        if v == '':
            return np.nan
        for f in (int, float):
            try:
                return f(v)
            except ValueError:
                pass
        return v

    with open(path, newline='') as f:
        return [{k: conv(v) for k, v in row.items()} for row in csv.DictReader(f)]


# ============================================================ the search
class Runner:
    def __init__(self, task, out, quick):
        self.task, self.out, self.quick = task, out, quick
        self.train = list(QUICK_TRAIN) if quick else sorted(train_data_keys)
        self.test = list(QUICK_TEST) if quick else sorted(test_data_keys)
        t = time.time()
        self.data = {d: load_block(task, d) for d in self.train + self.test}
        log('{} | engine {} | {} train designs ({} samples), {} test designs ({} samples) | {:.1f}s'.format(
            task.title, ENGINE, len(self.train), sum(len(self.data[d]['y']) for d in self.train),
            len(self.test), sum(len(self.data[d]['y']) for d in self.test), time.time() - t))
        cv = np.concatenate([task.class_value(self.data[d]['y']) for d in self.train])
        self.edges = np.quantile(cv, np.linspace(0, 1, N_CLASSES + 1)[1:-1])
        by_size = sorted(self.train, key=lambda d: -len(self.data[d]['y']))
        self.fold = {d: i % N_FOLDS for i, d in enumerate(by_size)}
        self.rows, self.per_design, self.done = [], [], {}
        self.t0 = time.time()

    # ------------------------------------------------------------ data
    def classes(self, y_ns):
        return np.digitize(self.task.class_value(y_ns), self.edges)

    def class_names(self):
        return class_names(self.edges)

    def matrix(self, d, featset, cols=None):
        b = self.data[d]
        return np.stack([b['cols'][c] for c in cols or self.task.featsets[featset]], 1), b['y']

    def stack(self, designs, featset, sampling):
        """Concatenated samples of some designs, with each design's row span."""
        rng = np.random.default_rng(SEED)
        Xs, Ys, offs, n = [], [], {}, 0
        for d in designs:
            X, Y = self.matrix(d, featset)
            if sampling == 'balanced' and len(X) > BALANCED_CAP:
                sel = np.sort(rng.choice(len(X), BALANCED_CAP, replace=False))
                X, Y = X[sel], Y[sel]
            Xs.append(X)
            Ys.append(Y)
            offs[d] = (n, n + len(X))
            n += len(X)
        return np.concatenate(Xs), np.concatenate(Ys), offs

    # ------------------------------------------------------- one fit
    def to_ns(self, P, sp, rng):
        """Target-space prediction -> ns.  It is first clamped to the fit
        labels' range plus a 10% margin, so a linear head cannot extrapolate
        into nonsense far outside anything seen in training."""
        lo, hi = rng
        P = self.task.targets[sp][1](np.clip(P, lo - 0.1 * (hi - lo), hi + 0.1 * (hi - lo)))
        return (np.maximum(P, 0) if self.task.clip_zero else P).astype(np.float32)

    def predict_heads(self, heads, X, nn):
        out = {}
        for sp, (h, rng) in heads.items():
            tag = '' if len(heads) == 1 else '[{}]'.format(sp)
            for name, P in h.predict(X, nn).items():
                out[name + tag] = self.to_ns(P, sp, rng)
        return out

    def fit_eval(self, cfg, fit_designs, sets, history=None, keep=False):
        """Fit one clustering configuration on fit_designs and score every
        head on each named design set.  Returns ({(set, head): Score}, info)."""
        task = self.task
        X, Y, offs = self.stack(fit_designs, cfg['featset'], cfg['sampling'])
        prep = Prep(task, cfg['featset'], cfg['transform']).fit(X)
        Xs = prep(X)
        del X
        est = SkKMeans(cfg['k'], init=cfg['init'], n_init=cfg['n_init'], algorithm=cfg['algorithm'])
        t0 = time.time()
        est.fit(Xs, callback=history(Xs, Y, prep) if history else None)
        info = {'fit_s': time.time() - t0, 'n_iter': int(est.n_iter_),
                'inertia': float(est.inertia_), 'n_fit': len(Xs), 'best_run': est.best_run_}
        C = est.centers_
        nn = neighbors(C, Xs, M_MAX)
        lab = nn[0][:, 0]
        heads = {}
        for sp, (fwd, _) in task.targets.items():
            T = fwd(Y)
            heads[sp] = (Heads(C, Xs, nn, T),(float(T.min()), float(T.max())))
        vote = majority(lab, len(C), self.classes(Y))
        if keep:
            info.update(prep=prep, centers=C, lab_fit=lab, Y_fit=Y, vote=vote)
        reuse = cfg['sampling'] == 'all'
        scores, kept = {}, {}
        for st, designs in sets.items():
            for d in designs:
                if reuse and d in offs:
                    lo, hi = offs[d]
                    Xd, Yd, nnd = Xs[lo:hi], Y[lo:hi], (nn[0][lo:hi], nn[1][lo:hi])
                else:
                    Xd, Yd = self.matrix(d, cfg['featset'])
                    Xd = prep(Xd)
                    nnd = neighbors(C, Xd, M_MAX)
                block = self.data[d]
                c_true = self.classes(Yd)
                preds = self.predict_heads(heads, Xd, nnd)
                for h, P in preds.items():
                    rec, cm, sums = design_scores(task, block, Yd, P, c_true, self.classes(P))
                    scores.setdefault((st, h), Score()).add(d, rec, cm, sums)
                c_vote = vote[nnd[0][:, 0]]
                rec, cm, _ = design_scores(task, block, Yd, None, c_true, c_vote)
                scores.setdefault((st, 'vote'), Score()).add(d, rec, cm, None)
                if keep and st == 'test':
                    k_ = kept.setdefault(st, {'y': [], 'pred': {}, 'vote': [], 'lab': [], 'xy': []})
                    k_['y'].append(Yd)
                    k_['vote'].append(c_vote)
                    k_['lab'].append(nnd[0][:, 0])
                    k_['xy'].append(self.matrix(d, None, cols=task.xy)[0])
                    for h, P in preds.items():
                        k_['pred'].setdefault(h, []).append(P)
        if keep:
            info['kept'] = {st: {'y': np.concatenate(v['y']), 'vote': np.concatenate(v['vote']),
                                 'lab': np.concatenate(v['lab']), 'xy': np.concatenate(v['xy']),
                                 'pred': {h: np.concatenate(p) for h, p in v['pred'].items()}}
                            for st, v in kept.items()}
        return scores, info

    # ------------------------------------------------ one configuration
    def run_cfg(self, cfg, stage):
        name = cfg_name(cfg)
        if name in self.done:
            return self.done[name]
        vid = len(self.done)
        t = time.time()
        val = {}
        for f in range(N_FOLDS):
            fit = [d for d in self.train if self.fold[d] != f]
            held = [d for d in self.train if self.fold[d] == f]
            sc, _ = self.fit_eval(cfg, fit, {'val': held})
            for key, s in sc.items():
                val.setdefault(key, Score()).merge(s)
        cv_s = time.time() - t
        full, info = self.fit_eval(cfg, self.train, {'train': self.train, 'test': self.test})
        scores = {**val, **full}
        names = sorted({h for _, h in scores},
                       key=lambda h: (h == 'vote', HEADS.index(split_head(h)[0])
                                      if h != 'vote' else 0, h))
        for h in names:
            base, sp = split_head(h)
            row = dict(variant=vid, stage=stage, engine=ENGINE,task=self.task.key,
                       **{k: cfg[k] for k in CFG_KEYS}, head=h, head_base=base,
                       target=sp or next(iter(self.task.targets)),
                       n_features=len(self.task.featsets[cfg['featset']]))
            for st in ('val', 'train', 'test'):
                for m, v in scores[(st, h)].summary().items():
                    row['{}_{}'.format(st, m)] = v
                for rec in scores[(st, h)].per:
                    self.per_design.append(dict(variant=vid, set=st, head=h, **rec))
            row.update(fit_s=info['fit_s'], n_iter=info['n_iter'], inertia=info['inertia'],
                       n_fit=info['n_fit'], cv_s=cv_s)
            self.rows.append(row)
        self.done[name] = vid
        self.flush()
        mine = [r for r in self.rows if r['variant'] == vid and r['head'] != 'vote']
        b = max(mine, key=lambda r: r['val_r2'])
        log('[{:3}] {:8} {:48} val R2 {:.4f}  test R2 {:.4f}  ({:18}) trad val {:.4f}  '
            'cv {:5.0f}s fit {:5.1f}s it {:3}  [{:.0f} min]'.format(
                vid, stage, name, b['val_r2'], b['test_r2'], b['head'],
                max(r['val_r2'] for r in mine if r['head_base'] == 'mean'),
                cv_s, info['fit_s'], info['n_iter'], (time.time() - self.t0) / 60))
        return vid

    def best(self, heads=None, rows=None):
        reg = [r for r in rows or self.rows
               if r['head'] != 'vote' and r['stage'] != 'k5'
               and (heads is None or r['head_base'] in heads)]
        return max(reg, key=lambda r: r['val_r2'])

    def search(self):
        k1 = 64 if self.quick else 256
        ks = (8, 64, 256) if self.quick else (4, 16, 64, 256, 1024, 4096)
        for fs in self.task.featsets:
            for tf in ('z', 'logz'):
                self.run_cfg(dict(BASE, featset=fs, transform=tf, k=k1), 'screen')
        pairs = {}
        for r in self.rows:
            if r['head'] != 'vote':
                key = (r['featset'], r['transform'])
                pairs[key] = max(pairs.get(key, -np.inf), r['val_r2'])
        for fs, tf in sorted(pairs, key=pairs.get, reverse=True)[:2]:
            for k in ks:
                self.run_cfg(dict(BASE, featset=fs, transform=tf, k=k), 'k_sweep')
        b = self.best()
        at = dict(BASE, featset=b['featset'], transform=b['transform'], k=b['k'])
        for change in (dict(init='random'), dict(n_init=4), dict(algorithm='minibatch'),
                       dict(sampling='balanced')):
            self.run_cfg(dict(at, **change), 'engine')

    def flush(self):
        write_csv(self.out / 'variants.csv', self.rows)
        write_csv(self.out / 'per_design.csv', self.per_design)

    # ----------------------------------------- the selected variant
    def tracker(self, cfg, head, hist):
        """History callback factory: after every k-means iteration, refit the
        traditional mean head and the selected head on the current centres
        and score both on the fit set and the test designs."""
        base, sp = split_head(head)
        sp = sp or next(iter(self.task.targets))
        names = ['mean'] + ([base] if base != 'mean' else [])
        m = max(head_m(h) for h in names)
        fwd = self.task.targets[sp][0]
        Xte, Yte, _ = self.stack(self.test, cfg['featset'], 'all')
        t_te = self.task.eval(Yte)

        def make(Xs, Y, prep):
            Xt = prep(Xte)
            T, t_tr = fwd(Y), self.task.eval(Y)
            rng = (float(T.min()), float(T.max()))

            def cb(run, it, C):
                nn_tr = neighbors(C, Xs, m)
                nn_te = neighbors(C, Xt, m)
                h = Heads(C, Xs, nn_tr, T, only=names)
                row = {'run': run, 'iter': it,
                       'inertia_per_sample': float(nn_tr[1][:, 0].sum(dtype=np.float64)) / len(Xs)}
                for st, X_, nn_, Y_, t_ in (('train', Xs, nn_tr, Y, t_tr), ('test', Xt, nn_te, Yte, t_te)):
                    P = h.predict(X_, nn_)
                    for hn in names:
                        p_ns = self.to_ns(P[hn], sp, rng)
                        p_ = self.task.eval(p_ns)
                        row['{}_{}_mse'.format(st, hn)] = float(((t_ - p_) ** 2).mean())
                        row['{}_{}_r2'.format(st, hn)] = r2_score(t_.reshape(-1), p_.reshape(-1))
                        row['{}_{}_mae_ns'.format(st, hn)] = float(np.abs(p_ns - Y_).mean())
                hist.append(row)
                if it % 10 == 0:
                    log('    run {} iter {:3}  inertia {:.5f}  train/test MSE {} {:.4f}/{:.4f}'.format(
                        run, it, row['inertia_per_sample'], names[-1],
                        row['train_{}_mse'.format(names[-1])], row['test_{}_mse'.format(names[-1])]))
            return cb
        return make

    def finish(self):
        sel = self.best()
        trad = self.best(heads={'mean'})
        cfg = {k: sel[k] for k in CFG_KEYS}
        log('selected (best validation R2): variant {} {} head {}'.format(sel['variant'], cfg_name(cfg), sel['head']))
        hist = []
        _, info = self.fit_eval(cfg, self.train, {'test': self.test},
                                history=self.tracker(cfg, sel['head'], hist), keep=True)
        write_csv(self.out / 'history.csv', hist)
        k5 = dict(cfg, k=N_CLASSES)
        self.run_cfg(k5, 'k5')
        _, info5 = self.fit_eval(k5, self.train, {'test': self.test}, keep=True)
        self.artifacts(sel, trad, cfg, info, hist, info5)
        log('done in {:.1f} min; results in {}'.format((time.time() - self.t0) / 60, os.path.relpath(self.out, script_dir)))

    def artifacts(self, sel, trad, cfg, info, hist, info5):
        task, out = self.task, self.out
        names = self.class_names()
        te = info['kept']['test']
        heads = [sel['head']] + (['mean'] if sel['head_base'] != 'mean' else [])
        # mean head in the same target space as the selected head
        if sel['head_base'] != 'mean':
            heads[1] = 'mean' + (sel['head'][len(sel['head_base']):])
        c_true = self.classes(te['y'])
        cms = [(h, confusion(c_true, self.classes(te['pred'][h]))) for h in heads]
        cms.append(('vote', confusion(c_true, te['vote'])))

        write_csv(out / 'classes.csv', [
            dict(cls=names[i], lo_ns=([0] + list(self.edges))[i],
                 hi_ns=(list(self.edges) + [np.inf])[i],
                 train_count=int((self.classes(info['Y_fit']) == i).sum()),
                 test_count=int((c_true == i).sum())) for i in range(N_CLASSES)])
        write_csv(out / 'confusion_matrix.csv', [
            dict(head=h, true_class=names[i], pred_class=names[j], count=int(cm[i, j]))
            for h, cm in cms for i in range(N_CLASSES) for j in range(N_CLASSES)])
        rep = []
        for h, cm in cms:
            _, _, _, (p, r, f) = f1_from_cm(cm)
            rep += [dict(head=h, cls=names[i], precision=p[i], recall=r[i], f1=f[i],
                         support=int(cm[i].sum())) for i in range(N_CLASSES)]
        write_csv(out / 'classification_report.csv', rep)

        clusters = self.cluster_table(info, te)
        write_csv(out / 'clusters.csv', clusters)
        clusters5 = self.cluster_table(info5, info5['kept']['test'])
        write_csv(out / 'clusters_k5.csv', clusters5)

        # what the plots need, so --replot can redraw them without refitting
        np.savez(out / 'selected_test_predictions.npz', y=te['y'], vote=te['vote'], lab=te['lab'],
                 **{'pred_{}'.format(i): te['pred'][h] for i, h in enumerate(heads)})
        np.savez(out / 'k5_test_clusters.npz', lab=info5['kept']['test']['lab'], xy=info5['kept']['test']['xy'])
        summary = dict(
            task=task.title, engine=ENGINE, quick=self.quick,
            label_space=task.space if task.allpin else 'ns',
            train_designs=self.train, test_designs=self.test,
            n_train_samples=int(sum(len(self.data[d]['y']) for d in self.train)),
            n_test_samples=int(sum(len(self.data[d]['y']) for d in self.test)),
            classes=[dict(cls=names[i], edge_low_ns=float(([0] + list(self.edges))[i]))
                     for i in range(N_CLASSES)],
            class_edges_ns=[float(e) for e in self.edges],
            selection_rule='highest mean per-design validation R2, 3-fold leave-designs-out CV on the training designs',
            selected=sel, best_traditional_mean_head=trad, selected_cfg=cfg,
            plot_heads=heads, history_run=info['best_run'],
            n_variants=len(self.done), n_rows=len(self.rows),
            minutes=(time.time() - self.t0) / 60)
        (out / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))

        plots(SimpleNamespace(
            out=out, task=task, engine=ENGINE, rows=self.rows, edges=self.edges,
            sel=sel, trad=trad, cfg=cfg, hist=hist, run=info['best_run'], heads=heads,
            te={'y': te['y'], 'vote': te['vote'], 'pred': {h: te['pred'][h] for h in heads}},
            te5={'lab': info5['kept']['test']['lab'], 'xy': info5['kept']['test']['xy']},
            clusters=clusters, clusters5=clusters5))

    def cluster_table(self, info, te):
        """One row per cluster: size, label statistics, majority class and
        purity on the fit set, test-set size, centroid in raw feature units."""
        k = len(info['centers'])
        lab, Y = info['lab_fit'], info['Y_fit']
        cls = self.classes(Y)
        cnt = np.bincount(lab, minlength=k)
        mu = cluster_means(lab, k, Y)
        late = self.task.class_value(Y)
        sq = np.bincount(lab, weights=late ** 2, minlength=k) / np.maximum(cnt, 1)
        lm = np.bincount(lab, weights=late, minlength=k) / np.maximum(cnt, 1)
        t = np.bincount(lab * N_CLASSES + cls, minlength=k * N_CLASSES).reshape(k, N_CLASSES)
        cen = info['prep'].inverse(info['centers'])
        test_cnt = np.bincount(te['lab'], minlength=k)
        names = self.class_names()
        order = np.argsort(lm)
        rows = []
        for rank, c in enumerate(order):
            r = dict(cluster=int(c), rank_by_delay=rank, size_fit=int(cnt[c]), size_test=int(test_cnt[c]),
                     late_delay_mean_ns=lm[c], late_delay_std_ns=np.sqrt(max(sq[c] - lm[c] ** 2, 0)))
            for i, ch in enumerate(('E_rise', 'E_fall', 'L_rise', 'L_fall')):
                r['mean_{}_ns'.format(ch)] = mu[c, i]
            r['majority_class'] = names[info['vote'][c]]
            r['purity'] = t[c].max() / max(cnt[c], 1)
            for i in range(N_CLASSES):
                r['n_C{}'.format(i + 1)] = int(t[c, i])
            for j, col in enumerate(info['prep'].cols):
                r['centroid_' + col] = cen[c, j]
            rows.append(r)
        return rows


# ================================================================ plots
SURF, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SERIES = ('#2a78d6', '#eb6834', '#1baf7a')      # reference categorical slots 1-3
SET_COLOR = {'train': SERIES[0], 'test': SERIES[1], 'val': SERIES[2]}
BLUES = ('#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b')


def _plt():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        'figure.facecolor': SURF, 'axes.facecolor': SURF, 'savefig.facecolor': SURF,
        'axes.edgecolor': AXIS, 'axes.linewidth': 0.8, 'axes.labelcolor': INK2,
        'axes.titlecolor': INK, 'axes.titlesize': 10.5, 'axes.labelsize': 9,
        'axes.titlelocation': 'left', 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.grid': True, 'axes.axisbelow': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
        'xtick.color': AXIS, 'ytick.color': AXIS, 'xtick.labelcolor': INK2,
        'ytick.labelcolor': INK2, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'text.color': INK, 'legend.frameon': False, 'legend.fontsize': 8,
        'lines.linewidth': 1.6, 'font.family': ['Segoe UI', 'DejaVu Sans'],
        'figure.titlesize': 12, 'figure.titleweight': 'bold'})
    return plt


def _blues():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list('blues', (SURF,) + BLUES)


def _save(plt, fig, path):
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plots(c):
    """Every figure of one run, from the namespace Runner.artifacts builds
    (or replot() reads back from disk)."""
    plt = _plt()
    out, task, sel, cfg, heads = c.out, c.task, c.sel, c.cfg, c.heads
    names = class_names(c.edges)
    tag = '{} | {} engine | {}'.format(task.title, c.engine, cfg_name(cfg))
    space = 'MSE in {}'.format(task.space) if task.allpin else 'MSE (ns^2)'
    c_true = np.digitize(task.class_value(c.te['y']), c.edges)
    cms = [(h, confusion(c_true, np.digitize(task.class_value(c.te['pred'][h]), c.edges))) for h in heads]
    cms.append(('vote', confusion(c_true, c.te['vote'])))

    # -- 1. error vs iteration --------------------------------------
    hr = [h for h in c.hist if h['run'] == c.run]
    it = [h['iter'] for h in hr]
    hnames = ['mean'] + ([split_head(sel['head'])[0]] if sel['head_base'] != 'mean' else [])
    step = 'epoch (mini-batch)' if cfg['algorithm'] == 'minibatch' else 'Lloyd iteration'
    fig, axs = plt.subplots(1, 2 + len(hnames), figsize=(4.2 * (2 + len(hnames)), 3.5))
    axs[0].plot(it, [h['inertia_per_sample'] for h in hr], color=SERIES[0])
    axs[0].set(title='K-means objective', xlabel=step, ylabel='inertia per sample (scaled units)')
    for ax, hn in zip(axs[1:], hnames):
        for st in ('train', 'test'):
            ax.plot(it, [h['{}_{}_mse'.format(st, hn)] for h in hr], color=SET_COLOR[st], label=st)
        ax.set(title='{} head: error'.format(hn), xlabel=step, ylabel=space)
        ax.legend()
    ax = axs[-1]
    for st in ('train', 'test'):
        ax.plot(it, [h['{}_{}_r2'.format(st, hnames[-1])] for h in hr], color=SET_COLOR[st], label=st)
    ax.set(title='{} head: R2'.format(hnames[-1]), xlabel=step, ylabel='pooled R2')
    ax.legend()
    fig.suptitle('Training and test error vs k-means iteration  ({})'.format(tag), x=0.01, ha='left')
    fig.tight_layout()
    _save(plt, fig, out / 'error_vs_iteration.png')

    # -- 2. error vs k ---------------------------------------------------
    same = [x for x in c.rows if x['featset'] == cfg['featset'] and x['transform'] == cfg['transform']
            and all(x[k] == BASE[k] for k in BASE) and x['stage'] != 'k5']
    panels = [sel['head']] + ([heads[1]] if len(heads) > 1 else [])
    fig, axs = plt.subplots(1, len(panels) + 1, figsize=(4.4 * (len(panels) + 1), 3.5))
    for ax, h in zip(axs, panels):
        rows = sorted([x for x in same if x['head'] == h], key=lambda x: x['k'])
        ks = [x['k'] for x in rows]
        for st in ('train', 'val', 'test'):
            ax.plot(ks, [1 - x['{}_r2'.format(st)] for x in rows], 'o-', ms=4,
                    color=SET_COLOR[st], label=st)
        ax.set_xscale('log', base=2)
        ax.set(title='{} head'.format(h), xlabel='k (clusters)', ylabel='1 - R2 (mean per design)')
        ax.legend()
    rows = sorted([x for x in same if x['head'] == sel['head']], key=lambda x: x['k'])
    axs[-1].plot([x['k'] for x in rows], [x['inertia'] / x['n_fit'] for x in rows], 'o-', ms=4,
                 color=SERIES[0])
    axs[-1].set_xscale('log', base=2)
    axs[-1].set(title='Elbow: inertia vs k', xlabel='k (clusters)', ylabel='inertia per sample')
    fig.suptitle('Error vs number of clusters  ({} | {}/{})'.format(task.title, cfg['featset'], cfg['transform']),
                 x=0.01, ha='left')
    fig.tight_layout()
    _save(plt, fig, out / 'error_vs_k.png')

    # -- 3. confusion matrices ----------------------------------------
    cmap = _blues()
    fig, axs = plt.subplots(1, len(cms), figsize=(4.3 * len(cms), 4.1))
    for ax, (h, cm) in zip(axs, cms):
        share = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(share, cmap=cmap, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(N_CLASSES):
            for j in range(N_CLASSES):
                ax.text(j, i, '{:.0f}%\n{}'.format(100 * share[i, j], cm[i, j]), ha='center', va='center',
                        fontsize=7, color='white' if share[i, j] > 0.55 else INK)
        acc, f1m, _, _ = f1_from_cm(cm)
        ax.set_xticks(range(N_CLASSES), ['C{}'.format(i + 1) for i in range(N_CLASSES)])
        ax.set_yticks(range(N_CLASSES), ['C{}'.format(i + 1) for i in range(N_CLASSES)])
        ax.set(title='{}\nacc {:.3f}  macro-F1 {:.3f}'.format(h, acc, f1m),
               xlabel='predicted class', ylabel='true class')
    fig.suptitle('Test-set confusion matrices (row share, count)  classes: ' + ', '.join(names),
                 x=0.01, ha='left', fontsize=9)
    fig.tight_layout()
    _save(plt, fig, out / 'confusion_matrix.png')

    # -- 4. F1 per class ----------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    wdt = 0.8 / len(cms)
    for q, (h, cm) in enumerate(cms):
        f1 = f1_from_cm(cm)[3][2]
        ax.bar(np.arange(N_CLASSES) + (q - (len(cms) - 1) / 2) * wdt, f1, wdt, color=SERIES[q % 3],
               edgecolor=SURF, linewidth=1.5, label='{} (macro {:.3f})'.format(h, f1_from_cm(cm)[1]))
    ax.set_xticks(range(N_CLASSES), names, fontsize=7.5)
    ax.set(ylim=(0, 1), ylabel='F1 (test)', title='F1 score per delay class')
    ax.legend(loc='upper center', ncol=len(cms), bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    _save(plt, fig, out / 'f1_per_class.png')

    # -- 5. predicted vs true ----------------------------------------------
    t, p = task.eval(c.te['y']).reshape(-1), task.eval(c.te['pred'][sel['head']]).reshape(-1)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    hb = ax.hexbin(t, p, gridsize=70, bins='log', cmap=cmap, mincnt=1)
    lo, hi = np.percentile(np.concatenate([t, p]), [0.1, 99.9])
    ax.plot([lo, hi], [lo, hi], color=MUTED, linewidth=1)
    ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel='true ({})'.format(task.space if task.allpin else 'ns'),
           ylabel='predicted', title='Test designs, all 4 corners/edges\n{}  R2 {:.4f}'.format(
               sel['head'], r2_score(t, p)))
    fig.colorbar(hb, ax=ax, label='samples (log)')
    fig.tight_layout()
    _save(plt, fig, out / 'pred_vs_true.png')

    # -- 6. the selected clustering -------------------------------------
    clusters = c.clusters
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={'width_ratios': (1, 1.1)})
    size = np.array([q['size_fit'] for q in clusters])
    lm = np.array([q['late_delay_mean_ns'] for q in clusters]) * 1000
    axs[0].scatter(np.maximum(size, 1), lm, s=14, color=SERIES[0], alpha=0.6, edgecolor=SURF, linewidth=0.4)
    # net delays are all positive and span decades; cell delays include 0
    axs[0].set(xscale='log', yscale='log' if task.clip_zero else 'linear',
               xlabel='cluster size (training samples)',
               ylabel='cluster mean late delay (ps)', title='{} clusters: size vs delay'.format(len(clusters)))
    groups = np.array_split(np.arange(len(clusters)), min(len(clusters), 24))
    cont = np.array([[sum(clusters[i]['n_C{}'.format(j + 1)] for i in g) for j in range(N_CLASSES)]
                     for g in groups], dtype=float)
    cont /= np.maximum(cont.sum(1, keepdims=True), 1)
    axs[1].imshow(cont, cmap=cmap, aspect='auto', vmin=0, vmax=1)
    axs[1].grid(False)
    axs[1].set_xticks(range(N_CLASSES), ['C{}'.format(i + 1) for i in range(N_CLASSES)])
    axs[1].set_yticks([0, len(groups) - 1], ['fastest', 'slowest'])
    axs[1].set(xlabel='true delay class', ylabel='clusters, ordered by mean delay ({} per row)'.format(
        int(np.ceil(len(clusters) / len(groups)))), title='Class mix inside the clusters (training set)')
    fig.suptitle('The clusters k-means formed  ({})'.format(tag), x=0.01, ha='left')
    fig.tight_layout()
    _save(plt, fig, out / 'clusters_selected.png')

    # -- 7. k = 5: the clusters as classes ----------------------------
    xy, lab = c.te5['xy'], c.te5['lab']
    pos = (xy > 0).all(1)           # log axes: a zero distance or zero LUT value has no place
    rng = np.random.default_rng(SEED)
    pick = rng.choice(np.flatnonzero(pos), min(30000, int(pos.sum())), replace=False)
    xy, lab = xy[pick], lab[pick]
    fig, axs = plt.subplots(1, N_CLASSES, figsize=(3.3 * N_CLASSES, 3.4), sharex=True, sharey=True)
    for ax, q in zip(axs, c.clusters5):
        mine = lab == q['cluster']
        ax.scatter(xy[~mine, 0], xy[~mine, 1], s=3, color=GRID, linewidth=0, rasterized=True)
        ax.scatter(xy[mine, 0], xy[mine, 1], s=4, color=SERIES[0], linewidth=0, alpha=0.7, rasterized=True)
        ax.set(xscale='log', yscale='log', xlabel=task.xy_label[0],
               title='cluster {} of 5: {:.3g} ps mean\nn_train={}  majority {}  purity {:.0f}%'.format(
                   q['rank_by_delay'] + 1, q['late_delay_mean_ns'] * 1000, q['size_fit'],
                   q['majority_class'].split()[0], 100 * q['purity']))
        ax.title.set_fontsize(8.5)
    axs[0].set_ylabel(task.xy_label[1])
    fig.suptitle('k = 5: the five clusters k-means finds, on the test designs  ({} | {}/{}){}'.format(
        task.title, cfg['featset'], cfg['transform'],
        '  [{} samples with a zero axis value not drawn]'.format(int((~pos).sum())) if (~pos).any() else ''),
        x=0.01, ha='left')
    fig.tight_layout()
    _save(plt, fig, out / 'clusters_k5.png')

    # -- 8. every variant: validation vs test ----------------------------
    reg = [x for x in c.rows if x['head'] != 'vote']
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter([x['val_r2'] for x in reg], [x['test_r2'] for x in reg], s=12, color=AXIS, linewidth=0,
               label='all {} variants'.format(len(reg)))
    for x, col, lb in ((c.trad, SERIES[0], 'best traditional (mean head)'),
                       (sel, SERIES[1], 'selected by validation')):
        ax.scatter([x['val_r2']], [x['test_r2']], s=60, color=col, edgecolor=SURF, linewidth=1.5, zorder=3,
                   label='{}: {}'.format(lb, x['head']))
    lo = min(min(x['val_r2'] for x in reg), min(x['test_r2'] for x in reg))
    ax.plot([lo, 1], [lo, 1], color=MUTED, linewidth=0.8)
    ax.set(xlabel='validation R2 (CV, mean per design)', ylabel='test R2 (mean per design)',
           title='Every variant: validation vs test')
    ax.legend(loc='lower right')
    fig.tight_layout()
    _save(plt, fig, out / 'variants_val_vs_test.png')


# ================================================================= main
def replot(task, out):
    """Redraw every figure of a finished run from its files, without refitting."""
    s = json.loads((out / 'summary.json').read_text())
    z = np.load(out / 'selected_test_predictions.npz')
    z5 = np.load(out / 'k5_test_clusters.npz')
    heads = s['plot_heads']
    plots(SimpleNamespace(
        out=out, task=task, engine=s['engine'], rows=read_csv(out / 'variants.csv'),
        edges=np.array(s['class_edges_ns']), sel=s['selected'], trad=s['best_traditional_mean_head'],
        cfg=s['selected_cfg'], hist=read_csv(out / 'history.csv'), run=s['history_run'], heads=heads,
        te={'y': z['y'], 'vote': z['vote'], 'pred': {h: z['pred_{}'.format(i)] for i, h in enumerate(heads)}},
        te5={'lab': z5['lab'], 'xy': z5['xy']},
        clusters=read_csv(out / 'clusters.csv'), clusters5=read_csv(out / 'clusters_k5.csv')))
    log('figures redrawn in', out)


def main(task, script):
    ap = argparse.ArgumentParser(description='k-means {} prediction ({} engine)'.format(task.title, ENGINE))
    ap.add_argument('--quick', action='store_true', help='7 small designs and a small grid (a smoke test)')
    ap.add_argument('--out', help='results directory (default results/<script>[_quick])')
    ap.add_argument('--replot', action='store_true', help='only redraw the figures of a finished run')
    args = ap.parse_args()
    out = Path(args.out) if args.out else Path(result_path) / (script + ('_quick' if args.quick else ''))
    if args.replot:
        return replot(task, out)
    out.mkdir(parents=True, exist_ok=True)
    with tee.StdoutTee(str(out / 'stdout.log'), mode='w'):
        r = Runner(task, out, args.quick)
        r.search()
        r.finish()


if __name__ == '__main__':
    main(NetDelayTask(), 'k_means_numpy_variation')
