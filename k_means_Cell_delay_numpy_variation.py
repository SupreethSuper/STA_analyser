"""K-means for pre-routing CELL DELAY prediction on the TimingPredict graphs.

Same scikit-learn model, heads, search, scoring and plots as
k_means_numpy_variation.py (imported from it); only the task differs.  One
sample per cell arc (cell input pin -> cell output pin, the `cell_out` edges);
the label is the arc's delay 4-vector `e_cell_delays` = [E_rise, E_fall,
L_rise, L_fall] in ns, scored in ns as train_gnn.py trains it.  Heads are
fitted in two target spaces, raw ns and asinh(ns / 10 ps), and every head
appears once per space: 'ridge_l1[asinh]'.

Features (pre-route only).  Every arc carries its own Liberty tables in the
512-wide `ef` (8 tables of 7 x 7, index_1 = input slew, index_2 = output
load; see liberty_lut.py).  A table only becomes a number once it is read at
an operating point, so the features read it the way an STA engine would:

  load      sum of the Liberty caps of the sink pins on the output net, plus
            alpha * net HPWL as a wire-capacitance estimate (alpha in pF/um
            is part of the search: 0, 1e-4, 2e-4)
  slew      the input pin's slew is estimated one hop upstream: the output
            transition table of the cell driving the input net, read at that
            cell's own load; two passes, starting from 0.1 ns everywhere
  d_*, t_*  each delay / transition table bilinearly interpolated at
            (estimated slew, load), with linear extrapolation like STA tools

Feature sets
  lut_mean       the 8 table means (data_graph.py's DeepGCNII summary) + load
  lut_load       the 4 delay tables at load, at three fixed slews
  lut_sta_a{i}   d_*, t_* at (estimated slew, load) with alpha_i, + context
  lut_sta_ctx    lut_sta_a1 + input pin cap, arcs per output pin, table means

Classes: 5 equal-frequency bins of max(L_rise, L_fall).  Results go to
results/k_means_Cell_delay_numpy_variation/.

    ..\\TimingPredict\\.venv312\\Scripts\\python.exe k_means_Cell_delay_numpy_variation.py            # full search
    ..\\TimingPredict\\.venv312\\Scripts\\python.exe k_means_Cell_delay_numpy_variation.py --quick    # 7 small designs
"""
import numpy as np

import k_means_numpy_variation as km     # its paths block holds every path this script uses

LUT = 7
N_SLOT = 8                          # [E, L] x [cell_rise, cell_fall, rise_transition, fall_transition]
AXES = N_SLOT * (1 + 2 * LUT)       # 120
DSLOT = (0, 1, 4, 5)                # delay table per label channel [E_rise, E_fall, L_rise, L_fall]
TSLOT = (2, 3, 6, 7)                # the matching output-transition table
CH = ('Er', 'Ef', 'Lr', 'Lf')
ALPHAS = (0.0, 1e-4, 2e-4)          # wire cap per um of net HPWL, pF/um
ROWS = (1, 3, 5)                    # slew rows read by lut_load (0.023, 0.122, 0.65 ns in sky130)
S0 = 0.1                            # slew assumed everywhere before the first pass, ns


def _bracket(axis, v):
    """Segment j and weight w with v = axis[j] + w (axis[j+1] - axis[j]);
    outside the axis the end segment is extended (linear extrapolation)."""
    j = np.clip((axis <= v[:, None]).sum(1) - 1, 0, LUT - 2)
    a0 = np.take_along_axis(axis, j[:, None], 1)[:, 0]
    a1 = np.take_along_axis(axis, j[:, None] + 1, 1)[:, 0]
    span = a1 - a0
    return j, np.where(span > 0, (v - a0) / np.where(span > 0, span, 1), 0)


def lut_row(lut, tid, slot, row, load):
    """Table `slot` of every arc at one fixed slew row, interpolated in load."""
    _, i2, vals = lut
    j, w = _bracket(i2[tid, slot], load)
    v0 = vals[tid, slot, row, j]
    v1 = vals[tid, slot, row, j + 1]
    return (v0 + w * (v1 - v0)).astype(np.float32)


def lut_bilinear(lut, tid, slot, slew, load):
    """Table `slot` of every arc read at (slew, load).  A table the arc lacks
    (is_valid = 0) has zero axes and values and reads as 0."""
    i1, i2, vals = lut
    jl, wl = _bracket(i2[tid, slot], load)
    js, ws = _bracket(i1[tid, slot], slew)
    v00, v01 = vals[tid, slot, js, jl], vals[tid, slot, js, jl + 1]
    v10, v11 = vals[tid, slot, js + 1, jl], vals[tid, slot, js + 1, jl + 1]
    lo = v00 + wl * (v01 - v00)
    hi = v10 + wl * (v11 - v10)
    return (lo + ws * (hi - lo)).astype(np.float32)


def input_slews(lut, tid, raw, load, passes=2):
    """Estimated input slew of every arc, per corner [E, L]: the worse output
    transition of the cell that drives the arc's input net, read at that
    cell's load -- a one-hop version of what STA propagates."""
    b, drv, n = raw['b'], raw['drv_a'], int(raw['n_nodes'])
    s_in = [np.full(len(tid), S0, np.float32)] * 2
    for _ in range(passes):
        new = []
        for c in range(2):
            tr = np.maximum(lut_bilinear(lut, tid, TSLOT[2 * c], s_in[c], load[:, 2 * c]),
                            lut_bilinear(lut, tid, TSLOT[2 * c + 1], s_in[c], load[:, 2 * c + 1]))
            out = np.zeros(n, np.float32)
            np.maximum.at(out, b, tr)
            s = np.where(drv >= 0, out[np.maximum(drv, 0)], 0)
            # a primary input (or nothing) drives the net: use the design's typical slew
            fill = np.median(s[s > 0]) if (s > 0).any() else S0
            new.append(np.clip(np.where(s > 0, s, fill), 1e-3, 1.5).astype(np.float32))
        s_in = new
    return s_in


def _d(i):
    return ['d_{}_a{}'.format(c, i) for c in CH]


def _t(i):
    return ['t_{}_a{}'.format(c, i) for c in CH]


MEANS = ['m{}'.format(s) for s in range(N_SLOT)]
CTX = ['fanout', 'hpwl']


ASINH_NS = 0.01        # asinh(y / 10 ps): log-like above 10 ps, linear near 0


def cellasinh(y):
    # not log(1e-4 + y): ~5% of arcs are exactly 0 and a few are negative
    # (wbqspiflash reaches -0.035 ns), which a log cannot take
    return np.arcsinh(y / ASINH_NS)


def cellasinh_inv(t):
    return ASINH_NS * np.sinh(t)


def _same(y):
    return y


class CellDelayTask:
    key = 'cell'
    title = 'cell delay'
    version = 1
    space = 'ns'
    featsets = {
        'lut_mean': MEANS + ['load_a0'] + CTX,
        'lut_load': ['dl_{}_r{}'.format(c, r) for c in CH for r in ROWS] + ['load_a1', 'fanout'],
        'lut_sta_a0': _d(0) + _t(0) + ['sE_a0', 'sL_a0', 'load_a0'] + CTX,
        'lut_sta_a1': _d(1) + _t(1) + ['sE_a1', 'sL_a1', 'load_a1'] + CTX,
        'lut_sta_a2': _d(2) + _t(2) + ['sE_a2', 'sL_a2', 'load_a2'] + CTX,
        'lut_sta_ctx': _d(1) + _t(1) + ['sE_a1', 'sL_a1', 'load_a1'] + CTX + ['incap', 'n_arcs'] + MEANS,
    }
    nolog = set()
    targets = {'raw': (_same, _same), 'asinh': (cellasinh, cellasinh_inv)}
    clip_zero = False                   # negative SDF cell delays do occur
    allpin = False
    xy = ('load_a1', 'd_Lr_a1')
    xy_label = ('output load incl. wire estimate (pF)', 'L rise delay read from the LUT (ns)')

    def __init__(self):
        # log1p works in ps and fF: delays/slews are ns and loads pF
        self.scale = {c: 1000.0 for fs in self.featsets.values() for c in fs
                      if c[:2] in ('m0', 'm1', 'm2', 'm3', 'm4', 'm5', 'm6', 'm7', 'd_', 't_', 'dl', 'sE', 'sL', 'lo')}

    def eval(self, y_ns):
        return y_ns

    def class_value(self, y_ns):
        return y_ns[:, 2:4].max(1)

    def build(self, design):
        """Cell arcs of one design with what their features need: the distinct
        512-wide table rows (an arc's tables depend only on its cell type) and
        the load / net context of the output pin."""
        g = km.load_graph(design)
        a, b = (t.numpy().astype(np.int64) for t in g.edges(etype='cell_out'))
        s, d = (t.numpy().astype(np.int64) for t in g.edges(etype='net_out'))
        nf = g.ndata['nf'].numpy()
        ef = g.edges['cell_out'].data['ef'].numpy().astype(np.float32)
        y = g.edges['cell_out'].data['e_cell_delays'].numpy().astype(np.float32)
        n = g.num_nodes()
        del g
        types, tid = np.unique(ef, axis=0, return_inverse=True)
        del ef
        x, yy = nf[:, 3], nf[:, 5]
        man = np.abs(x[d] - x[s]) + np.abs(yy[d] - yy[s])
        hpwl = km.net_geometry(n, s, d, x, yy, man)[0]
        capsum = np.stack([np.bincount(s, weights=nf[d, 6 + c], minlength=n) for c in range(4)], 1)
        drv = np.full(n, -1, np.int64)
        drv[d] = s
        return dict(
            tid=tid.reshape(-1).astype(np.int32), types=types, b=b.astype(np.int32),
            drv_a=drv[a].astype(np.int32), capsum_b=capsum[b].astype(np.float32),
            hpwl_b=hpwl[b].astype(np.float32), fan_b=np.bincount(s, minlength=n)[b].astype(np.float32),
            incap_a=(nf[a, 6:10].mean(1) * 1000).astype(np.float32),
            n_arcs_b=np.bincount(b, minlength=n)[b].astype(np.float32),
            y=y, n_nodes=np.array(n))

    def columns(self, raw):
        T, tid = raw['types'], raw['tid'].astype(np.int64)
        ax = T[:, :AXES].reshape(-1, N_SLOT, 1 + 2 * LUT)
        lut = (ax[:, :, 1:1 + LUT], ax[:, :, 1 + LUT:], T[:, AXES:].reshape(-1, N_SLOT, LUT, LUT))
        means = lut[2].mean((2, 3))
        cols = {'m{}'.format(sl): means[tid, sl] for sl in range(N_SLOT)}
        cols.update(fanout=raw['fan_b'], hpwl=raw['hpwl_b'], incap=raw['incap_a'], n_arcs=raw['n_arcs_b'])
        for i, alpha in enumerate(ALPHAS):
            load = raw['capsum_b'] + alpha * raw['hpwl_b'][:, None]       # (e, 4) pF
            cols['load_a{}'.format(i)] = load.mean(1)
            if i == 1:
                for ch, sl in enumerate(DSLOT):
                    for row in ROWS:
                        cols['dl_{}_r{}'.format(CH[ch], row)] = lut_row(lut, tid, sl, row, load[:, ch])
            s_in = input_slews(lut, tid, raw, load)
            for ch in range(4):
                cols['d_{}_a{}'.format(CH[ch], i)] = lut_bilinear(lut, tid, DSLOT[ch], s_in[ch // 2], load[:, ch])
                cols['t_{}_a{}'.format(CH[ch], i)] = lut_bilinear(lut, tid, TSLOT[ch], s_in[ch // 2], load[:, ch])
            cols['sE_a{}'.format(i)], cols['sL_a{}'.format(i)] = s_in
        return {k: np.asarray(v, np.float32) for k, v in cols.items()}


if __name__ == '__main__':
    km.main(CellDelayTask(), 'k_means_Cell_delay_numpy_variation')
