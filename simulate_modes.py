# -*- coding: utf-8 -*-
"""Monte Carlo simulations for distributed graph-state synthesis.

Compares the Steiner baseline with the rank-2 protocol on Waxman networks.
Edit the experiment settings in main() before running.
"""

import os
import math
import csv
import pickle
import warnings

import numpy as np
import networkx as nx

import matplotlib
matplotlib.use('Agg')  # headless-safe
import matplotlib.pyplot as plt

from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **k):
        return x

warnings.filterwarnings("ignore")

# Physical-network parameters
ATTN = 0.5
PHYS_MULTIPLIER = 3
K_TARGET = 3
ALPHA_W = 0.10

# NetworkX implementation of Mehlhorn's Steiner approximation.
STEINER_METHOD = 'mehlhorn'

# Metrics
COLS = ['Steiner', 'prop']
METRICS = ('cost', 'ncz', 'nmeas', 'shot_depth', 'n_shots', 'step_depth')

# Network and target graphs
def gen_positions(N, min_dist=0.06, seed=None):
    """Sample node positions in the unit square."""
    rng = np.random.default_rng(seed)
    pos, cm, att = [], min_dist, 0
    while len(pos) < N and att < 2000:
        p = rng.random(2)
        if len(pos) == 0 or all(np.linalg.norm(p - np.array(o)) >= cm for o in pos):
            pos.append(p)
        att += 1
        if att % 200 == 0:
            cm *= 0.95
    while len(pos) < N:
        pos.append(rng.random(2))
    return {i: tuple(p) for i, p in enumerate(pos)}


def build_physical_network(N_phys, alpha_w, seed=None):
    """Build a connected Waxman physical network."""
    rng = np.random.default_rng(seed)
    pos = gen_positions(N_phys, seed=seed)
    G = nx.Graph()
    G.add_nodes_from(pos)
    nx.set_node_attributes(G, pos, 'pos')

    L = math.sqrt(2.0)
    pts = np.array([pos[i] for i in range(N_phys)])
    iu, ju = np.triu_indices(N_phys, 1)
    d = np.linalg.norm(pts[iu] - pts[ju], axis=1)
    keep = (d <= 3.5 * alpha_w * L) if alpha_w < 0.5 else np.ones_like(d, dtype=bool)
    base = np.where(keep, np.exp(-d / (alpha_w * L)), 0.0)
    target_edges = K_TARGET * N_phys / 2.0
    s = base.sum()
    beta = min(1.0, target_edges / s) if s > 0 else 0.6
    probs = beta * base
    ei = np.where(rng.random(len(d)) < probs)[0]
    G.add_edges_from([(int(iu[k]), int(ju[k])) for k in ei])

    # Connect components through the nearest node pair.
    while not nx.is_connected(G):
        cs = list(nx.connected_components(G))
        a = list(cs[0])
        r = list(set().union(*cs[1:]))
        ca = np.array([pos[u] for u in a])
        cr = np.array([pos[v] for v in r])
        dd = np.linalg.norm(ca[:, None, :] - cr[None, :, :], axis=-1)
        k = np.unravel_index(np.argmin(dd), dd.shape)
        G.add_edge(a[k[0]], r[k[1]])

    for u, v in G.edges():
        dist = float(np.linalg.norm(np.array(pos[u]) - np.array(pos[v])))
        G[u][v]['weight'] = math.exp(ATTN * dist)
        G[u][v]['resource'] = math.exp(ATTN * dist)

    sp = dict(nx.all_pairs_dijkstra_path(G, weight='weight'))
    path_edges = {}
    edge_cost = {}
    for u, v, d_data in G.edges(data=True):
        edge_cost[(min(u, v), max(u, v))] = d_data['resource']
    for i in G.nodes():
        for j in G.nodes():
            if i == j:
                path_edges[(i, j)] = set()
            else:
                p = sp[i][j]
                path_edges[(i, j)] = set(
                    (min(p[k], p[k + 1]), max(p[k], p[k + 1])) for k in range(len(p) - 1))
    return G, sp, path_edges, edge_cost


def build_target_ER(N, p, seed=None):
    """Generate a connected ER target graph and return its adjacency matrix."""
    rng = np.random.default_rng(seed)
    iu, ju = np.triu_indices(N, 1)
    G = nx.Graph()
    G.add_nodes_from(range(N))
    G.add_edges_from((int(iu[k]), int(ju[k]))
                     for k in np.where(rng.random(len(iu)) < p)[0])
    if N > 1 and not nx.is_connected(G):
        cc = [list(c) for c in nx.connected_components(G)]
        for i in range(len(cc) - 1):
            G.add_edge(cc[i][0], cc[i + 1][0])
    M = nx.to_numpy_array(G).astype(int)
    np.fill_diagonal(M, 0)
    M = np.triu(M, 1)
    return M + M.T


# Gate and measurement counts
def census_structure(struct, target_set):
    """Count CZ gates and Pauli measurements for one routing structure."""
    ncz = nmeas = 0
    for v in struct.nodes():
        d = struct.degree(v)
        if d <= 1:
            continue
        ncz += d - 1
        nmeas += (d - 1) if v in target_set else d
    return ncz, nmeas


def _steiner_tree_safe(G, sp, root, leaves):
    """Mehlhorn Steiner tree with a shortest-path fallback."""
    terms = list(dict.fromkeys([root] + [l for l in leaves if l != root]))
    if len(terms) < 2:
        return nx.Graph()
    try:
        return nx.algorithms.approximation.steiner_tree(
            G, terms, weight='weight', method=STEINER_METHOD)
    except Exception:
        t = nx.Graph()
        for lf in leaves:
            if lf == root:
                continue
            p = sp[root][lf]
            t.add_edges_from((p[k], p[k + 1]) for k in range(len(p) - 1))
        return t


# Steiner baseline
def meignant_steiner(G, sp, M, tgt):
    """Greedy star decomposition with Steiner routing."""
    cost = shot_depth = 0.0
    ncz = nmeas = n_shots = 0
    res = M.copy()
    target_set = set(tgt)

    while np.any(res == 1):
        c = int(np.argmax(res.sum(1)))
        nb = np.where(res[c] == 1)[0]
        if len(nb) == 0:
            break
        cg = int(tgt[c])
        leaves = [int(tgt[x]) for x in nb]

        tree = _steiner_tree_safe(G, sp, cg, leaves)

        if tree.number_of_edges() > 0:
            for u, v in tree.edges():
                cost += G[u][v]['resource']
            shot_depth += max(dict(tree.degree()).values())
            n_shots += 1

            cz, ms = census_structure(tree, target_set)
            ncz += cz
            nmeas += ms

        for x in nb:
            res[c, x] = 0
            res[x, c] = 0

    step_depth = shot_depth / n_shots if n_shots > 0 else 0.0
    return (float(cost), int(ncz), int(nmeas), float(shot_depth),
            int(n_shots), float(step_depth))


# Rank-2 protocol
def prop_select_exact(G, sp, M, tgt, anc, path_edges, edge_cost, max_steps=None):
    """Select pivots and ancilla locations for the rank-2 protocol."""
    Rm = M.copy()
    steps = []
    anc = list(anc)
    if len(anc) < 2 or not np.any(Rm != 0):
        return steps

    N = Rm.shape[0]
    if max_steps is None:
        max_steps = N

    # Cache shortest-path edge masks used in the placement search.
    edge_list = sorted(edge_cost.keys())
    edge_idx = {e: k for k, e in enumerate(edge_list)}
    n_edges = len(edge_list)
    cost_vec = np.array([edge_cost[e] for e in edge_list], dtype=float)

    mask_cache = {}

    def pe_mask(a, b):
        m = mask_cache.get((a, b))
        if m is None:
            m = np.zeros(n_edges, dtype=bool)
            for e in path_edges[(a, b)]:
                idx = edge_idx.get(e)
                if idx is not None:
                    m[idx] = True
            mask_cache[(a, b)] = m
        return m

    anc_arr = np.array(anc, dtype=int)
    n_anc = len(anc_arr)

    step = 0
    while np.any(Rm != 0) and step < max_steps:
        edges_to_resolve = np.argwhere(np.triu(Rm, 1) == 1)
        if len(edges_to_resolve) == 0:
            break

        # Pivot with maximum net progress.
        best_edge, best_net, best_Rn = None, -np.inf, None
        for i, j in edges_to_resolve:
            i, j = int(i), int(j)
            col_i, col_j = Rm[:, i], Rm[:, j]
            Rn = (Rm + ((np.outer(col_j, col_i) + np.outer(col_i, col_j)) % 2)) % 2
            net = (np.sum((Rm == 1) & (Rn == 0)) - np.sum((Rm == 0) & (Rn == 1))) / 2.0
            if net > best_net:
                best_net, best_edge, best_Rn = net, (i, j), Rn
        if best_edge is None:
            break
        i, j = best_edge

        # Ancilla placement for the selected pivot.
        u_phys = [int(tgt[x]) for x in np.where(Rm[:, j] == 1)[0]]
        v_phys = [int(tgt[x]) for x in np.where(Rm[:, i] == 1)[0]]

        maskA = np.zeros((n_anc, n_edges), dtype=bool)
        maskB = np.zeros((n_anc, n_edges), dtype=bool)
        for k, s in enumerate(anc_arr):
            s = int(s)
            for lf in u_phys:
                maskA[k] |= pe_mask(s, lf)
            for lf in v_phys:
                maskB[k] |= pe_mask(s, lf)

        best_sA = best_sB = -1
        best_cost = np.inf
        for ka in range(n_anc):
            sA = int(anc_arr[ka])
            AB = maskB | maskA[ka]
            bb = np.zeros((n_anc, n_edges), dtype=bool)
            for kb in range(n_anc):
                if kb != ka:
                    bb[kb] = pe_mask(sA, int(anc_arr[kb]))
            costs = (AB | bb).astype(float) @ cost_vec
            costs[ka] = np.inf
            kb = int(np.argmin(costs))
            if costs[kb] < best_cost:
                best_cost = costs[kb]
                best_sA, best_sB = sA, int(anc_arr[kb])

        Rm = best_Rn
        steps.append((best_sA, best_sB, list(u_phys), list(v_phys)))
        step += 1

    return steps


def prop_realize_census(G, sp, steps, tgt):
    """Evaluate the realized routing graph for each rank-2 step."""
    cost = shot_depth = 0.0
    ncz = nmeas = n_shots = 0
    target_set = set(tgt)

    for sA, sB, u_phys, v_phys in steps:
        tree_A = _steiner_tree_safe(G, sp, sA, u_phys)
        tree_B = _steiner_tree_safe(G, sp, sB, v_phys)

        realized = nx.Graph()
        realized.add_edges_from(tree_A.edges())
        realized.add_edges_from(tree_B.edges())
        bb = sp[sA][sB]
        realized.add_edges_from((bb[k], bb[k + 1]) for k in range(len(bb) - 1))

        if realized.number_of_edges() > 0:
            for (uu, vv) in realized.edges():
                cost += G[uu][vv]['resource']
            shot_depth += max(dict(realized.degree()).values())
            n_shots += 1

            cz, ms = census_structure(realized, target_set)
            ncz += cz
            nmeas += ms

    step_depth = shot_depth / n_shots if n_shots > 0 else 0.0
    return (float(cost), int(ncz), int(nmeas), float(shot_depth),
            int(n_shots), float(step_depth))


def prop_full(G, sp, M, tgt, anc, path_edges, edge_cost):
    steps = prop_select_exact(G, sp, M, tgt, anc, path_edges, edge_cost)
    return prop_realize_census(G, sp, steps, tgt)


# Monte Carlo worker
def _worker_trial(args):
    N, trial, ps_for_N, alpha_w, base_seed = args
    N_phys = PHYS_MULTIPLIER * N

    phys_seed = base_seed + 1000 * N + trial
    target_seed = base_seed + 7 * N + trial

    G, sp, path_edges, edge_cost = build_physical_network(
        N_phys, alpha_w, seed=phys_seed)
    rng = np.random.default_rng(phys_seed)
    perm = rng.permutation(N_phys)
    tgt, anc = perm[:N], perm[N:]

    res_dict = {p: {} for p in ps_for_N}
    for p in ps_for_N:
        # Same random numbers across p give nested ER instances.
        M = build_target_ER(N, p, seed=target_seed)
        res_b = meignant_steiner(G, sp, M, tgt)
        res_p = prop_full(G, sp, M, tgt, anc, path_edges, edge_cost)
        res_dict[p] = {'Steiner': res_b, 'prop': res_p}
    return N, trial, res_dict


# Parallel simulation
def run_grid_parallel(Ns, ps_by_N, num_trials, alpha_w, base_seed, ckpt_dir, max_workers):
    acc = {
        N: {
            p: {c: {m: [] for m in METRICS} for c in COLS}
            for p in ps_by_N[N]
        }
        for N in Ns
    }
    os.makedirs(ckpt_dir, exist_ok=True)
    completed = {}

    def add_result(N, result):
        for p in ps_by_N[N]:
            for col in COLS:
                vals = dict(zip(METRICS, result[p][col]))
                for m in METRICS:
                    acc[N][p][col][m].append(vals[m])

    # Existing full-grid checkpoints can be reused in mode 2.
    # Sparse mode-2 checkpoints are not accepted as complete in mode 1.
    for N in Ns:
        required_ps = ps_by_N[N]
        for trial in range(num_trials):
            path = os.path.join(ckpt_dir, f'N{N}_trial{trial}.pkl')
            if not os.path.exists(path):
                continue
            try:
                with open(path, 'rb') as f:
                    result = pickle.load(f)
                if not all(p in result for p in required_ps):
                    continue
                filtered = {p: result[p] for p in required_ps}
                completed[(N, trial)] = filtered
                add_result(N, filtered)
            except Exception:
                pass

    tasks = [
        (N, trial, tuple(ps_by_N[N]), alpha_w, base_seed)
        for N in Ns
        for trial in range(num_trials)
        if (N, trial) not in completed
    ]

    if max_workers is None:
        max_workers = min(8, os.cpu_count() or 4)

    n_points = sum(len(ps_by_N[N]) for N in Ns)
    print(f'parameter points={n_points}, comparisons={n_points * num_trials}')
    print(f'physical-network jobs={len(Ns) * num_trials}, '
          f'cached={len(completed)}, run={len(tasks)}, workers={max_workers}')

    if tasks:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_worker_trial, t): t for t in tasks}
            for future in tqdm(as_completed(futures), total=len(futures)):
                N, trial, result = future.result()
                path = os.path.join(ckpt_dir, f'N{N}_trial{trial}.pkl')
                with open(path + '.tmp', 'wb') as f:
                    pickle.dump(result, f)
                os.replace(path + '.tmp', path)
                add_result(N, result)

    return acc


def summarize(acc, Ns, ps_by_N):
    """Return means and 95% confidence intervals."""
    stats = {}
    for N in Ns:
        stats[N] = {}
        for p in ps_by_N[N]:
            stats[N][p] = {}
            for c in COLS:
                stats[N][p][c] = {}
                for m in METRICS:
                    a = np.array(acc[N][p][c][m], dtype=float)
                    stats[N][p][c][m] = (
                        float(a.mean()) if len(a) else 0.0,
                        float(1.96 * a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0)
    return stats


# CSV I/O
def export_csv(stats, Ns, ps_by_N, path, regime='near-RGG'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        head = ['regime', 'N', 'p', 'col']
        for m in METRICS:
            head += [f'{m}_mean', f'{m}_ci95']
        w.writerow(head)
        for N in Ns:
            for p in ps_by_N[N]:
                for c in COLS:
                    row = [regime, N, p, c]
                    for m in METRICS:
                        mean, ci = stats[N][p][c][m]
                        row += [f'{mean:.4f}', f'{ci:.4f}']
                    w.writerow(row)


def load_csv(path):
    """Load a CSV generated by this version of the script."""
    D, Ns, ps = {}, set(), set()
    required = ['regime', 'N', 'p', 'col']
    for m in METRICS:
        required.extend([f'{m}_mean', f'{m}_ci95'])

    with open(path, newline='') as f:
        rd = csv.DictReader(f)
        missing = [name for name in required if name not in (rd.fieldnames or [])]
        if missing:
            raise ValueError(
                'CSV format mismatch. Regenerate the data with this version of '
                f'simulate.py. Missing columns: {", ".join(missing)}')

        for row in rd:
            N, p, col = int(row['N']), float(row['p']), row['col']
            Ns.add(N)
            ps.add(p)
            D.setdefault(N, {}).setdefault(p, {})[col] = {
                m: (float(row[f'{m}_mean']), float(row[f'{m}_ci95']))
                for m in METRICS
            }
    return D, sorted(Ns), sorted(ps)


# Figures
C_ST = '#17A589'
C_PR = '#76448A'


def plot_paper_figures(stats, Ns, ps, Nrep, prep, out_dir, plot_time=True, plot_overhead=True):
    plt.rcParams.update({
        'font.size': 12, 'font.family': 'serif', 'mathtext.fontset': 'cm',
        'axes.labelsize': 13, 'axes.titlesize': 14, 'legend.fontsize': 11
    })

    def series(metric, by, fixed, col):
        if by == 'p':
            return (np.array(ps, dtype=float),
                    [stats[fixed][p][col][metric][0] for p in ps],
                    [stats[fixed][p][col][metric][1] for p in ps])
        return (np.array(Ns, dtype=float),
                [stats[N][fixed][col][metric][0] for N in Ns],
                [stats[N][fixed][col][metric][1] for N in Ns])

    def draw_panel(ax, metric, by, fixed, ylabel, title, legend=False):
        for col, c, mk, lab in [('Steiner', C_ST, 's-', 'Steiner (baseline)'),
                                ('prop', C_PR, 'o-', 'Proposed (rank-2)')]:
            xs, y, e = map(np.array, series(metric, by, fixed, col))
            ax.plot(xs, y, mk, color=c, lw=2.5, ms=7, label=lab, zorder=4)
            ax.fill_between(xs, y - e, y + e, color=c, alpha=0.15, lw=0, zorder=2)
        ax.set_xlabel(r'Target density $p$' if by == 'p' else r'Network size $N$')
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=10)
        ax.grid(True, ls='--', alpha=0.5, zorder=1)
        tag = rf'$N={fixed}$' if by == 'p' else rf'$p={fixed}$'
        ax.text(0.05, 0.93, tag, transform=ax.transAxes,
                bbox=dict(facecolor='white', alpha=0.85, edgecolor='gray', pad=3))
        if legend:
            ax.legend(loc='best', framealpha=0.9)
        return xs

    if plot_time:
        fig, ax = plt.subplots(2, 3, figsize=(16, 9))
        draw_panel(ax[0, 0], 'n_shots', 'p', Nrep, 'Number of steps', 'Serial Steps', True)
        ax[0, 0].axhline(Nrep - 1, color=C_ST, ls=':', lw=2, alpha=0.6)
        ax[0, 0].axhline(Nrep // 2, color=C_PR, ls=':', lw=2, alpha=0.6)
        draw_panel(ax[0, 1], 'step_depth', 'p', Nrep, 'Slots per step', 'Per-Step Slot Count')
        draw_panel(ax[0, 2], 'shot_depth', 'p', Nrep, 'Time slots', 'Time-Slot Depth')

        xs = draw_panel(ax[1, 0], 'n_shots', 'N', prep, 'Number of steps', '')
        ax[1, 0].plot(xs, xs - 1, ':', color=C_ST, lw=2, alpha=0.6)
        ax[1, 0].plot(xs, np.floor(xs / 2), ':', color=C_PR, lw=2, alpha=0.6)
        draw_panel(ax[1, 1], 'step_depth', 'N', prep, 'Slots per step', '')
        draw_panel(ax[1, 2], 'shot_depth', 'N', prep, 'Time slots', '')

        for i, a in enumerate(ax.flat):
            a.text(-0.20, 1.05, f"({chr(ord('a') + i)})", transform=a.transAxes,
                   fontsize=15, fontweight='bold', va='top')

        fig.subplots_adjust(top=0.92, bottom=0.1, left=0.06, right=0.98,
                            hspace=0.25, wspace=0.3)
        fig.savefig(os.path.join(out_dir, 'Fig1_Time_Performance.png'), dpi=300,
                    bbox_inches='tight')
        plt.close(fig)

    if plot_overhead:
        fig, ax = plt.subplots(2, 3, figsize=(16, 9))
        draw_panel(ax[0, 0], 'cost', 'p', Nrep, 'Routing cost', 'Network Resource Cost', True)
        draw_panel(ax[0, 1], 'ncz', 'p', Nrep, 'Total CZ gates', 'Global CZ Gates')
        draw_panel(ax[0, 2], 'nmeas', 'p', Nrep, 'Total measurements', 'Global Pauli Measurements')
        draw_panel(ax[1, 0], 'cost', 'N', prep, 'Routing cost', '')
        draw_panel(ax[1, 1], 'ncz', 'N', prep, 'Total CZ gates', '')
        draw_panel(ax[1, 2], 'nmeas', 'N', prep, 'Total measurements', '')

        for i, a in enumerate(ax.flat):
            a.text(-0.20, 1.05, f"({chr(ord('a') + i)})", transform=a.transAxes,
                   fontsize=15, fontweight='bold', va='top')

        fig.subplots_adjust(top=0.92, bottom=0.1, left=0.06, right=0.98,
                            hspace=0.25, wspace=0.3)
        fig.savefig(os.path.join(out_dir, 'Fig2_Global_Overheads.png'), dpi=300,
                    bbox_inches='tight')
        plt.close(fig)

# Main

def main():
    # Experiment settings
    NUM_TRIALS = 300
    BASE_SEED = 1234
    MAX_WORKERS = 8

    # 1 = full 9 x 9 (N, p) grid, same as the original code.
    # 2 = only the 17 (N, p) points used by the paper figures.
    EXPERIMENT_MODE = 2

    NS = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    PS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    N_FIXED = 50
    P_FIXED = 0.5

    RUN_SIMULATION = False
    PLOT_TIME = True
    PLOT_OVERHEAD = True

    OUTPUT_DIR = './outputs'

    if EXPERIMENT_MODE == 1:
        # Original calculation: all 81 parameter points.
        PS_BY_N = {N: list(PS) for N in NS}
        csv_name = 'grid_results.csv'
        mode_label = '1 (full 9x9 grid)'
    elif EXPERIMENT_MODE == 2:
        # Figure-only calculation:
        # N=50 sweeps all p; all other N values use only p=0.5.
        PS_BY_N = {
            N: (list(PS) if N == N_FIXED else [P_FIXED])
            for N in NS
        }
        csv_name = 'figure_points_results.csv'
        mode_label = '2 (figure points only)'
    else:
        raise ValueError('EXPERIMENT_MODE must be 1 or 2.')

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ckpt_dir = os.path.join(OUTPUT_DIR, f'checkpoints_seed{BASE_SEED}')
    csv_path = os.path.join(OUTPUT_DIR, csv_name)

    n_points = sum(len(PS_BY_N[N]) for N in NS)

    print(f'mode={mode_label}')
    print(f'trials={NUM_TRIALS}, seed={BASE_SEED}')
    print(f'N={NS}')
    print(f'p={PS}')
    print(f'parameter points={n_points}')
    print(f'total comparisons={n_points * NUM_TRIALS}')

    if RUN_SIMULATION:
        acc = run_grid_parallel(
            NS, PS_BY_N, NUM_TRIALS, ALPHA_W, BASE_SEED, ckpt_dir, MAX_WORKERS)
        stats = summarize(acc, NS, PS_BY_N)
        export_csv(stats, NS, PS_BY_N, csv_path)
    else:
        stats, NS, PS = load_csv(csv_path)

    if PLOT_TIME or PLOT_OVERHEAD:
        plot_paper_figures(
            stats, NS, PS, N_FIXED, P_FIXED, OUTPUT_DIR,
            plot_time=PLOT_TIME, plot_overhead=PLOT_OVERHEAD)

    print(f'\nN={N_FIXED}, p={P_FIXED}')
    for m in METRICS:
        st = stats[N_FIXED][P_FIXED]['Steiner'][m][0]
        pr = stats[N_FIXED][P_FIXED]['prop'][m][0]
        imp = (st - pr) / st * 100 if st > 0 else 0.0
        print(f'{m:>10s}: Steiner={st:10.4f}  rank-2={pr:10.4f}  change={imp:7.2f}%')


if __name__ == '__main__':
    main()
