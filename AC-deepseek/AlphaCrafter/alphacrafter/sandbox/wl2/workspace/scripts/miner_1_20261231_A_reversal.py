"""miner_1 2026-12-31 Factor Idea A: short-horizon reversal.

Motivation: 2026 H2 tape is reversal chop - momentum leaders (WTI/N225/SPX) rolled
over after 12-03 entry while ETH extended; short-horizon mean reversion may
predict the next 10d better than continuation in this regime.
Construction: factor = -ret_h (h in {3,5,10}), i.e. high factor = recent loser.
Validate on full 2020-2026 sample with regime splits; admission |IC|>=0.007, |ICIR|>=0.084.
"""
import sys, json
sys.path.insert(0, "scripts")
from miner_1_20261119_lib import (series, fwd_by_horizon, cross_sectional_rank,
                                  spearman_ic_matrix, summarize, decay_curve,
                                  turnover_10d_rank, coverage_stats,
                                  library_pairwise_corr, to_grid, ASSETS,
                                  GATE_IC, GATE_ICIR)

import numpy as np
import pandas as pd

def rev_factor(series, h):
    out = {}
    for s, df in series.items():
        close = df["close"]
        out[s] = -1.0 * (close / close.shift(h) - 1.0)
    return out

for h in [3, 5, 10]:
    cand = rev_factor(series, h)
    mat = to_grid(cand)
    rank_mat = cross_sectional_rank(mat)
    fwd10 = fwd_by_horizon[10]
    ics = spearman_ic_matrix(mat, fwd10)
    summ = summarize(ics, "rev_%dd" % h)
    if summ is None:
        print("rev_%dd NO VALID IC DATES" % h); continue
    cov_ad, cov_d8 = coverage_stats(mat)
    to = turnover_10d_rank(rank_mat)
    dec = decay_curve(mat, fwd_by_horizon)
    lpc, lpc_name, lpc_max = library_pairwise_corr(mat)
    passed = abs(summ["ic"]) >= GATE_IC and abs(summ["icir"]) >= GATE_ICIR
    print("=" * 70)
    print("FACTOR: rev_%dd  IC=%.4f ICIR=%.4f hit=%.3f n=%d" % (h, summ["ic"], summ["icir"], summ["hit"], summ["n_ic_dates"]))
    print("  coverage_asset_days=%.3f dates_ge8=%.3f turnover_10d=%.4f" % (cov_ad, cov_d8, to))
    print("  decay:", dec)
    print("  regime:", json.dumps(summ["regime"]))
    print("  max_lib_corr=%.4f (%s)" % (lpc_max, lpc_name))
    print("  GATE PASS:", passed)
    if passed:
        np.save("factors/rev_%dd.signal.npy" % h, rank_mat)
        print("  artifact saved factors/rev_%dd.signal.npy" % h)
