"""miner_1 candidate evaluation helper (2033-05-12).

Validates one candidate factor across:
  WARM  : 2020-01-01..2026-07-15 (canonical admission reference, all 15 assets)
  OOS   : 2026-07-16..last     (online period, live assets only)
  RECENT: last ~365d           (live assets only)
Admission gate (shared): |IC|>=0.007 and |ICIR|>=0.084 at h=10 on WARM.
Library correlation: max abs mean daily cross-sectional Spearman vs persisted
signal artifacts (factors/*_signal.npy on the canonical grid).
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, forward_returns, rank_ic_series,
                           VAL_START, VAL_END, canonical_grid, signal_matrix)

FROZEN = {'HSI', 'SX5E', 'BTC', 'US10Y', 'CN10Y'}
LIVE = [s for s in WATCHLIST if s not in FROZEN]
H = 10


def stats(ic):
    if len(ic) < 2:
        return {'ic': float('nan'), 'icir': float('nan'), 'hit': float('nan'), 'n': int(len(ic))}
    sd = ic.std(ddof=1)
    return {'ic': float(ic.mean()),
            'icir': float(ic.mean() / sd) if sd > 0 else float('nan'),
            'hit': float((ic > 0).mean()), 'n': int(ic.notna().sum())}


def load_library_artifacts():
    """Load persisted signal artifacts -> {factor_id: (n_dates,15) matrix} on canonical grid."""
    out = {}
    for p in sorted(Path('factors').glob('*_signal.npy')):
        fid = p.name.replace('_signal.npy', '')
        if fid in ('factor_ensemble',):
            continue
        out[fid] = np.load(p, allow_pickle=False)
    return out


def library_corr_matrix(cand_mat, lib_mats, min_valid=8):
    """Max abs mean daily Spearman between candidate and each library artifact."""
    best, best_id = 0.0, None
    n = cand_mat.shape[0]
    for fid, lm in lib_mats.items():
        if lm.shape[0] != n:
            continue
        corrs = []
        for i in range(n):
            a = cand_mat[i]; b = lm[i]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() >= min_valid:
                ra = pd.Series(a[m]).rank().values
                rb = pd.Series(b[m]).rank().values
                c = np.corrcoef(ra, rb)[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id


def eval_candidate(factor_id, panel_fn, prices=None, print_out=True):
    """panel_fn(prices) -> wide panel indexed by date with WATCHLIST columns."""
    if prices is None:
        prices = load_prices(days=4000)
    max_date = max(dd.index.max() for dd in prices.values())
    panel = panel_fn(prices)

    fwd = forward_returns(prices, H)
    oos_start = VAL_END + pd.Timedelta(days=1)
    recent_start = max_date - pd.Timedelta(days=365)

    warm_p = panel[(panel.index >= VAL_START) & (panel.index <= VAL_END)]
    warm_ic = rank_ic_series(warm_p, fwd.reindex(warm_p.index), min_valid=8)
    oos_p = panel[(panel.index >= oos_start)]
    oos_ic = rank_ic_series(oos_p[LIVE], fwd.reindex(oos_p.index)[LIVE], min_valid=8)
    rec_p = panel[(panel.index >= recent_start)]
    rec_ic = rank_ic_series(rec_p[LIVE], fwd.reindex(rec_p.index)[LIVE], min_valid=8)

    s_warm = stats(warm_ic); s_oos = stats(oos_ic); s_rec = stats(rec_ic)

    # coverage on warm window
    fac_w = warm_p
    total_cells = fac_w.shape[0] * fac_w.shape[1]
    valid_cells = int(fac_w.notna().sum().sum())
    coverage = valid_cells / total_cells if total_cells else 0.0
    ge8 = float((fac_w.notna().sum(axis=1) >= 8).mean())

    # turnover: mean abs rank change over 10d steps on warm window
    ranked = fac_w.rank(axis=1)
    turnover = float(ranked.diff(10).abs().mean().mean()) if len(ranked) > 10 else float('nan')

    # library correlation on canonical grid
    grid = canonical_grid(prices)
    cand_mat = signal_matrix(panel, grid, prices)
    lib_mats = load_library_artifacts()
    rho, rho_id = library_corr_matrix(cand_mat, lib_mats)

    res = {
        'factor_id': factor_id,
        'warm': s_warm, 'oos_live': s_oos, 'recent_live': s_rec,
        'coverage_asset_days': coverage, 'coverage_dates_ge8': ge8,
        'turnover_10d_rank': turnover,
        'max_abs_library_correlation': rho, 'max_corr_library_id': rho_id,
        'panel_shape': list(panel.shape), 'last_date': str(max_date.date()),
    }
    if print_out:
        print(f"=== {factor_id} ===", flush=True)
        print(f"  warm   : ic={s_warm['ic']:+.4f} icir={s_warm['icir']:+.3f} hit={s_warm['hit']:.3f} n={s_warm['n']}", flush=True)
        print(f"  oos    : ic={s_oos['ic']:+.4f} icir={s_oos['icir']:+.3f} hit={s_oos['hit']:.3f} n={s_oos['n']}", flush=True)
        print(f"  recent : ic={s_rec['ic']:+.4f} icir={s_rec['icir']:+.3f} hit={s_rec['hit']:.3f} n={s_rec['n']}", flush=True)
        print(f"  coverage={coverage:.3f} ge8={ge8:.3f} turnover={turnover:.3f} "
              f"max_abs_lib_corr={rho:.3f}({rho_id})", flush=True)
        ok = abs(s_warm['ic']) >= 0.007 and abs(s_warm['icir']) >= 0.084
        print(f"  ADMISSION |IC|={abs(s_warm['ic']):.4f}>=0.007 {abs(s_warm['ic'])>=0.007} "
              f"|ICIR|={abs(s_warm['icir']):.4f}>=0.084 {abs(s_warm['icir'])>=0.084} -> {'PASS' if ok else 'FAIL'}", flush=True)
    return res
