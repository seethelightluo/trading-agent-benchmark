"""miner_1 drift audit of all 22 library factors as of 2033-11-24.

Recomputes each persisted factor signal on full history and evaluates:
  WARM  : 2020-01-01..2026-07-15 (canonical admission window, all 15 assets)
  OOS   : 2026-07-16..last       (online period, live assets only)
  RECENT: last ~365d             (live assets only)
Gate: |IC|>=0.007 and |ICIR|>=0.084 at h=10 on WARM.
"""
import sys, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, forward_returns, rank_ic_series,
                           VAL_START, VAL_END)
from miner1_libfuncs import FUNCS, build_refs, FROZEN, LIVE
from miner1_eval_helper import stats, load_library_artifacts

H = 10

def main():
    prices = load_prices(days=4000)
    refs = build_refs(prices)
    max_date = max(dd.index.max() for dd in prices.values())
    fwd = forward_returns(prices, H)
    oos_start = VAL_END + pd.Timedelta(days=1)
    recent_start = max_date - pd.Timedelta(days=365)
    print(f"data last date: {max_date.date()}, recent_start={recent_start.date()}, "
          f"live assets: {LIVE}", flush=True)

    rows = []
    for name, fn in FUNCS.items():
        try:
            panel = fn.__module__ and None
            # call via factor_to_panel
            from factor_common import factor_to_panel
            panel = factor_to_panel(lambda df, s, fn=fn: fn(df, s, ref=refs), prices)
        except Exception as e:
            print(f"{name}: ERROR {e}", flush=True)
            continue
        warm_p = panel[(panel.index >= VAL_START) & (panel.index <= VAL_END)]
        warm_ic = rank_ic_series(warm_p, fwd.reindex(warm_p.index), min_valid=8)
        oos_p = panel[(panel.index >= oos_start)]
        oos_ic = rank_ic_series(oos_p[LIVE], fwd.reindex(oos_p.index)[LIVE], min_valid=8)
        rec_p = panel[(panel.index >= recent_start)]
        rec_ic = rank_ic_series(rec_p[LIVE], fwd.reindex(rec_p.index)[LIVE], min_valid=8)
        sw, so, sr = stats(warm_ic), stats(oos_ic), stats(rec_ic)
        rows.append({
            'factor': name,
            'warm_ic': sw['ic'], 'warm_icir': sw['icir'], 'warm_hit': sw['hit'], 'warm_n': sw['n'],
            'oos_ic': so['ic'], 'oos_icir': so['icir'], 'oos_hit': so['hit'], 'oos_n': so['n'],
            'rec_ic': sr['ic'], 'rec_icir': sr['icir'], 'rec_hit': sr['hit'], 'rec_n': sr['n'],
        })
        flag = 'PASS' if (abs(sw['ic']) >= 0.007 and abs(sw['icir']) >= 0.084) else 'FAIL'
        drift = 'ok' if (abs(so['ic']) >= 0.007 and abs(so['icir']) >= 0.084) else 'DRIFT'
        print(f"{name:28s} warm ic={sw['ic']:+.4f} icir={sw['icir']:+.3f} | oos ic={so['ic']:+.4f} "
              f"icir={so['icir']:+.3f} n={so['n']} | rec ic={sr['ic']:+.4f} icir={sr['icir']:+.3f} "
              f"n={sr['n']} | {flag} oos:{drift}", flush=True)

    out = Path('scripts/miner_1_20331124_drift_audit.json')
    out.write_text(json.dumps(rows, indent=1, default=str))
    print(f"saved -> {out}", flush=True)

if __name__ == '__main__':
    main()
