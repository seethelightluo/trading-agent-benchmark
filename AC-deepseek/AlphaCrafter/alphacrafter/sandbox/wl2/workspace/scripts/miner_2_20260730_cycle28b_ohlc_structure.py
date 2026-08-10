"""miner_2 cycle 28b: intraday / overnight price-structure family (OHLC anatomy).

Motivation: surviving library uses close (trend), close+DXY (beta), close+volume.
Where within the daily range returns accrue (overnight gap vs intraday drift) and
the shape of the daily candle (body vs shadow) is untouched and regime-informative.
Candidates (per-asset, own calendar):
  - overnight_gap_20:  mean(|open[t]/close[t-1]-1|) over 20d (gap magnitude)
  - overnight_share_60: var(overnight ret)/var(total ret) over 60d (news-priced-overnight share)
  - intraday_drift_20:  mean(close/open-1) over 20d (intraday drift, gap-free momentum)
  - body_ratio_20:      mean(|close-open|/(high-low)) over 20d (body dominance)
  - range_eff_60:       close-close realized vol / Parkinson (H-L) vol over 60d
Sign left raw; ensemble assigns direction from IC sign.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner2_lib import (load_close_panel, load_ohlc_panels, per_asset,
                        validate_factor, load_library_signals, report,
                        forward_returns, compute_ic, regime_breakdown)

panel = load_close_panel()
ohlc = load_ohlc_panels()
lib = load_library_signals(panel)
fwd_cache = {str(h): forward_returns(panel, h) for h in (1, 2, 3, 5, 10, 20)}

open_p, high_p, low_p, close_p = ohlc["open"], ohlc["high"], ohlc["low"], ohlc["close"]
prev_close = close_p.shift(1)
overnight = open_p / prev_close - 1.0
intraday = close_p / open_p - 1.0
total = close_p / prev_close - 1.0

idx = panel.index


def _per_pair(a_df, b_df, func):
    out = {}
    for a in a_df.columns:
        sa, sb = a_df[a].dropna(), b_df[a].dropna()
        out[a] = func(sa, sb).reindex(idx)
    return pd.DataFrame(out, index=idx)


def var_ratio(o, t, w=60):
    return o.rolling(w, min_periods=max(20, w // 2)).var() / t.rolling(w, min_periods=max(20, w // 2)).var()


def range_eff(hl, cc, w=60):
    cc_vol = cc.rolling(w, min_periods=max(20, w // 2)).std()
    pk = (np.log(hl) ** 2).rolling(w, min_periods=max(20, w // 2)).mean().apply(np.sqrt)
    return cc_vol / pk


cands = {
    "overnight_gap_20": per_asset(overnight, lambda s: s.rolling(20, min_periods=10).apply(lambda x: np.nanmean(np.abs(x)), raw=True)),
    "overnight_share_60": _per_pair(overnight, total, var_ratio),
    "intraday_drift_20": intraday.rolling(20, min_periods=10).mean(),
    "body_ratio_20": per_asset((close_p - open_p).abs() / (high_p - low_p),
                               lambda s: s.rolling(20, min_periods=10).apply(lambda x: np.nanmean(x), raw=True)),
    "range_eff_60": _per_pair(high_p / low_p, total, range_eff),
}

print("panel dates:", len(panel), "assets:", len(panel.columns))
print("\n=== VALIDATION (admission horizon 10d; gate |IC|>=0.007 & |ICIR|>=0.084) ===")
results = {}
for name, f in cands.items():
    m = validate_factor(f, panel, library=lib, fwd_cache=fwd_cache)
    p = report(name, m)
    results[name] = (m, p)

print("\n=== REGIME BREAKDOWNS for borderline ===")
for name, (m, p) in results.items():
    if abs(m["ic"]) >= 0.007 or abs(m["icir"]) >= 0.05:
        ic_ser = compute_ic(cands[name], fwd_cache["10"]).dropna()
        rb = regime_breakdown(ic_ser)
        print(name, "PASS" if p else "FAIL", "| full", m["ic"], m["icir"],
              "| regimes", {k: (v["ic"], v["icir"]) for k, v in rb.items()},
              "| maxlibcorr", m.get("max_abs_library_correlation"))

print("\n=== SUMMARY ===")
for name, (m, p) in results.items():
    print(f"{name:22s} PASS={p} | ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov_asset={m['coverage_asset_days']:.3f} to10={m.get('turnover_10d_rank')} "
          f"maxlibcorr={m.get('max_abs_library_correlation')} decay={m['decay_ic_by_horizon']}")
