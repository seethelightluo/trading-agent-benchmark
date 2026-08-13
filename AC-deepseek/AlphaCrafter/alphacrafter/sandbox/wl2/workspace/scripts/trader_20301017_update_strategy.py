"""Trader cycle 2030-10-17: switch strategy.py live ensemble from the stale
2026-11-19 10f set to the refreshed 2030-10-17 5f set, and add live
computations for mom_180d_skip5 and range_pos_252 (canonical formulas from
factors/*.json). Keeps all infra (rank-z, softmax, defensive tilt, caps,
turnover dampener, rebalance_to_weights gate) unchanged."""
from pathlib import Path

p = Path("strategy.py")
s = p.read_text()

# 1. LIVE_FIDS -> new 5-factor set
import re
s2 = re.sub(
    r'LIVE_FIDS = \{[^}]+\}',
    'LIVE_FIDS = {\n    "downbeta_spx_60", "max_consec_gain_20", "spx_corr60",\n    "mom_180d_skip5", "range_pos_252",\n}',
    s, count=1)
assert s2 != s, "LIVE_FIDS replacement failed"
s = s2

# 2. Replace _live_factors body (between its def and def _fit_weights)
start = s.index("def _live_factors(assets):")
end = s.index("def _fit_weights(pref, cap=CAP")
new_func = '''def _live_factors(assets):
    """Recompute the 5 ensemble factor signals live from price data visible at
    the decision date (2030-10-17 ensemble: downbeta_spx_60, max_consec_gain_20,
    spx_corr60, mom_180d_skip5, range_pos_252). Formulas match the persisted
    factor JSONs (v2.0.0). Perfectly-flat trailing 15d series (feed artifact)
    -> NaN (neutral rank 0.5)."""
    import numpy as np
    import pandas as pd
    closes = {}
    for a in assets:
        try:
            df = get_stock_daily_data(a, days=300)
        except Exception:
            df = None
        if df is not None and "close" in df and len(df) >= 130:
            closes[a] = df.set_index(pd.to_datetime(df["date"]))["close"].astype(float)
    if len(closes) < 8:
        return {}
    spx = closes.get("SPX")
    spx_ret = spx.pct_change() if spx is not None else None

    def longest_run(x):
        m = 0.0
        cur = 0
        for v in x:
            if v == 1:
                cur += 1
                if cur > m:
                    m = cur
            else:
                cur = 0
        return m

    per = {a: {} for a in assets}
    for a, c in closes.items():
        ret = c.pct_change()
        f = per[a]
        flat = bool(len(ret) >= 15 and float(ret.tail(15).std()) < 1e-12)
        f["_flat"] = flat
        if flat:
            continue  # all live factor values stay NaN -> neutral rank
        pos = (ret > 0).astype(int)
        f["max_consec_gain_20"] = pos.rolling(21, min_periods=10).apply(longest_run, raw=True)
        f["mom_180d_skip5"] = c.shift(5) / c.shift(185) - 1.0
        rng_min = c.rolling(252, min_periods=30).min()
        rng_max = c.rolling(252, min_periods=30).max()
        f["range_pos_252"] = (c - rng_min) / (rng_max - rng_min).replace(0, np.nan)
        if spx_ret is not None:
            f["spx_corr60"] = ret.rolling(60, min_periods=15).corr(spx_ret)
            m2 = pd.concat([ret, spx_ret], axis=1, join="inner").dropna()
            m2.columns = ["a", "s"]

            def downbeta(x):
                sub = m2.loc[x.index]
                sub = sub[sub["s"] < 0]
                if len(sub) < 15:
                    return np.nan
                if sub["s"].var() < 1e-12:
                    return np.nan
                return float(sub["a"].cov(sub["s"]) / sub["s"].var())

            f["downbeta_spx_60"] = m2["a"].rolling(60, min_periods=20).apply(downbeta, raw=False)

    out = {}
    for fid in LIVE_FIDS:
        vals = []
        for a in assets:
            f = per.get(a, {})
            if f.get("_flat", False):
                vals.append(float("nan"))
                continue
            s = f.get(fid)
            if s is not None:
                if hasattr(s, "iloc"):
                    if len(s) > 0:
                        v = float(s.iloc[-1])
                        vals.append(v if v == v else float("nan"))
                    else:
                        vals.append(float("nan"))
                else:
                    v = float(s)
                    vals.append(v if v == v else float("nan"))
            else:
                vals.append(float("nan"))
        if sum(1 for v in vals if v == v) >= LIVE_MIN_FINITE:
            out[fid] = vals
    return out


'''
s = s[:start] + new_func + s[end:]

# 3. Docstring ensemble description
old_doc = '''Ensemble (2026-11-19, quality_ic_tilt, 10f cap): downbeta_spx_60 .1361(+1),
max_consec_gain_20 .1309(+1), mom20_volproxy60 .1050(+1), gain_loss_20 .0990(+1),
vol_of_vol20x60 .0975(+1), spx_corr60 .0972(+1), usdjpy_beta_cond_120x60 .0868(+1),
mom30_vol60 .0840(+1), days_since_high_60 .0826(-1), max_consec_loss_20 .0809(-1).'''
new_doc = '''Ensemble (2030-10-17, quality_ic_tilt, 5f cap): downbeta_spx_60 .3263(+1),
max_consec_gain_20 .2657(+1), spx_corr60 .1739(+1), mom_180d_skip5 .1500(+1),
range_pos_252 .0841(+1). First refreshed ensemble after ~37 stale cycles;
the 2026-11-19 10f momentum set (mom20_volproxy60, gain_loss_20,
vol_of_vol20x60, usdjpy_beta_cond_120x60, mom30_vol60, days_since_high_60,
max_consec_loss_20) is DROPPED (miner_1 gate decay through 2030-10-16).'''
assert old_doc in s, "docstring anchor not found"
s = s.replace(old_doc, new_doc)

p.write_text(s)
print("strategy.py updated OK")
