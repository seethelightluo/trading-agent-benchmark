"""miner_2 cycle-22 screen: candidate cross-asset factors, validation to 2026-10-07.

Uses the shared dense-per-asset methodology (factor computed on each asset's own
trading calendar, then reindexed onto the union panel for cross-sectional rank IC).
Admission gate: |IC| >= 0.0070, |ICIR| >= 0.0840 at h=10; n>=8 assets per date.
Window 2020-01-01 .. 2026-10-07 (visible_through for sim date 2026-10-08).
No persistence in this screen; only exploration.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import factor_validation_lib as fvl

# extend the shared lib validation window to the current visible date
fvl.CURRENT_DATE = pd.Timestamp("2026-10-07")
fvl.load_closes.__defaults__ = None  # just to avoid confusion; we reload below

close, vol, open_, high, low = fvl.load_closes(end_date=pd.Timestamp("2026-10-07"))

def _load_index(name):
    df = pd.read_csv(f"../persistent/index_data/{name}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp("2026-10-07")].set_index("date").sort_index()
    return df["close"].astype(float)

macro = {k: _load_index(k) for k in ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]}

HORIZONS = (1, 2, 3, 5, 10, 20)


def _align(s, c):
    return s.reindex(c.index).ffill()


# ---------------- candidate factor families ----------------
def f_ret_vol_corr_40(c, v, o, h, l, m, win=40, short=5, volwin=20):
    r = c.pct_change()
    r5 = r.rolling(short).sum()
    v20 = r.rolling(volwin).std()
    return r5.rolling(win).corr(v20).replace([np.inf, -np.inf], np.nan)


def f_drawdown_120_abs(c, v, o, h, l, m, win=120):
    return (c / c.rolling(win).max() - 1.0).replace([np.inf, -np.inf], np.nan)


def f_gap_ratio_20(c, v, o, h, l, m, win=20):
    if o is None:
        return pd.Series(np.nan, index=c.index)
    gap = (o / c.shift(1) - 1.0).abs()
    full = c.pct_change().abs()
    return (gap.rolling(win).mean() / full.rolling(win).mean().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def f_skew_5_60(c, v, o, h, l, m, short=5, volwin=60):
    r = c.pct_change()
    return (r.rolling(short).skew() / r.rolling(volwin).std().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def f_vol_trend_20x60(c, v, o, h, l, m, short=20, long=60):
    if v is None:
        return pd.Series(np.nan, index=c.index)
    return (v.rolling(short).mean() / v.rolling(long).mean().replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def f_dual_mom_10x60(c, v, o, h, l, m, short=10, long=60):
    r = c.pct_change()
    return (r.rolling(short).sum() - r.rolling(long).sum()).replace([np.inf, -np.inf], np.nan)


def f_overnight_ratio_30(c, v, o, h, l, m, win=30):
    if o is None:
        return pd.Series(np.nan, index=c.index)
    ovn = (o / c.shift(1) - 1.0)
    intra = c / o - 1.0
    num = ovn.rolling(win).sum()
    den = ovn.abs().rolling(win).sum() + intra.abs().rolling(win).sum()
    return (num / den.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)


def f_yield_spread_20x60(c, v, o, h, l, m, short=20, long=60):
    # for yield instruments: short-minus-long change in yield level (beta-neutral)
    chg = c.diff()
    return (chg.rolling(short).mean() - chg.rolling(long).mean()).replace([np.inf, -np.inf], np.nan)


def f_hl_range_pos_30(c, v, o, h, l, m, win=30):
    if h is None or l is None:
        return pd.Series(np.nan, index=c.index)
    pos = (c - l.rolling(win).min()) / (h.rolling(win).max() - l.rolling(win).min()).replace(0, np.nan)
    return pos.replace([np.inf, -np.inf], np.nan)


def f_cross_asset_spread(c, v, o, h, l, m, name, win=60):
    """Spread between asset's own momentum and a macro reference momentum (pairing)."""
    # use DXY as default reference (risk-off barometer)
    ref = _align(m[name], c)
    rm = ref.pct_change().rolling(10).sum()
    am = c.pct_change().rolling(10).sum()
    return (am - rm).replace([np.inf, -np.inf], np.nan)


FACTORS = {
    "ret_vol_corr_40": (f_ret_vol_corr_40, {}),
    "drawdown_120_abs": (f_drawdown_120_abs, {}),
    "gap_ratio_20": (f_gap_ratio_20, {}),
    "skew_5_60": (f_skew_5_60, {}),
    "vol_trend_20x60": (f_vol_trend_20x60, {}),
    "dual_mom_10x60": (f_dual_mom_10x60, {}),
    "overnight_ratio_30": (f_overnight_ratio_30, {}),
    "hl_range_pos_30": (f_hl_range_pos_30, {}),
    "mom_minus_dxy_10": (f_cross_asset_spread, {"name": "DXY"}),
}

print(f"union panel: {close.shape[0]} dates x {close.shape[1]} assets")
print(f"window: {close.index.min().date()} -> {close.index.max().date()}")

results = {}
for name, (fn, params) in FACTORS.items():
    res = fvl.validate_factor(fn, close, vol, open_, high, low, macro,
                              horizons=HORIZONS, admission_horizon=10, **params)
    results[name] = res
    fvl.print_result(name, res)

print("\n=== PASS/FAIL summary (h=10 gate) ===")
for name, res in results.items():
    ok = abs(res["ic"]) >= fvl.IC_GATE and abs(res["icir"]) >= fvl.ICIR_GATE
    print(f"{name:22s} IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} n={res['n_ic_dates']:5d} "
          f"cov_ad={res['coverage_asset_days']:.3f} cov_ge8={res['coverage_dates_ge8']:.3f} -> {'PASS' if ok else 'fail'}")