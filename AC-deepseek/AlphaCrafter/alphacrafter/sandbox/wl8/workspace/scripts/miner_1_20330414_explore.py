"""
miner_1 2033-04-14 exploration: fresh candidate factors on the 15-instrument
cross-asset tradable universe. All data observed through 2033-04-13
(visible_through from persistent/date.json); factor at t uses data <= t only,
forward return t..t+10 (admission horizon).

Motivation: factors/ is empty (all evicted/deprecated), trader runs fallback
mom10/vix-beta/yield-beta ensemble. Need robust low-correlation factors with
|IC| >= 0.0070 and |ICIR| >= 0.0840 at 10d horizon, stable across regimes
(2023-2033 includes vol spikes, crypto cycles, yield regime shifts, 2026 crash).

Candidates (interpretable, price/OHLC based):
 A) gap_vol_beta_20x60 : 20d mean |open/prev_close-1| scaled by 60d vol (gap shock)
 B) ret_accel_5x20     : 5d total return minus 20d total return (momentum acceleration)
 C) skew_60            : skewness of 1d returns over 60d calibrated by vol (tail tilt)
 D) gk_vol_ratio_10x40 : Garman-Klass 10d vol / 40d vol (relative vol expansion)
 E) close_loc_20       : mean (close-low)/(high-low) over 20d (buying pressure)
 F) lo_ret_20x60       : 20d total return minus 60d total return (reversal / low-return catch-up)
"""
import sys, os, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import miner_1_common as common

IC_TH = common.IC_THRESHOLD
ICIR_TH = common.ICIR_THRESHOLD
H = common.ADMISSION_HORIZON

np.set_printoptions(suppress=True)
pd.set_option("display.width", 200)

panel, vpanel = common.load_panel()
print(f"[data] panel shape {panel.shape}, dates {panel.index.min().date()}..{panel.index.max().date()}, n_assets {panel.shape[1]}")

# macro for potential conditioning (observation-only use)
vix = common.load_macro_panel("VIX")

# ---------- factor builders (values at t use data <= t) ----------

def build_A(panel, vpanel):
    """gap_vol_beta_20x60: mean |open/prev_close - 1| over 20d, scaled by 60d close-vol."""
    opens, prev = {}, {}
    for s in common.WATCHLIST:
        df = pd.read_csv(f"{common.STOCK_DIR}/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= panel.index.min()) & (df["date"] <= panel.index.max())].sort_values("date")
        opens[s] = df.set_index("date")["open"].astype(float)
    op = pd.DataFrame(opens).reindex(panel.index)
    prev_close = panel.shift(1)
    gap = (op / prev_close - 1.0).abs()
    g20 = gap.rolling(20, min_periods=10).mean()
    vol60 = panel.pct_change().rolling(60, min_periods=30).std()
    return g20 / vol60

def build_B(panel, vpanel):
    """ret_accel_5x20: 5d return minus 20d return (acceleration)."""
    r5 = panel.pct_change(5)
    r20 = panel.pct_change(20)
    return r5 - r20

def build_C(panel, vpanel):
    """skew_60: skewness of 1d returns over 60d (tail tilt; negative skew = crash-prone)."""
    r = panel.pct_change()
    skew = r.rolling(60, min_periods=30).skew()
    return skew

def build_D(panel, vpanel):
    """gk_vol_ratio_10x40: Garman-Klass 10d vol / 40d vol (relative vol expansion)."""
    hl, hc, lc = {}, {}, {}
    for s in common.WATCHLIST:
        df = pd.read_csv(f"{common.STOCK_DIR}/{s}.csv")
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= panel.index.min()) & (df["date"] <= panel.index.max())].sort_values("date")
        d = df.set_index("date")
        hl[s] = (np.log(d["high"]) - np.log(d["low"])).astype(float)
        hc[s] = (np.log(d["high"]) - np.log(d["close"])).astype(float)
        lc[s] = (np.log(d["low"]) - np.log(d["close"])).astype(float)
    HL = pd.DataFrame(hl).reindex(panel.index)
    HC = pd.DataFrame(hc).reindex(panel.index)
    LC = pd.DataFrame(lc).reindex(panel.index)
    gk = 0.5 * HL.pow(2) - (2 * np.log(2) - 1) * HC.pow(2) - LC.pow(2)
    gk = gk.clip(low