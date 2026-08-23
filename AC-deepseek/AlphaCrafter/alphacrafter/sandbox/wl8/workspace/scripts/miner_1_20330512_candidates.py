"""miner_1 2033-05-12 exploration cycle. 7 candidate factors on 15-instrument cross-asset tradable universe.
Data through visible_through=2033-05-11. Admission horizon H=10d. Gate: |IC|>=0.0070, |ICIR|>=0.0840."""
import json, math
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, "scripts")
import miner_1_common as common

IC_TH = common.IC_THRESHOLD
ICIR_TH = common.ICIR_THRESHOLD
H = common.ADMISSION_HORIZON

panel, vpanel = common.load_panel()
print(f"[data] panel shape {panel.shape}, dates {panel.index.min().date()}..{panel.index.max().date()} n_dates={panel.shape[0]} n_assets={panel.shape[1]}")

ohlc = {}
for s in common.WATCHLIST:
    df = pd.read_csv(f"{common.STOCK_DIR}/{s}.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= panel.index.min()) & (df["date"] <= panel.index.max())].sort_values("date")
    ohlc[s] = df.set_index("date")[["open", "high", "low", "close"]].astype(float).reindex(panel.index)

def series(col):
    return pd.DataFrame({s: ohlc[s][col] for s in common.WATCHLIST}).reindex(panel.index)

OP, HI, LO, CL = series("open"), series("high"), series("low"), series("close")

def build_A():
    prev_close = CL.shift(1)
    gap = (OP / prev_close - 1.0).abs()
    g20 = gap.rolling(20, min_periods=10).mean()
    vol60 = CL.pct_change().rolling(60, min_periods=30).std()
    return g20 / vol60

def build_B():
    return CL.pct_change(5) - CL.pct_change(20)

def build_C():
    return CL.pct_change().rolling(60, min_periods=30).skew()

def build_D():
    hl = np.log(HI) - np.log(LO)
    hc = np.log(HI) - np.log(CL)
    lc = np.log(LO) - np.log(CL)
    gk = (0.5 * hl.pow(2) - (2 * math.log(2) - 1) * hc.pow(2) - lc.pow(2)).clip(lower=0)
    gk10 = gk.rolling(10, min_periods=6).mean()
    gk40 = gk.rolling(40, min_periods=20).mean()
    return np.sqrt(gk10 / gk40)

def build_E():
    rng = (HI - LO).replace(0, np.nan)
    return ((CL - LO) / rng).rolling(20, min_periods=10).mean()

def build_F():
    return CL.pct_change(20) - CL.pct_change(60)

def build_G():
    hlw = np.log(HI) - np.log(LO)
    m5 = hlw.rolling(5, min_periods=3).mean()
    m20 = hlw.rolling(20, min_periods=10).mean()
    return m5 / m20

builders = {
    "gap_vol_ratio_20x60": build_A,
    "ret_accel_5x20": build_B,
    "skew_60": build_C,
    "gk_vol_ratio_10x40": build_D,
    "close_loc_20": build_E,
    "lo_ret_20x60": build_F,
    "hl_width_ratio_5x20": build_G,
}

fwd10 = common.forward_returns(panel, horizon=H)
fwd20 = common.forward_returns(panel, horizon=20)
fwd5 = common.forward_returns(panel, horizon=5)

for fid, builder in builders.items():
    try:
        f = builder()
    except Exception as e:
        print(f"[{fid}] BUILD ERROR: {e}")
        continue
    ics = common.spearman_ic_series(f, fwd10)
    m = common.ic_metrics(ics)
    cov = common.coverage(f, panel)
    to = common.turnover_rank_chg(f, panel)
    ic5 = common.spearman_ic_series(f, fwd5)
    ic20 = common.spearman_ic_series(f, fwd20)
    dec5 = float(ic5.mean()) if len(ic5) >= common.MIN_IC_DATES else float("nan")
    dec20 = float(ic20.mean()) if len(ic20) >= common.MIN_IC_DATES else float("nan")
    last = ics.index[-1] - pd.Timedelta(days=400)
    ics_l = ics.loc[ics.index >= last]
    ml = common.ic_metrics(ics_l) if len(ics_l) >= 30 else {"ic": float("nan"), "icir": float("nan"), "n_ic_dates": len(ics_l), "hit": float("nan")}
    reg = common.regime_slices(ics)
    print("=" * 100)
    print(f"[{fid}] IC={m['ic']:.4f} ICIR={m['icir']:.4f} n_dates={m['n_ic_dates']} hit={m['hit']:.3f} t={m['tstat']:.2f} coverage={cov:.3f} turnover={to:.3f}")
    print(f"    decay h5={dec5:.4f} h10={m['ic']:.4f} h20={dec20:.4f} | last~400d: IC={ml['ic']:.4f} ICIR={ml['icir']:.4f} n={ml['n_ic_dates']}")
    print(f"    regimes: {reg}")
    gate = (abs(m["ic"]) >= IC_TH) and (abs(m["icir"]) >= ICIR_TH) and m["n_ic_dates"] >= common.MIN_IC_DATES
    print(f"    GATE {'PASS' if gate else 'FAIL'} (|IC|>={IC_TH} & |ICIR|>={ICIR_TH} & n>={common.MIN_IC_DATES})")