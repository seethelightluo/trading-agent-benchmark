"""Shared helpers for miner_1 factor research."""
import os
import numpy as np
import pandas as pd

CUT = pd.Timestamp("2026-07-15")
START = pd.Timestamp("2020-01-01")
SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
DATA_DIR = "../persistent/stock_data"
IDX_DIR = "../persistent/index_data"


def load_close(which=SYMBOLS, dir_=DATA_DIR):
    out = {}
    for s in which:
        d = pd.read_csv(os.path.join(dir_, f"{s}.csv"))
        d["date"] = pd.to_datetime(d["date"])
        d = d[d["date"] <= CUT].sort_values("date").set_index("date")
        for col in ["close", "open", "high", "low", "volume"]:
            d[col] = pd.to_numeric(d[col], errors="coerce")
        out[s] = d
    return out


def build_returns(closes, fwd_days=1):
    """Forward return (log or simple) over fwd_days, aligned to symbol frames."""
    fwd = {}
    for s, df in closes.items():
        ret = df["close"].shift(-fwd_days) / df["close"] - 1.0
        fwd[s] = ret
    return pd.DataFrame(fwd)


def factor_panel(closes, factor_fn):
    """factor_fn(df) -> Series of factor values indexed by date. Returns DataFrame dates x symbols."""
    cols = {}
    for s, df in closes.items():
        try:
            fv = factor_fn(df)
            if fv is not None and len(fv):
                cols[s] = fv
        except Exception as e:
            print(f"  [warn] {s}: {e}")
    return pd.DataFrame(cols)


def ic_analysis(factor_df, closes, fwd_days=1, min_names=8, rank=True):
    """Daily cross-sectional IC between factor and forward return."""
    fwd = build_returns(closes, fwd_days)
    dates = factor_df.index.intersection(fwd.index)
    ics = []
    obs = []
    for dt in dates:
        f = factor_df.loc[dt].dropna()
        r = fwd.loc[dt].reindex(f.index).dropna()
        common = f.index.intersection(r.index)
        if len(common) < min_names:
            continue
        x = f[common].astype(float)
        y = r[common].astype(float)
        if rank:
            x = x.rank()
            y = y.rank()
        if x.std() == 0 or y.std() == 0:
            continue
        ic = np.corrcoef(x, y)[0, 1]
        if np.isfinite(ic):
            ics.append(ic)
            obs.append(len(common))
    ics = np.array(ics)
    obs = np.array(obs)
    if len(ics) == 0:
        return {"n_dates": 0, "n_obs": 0, "ic": np.nan, "icir": np.nan,
                "hit": np.nan, "ic_std": np.nan}
    return {"n_dates": int(len(ics)), "n_obs": int(obs.sum()),
            "ic": float(ics.mean()), "icir": float(ics.mean() / ics.std()) if ics.std() > 0 else np.nan,
            "hit": float((ics > 0).mean()), "ic_std": float(ics.std())}


def decay_analysis(factor_df, closes, horizons=(1, 2, 3, 5, 10, 20, 30), min_names=8):
    out = {}
    for h in horizons:
        r = ic_analysis(factor_df, closes, fwd_days=h, min_names=min_names)
        out[h] = r["ic"]
    return out


def coverage(factor_df, closes):
    n_total = 0
    n_valid = 0
    for s in closes:
        if s in factor_df.columns:
            v = factor_df[s].dropna()
            n_valid += len(v)
        n_total += len(closes[s])
    return n_valid / n_total if n_total else np.nan


def turnover(factor_df, rebal=10):
    """Mean fraction of symbols whose cross-sectional decile/rank direction flips or rank change."""
    ranks = factor_df.rank(axis=1)
    chg = []
    for i in range(rebal, len(ranks)):
        prev = ranks.iloc[i - rebal].dropna()
        cur = ranks.iloc[i].dropna()
        common = prev.index.intersection(cur.index)
        if len(common) < 2:
            continue
        # normalized rank change (0..1), 1 means full flip
        c = (cur[common] - prev[common]).abs() / (len(common) - 1)
        chg.append(c.mean())
    return float(np.mean(chg)) if chg else np.nan


def summary(factor_df, closes, fwd_days=1, label=""):
    cov = coverage(factor_df, closes)
    to = turnover(factor_df)
    ic1 = ic_analysis(factor_df, closes, fwd_days=1)
    ic5 = ic_analysis(factor_df, closes, fwd_days=5)
    dec = decay_analysis(factor_df, closes)
    print(f"=== {label} ===")
    print(f"  coverage          : {cov:.3f}")
    print(f"  turnover(10d)     : {to:.3f}")
    for tag, r in [("1d", ic1), ("5d", ic5)]:
        print(f"  fwd {tag:>2} : IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} hit={r['hit']:.3f} "
              f"n_dates={r['n_dates']} n_obs={r['n_obs']}")
    print("  decay IC:", {k: round(v, 4) for k, v in dec.items()})
    return {"cov": cov, "turnover": to, "ic1": ic1, "ic5": ic5, "decay": dec}
