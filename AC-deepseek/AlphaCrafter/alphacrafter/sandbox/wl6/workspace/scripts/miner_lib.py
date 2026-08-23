"""Shared helper harness for factor validation (miner_1)."""
import os, numpy as np, pandas as pd

WATCH = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX",
         "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
DATA = "../persistent/stock_data"

def load_close_panel(end="2034-09-26", start="2020-01-01"):
    """Return close panel (index=date, cols=assets), sorted ascending, and pct returns."""
    closes = {}
    for s in WATCH:
        p = os.path.join(DATA, s + ".csv")
        df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
        df = df[~df.index.duplicated(keep="last")]
        closes[s] = df["close"]
    panel = pd.DataFrame(closes).sort_index()
    panel = panel.loc[(panel.index >= pd.Timestamp(start)) & (panel.index <= pd.Timestamp(end))]
    # daily returns
    rets = panel.pct_change()
    return panel, rets

def cross_sectional_ic(signal, forward_ret, min_assets=8):
    """signal: df (idx=date, cols=assets) of factor values.
       forward_ret: df of forward returns already computed, same index/cols.
       Returns per-date IC series and summary metrics for horizon-based forward returns."""
    # align
    sig = signal.reindex(index=forward_ret.index, columns=forward_ret.columns)
    fwd = forward_ret.reindex(index=sig.index, columns=sig.columns)
    dates, ics = [], []
    for dt in fwd.index:
        x = sig.loc[dt]
        y = fwd.loc[dt]
        m = x.notna() & y.notna()
        n = int(m.sum())
        if n < min_assets:
            continue
        eps = 1e-9
        if x[m].std() < eps or y[m].std() < eps:
            continue
        ic = np.corrcoef(x[m], y[m])[0, 1]
        if np.isfinite(ic):
            dates.append(dt); ics.append(ic)
    ics = np.array(ics)
    out = pd.Series(ics, index=pd.Index(dates, name="date"))
    return out

def summarize(ic_series, horizon):
    ic = float(ic_series.mean())
    sd = float(ic_series.std())
    icir = ic / sd if sd > 0 else 0.0
    hit = float((ic_series > 0).mean()) if len(ic_series) else float("nan")
    n = len(ic_series)
    return {"ic": ic, "icir": icir, "ic_hit_ratio": hit, "n_ic_dates": n,
            "admission_horizon": horizon}

def compute_forward_rets(rets, horizon):
    """Forward return over horizon days using close-based daily rets (compounded)."""
    return (1 + rets).rolling(horizon).apply(lambda x: x.prod() - 1, raw=True).shift(-horizon)

def decay_analysis(factor_fn, rets, horizons=(1,2,3,5,10,20)):
    """Return dict horizon->IC using forward returns computed from factor signal at date t."""
    out = {}
    end = rets.index.max()
    for h in horizons:
        fwd = compute_forward_rets(rets, h)
        sig = factor_fn()
        ic = cross_sectional_ic(sig, fwd)
        out[str(h)] = round(float(ic.mean()), 4) if len(ic) else None
    return out

print("miner_lib loaded")