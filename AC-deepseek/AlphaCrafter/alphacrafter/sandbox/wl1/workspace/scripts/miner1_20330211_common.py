"""Shared validation helpers for miner1 2033-02-11 cycle."""
import numpy as np
import pandas as pd

TRADABLE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX",
            "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

def load_panel():
    with open('scripts/panel_cache_20330211.pkl', 'rb') as f:
        return pd.read_pickle(f)

def daily_ic(factor_df, fwd_ret, min_valid=8):
    ic, dates = [], []
    for dt in factor_df.index:
        f = factor_df.loc[dt]
        r = fwd_ret.loc[dt]
        m = f.notna() & r.notna()
        if m.sum() >= min_valid:
            ic.append(f[m].rank().corr(r[m].rank()))
            dates.append(dt)
    return pd.Series(ic, index=pd.DatetimeIndex(dates))

def summarize(ic_s, label=""):
    ic = ic_s.dropna()
    if len(ic) == 0:
        return {"label": label, "n": 0, "ic": np.nan, "icir": np.nan, "hit": np.nan}
    mean_ic = ic.mean()
    std_ic = ic.std(ddof=1)
    return {"label": label, "n": len(ic), "ic": mean_ic,
            "icir": mean_ic / std_ic if std_ic > 0 else np.nan,
            "hit": (ic > 0).mean()}

def gate_pass(s, gate_ic=0.0070, gate_icir=0.0840):
    return abs(s["ic"]) >= gate_ic and abs(s["icir"]) >= gate_icir

def full_report(fac, px, horizons=(1, 5, 10), label="", recent_days=250):
    """fac: factor DataFrame indexed by date, cols = assets."""
    out = {}
    for h in horizons:
        fwd = px.shift(-h) / px - 1.0
        ic = daily_ic(fac, fwd)
        s = summarize(ic, f"{label}_h{h}")
        s["gate"] = gate_pass(s)
        out[f"h{h}"] = s
        # recent window
        if len(ic) > recent_days:
            s2 = summarize(ic.iloc[-recent_days:], f"{label}_h{h}_recent")
            out[f"h{h}_recent"] = s2
    cov = fac.notna().mean(axis=1)
    out["coverage"] = {"mean": float(cov.mean()), "min": float(cov.min())}
    ranks = fac.rank(axis=1)
    chg = ranks.diff().abs().mean(axis=1)
    out["turnover"] = float((chg / (fac.shape[1] - 1)).mean())
    return out

def print_report(rep, label=""):
    for k, v in rep.items():
        if isinstance(v, dict) and "ic" in v:
            print(f"  {k}: n={v['n']} IC={v['ic']:+.4f} ICIR={v['icir']:+.3f} hit={v['hit']:.3f} gate={'PASS' if v.get('gate') else 'fail'}")
        elif isinstance(v, dict) and "mean" in v:
            print(f"  {k}: coverage mean={v['mean']:.3f} min={v['min']:.3f}")
        else:
            print(f"  {k}: {v:.4f}")
