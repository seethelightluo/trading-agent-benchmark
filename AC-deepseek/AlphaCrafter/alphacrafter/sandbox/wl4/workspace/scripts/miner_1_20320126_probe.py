"""miner_1 probe (2032-01-26) - data availability + re-validation of current effective factors."""
import sys, time, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

sys.path.insert(0, "scripts")
from factor_research_lib import (load_panels, close_panel, forward_returns,
                                 rank_ic_series, summarize_ic, coverage_metrics,
                                 turnover_rank, decay_profile, TRADABLE)

t0 = time.time()
panels = load_panels(days=3500)
closes = close_panel(panels)
rets = closes.pct_change()
mkt_ret = rets.mean(axis=1)
print(f"closes {closes.shape} | {closes.index.min().date()}..{closes.index.max().date()} | {time.time()-t0:.1f}s", flush=True)
print("assets present:", list(closes.columns), flush=True)

H = 10
fwd = forward_returns(closes, H)

def rolling_beta(y, x, win=60, min_obs=40):
    out = {}
    for a in y.columns:
        z = pd.concat([y[a].rename("y"), x.rename("x")], axis=1).dropna()
        cov = z["y"].rolling(win).cov(z["x"])
        var = z["x"].rolling(win).var()
        b = (cov / var).where(z["x"].rolling(win).count() >= min_obs)
        out[a] = b
    return pd.DataFrame(out, index=y.index)

existing = {}
existing["vol_adj_mom_accel_20x60"] = (closes/closes.shift(20)-1 - (closes/closes.shift(60)-1)) / rets.rolling(20).std()
existing["dn_mkt_beta_60d"] = rolling_beta(rets, mkt_ret.clip(upper=0), 60)
existing["rate_beta_cn10y_60d"] = rolling_beta(rets, closes["CN10Y"].pct_change(), 60)

def report(name, sig, expected_sign=1):
    ics = rank_ic_series(sig, fwd)
    s = summarize_ic(ics, expected_sign=expected_sign)
    cov = coverage_metrics(sig)
    to = turnover_rank(sig, 10)
    flag = "  <== FULL-PASS" if (abs(s["ic"]) >= 0.0070 and abs(s["icir"]) >= 0.0840) else ""
    print(f"{name:26s} IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit_ratio']:.2f} n={s['n_ic_dates']} "
          f"cov_ge8={cov['coverage_dates_ge8']:.2f} to={to if to is not None else float('nan'):.2f}{flag}", flush=True)
    return s, ics

print("\n=== RE-VALIDATION OF EFFECTIVE FACTORS (full history) ===", flush=True)
for name, sig in existing.items():
    exp = 1 if name != "rate_beta_cn10y_60d" else -1
    report(name, sig, expected_sign=exp)

print("\n=== RECENT 250d / 500d DRIFT ===", flush=True)
for name, sig in existing.items():
    exp = 1 if name != "rate_beta_cn10y_60d" else -1
    for win in [250, 500]:
        sub = sig.loc[sig.index[-win]:]
        ics = rank_ic_series(sub, fwd.loc[sub.index])
        s = summarize_ic(ics, expected_sign=exp)
        print(f"{name:26s} win={win}: IC={s['ic']:+.4f} ICIR={s['icir']:+.3f} hit={s['ic_hit_ratio']:.2f} n={s['n_ic_dates']}", flush=True)

print("\nlast close per asset:", {a: str(closes[a].dropna().index[-1].date()) for a in closes.columns}, flush=True)
