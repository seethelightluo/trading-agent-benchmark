"""miner_1 probe (2031-11-17) - data availability + current effective factor re-validation.

Visible data through previous completed trading day (2031-11-14). No lookahead,
no live-account interaction.
"""
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

# volume availability
vol_panel = pd.concat({a: panels[a]["volume"].astype(float) for a in TRADABLE if a in panels},
                      axis=1).sort_index().reindex(closes.index)
print("volume notna per asset:", vol_panel.notna().sum().to_dict(), flush=True)

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
    return s, ics, sig

print("\n=== RE-VALIDATION current effective factors (data thru 2031-11-14) ===", flush=True)
for name, sig in existing.items():
    s, ics, sig = report(name, sig, expected_sign=1 if name != "rate_beta_cn10y_60d" else -1)
    recent = {w: (ics.iloc[-w:].mean(), ics.iloc[-w:].mean()/ics.iloc[-w:].std(ddof=1)) for w in (126, 252) if len(ics) > w}
    print(f"   recent126 IC={recent.get(126,(float('nan'),float('nan')))[0]:+.4f} ICIR={recent.get(126,(float('nan'),float('nan')))[1]:+.3f} | "
          f"recent252 IC={recent.get(252,(float('nan'),float('nan')))[0]:+.4f} ICIR={recent.get(252,(float('nan'),float('nan')))[1]:+.3f}", flush=True)

print(f"\ndone {time.time()-t0:.1f}s", flush=True)
