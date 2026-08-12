"""Diagnostic: capture what strategy.py proposed at 2028-06-29 block start.

Stubs rebalance_to_weights to record the target; runs the hook logic with the
same date context the step used (current date = 2028-06-29 pre-step is gone;
we simulate by directly invoking the strategy's internal computation path with
the live account watchlist and current data).
"""
import json
import math

import strategy as S
from alphacrafter.sim.utils import get_account_dict

calls = []
S.rebalance_to_weights = lambda w, **kw: calls.append((w, kw))

# force block-start behavior
S.is_block_start = lambda: True

assets = list(get_account_dict()["watch_list"])
frames = {a: S.get_df(a) for a in assets}
close = {a: S.series(frames[a]) for a in assets}
open_ = {a: S.series(frames[a], "open") for a in assets}
frozen = S.detect_frozen(close)
live = [a for a in assets if a not in frozen]
print("frozen:", sorted(frozen))
print("live:", sorted(live))

ret = {a: close[a].pct_change() for a in assets}
panel = S.pd.concat([ret[a].rename(a) for a in assets], axis=1, join="inner").dropna()
print("panel rows:", len(panel))

ens = S.load_ensemble()
print("ensemble:", [(f, w, d) for f, w, d in ens])

# regime
lp = panel[live]
market = lp.mean(axis=1)
wealth = (1.0 + market).cumprod()
mdd = float((wealth / wealth.rolling(60).max() - 1.0).tail(20).min())
mkt20 = float(market.tail(20).mean())
vol20 = float(lp.tail(20).std().mean())
vol_med = float(lp.tail(120).std().median(axis=0))
risk_off = (mkt20 < 0.0 and mdd < -0.025) or (vol20 > 1.25 * max(vol_med, 1e-6))
risk_on = mkt20 > 0.0 and mdd > -0.015
vix = S.series(S.get_df("VIX"))
vix_level = float(vix.iloc[-1]) if vix is not None and len(vix) else None
eq_live = [a for a in S.EQ_ASSETS if a in live]
eq_ret21 = float(S.np.mean([close[a].iloc[-1] / close[a].iloc[-22] - 1.0 for a in eq_live]))
stress = risk_off and ((vix_level is not None and vix_level >= S.VIX_STRESS) or eq_ret21 < S.EQ_RET21_STRESS)
print(f"regime: mkt20={mkt20*100:.2f}% mdd={mdd*100:.2f}% vol20={vol20*100:.2f}% "
      f"vol_med={vol_med*100:.2f}% risk_off={risk_off} risk_on={risk_on} "
      f"VIX={vix_level:.1f} eq_ret21={eq_ret21*100:.2f}% stress={stress}")

# run the full strategy hook (will call stubbed rebalance)
S.strategy_hook()

if calls:
    w, kw = calls[-1]
    print("\nPROPOSED TARGET (sum=%.6f):" % sum(w.values()))
    for a in sorted(w, key=lambda x: -w[x]):
        print(f"  {a:12s} {w[a]*100:6.2f}%")
    print("forecast k scale kw:", kw.get("factor_ids"))
    f = kw.get("forecast_returns")
    if f:
        print("forecast sample:", {a: round(f[a], 5) for a in list(f)[:5]})
else:
    print("\nNO rebalance call made")
