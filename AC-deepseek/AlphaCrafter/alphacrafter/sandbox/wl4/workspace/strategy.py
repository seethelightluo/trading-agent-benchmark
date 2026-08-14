"""v64 (2035-10-15):
v63 + four risk-adjustment fires from the post-v63 blocks (last live step was
05-28..06-11; safety advances 07-09..10-15 moved the window; block returns
measured on 10-trading-day price windows ending 2035-10-12; SCREENER ensemble
unchanged 0.60/0.40 = vol_adj_mom_accel_20x60 dir=+1 PRIMARY w=0.60,
dn_mkt_beta_60d dir=+1 w=0.40, loaded live from factors/factor_ensemble.json;
account 1216697.17 -> 1277634.57 (+5.01% cum over the frozen window, cash 0,
gross 100%, 15 positions, no orders)):
  - WTI x0.15 -> x0.20  RE-BOOST 3 consecutive positive blocks (+9.12% v63,
    +20.14%, +2.72%); fires the v60 watch "WTI re-boost x0.20 on 2 cons pos"
    (pending v64 confirmed).
  - SOX x0.25 -> x0.20  3rd consecutive negative (-7.42%/-0.79%/-15.55%),
    last block -15.55% < -10%; fires the v60 watch "SOX x0.20 on 3rd cons neg
    or <-10%".
  - 000688.SH x0.75 -> x0.25  3rd consecutive negative (-1.10%/-3.01%/
    -8.10%), last block -8.10% < -8%; fires the v60 watch "000688 x0.25 on
    4th cons neg or <-8%".
  - N225 x0.50 -> x0.60  2 consecutive positive (+4.67%/+2.08%); fires the
    v63 watch "N225 x0.60 on 1 more pos".
  Kept: SPX x1.00 (1st mild neg -1.05% after +0.36%; x0.85 on 2nd cons neg or
  <-8%), XAU x1.00 (1st pos +2.28% after -3.27%; x0.85 on 2nd cons neg or
  <-5%), SX5E x0.70 (1st pos +2.87% after 3 negs; x0.50 on 2nd cons neg or
  <-8%), NDX x1.00 (1st pos +0.45% after -1.79%; x0.45 on 2nd cons neg or
  <-6%), COPPER x0.55 (1st pos +3.84% after -0.02%; x0.40 on 4th cons neg or
  <-8%, re-boost x0.85 on 2 cons pos), US10Y x0.30 (1st pos +0.52% after
  -8.01%; x0.20 on 3rd cons neg, x0.35 on 2 cons pos).
  Frozen stale names unchanged: 000300.SH/HSI/BTC/ETH/CN10Y (~39% book
  neutral rank, pnl 0; multipliers held x0.106/x0.106/x0.016/x0.080/x0.074
  via caps).
  Regime: VIX 13.33 (+48% over 10d, breaking up from ~9.0 floor, still below
  the >14 defensive-re-tilt watch); risk rotation - SOX/000688 (semis/China
  tech) crashed while WTI/COPPER/N225/SX5E/XAU led; USD weakened (EURUSD
  +1.62%, DXY -1.50%). Momentum factor ranked SOX high (rebound risk) but the
  defensive-multiplier cut dominates. FEEDBACK TO SCREENER: 0.60 momentum /
  0.40 dn_mkt_beta held up (defensive floor + WTI/N225 re-boosts); momentum
  leaders rotated from tech to commodities - cross-sectional breadth intact.
"""
"""v63 (2035-06-11):
v62 + three risk-adjustment fires from the 2035-05-28..06-11 block (proposal
on 05-28 EXECUTED, cost = 05-25 closes; account 1223573.54 -> 1216697.17,
-0.56% block, Sharpe -3.34 DD 1.09%; SCREENER ensemble unchanged 0.60/0.40 =
vol_adj_mom_accel_20x60 dir=+1 PRIMARY w=0.60, dn_mkt_beta_60d dir=+1 w=0.40,
loaded live from factors/factor_ensemble.json; root+factors synced
byte-identical; first live block after 4 safety-advance cycles 03-05..05-28):
  - 000688.SH x0.60 -> x0.75  4th consecutive positive (+6.22%/+5.52%/
    +0.99%/+0.43%); fires the v60 watch "000688 x0.75 on 1 more pos".
  - NDX x0.60 -> x1.00  RE-BOOST 2 consecutive positive (+5.36%/+2.01%);
    fires the v60 watch "NDX re-boost x1.00 on 2 cons pos".
  - COPPER x0.70 -> x0.55  3rd consecutive negative (-3.53%/-4.68%/-3.56%);
    fires the v60 watch "COPPER x0.55 on 3rd cons neg or <-8%".
  Kept: SPX x1.00 (2nd cons pos +5.89%/+2.01%; x0.85 on 2nd cons neg or
  <-8%), XAU x1.00 (1st neg -3.39% after 4-pos run; x0.85 on 2nd cons neg or
  <-5%), SX5E x0.70 (1st neg -5.79% LARGE single after 2-pos run; x0.50 on
  2nd cons neg or <-8%), SOX x0.25 (1st pos +8.65% after 2 negs; re-boost
  x0.35 on 2 cons pos), WTI x0.15 (1st pos +9.12% after -11.38% cut;
  re-boost x0.20 on 2 cons pos), N225 x0.50 (1st neg -1.26% after 3-pos run;
  x0.60 on 1 more pos, x0.25 on 4th cons neg or <-6%), US10Y x0.30 (2nd cons
  neg -0.46%/-4.35%; x0.20 on 3rd cons neg, x0.35 on 2 cons pos).
  Frozen stale names unchanged: 000300.SH/HSI/BTC/ETH/CN10Y (~39% book
  neutral rank, pnl 0; BTC/ETH/CN10Y multipliers held x0.15/x0.75/x0.70).
  Block: defensive re-tilt HURT - XAU -3.39% (w~0.121 cap) and US10Y -4.35%
  (w~0.038) both negative while momentum names led (WTI +9.12% w~0.026,
  SOX +8.65% w~0.022, SPX +2.01% w~0.099, NDX +2.01% w~0.081); SX5E -5.79%
  (w~0.053 MAIN DRAG ~-0.31%) and COPPER -3.56% (w~0.114 ~-0.41%) dragged.
  Ensemble 0.60/0.40 defensive floor (dn_mkt_beta 0.40) underperformed this
  block - momentum names were the winners. FEEDBACK TO SCREENER: defensive
  re-tilt not validated in current regime; consider trimming dn_mkt_beta
  weight back toward 0.30-0.35 (momentum leaders SPX/NDX/SOX/WTI resilient).
  VIX watch unchanged: break >14 => defensive re-tilt.
"""


"""v56 (2034-09-18):
v55 + three risk-adjustment fires from the 2034-09-04..09-18 block (proposal
on 09-04 EXECUTED, cost = 09-03 closes; account ~1159308 -> 1163713.34,
+0.38% block, Sharpe 2.83 DD 0.46%; SCREENER 2034-09-04 RE-TILT ensemble
0.65/0.35 = vol_adj_mom_accel_20x60 dir=+1 PRIMARY w=0.65, dn_mkt_beta_60d
dir=+1 w=0.35, loaded live from factors/factor_ensemble.json; root+factors
synced byte-identical md5 01e704e9):
  - N225 x0.50 -> x0.40  2nd consecutive negative block (-6.50% 03-06..03-20,
    then -1.87% 09-04..09-18); fires the v53 watch "N225 x0.40 on 2nd cons
    neg or another large <-6%".
  - SX5E x0.70 -> x0.60  2nd consecutive negative block (-4.96% 03-20, then
    -2.48%); fires the v53 watch "SX5E x0.60 on 2nd cons neg or large <-8%".
  - 000688.SH x0.70 -> x0.45  2nd consecutive negative block (-3.00% 03-20,
    then -0.37%); fires the v53 watch "000688 x0.45 on 2nd cons neg or
    <-8%" (standing pattern: mild 2nd cons neg fires, mirror v46 -0.99%).
  Kept: XAU x1.00 (2nd cons pos +6.95% w~0.084; x0.85 on 2nd cons neg or
  <-5%), COPPER cap 0.12 (1st neg -5.39% w~0.077, largest drag ~-0.42%;
  x0.85 on 2nd cons neg or <-8%), US10Y x0.30 (1st neg -4.59% after
  +1.81%; x0.25 on 2nd cons neg), SOX x0.30 (1st neg -4.94% after 2-pos
  run; x0.25 on 2nd cons neg or <-10%), NDX x0.60 (1st pos +5.39% after
  mild neg; re-boost x1.00 on 2 cons pos), WTI x0.20 (3rd stable pos
  +5.76%; re-boost x0.25 on 1 more pos), SPX x0.75 (1st pos +4.67% after
  -1.46%; x0.85 on 2 cons pos, x0.65 on 2nd cons neg or <-8%).
  Frozen stale names unchanged: 000300.SH/HSI/BTC/ETH/CN10Y (~33% book
  neutral rank, pnl 0; HSI/000300/CN10Y/BTC/ETH identical frozen mktvals).
  Momentum factor mixed: SOX/COPPER stalled (-4.94%/-5.39%) while NDX/SPX/
  XAU led — early mean-reversion flag from screener partially materialized.
  Ensemble: 0.65/0.35 (SCREENER 2034-09-04 re-tilt) - loaded live.
"""

"""v57 (2034-10-02):
v56 + seven risk-adjustment fires from the 2034-09-18..10-02 block (proposal
on 09-18 PERSISTED/SKIPPED by deterministic gate - holdings unchanged at the
09-04 target; account 1163713.34 -> 1165658.00, +0.17% block, Sharpe 0.66 DD
0.79%; SCREENER ensemble unchanged 0.65/0.35 = vol_adj_mom_accel_20x60 dir=+1
PRIMARY w=0.65, dn_mkt_beta_60d dir=+1 w=0.35, loaded live from
factors/factor_ensemble.json; root+factors synced byte-identical):
  - SOX x0.30 -> x0.25  2nd consecutive negative (-4.94%, then -18.85% LARGE
    <-10%); fires the v56 watch "SOX x0.25 on 2nd cons neg or <-10%".
  - COPPER (cap) -> x0.85  2nd consecutive negative (-5.39%, then -8.05% at
    the ~-8% threshold); fires the v56 watch "COPPER x0.85 on 2nd cons neg
    or <-8%". First explicit COPPER multiplier entry.
  - N225 x0.40 -> x0.30  3rd consecutive negative (-6.50%/-1.87%/-0.26%);
    fires the v56 watch "N225 x0.30 on 3rd cons neg".
  - 000688.SH x0.45 -> x0.30  3rd consecutive negative (-3.00%/-0.37%/
    -1.07%); fires the v56 watch "000688 x0.30 on 3rd cons neg".
  - US10Y x0.30 -> x0.25  2nd consecutive negative (-4.59%/-5.45%); fires
    the v56 watch "US10Y x0.25 on 2nd cons neg".
  - WTI x0.20 -> x0.25  RE-BOOST 2 consecutive positive (+5.76%/+20.16%);
    fires the v56 watch "WTI x0.25 on 1 more cons pos".
  - NDX x0.60 -> x1.00  RE-BOOST 2 consecutive positive (+5.39%/+4.60%);
    fires the v56 watch "NDX x1.00 on 2 cons pos".
  - SPX x0.75 -> x0.85  RE-BOOST 2 consecutive positive (+4.67%/+7.26%);
    fires the v56 watch "SPX x0.85 on 2 cons pos".
  Kept: XAU x1.00 (2nd cons pos +5.93%; x0.85 on 2nd cons neg or <-5%),
  SX5E x0.60 (1st pos +3.08% after 2 negs; re-boost x0.70 on 2 cons pos,
  x0.50 on 2nd cons neg).
  Frozen stale names unchanged: 000300.SH/HSI/BTC/ETH/CN10Y (~33% book
  neutral rank, pnl 0).
  Block: risk-on rotation continued - WTI +20.16% / SPX +7.26% / XAU +5.93%
  / NDX +4.60% / SX5E +3.08% led; SOX -18.85% (MAIN DRAG, contained by
  x0.30 mult) and COPPER -8.05% dragged; US10Y -5.45% on rate creep. The
  screener's mean-reversion flag fully materialized (SOX/COPPER momentum
  leaders reversing hard, defensive/quality leading).
  Ensemble: 0.65/0.35 (SCREENER 2034-09-04 re-tilt) - loaded live.
"""

"""v55 (2034-09-04):
v54 + SCREENER RE-TILT ensemble (weights only, no logic change): 0.65/0.35 =
vol_adj_mom_accel_20x60 dir=+1 PRIMARY w=0.65, dn_mkt_beta_60d dir=+1 w=0.35;
rate_beta_cn10y_60d remains EXCLUDED (CN10Y frozen/stale at 1.3437, degenerate
signal; demoted not evicted). Loaded live from factors/factor_ensemble.json
(root + factors synced byte-identical md5 01e704e9). Momentum up-tilt vs
prior 0.60/0.40: 33pp 20d cross-sectional spread (SOX +23.7% vs N225 -9.8%),
VIX pinned at 9.0 floor since 08-23 (defensive hedge need reduced), COPPER
+24.8% (75d) trend intact. Retained 0.35 downside-beta overlay for the
elevated-rate backdrop (US10Y 6.55% creeping up). DEFENSIVE_MULT/weights/caps
unchanged from v54/v53.
"""

"""v54 (2034-06-12):
v53 + SCREENER RE-TILT ensemble (weights only, no logic change): 0.55/0.45 =
vol_adj_mom_accel_20x60 dir=+1 PRIMARY w=0.55, dn_mkt_beta_60d dir=+1 w=0.45;
rate_beta_cn10y_60d DROPPED (CN10Y frozen/stale, degenerate signal; lowest
quality). Loaded live from factors/factor_ensemble.json (root + factors synced
byte-identical). High-vol regime (VIX 36.9): dn_mkt_beta floor at 0.45 per
screener; momentum weight capped below strict q-tilt to avoid single-factor
concentration. DEFENSIVE_MULT/weights/caps unchanged from v53.
"""

"""v53 (2034-03-20):
v52 + three risk-adjustment fires from the 2034-03-06..03-20 block (first
live cycle after the 2033-06..2034-03 safety-advance freeze; proposal on
03-06 EXECUTED, cost = 03-03 closes; account 1164607.61 -> 1167314.41,
+0.23% block, Sharpe 1.12 DD 0.90%; SCREENER ensemble 0.50/0.30/0.20
unchanged since 2033-10-31, loaded live from factors/factor_ensemble.json):
  - N225 x0.60 -> x0.50  large single negative -6.50% (03-06..03-20 close,
    MAIN DRAG ~-0.20% at w~0.030 after the -1.05% mild neg in the last
    recorded block); fires the standing chain-cut pattern "another large
    <-6%" (mirror x1.00->x0.85->x0.70->x0.60->x0.50).
  - SOX x0.20 -> x0.30  RE-BOOST 2 consecutive positive blocks (+9.83%
    05-30..06-13, +9.95% 03-06..03-20); fires the v51 watch "SOX re-boost
    x0.30 on 2 cons pos".
  - WTI x0.15 -> x0.20  RE-BOOST 2 stable positive blocks (+17.37%
    05-30..06-13, +2.04% 03-06..03-20); fires the v51 watch "WTI re-boost
    x0.20 after 2 stable".
  Kept: XAU x1.00 (pos +3.89% at cap; x0.85 on 2nd cons neg or <-5%),
  COPPER cap 0.12 (pos +3.18% w~0.124; x0.85 on 2nd cons neg or <-8%),
  US10Y x0.30 (pos +1.81% 1st after -0.83%; re-boost x0.35 on 2 cons pos),
  NDX x0.60 (mild neg -0.31% after cut; re-boost x1.00 on 2 cons pos),
  SPX x0.75 (mild neg -1.46% after 3-pos run; x0.65 on 2nd cons neg or
  <-8%), SX5E x0.70 (neg -4.96% 1st after +7.96% TOP; x0.60 on 2nd cons
  neg or large <-8%), 000688.SH x0.70 (neg -3.00% 1st after 3-pos run;
  x0.45 on 2nd cons neg or <-8%).
  Frozen stale names unchanged: 000300.SH/HSI/BTC/ETH/CN10Y (~36% book
  neutral rank, pnl 0).
  Ensemble: 0.50/0.30/0.20 (SCREENER 2033-10-31 re-tilt) - loaded live
  from factors/factor_ensemble.json.
"""

"""v52 (2033-10-31): v51 + SCREENER ensemble re-tilt (weights only):
0.50/0.30/0.20 = vol_adj_mom_accel_20x60 dir=+1 PRIMARY w=0.50,
dn_mkt_beta_60d dir=+1 w=0.30, rate_beta_cn10y_60d dir=-1 w=0.20; loaded
live from factors/factor_ensemble.json (root + factors byte-identical).
No logic change; DEFENSIVE_MULT/weights/caps unchanged.
"""

"""v51 (2033-06-13) / v50 (2033-05-16) / v49 (2033-04-04) / v48-v37-lite:
See memory.txt for full historical entries. All prior versions used the same
core logic (cross-sectional ensemble ranks -> defensive-multiplier base ->
inverse-vol tilt -> capped normalize) with per-cycle risk-adjustment fires on
DEFENSIVE_MULT (chain-cut on consecutive negatives, re-boost on consecutive
positives). Frozen stale names 000300.SH/HSI/BTC/ETH/CN10Y have carried
neutral rank since ~2028. Ensembles historically 0.40/0.35/0.25 -> 0.50/0.30/
0.20 -> 0.55/0.45 -> 0.60/0.40 -> 0.65/0.35 (current).
"""
from math import isfinite
import json
from pathlib import Path

import pandas as pd
from alphacrafter.sim.utils import (
    get_account_dict,
    get_stock_daily_data,
    get_index_daily_data,
    rebalance_to_weights,
    register_hook,
)

N_ASSETS = 15
CAP = 0.12          # per-asset weight cap (v10: 0.13 -> 0.12; v11 kept)
FLOOR = 0.5 / N_ASSETS
SPREAD = 0.06       # max score-driven spread above floor before vol tilt (v8: 0.08 -> 0.06)
MIN_OBS = 40        # min obs for 60d beta factors
CORR_WIN = 20       # vol-price corr window
CORR_MIN = 10       # min obs for corr
VOL_EXP = 0.6       # inverse-vol exponent (v5: 0.5 -> 0.6)
STALE_N = 5         # consecutive identical closes => stale quote
# v27 defensive multipliers (fired 2029-04-09 after the 03-26..04-09 block +1.35%):
#   BTC x0.30 -> x0.25 (7th cons neg incl -6.50%), US10Y x1.25 -> x1.00 (2 cons neg
#   -3.86%/-2.01%), CN10Y x0.60 -> x0.70 (2 cons pos +2.46%/+17.90% mean-reversion)
# Kept: XAU x1.00 (+5.29% pos), SOX x0.45 (+10.80% pos), NDX x0.75 (single -4.54%),
#   WTI x0.80 (mild -0.48% after +28.26%), SPX x0.85 (single -2.01%), 000688.SH x0.70
#   (single +0.81%), SX5E x1.00 (single -2.86%), ETH x0.75 (frozen)
# Prior v26 (2029-03-26): BTC x0.35->x0.30, SOX x0.55->x0.45, SPX x0.65->x0.85,
#   WTI x0.65->x0.80, 000688 x0.85->x0.70
# Prior v25 (2029-02-12): XAU x0.85->x1.00, SX5E x0.85->x1.00, NDX x0.65->x0.75,
#   BTC x0.40->x0.35, CN10Y x0.70->x0.60
# Prior v24 (2028-11-06). Prior v23 (2028-10-23):
#   XAU x1.00 -> x0.85 (6th consecutive neg), BTC x0.55 -> x0.45 (2nd cons neg),
#   SPX x1.00 -> x0.65 (3rd consecutive neg)
# Prior v22 (2028-10-09): SPX x1.00 -> x0.85 (2nd cons neg), WTI x0.65 -> x0.80
#   (re-boost after 2 stable blocks)
# Prior v20 (2028-09-11): XAU x1.25 -> x1.00 (3 cons neg), WTI x0.80 -> x0.65
#   (2 consecutive large air-pockets)
# Prior v19 (2028-08-28): BTC x0.65 -> x0.55, CN10Y x1.00 -> x0.85,
#   US10Y x1.00 -> x1.25
# Prior v17 (2028-06-05): XAU x1.00 -> x1.25, US10Y x1.25 -> x1.00,
#   BTC x0.75 -> x0.65, SX5E x1.00 -> x0.85
DEFENSIVE_MULT = {
    "XAU": 1.00, "US10Y": 0.30, "CN10Y": 0.70,   # safe havens (v60: XAU kept x1.00 4th cons pos +0.53%, x0.85 on 2nd cons neg or <-5%; US10Y kept x0.30 1st neg -0.46% after 2 pos, x0.35 on 2 cons pos, x0.20 on 3rd cons neg; CN10Y kept x0.70 frozen stale)
    "SOX": 0.20, "NDX": 1.00, "ETH": 0.75, "WTI": 0.20, "BTC": 0.15, "N225": 0.60,  # high-beta (v64: SOX x0.25->x0.20 3rd cons neg -7.42%/-0.79%/-15.55% <-10%, x0.15 on 4th cons neg or <-15%, re-boost x0.30 on 2 cons pos; NDX kept x1.00 1 pos +0.45% after -1.79% neg, x0.45 on 2nd cons neg or <-6%; WTI x0.15->x0.20 RE-BOOST 3 cons pos +9.12%/+20.14%/+2.72%, x0.10 on 2nd cons neg or <-8%, x0.25 on 2 more pos; N225 x0.50->x0.60 2 cons pos +4.67%/+2.08%, x0.70 on 1 more pos, x0.25 on 4th cons neg or <-6%; BTC/ETH frozen stale)
    "SX5E": 0.70, "SPX": 1.00, "000688.SH": 0.25, "COPPER": 0.55,  # v60: SX5E x0.60->x0.70 2nd cons pos +1.06%/+1.03%, x0.80 on 1 more pos, x0.50 on 2nd cons neg; SPX kept x1.00 1st pos +5.89% after -3.27%, x0.85 on 2nd cons neg or <-8%; 000688 x0.75->x0.25 3rd cons neg -1.10%/-3.01%/-8.10% <-8% (fires v60 watch), x0.15 on 4th cons neg or <-12%, re-boost x0.45 on 2 cons pos; COPPER kept x0.55 1st pos +3.84% after -0.02% flat, x0.40 on 4th cons neg or <-8%, re-boost x0.85 on 2 cons pos
}


def stock(a, n=170):
    try:
        return get_stock_daily_data(a, days=n)
    except Exception:
        return None


def index(a, n=170):
    try:
        return get_index_daily_data(a, days=n)
    except Exception:
        return None


def ranks(values, assets):
    valid = sorted((float(v), a) for a, v in values.items()
                   if v is not None and isfinite(float(v)))
    out = {a: 0.5 for a in assets}
    for i, (_, a) in enumerate(valid):
        out[a] = i / max(1, len(valid) - 1)
    return out


def rolling_beta(y, x, win=60, min_obs=MIN_OBS):
    z = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna().tail(win)
    if len(z) < min_obs:
        return None
    var = float(z.x.var())
    if var <= 1e-14:
        return None
    return float(z.y.cov(z.x) / var)


def load_ensemble():
    try:
        raw = json.loads((Path(__file__).parent / "factors" / "factor_ensemble.json").read_text())
        return [(str(it["factor_id"]), float(it["weight"]), int(it.get("direction", 1)))
                for it in raw.get("selected_factors", [])
                if isinstance(it, dict) and it.get("factor_id")]
    except (OSError, ValueError, TypeError):
        return []


def is_stale(closes, a, n=STALE_N):
    c = closes.get(a)
    if c is None or len(c) < n + 2:
        return False
    tail = c.astype(float).iloc[-n - 1:]
    return bool((tail.diff().dropna().abs() < 1e-12).all())


def vol_adj_mom_accel(c, fast=20, slow=60, vol_win=20):
    """(close/close.shift(20)-1 - (close/close.shift(60)-1)) / rolling_std(ret,20).
    Positive = recently accelerating per unit risk; last valid value."""
    r = c.pct_change()
    mom_fast = c / c.shift(fast) - 1.0
    mom_slow = c / c.shift(slow) - 1.0
    vol = r.rolling(vol_win).std()
    z = (mom_fast - mom_slow) / vol
    z = z.dropna()
    if len(z) < 1:
        return None
    val = float(z.iloc[-1])
    return val if isfinite(val) else None



def compute_factor_values(assets, stale, closes, panel, eurusd_ret, cn10y_ret, vol):
    """Raw cross-sectional factor values for the factor library.
    Stale assets get no entries -> neutral rank 0.5 downstream."""
    mkt = panel.mean(axis=1)
    dn_x = mkt.clip(upper=0.0)
    vals = {
        "vol_price_corr_20": {},
        "eurusd_beta_60d": {},
        "rate_beta_cn10y_60d": {},
        "dn_mkt_beta_60d": {},
        "vol_adj_mom_accel_20x60": {},
    }
    for a in assets:
        if a in stale:
            continue
        c = closes.get(a)
        v = vol.get(a)
        if c is not None and v is not None:
            z = pd.concat([c.pct_change().rename("r"), v.astype(float).rename("v")],
                          axis=1).dropna().tail(CORR_WIN)
            if len(z) >= CORR_MIN:
                _c = z.r.corr(z.v)
                if _c is not None and isfinite(float(_c)):
                    vals["vol_price_corr_20"][a] = float(_c)
        c2 = closes.get(a)
        if c2 is not None:
            ma = vol_adj_mom_accel(c2)
            if ma is not None:
                vals["vol_adj_mom_accel_20x60"][a] = ma
        y = panel[a]
        vals["dn_mkt_beta_60d"][a] = rolling_beta(y, dn_x)
        if eurusd_ret is not None:
            vals["eurusd_beta_60d"][a] = rolling_beta(y, eurusd_ret)
        if cn10y_ret is not None:
            vals["rate_beta_cn10y_60d"][a] = rolling_beta(y, cn10y_ret)
    return vals


def capped_normalize(w, cap=CAP):
    """Normalize weights to sum 1, then water-fill cap at `cap`."""
    w = {a: max(0.0, float(x)) for a, x in w.items()}
    total = sum(w.values())
    if total <= 0:
        return {a: 1.0 / len(w) for a in w}
    w = {a: x / total for a, x in w.items()}
    for _ in range(200):
        excess = sum(max(0.0, x - cap) for x in w.values())
        if excess < 1e-12:
            break
        clipped = {a: min(cap, x) for a, x in w.items()}
        room = [a for a, x in clipped.items() if x < cap - 1e-12]
        if not room:
            w = clipped
            break
        room_total = sum(w[a] for a in room)
        if room_total <= 0:
            w = clipped
            break
        for a in room:
            clipped[a] += excess * (w[a] / room_total)
        w = clipped
    total = sum(w.values())
    if abs(total - 1.0) > 1e-9 and total > 0:
        w = {a: x / total for a, x in w.items()}
    return w


def compute_target(assets):
    """Return (weights, forecast_returns, factor_ids, info)."""
    frames = {a: stock(a) for a in assets}
    closes = {a: (f.close.astype(float) if f is not None and "close" in f else None)
              for a, f in frames.items()}
    vol = {a: (f.volume.astype(float) if f is not None and "volume" in f else None)
           for a, f in frames.items()}

    # v5 stale-quote guard
    stale = {a for a in assets if is_stale(closes, a)}
    live = [a for a in assets if a not in stale]

    usable = [c.pct_change().rename(a) for a, c in closes.items()
              if a in live and c is not None and len(c) >= 30]
    panel = (pd.concat(usable, axis=1, join="inner").dropna().tail(130)
             if len(usable) >= 8 else pd.DataFrame())
    if len(panel) < 50:
        return ({a: 1.0 / len(assets) for a in assets},
                {a: 0.0 for a in assets}, [], {"fallback": "short_panel",
                                               "stale": sorted(stale)})

    ef = index("EURUSD")
    eurusd_ret = (ef.close.astype(float).pct_change()
                  if ef is not None and "close" in ef else None)
    cn10y_ret = (closes["CN10Y"].pct_change() if closes.get("CN10Y") is not None else None)

    ens = load_ensemble()
    factor_ids = [fid for fid, _, _ in ens]
    if not factor_ids:
        # defensive fallback: slight safe-haven tilt, zero forecast
        w = {a: 1.0 / len(assets) for a in assets}
        for a in ("XAU", "US10Y", "CN10Y"):
            if a in w:
                w[a] += 0.02
        w = capped_normalize(w, cap=0.16)
        return (w, {a: 0.0 for a in assets}, [], {"fallback": "no_ensemble",
                                                  "stale": sorted(stale)})

    vals = compute_factor_values(assets, stale, closes, panel, eurusd_ret, cn10y_ret, vol)

    # composite score = sum(weight * direction * rank)
    score = {a: 0.0 for a in assets}
    for fid, wgt, drc in ens:
        r = ranks(vals.get(fid, {}), assets)
        for a in assets:
            score[a] += wgt * drc * r[a]

    s_vals = [score[a] for a in assets]
    lo, hi = min(s_vals), max(s_vals)
    vol20 = {a: max(float(panel[a].tail(20).std()), 0.004) for a in live}
    if vol20:
        med_vol = sorted(vol20.values())[len(vol20) // 2]
    else:
        med_vol = 0.01
    for a in stale:
        vol20[a] = med_vol  # frozen quotes: neutral average risk

    # base weight: floor + score-driven spread (v10: defensive multiplier applied
    # here, before the inverse-vol tilt; weights re-normalized downstream)
    base = {a: FLOOR + SPREAD * ((score[a] - lo) / (hi - lo + 1e-12)) for a in assets}
    for a, m in DEFENSIVE_MULT.items():
        if a in base:
            base[a] *= m
    tilted = {a: base[a] / (vol20[a] ** VOL_EXP) for a in assets}
    weights = capped_normalize(tilted)

    # forecast returns (10-day proxy): z-scored score * typical 10d cross-sectional vol
    mean_s = sum(s_vals) / len(s_vals)
    std_s = (sum((v - mean_s) ** 2 for v in s_vals) / len(s_vals)) ** 0.5 or 1e-12
    scale = float(panel.tail(60).std(axis=1, ddof=0).median()) * (10.0 ** 0.5) or 0.01
    forecast_returns = {}
    for a in assets:
        z = max(-2.5, min(2.5, (score[a] - mean_s) / std_s))
        forecast_returns[a] = z * scale
    return weights, forecast_returns, factor_ids[:10], {
        "scores": {a: round(float(score[a]), 4) for a in assets},
        "scale": round(float(scale), 5),
        "vol20": {a: round(float(v), 4) for a, v in vol20.items()},
        "stale": sorted(stale),
    }


@register_hook
def strategy_hook():
    assets = list(get_account_dict()["watch_list"])
    weights, forecast_returns, factor_ids, info = compute_target(assets)
    rebalance_to_weights(
        weights,
        forecast_returns=forecast_returns,
        factor_ids=factor_ids,
        horizon_days=10,
    )


if __name__ == "__main__":
    from alphacrafter.sim.utils import get_account_dict
    _assets = list(get_account_dict()["watch_list"])
    _w, _f, _ids, _info = compute_target(_assets)
    print("factor_ids:", _ids)
    print("info:", json.dumps(_info, indent=1)[:1500])
    print("weights sum:", round(sum(_w.values()), 6))
    for _a in _assets:
        print(f"  {_a:10s} w={_w[_a]:.4f} f={_f[_a]:+.5f}")
