"""Trader strategy v24 on top of v23 (fired 2028-11-06 after the 10-23..11-06
block -2.95%, DD ~2.5%, broad risk-off: 000688 -10.9%, N225 -7.1%, WTI -12.8%,
BTC -8.0%, CN10Y -7.9%, NDX -6.5%, SOX -7.0%, SPX -1.9%; winners XAU +4.4%,
COPPER +4.9%, SX5E +3.7%, US10Y +0.7%):
  v24 changes (evidence from the 10-23..11-06 block; v23 proposal was skipped
  by the deterministic gate -> actual holdings were the v22 target, so all
  multi changes below apply at the NEXT rebalance):
    1. CN10Y x0.85 -> x0.70 (PRE-AGREED WATCH FIRED: 2nd consecutive large
       negative -10.42% (10-09..10-23) / -7.91% (10-23..11-06); v23 watch said
       cut to x0.70 on any 2nd consecutive negative or another large print;
       largest two consecutive block prints in book history for CN10Y).
    2. 000688 x1.00 -> x0.85 (NEW WATCH-CUT FIRED: 2nd consecutive negative
       -4.18% / -10.88% with the 2nd a large air-pocket; mirrors the v22 SPX /
       v17 SX5E 2-consecutive precedent).
    3. WTI x0.80 -> x0.65 (LARGE AIR-POCKET AFTER RE-BOOST: -12.79% this block
       immediately after the +9.89% re-boost payoff; only 1 stable block before
       the large print, so the re-boost premise failed; revert to the v20 cut
       depth; drag was the 2nd-largest this block ~-0.76%).
    4. BTC x0.45 -> x0.40 (3rd consecutive negative -1.20%/-3.95%/-7.96% with a
       large print; BTC remains the persistent whipsaw drag, deepen again).
    5. NDX x0.75 -> x0.65 (2nd consecutive negative -3.25%/-6.53%; mirrors the
       SPX 2-consecutive precedent).
    6. SOX x0.65 -> x0.55 (2nd consecutive negative -3.50%/-7.02%; repeat-
       offender tech cut deepened, v11 precedent chain).
  Kept: XAU x0.85 (+4.36% - the 10-23 cut was well-timed at the bottom of a
  6-consecutive-negative streak; re-boost to x1.00 only after 2 consecutive
  positive blocks), US10Y x1.25 (+0.68%, safe haven functioning), SPX x0.65
  (4th consecutive but mild -2.66%/-1.00%/-1.91%; drag ~-0.11%; deepen to x0.55
  only on a large print < -5% or 2 more consecutive negatives), SX5E x0.85
  (+3.73%, watch cleared), ETH x0.75 (frozen), CAP 0.12, SPREAD 0.06, vol exp
  0.6, stale-guard.
  New watches: N225 (single large -7.08% after +0.95% -> cut to x0.85 on a 2nd
  consecutive large air-pocket), COPPER (single +4.85% after -5.24% -> watch
  cleared, cut only on 2nd consecutive large), XAU re-boost trigger (2
  consecutive positives), SPX deepen trigger (large print or 2 more
  consecutive), BTC deepen trigger (4th consecutive negative -> x0.35).
  Screener re-tilt (factors/factor_ensemble.json, updated 2028-12-04):
    vol_adj_mom_accel_20x60  w=0.46 -> 0.40  dir=+1  PRIMARY (trimmed further:
      momentum acceleration unreliable while BTC -49% 120d and CN10Y -34% 120d)
    dn_mkt_beta_60d          w=0.34 -> 0.36  dir=+1  (boosted: low-downside-beta
      remains the drawdown anchor through the CN10Y/WTI/equity risk-off)
    rate_beta_cn10y_60d      w=0.20 -> 0.24  dir=-1  (boosted: CN10Y crashed
      -27.7% 60d, rate-hedge tilt pays while CN10Y keeps sliding)

Trader strategy v23 on top of v22 (fired 2028-10-23 after the 10-09..10-23
block -1.38%, DD 1.52%):
  v23 changes (evidence from the 10-09..10-23 block):
    1. XAU x1.00 -> x0.85 (PRE-AGREED TRIGGER FIRED: 6th consecutive negative
       10d block -0.57%/-1.64%/-5.44%/-2.93%/-1.49%/-3.39% (07-31..10-23),
       last print -3.39% is a large print; the v22 watch said cut to x0.85 on
       a 6th consecutive any-size or large print).
    2. BTC x0.55 -> x0.45 (PRE-AGREED WATCH FIRED: 2nd consecutive negative
       -1.20% (09-25..10-09) / -3.95% (10-09..10-23) after the +16.04% rebound;
       v22 watch said deepen to x0.45 on 2nd consecutive negative).
    3. SPX x0.85 -> x0.65 (PRE-AGREED WATCH FIRED: 3rd consecutive negative
       -3.52% (09-11..09-25) / -2.66% (09-25..10-09) / -1.00% (10-09..10-23);
       v22 watch said deepen x0.85->x0.65 on 3rd consecutive).
  Kept: US10Y x1.25 (+2.47%, safe haven functioning), CN10Y x0.85 (watch: huge
  single air-pocket -10.42% after +0.48% - LARGEST block print in book history;
  cut to x0.70 on any 2nd consecutive negative or another large print), SOX
  x0.65, NDX/ETH x0.75, WTI x0.80 (+9.89% - re-boost paying off; keep, needs 2
  more stable blocks before x1.00 re-boost), SX5E x0.85 (single -3.29% after
  +3.55% -> watch), CAP 0.12, SPREAD 0.06, vol exp 0.6, stale-guard.
  New watches: CN10Y (see above), SX5E (2nd consecutive or large print ->
  x0.70), COPPER (single -5.24% after +2.83% -> watch), 000688 (single -4.18%
  after +2.82% -> watch).

Trader strategy v22 on top of v21 (fired 2028-10-09 after the 09-25..10-09
block +0.55%, DD 1.36%):
  v22 changes (evidence from the 09-25..10-09 block):
    1. SPX x1.00 -> x0.85 (WATCH-CUT FIRED: 2nd consecutive negative 10d block
       -3.52% (09-11..09-25) / -2.66% (09-25..10-09) after +9.49%; mirrors the
       v17 SX5E precedent (2 consecutive negatives -> x0.85); US large-cap
       cooling per Screener (SPX flat 40d, pulled back from 6621)).
    2. WTI x0.65 -> x0.80 (RE-BOOST FIRED: air-pocket pattern abated, 2 stable
       blocks -2.01% (09-11..09-25) / +0.94% (09-25..10-09) after the -13.85%
       large print; v16 precedent (2 stable blocks -> re-boost to x0.80)).
  Kept: XAU x1.00 (5th consecutive negative -1.49% but mild, de-boost working
  (drag ~-0.12%); cut to x0.85 only on a 6th consecutive negative (any size)
  or a large print), US10Y x1.25 (+1.46%, safe haven still functioning),
  CN10Y x0.85 (+0.48%, watch cleared), SOX x0.65, NDX/ETH x0.75, BTC x0.55
  (-1.20% after +16.04% rebound, re-boost still needs 2 stable blocks),
  SX5E x0.85, CAP 0.12, SPREAD 0.06, vol exp 0.6, stale-guard.
  Watches cleared: COPPER (+2.83%, 2-moderate streak broken -> no cut),
  CN10Y (+0.48%, no 2nd consecutive), 000688 (+2.82%, 2nd positive after the
  -6.46% air-pocket -> watch cleared).

Trader strategy v21 on top of v20: SCREENER RE-TILT 2028-09-25
(quality-IC tilt for the 09-25..10-09 block; high-VIX risk-off regime).
  v21 changes (weights only, loader reads factors/factor_ensemble.json live):
    vol_adj_mom_accel_20x60  w=0.52 -> 0.50  dir=+1  PRIMARY (trimmed: pure
      quality tilt would over-concentrate in a BTC/commodity whipsaw regime)
    dn_mkt_beta_60d          w=0.28 -> 0.30  dir=+1  (boosted: low-downside-beta
      is the drawdown anchor while VIX 40.7 and risk-off)
    rate_beta_cn10y_60d      w=0.20 (unchanged) dir=-1
  Kept v20 DEFENSIVE_MULT: XAU x1.00, US10Y x1.25, CN10Y x0.85, SOX x0.65,
  NDX/ETH x0.75, WTI x0.65, BTC x0.55, SX5E x0.85, CAP 0.12, SPREAD 0.06,
  vol exp 0.6, stale-guard, full-investment 15-asset cross-section, 10-day
  cadence.
  Watches carried into this block: COPPER (2 consecutive moderate -4.84%/
  -4.75% -> cut to x0.85 on 3rd consecutive or a large print), XAU (4th
  consecutive negative, de-boost working -> cut to x0.85 only on a 5th
  consecutive with large print), SPX (reversal -3.52% after +9.49% -> any
  change needs a 2nd consecutive negative), CN10Y (flipped -2.42% after
  +2.28% -> cut only on 2nd consecutive negative), BTC (single positive after
  2 large air-pockets -> re-boost only after 2 stable blocks).

Trader strategy v20 on top of v19: XAU SAFE-HAVEN DE-BOOST + WTI DEEPER
CUT fired 2028-09-11 after the 08-28..09-11 block (-1.16%, DD 1.63%).

v20 changes (evidence from the 08-28..09-11 block, mirror of the v18/v15
contingency framework):
  1. XAU x1.25 -> x1.00 (SAFE-HAVEN DE-BOOST FIRED: 3 consecutive negative 10d
     blocks -0.57% (07-31..08-14) / -1.64% (08-14..08-28) / -5.44% (08-28..
     09-11) on ~11.5% wt, MAIN DRAG ~-0.63% of the block; the v18 US10Y
     de-boost precedent (2 consecutive negatives -> x1.00) is now exceeded
     with a large third print, so the boost is removed).
  2. WTI x0.80 -> x0.65 (DEEPER CUT FIRED: 2 consecutive large air-pockets
     -6.97% (08-14..08-28) then -13.85% (08-28..09-11) on ~3.2% wt, ~-0.45%
     drag; matches the v15 WTI escalation precedent (2 consecutive large
     air-pockets -> x0.80->x0.65)).
  Kept: US10Y x1.25 (3 consecutive positives +0.99%/+0.89%/+8.66% then mild
  -0.14%, still functioning safe haven), CN10Y x0.85 (turned positive +2.28%
  after 3 negatives, keep the reduced boost), SOX x0.65, NDX/ETH x0.75, BTC
  x0.55 (positive +2.22% this block), SX5E x0.85, CAP 0.12, SPREAD 0.06,
  vol exp 0.6, stale-guard, full-investment 15-asset cross-section, 10-day
  cadence.
  000688 watch: single air-pocket -6.46% on ~4.2% wt; cut to x0.85 only on a
  2nd consecutive large air-pocket.
  SX5E watch: single moderate -4.93% on ~5.2% wt; deeper cut only on a 2nd
  consecutive negative.
  COPPER watch: single moderate -4.84% on ~7.2% wt; cut only on a 2nd
  consecutive large air-pocket.

Trader strategy v19 on top of v18: BTC ESCALATION + CN10Y DEEPER CUT
+ US10Y SAFE-HAVEN BOOST fired 2028-08-28 after the 08-14..08-28 block
(+0.08%, DD 0.56%).
  v19 changes: BTC x0.65 -> x0.55 (2nd consecutive large air-pocket
  -10.01%/-21.71%), CN10Y x1.00 -> x0.85 (3 consecutive negatives
  -1.60%/-2.69%/-5.12%), US10Y x1.00 -> x1.25 (3 consecutive positives
  +0.99%/+0.89%/+8.66%).

Trader strategy v18 on top of v17: CN10Y SAFE-HAVEN DE-BOOST fired 2028-08-14
after the 07-31..08-14 block (-0.99%, DD 1.83%).
  v18 changes: CN10Y x1.25 -> x1.00 (CN10Y printed 2 consecutive negative 10d
  blocks -1.60% / -2.69% while US10Y outperformed +0.99% / +0.89%).
  Kept: XAU x1.25, US10Y x1.00, SOX x0.65, NDX/ETH x0.75, WTI x0.80, BTC
  x0.65, SX5E x0.85, CAP 0.12, SPREAD 0.06, vol exp 0.6, stale-guard.

Trader strategy v17 on top of v16: SAFE-HAVEN RE-ROTATION + BTC ESCALATION
+ SX5E WATCH-CUT fired 2028-06-05 after the 05-22..06-05 block (-2.01%, DD 2.01%).
  v17 changes: XAU x1.00 -> x1.25 (2 consecutive positive blocks +4.22%/+0.48%),
  US10Y x1.25 -> x1.00 (2 consecutive negative blocks -3.83%/-5.99%), BTC
  x0.75 -> x0.65 (2nd consecutive large air-pocket), SX5E x1.00 -> x0.85
  (2 consecutive negative blocks -2.10%/-7.36%).

v16 changes (fired 2028-05-08 after the 04-24..05-08 block -0.97%, DD 1.17%):
  XAU x1.25 -> x1.00 (5 consecutive negative blocks 02-28..05-05), WTI
  x0.65 -> x0.80 (air-pocket pattern abated: +4.66%/-3.60%/+5.48% last 3 blocks).

v15 history: PRE-AGREED WTI CONTINGENCY FIRED 2028-03-27 - WTI printed a SECOND
consecutive large air-pocket (-12.4% in 02-28..03-13, -10.5% in 03-13..03-27)
while already at x0.80; deepened the cut x0.80 -> x0.65 (same depth as the SOX
repeat-offender cut).

v14 history: Screener RE-TILT 2028-03-13 dropped
vol_price_corr_20 and promoted vol_adj_mom_accel_20x60 (w=0.50) to PRIMARY;
dn_mkt_beta_60d w=0.36->0.28, rate_beta_cn10y_60d w=0.28->0.22. Code change:
added vol_adj_mom_accel_20x60 computation. Kept v13 defensive multipliers
(BTC x0.75, WTI x0.65, SOX x0.65, NDX/ETH x0.75, XAU/US10Y/CN10Y x1.25).

Ensemble from factors/factor_ensemble.json (quality-IC tilt, re-tilted 2028-12-04):
  vol_adj_mom_accel_20x60  w=0.40  dir=+1  vol-adjusted momentum acceleration -- PRIMARY (trimmed
                                            further: momentum unreliable through the BTC/CN10Y crash)
  dn_mkt_beta_60d          w=0.36  dir=+1  low downside-market-beta (safe-haven anchor, boosted)
  rate_beta_cn10y_60d      w=0.24  dir=-1  low CN10Y-beta tilt (rate-hedge, boosted as CN10Y crashed)
  Loader reads JSON live; root + factors/ synced byte-identical.

Legacy header: v13 (BTC x0.75) on top of v12-TEST: defensive escalation + deeper
SOX cut (pre-agreed contingency fired 2027-10-11 after SOX 3rd air-pocket
+11%/-12.6%/-15.7% despite x0.75) + BTC cut (pre-agreed contingency fired
2028-01-31 after BTC 2nd consecutive large air-pocket: -20.1% in 01-17..01-31,
-17.6% MTD per Screener).

v5 changes (triggered 2026-11-09 after 3 consecutive negative live blocks):
  1. Inverse-vol exponent 0.5 -> 0.6 (stronger vol dampening)
  2. Per-asset cap 0.18 -> 0.15 (lower concentration)
  3. Stale-quote guard: assets with STALE_N consecutive identical closes are
     excluded from factor computation (neutral rank 0.5), dropped from the
     market panel, and assigned cross-sectional median vol20. Prevents frozen
     quotes (HSI/ETH since ~2026-10-14) from distorting betas / inflating
     low-vol weights.

v6 changes (triggered 2027-01-18 after a negative block with risk-on stall):
  SPREAD 0.14 -> 0.10.

v7 changes (triggered 2027-03-01 after the 02-15..03-01 negative block):
  SPREAD 0.10 -> 0.08.

v8 changes (triggered 2027-04-12 after the 4th consecutive negative live block):
  SPREAD 0.08 -> 0.06.

v9 changes (triggered 2027-06-07 after the 05-24..06-07 negative block):
  Per-asset cap 0.15 -> 0.13; defensive multiplier on base weights:
  XAU/US10Y/CN10Y x1.15, SOX/NDX/ETH x0.85.

v10 changes (triggered 2027-06-21 - pre-agreed escalation contingency FIRED):
  Per-asset cap 0.13 -> 0.12; defensive multiplier escalation:
  XAU/US10Y/CN10Y x1.15 -> x1.25, SOX/NDX/ETH x0.85 -> x0.75.

v11 changes (triggered 2027-10-11 - pre-agreed SOX contingency FIRED):
  SOX x0.75 -> x0.65 (deeper high-beta tech cut).

v13 changes (triggered 2028-01-31 - pre-agreed BTC contingency FIRED):
  BTC x1.00 -> x0.75 (deeper crypto cut).

Full-investment long-only 15-asset cross-sectional strategy; non-negative
weights sum to 1 (cash=0). Rebalance cadence 10 trading days (handled by
rebalance_to_weights horizon_days). Bearish views expressed by defensive tilt,
never by cash or shorts.
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
# v24 defensive multipliers (fired 2028-11-06). Prior v23 (2028-10-23):
#   XAU x1.00 -> x0.85 (6th consecutive neg), BTC x0.55 -> x0.45 (2nd cons neg),
#   SPX x0.85 -> x0.65 (3rd consecutive neg)
# Prior v22 (2028-10-09): SPX x1.00 -> x0.85 (2nd cons neg), WTI x0.65 -> x0.80
#   (re-boost after 2 stable blocks)
# Prior v20 (2028-09-11): XAU x1.25 -> x1.00 (3 cons neg), WTI x0.80 -> x0.65
#   (2 consecutive large air-pockets)
# Prior v19 (2028-08-28): BTC x0.65 -> x0.55, CN10Y x1.00 -> x0.85,
#   US10Y x1.00 -> x1.25
# Prior v17 (2028-06-05): XAU x1.00 -> x1.25, US10Y x1.25 -> x1.00,
#   BTC x0.75 -> x0.65, SX5E x1.00 -> x0.85
DEFENSIVE_MULT = {
    "XAU": 0.85, "US10Y": 1.25, "CN10Y": 0.70,   # safe havens (v24: CN10Y cut to x0.70 - 2nd consecutive large -10.42%/-7.91%; XAU kept x0.85 +4.36% well-timed bottom; US10Y kept x1.25 +0.68%)
    "SOX": 0.55, "NDX": 0.65, "ETH": 0.75, "WTI": 0.65, "BTC": 0.40,  # high-beta cuts (v24: SOX x0.65->x0.55 2nd cons -3.50%/-7.02%; NDX x0.75->x0.65 2nd cons -3.25%/-6.53%; WTI x0.80->x0.65 large -12.79% after re-boost; BTC x0.45->x0.40 3rd cons -1.20%/-3.95%/-7.96%)
    "SX5E": 0.85, "SPX": 0.65, "000688.SH": 0.85,  # v24: 000688.SH new x0.85 - 2nd cons -4.18%/-10.88% large air-pocket; SPX kept x0.65 4th cons mild; SX5E kept x0.85 +3.73% watch cleared
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
    """Raw cross-sectional factor values for the 3-factor ensemble.
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
