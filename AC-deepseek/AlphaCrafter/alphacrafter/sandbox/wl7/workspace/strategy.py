"""Trader strategy - 7-factor cross-asset ensemble (2030-04-09 cycle).

Ensemble (factor_ensemble.json, quality_ic_tilt, loaded dynamically):
  downside_vol_ratio_20   0.26  (dir+1, champion - live defensive quality
                                 throughout the entire 2028-30 bear)
  beta_ew_60d             0.20  (dir+1 - cross-sectional beta quality #2,
                                 high-dispersion tape keeps it meaningful)
  rel_mom_20d_skip5       0.18  (dir+1 - trimmed after China-leg whipsaws,
                                 crypto/gold momentum legs remain useful)
  max_ret_20d             0.10  (dir+1 - shares rel_mom exposure, trimmed)
  dxy_beta_cond_60x20     0.10  (dir -1 - USD 104.99 flat/-0.4%: hedge idle,
                                 kept small as bear-tape insurance)
  corr_ew_60              0.09  (dir+1 - low-churn diversifier, ~0.01 corr
                                 keeps comovement signal starved but stable)
  kurt_20d_skip5          0.07  (dir+1 - small tail hedge, VIX 34.8 still
                                 EXTREME - kept)

Weights are loaded dynamically from factor_ensemble.json at runtime.
Long-only, fully invested across the 15 watchlist assets (no cash sleeve).

CRITICAL FIX 2030-04-09: all pd.Series(...) constructions now pass a numpy
array (Series-alignment was silently producing all-NaN close/return data
inside the simulator environment; every decision from 2030-01-15 had empty
factor_ids, zero gross edge and the gate skipped, letting the safety-
advance advance windows without live P&L harvesting).

Regime context (data thru 2030-04-08): VIX 34.8 EXTREME (+11.3%/20d
re-escalating again; 02-12->04-08 peaked 79->cooled to 45.8 on 03-20->
spiked back to 38.7 by 04-01 -> 34.8 now) -> risk-off trigger HIT ->
def_floor 0.12. DXY 104.99 (-0.4%/20d, roughly flat after 103-106 range).
Crypto stable 25-40k band, whipsawing intra-block (BTC -5.0% then +7.6%
then -1.5%); 000300.SH +1.4%/20d recovering; US10Y price -1.0%/20d steady
(after two crash blocks then a bounce, now range-bound; cost-basis plr
-36.2% still structural drag). FROZEN feeds: 000688.SH, SOX, NDX, CN10Y
(~33% dead weight, structural drag unchanged).

Defensive floor XAU-anchored (US10Y cap 10% maintained, screener priority):
- XAU 70% of defensive budget (primary floor); US10Y 10% cap (bond floor
  range-bound, cost-basis deepest structural drag); CN10Y frozen -> 0.
  Global per-asset cap 0.15 trims XAU to 15% max.

Below-MA20 trend-exclusion guard (2030-02-26, EXTENDED per screener):
- Legs below their 20d MA get cross-sectional score rank halved, mitigating
  rel_mom/max_ret whipsaw mean-reversion. Extended to regional equity legs
  (000300.SH, N225, HSI, SX5E) and commodity/crypto (WTI, COPPER, BTC,
  ETH). SPX excluded (primary benchmark reference).

Per-asset cap 0.15; submits one complete target via rebalance_to_weights helper.
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

DEF = ("XAU", "US10Y", "CN10Y")
TREND_GUARD = {"WTI", "COPPER", "BTC", "ETH", "000300.SH", "N225", "HSI", "SX5E"}
CAP = 0.15
MIN_ROWS = 61


def _stock(sym, days=200):
    try:
        return get_stock_daily_data(sym, days=days)
    except Exception:
        return None


def _index(sym, days=200):
    try:
        return get_index_daily_data(sym, days=days)
    except Exception:
        return None


def _series(df, name=None):
    """Build a float Series WITHOUT pandas Series-alignment traps.

    Passing `df["close"]` (a Series) together with `index=` re-aligns and
    silently yields all-NaN under this environment's pandas/dtypes; pass the
    numpy array instead so values are assigned positionally.
    """
    if df is None or "close" not in df or len(df) < MIN_ROWS:
        return None
    vals = df["close"].to_numpy(dtype=float)
    idx = pd.to_datetime(df["date"].to_numpy())
    s = pd.Series(vals, index=idx)
    return s.rename(name) if name else s


def ranks(values, assets):
    """Cross-sectional rank map asset -> [0,1]; missing get 0.5."""
    valid = sorted((float(v), a) for a, v in values.items()
                   if v is not None and isfinite(float(v)))
    out = {a: 0.5 for a in assets}
    n = len(valid)
    if n == 0:
        return out
    for i, (_, a) in enumerate(valid):
        out[a] = i / (n - 1) if n > 1 else 0.5
    return out


def capped_normalize(w, pref):
    w = {a: max(0.0, float(x)) for a, x in w.items()}
    for _ in range(60):
        excess = sum(max(0.0, x - CAP) for x in w.values())
        w = {a: min(CAP, x) for a, x in w.items()}
        room = [a for a, x in w.items() if x < CAP - 1e-12]
        if excess < 1e-12 or not room:
            break
        den = sum(max(0.0, pref.get(a, 0.0)) for a in room)
        for a in room:
            w[a] += excess * (max(0.0, pref.get(a, 0.0)) / den if den else 1.0 / len(room))
    total = sum(w.values())
    return {a: x / total for a, x in w.items()} if total > 0 else {a: 1.0 / len(w) for a in w}


def def_alloc(budget, series_map, assets, scale=0.10):
    """XAU-anchored defensive allocation (US10Y cap 10% maintained).

    XAU 70% of defensive budget (primary floor); US10Y 10% cap (bond floor
    range-bound after crashes; cost-basis still deepest structural drag);
    CN10Y frozen (flat since 2028-01-03) -> 0. Global per-asset cap 0.15
    trims XAU to 15% max.
    """
    alloc = {a: 0.0 for a in assets}
    alloc["XAU"] = 0.70 * budget
    alloc["US10Y"] = 0.10 * budget
    # CN10Y frozen -> 0 (share returns to general pool)
    return alloc


@register_hook
def strategy_hook():
    account = get_account_dict()
    assets = list(account.get("watch_list", []))
    if not assets:
        return

    # ---- aligned daily return panel -------------------------------------
    frames = {a: _stock(a) for a in assets}
    series = {a: _series(f) for a, f in frames.items()}
    usable = {a: s.pct_change().rename(a) for a, s in series.items() if s is not None}
    if len(usable) < 8:
        eq = {a: 1.0 / len(assets) for a in assets}
        rebalance_to_weights(eq, forecast_returns={a: 0.0 for a in assets},
                             factor_ids=[], horizon_days=10)
        return
    R = pd.concat(usable, axis=1, join="inner").dropna().tail(150)
    if len(R) < MIN_ROWS:
        eq = {a: 1.0 / len(assets) for a in assets}
        rebalance_to_weights(eq, forecast_returns={a: 0.0 for a in assets},
                             factor_ids=[], horizon_days=10)
        return

    cp = (1.0 + R).cumprod()
    mkt = R.mean(axis=1)

    # 1) rel_mom_20d_skip5 : 20d momentum ending 5d ago, demeaned cross-sectionally
    mom = cp.shift(5) / cp.shift(25) - 1.0
    rel_mom = mom.sub(mom.median(axis=1), axis=0)

    # 2) beta_ew_60d : rolling 60d beta vs EW market
    mvar = mkt.rolling(60).var()
    beta_ew = R.rolling(60).cov(mkt).div(mvar, axis=0)

    # 3) downside_vol_ratio_20 : -(downside semi-vol / total vol)
    neg = R.clip(upper=0.0)
    semi = (neg ** 2).rolling(20).mean() ** 0.5
    tot = R.rolling(20).std()
    dvr = -(semi / tot)

    # 4) max_ret_20d
    mx = R.rolling(20).max()

    # 5) dxy_beta_cond_60x20 : -beta(asset, DXY, 60) * DXY 20d move (dir applied from ensemble)
    dxy_cond = None
    dfx = _index("DXY")
    if dfx is not None and len(dfx) >= MIN_ROWS:
        dc = _series(dfx)
        if dc is not None:
            dxy_ret = dc.pct_change().reindex(R.index)
            dxy_20 = (dc / dc.shift(20) - 1.0).reindex(R.index)
            if dxy_ret.notna().sum() >= 40 and dxy_20.notna().sum() >= 40:
                dvar = dxy_ret.rolling(60).var()
                bfx = R.rolling(60).cov(dxy_ret).div(dvar, axis=0)
                dxy_cond = -bfx * dxy_20

    # 6) corr_ew_60 : mean pairwise 60d correlation
    corr_parts = []
    for a in R.columns:
        others = [R[a].rolling(60).corr(R[b]) for b in R.columns if b != a]
        corr_parts.append(pd.concat(others, axis=1).mean(axis=1).rename(a))
    corr_ew = pd.concat(corr_parts, axis=1)

    # 7) kurt_20d_skip5 : 20d kurtosis of returns ending 5d ago
    kurt = R.shift(5).rolling(20).kurt()

    factor_values = {
        "rel_mom_20d_skip5": rel_mom,
        "beta_ew_60d": beta_ew,
        "downside_vol_ratio_20": dvr,
        "max_ret_20d": mx,
        "dxy_beta_cond_60x20": dxy_cond,
        "corr_ew_60": corr_ew,
        "kurt_20d_skip5": kurt,
    }

    # ---- ensemble weights -------------------------------------------------
    try:
        ens = json.loads((Path(__file__).parent / "factor_ensemble.json").read_text())
        sel = [(str(it["factor_id"]), float(it["weight"]), int(it.get("direction", 1)))
               for it in ens.get("selected_factors", [])
               if isinstance(it, dict) and it.get("factor_id")]
    except (OSError, ValueError, TypeError):
        sel = []
    if not sel:
        eq = {a: 1.0 / len(assets) for a in assets}
        rebalance_to_weights(eq, forecast_returns={a: 0.0 for a in assets},
                             factor_ids=[], horizon_days=10)
        return

    score = {a: 0.0 for a in assets}
    active = []
    for fid, w, d in sel:
        fr = factor_values.get(fid)
        if fr is None or len(fr) == 0:
            continue
        last = fr.iloc[-1]
        if last.isna().all():
            continue
        rk = ranks(last.to_dict(), assets)
        for a in assets:
            score[a] += w * (d * rk[a])
        active.append(fid)
    if not active:
        eq = {a: 1.0 / len(assets) for a in assets}
        rebalance_to_weights(eq, forecast_returns={a: 0.0 for a in assets},
                             factor_ids=[], horizon_days=10)
        return

    # ---- regime assessment (defensive tilt) --------------------------------
    # VIX <14 strictly calm (eased floor 0.09), 14-30 medium (0.10),
    # >30 OR negative mkt trend risk-off (0.12). VIX 34.8 EXTREME
    # (re-escalating after 45.8->38.7 swing) -> risk-off trigger HIT ->
    # def_floor 0.12 until VIX < 30 confirmed.
    mkt_20 = float(mkt.tail(20).mean())
    vix_df = _index("VIX")
    try:
        vix = float(vix_df["close"].iloc[-1]) if vix_df is not None and len(vix_df) else 25.0
    except Exception:
        vix = 25.0
    risk_off = mkt_20 < -0.0005 or vix > 30.0
    risk_on = mkt_20 > 0.001 and vix < 14.0
    def_floor = 0.12 if risk_off else (0.09 if risk_on else 0.10)

    # ---- target weights -----------------------------------------------------
    score_rk = ranks(score, assets)
    # below-MA20 trend-exclusion guard (EXTENDED to regional equity legs per
    # 02-26 screener: 000300 -11.9% block confirmed whipsaw scope; SPX kept
    # out as primary benchmark reference)
    ma20 = cp.rolling(20).mean().iloc[-1]
    adj_rk = dict(score_rk)
    for a in TREND_GUARD:
        if a in cp.columns and a in ma20.index and cp[a].iloc[-1] < ma20[a]:
            adj_rk[a] *= 0.5
    def_w = def_alloc(def_floor * len(DEF), series, list(DEF))
    pref = {}
    for a in assets:
        pref[a] = def_w.get(a, def_floor) if a in DEF else 0.05 + 0.95 * adj_rk[a]
    weights = capped_normalize(dict(pref), pref)

    # ---- deterministic forecast returns (z-scored score * vol scale) --------
    vals = [score[a] for a in assets]
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / len(vals)
    std = var ** 0.5 if var > 0 else 1.0
    xvol = R.tail(60).std(axis=1).median()
    scale = min(max(float(xvol), 0.005), 0.02) if xvol == xvol else 0.01
    forecast_returns = {a: ((score[a] - mean) / max(std, 1e-12)) * scale for a in assets}

    # force exact sum-to-one
    weights[assets[-1]] += 1.0 - sum(weights.values())

    rebalance_to_weights(
        weights,
        forecast_returns=forecast_returns,
        factor_ids=active[:10],
        horizon_days=10,
    )