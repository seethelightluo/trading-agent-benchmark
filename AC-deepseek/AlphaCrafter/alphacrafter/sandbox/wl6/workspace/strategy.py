"""Trader strategy: Screener quality-IC factor ensemble (updated 2030-07-11).

Cross-sectional composite across all 15 tradable benchmark assets; long-only,
fully invested (no cash sleeve). One target submitted daily via
rebalance_to_weights; the benchmark helper gates cadence (10d), turnover cost
(3bp one-way) and gross edge. Defensive tilt expressed through weights, never cash.

Active ensemble (loaded dynamically from factor_ensemble.json, 2030-07-11):
  beta_vix_60d_neg(+0.28), mom_120d_skip5(+0.16), sign_ewma_60d(+0.15),
  vol_beta_spx_60d(+0.15), mom_10d_skip5(+0.14), down_vol_ratio_20x120(+0.08),
  skew_20d_neg(-0.04 dir=-1). (7 factors, sum=1.0, <=10 cap ok.
  2030-07-11 screener refresh under regime=highvol_choppy_sideways_VIX41.6_
  off_54_peak_SPX_5040_-3.3pct_60d_NDX_+6.1pct_40d_lead_XAU_strong_BTC_
  -15.5pct_COPPER_WTI_weak_dispersion_high: ensemble HELD at 7 factors per
  trader 'keep_ensemble_7f' feedback after +3.54% block (sharpe 6.99, maxdd
  0.88%, net 1,005,098->1,040,638). Changes vs 2030-04-18: (1) mom_10d_skip5
  0.15->0.16 - short-horizon timing 'excellent' (ETH/NDX caught 2nd block
  running; COPPER/SPX/US10Y trims well-timed); (2) mom_120d_skip5 0.13->0.12 -
  medium-horizon momentum demoted (WTI boosted then -6.93%, N225 boosted then
  -2.82%); (3) beta_vix_60d_neg anchor held 0.28 (VIX 41.6 defensive first);
  (4) sign_ewma_60d held 0.18; (5) vol_beta_spx_60d held 0.16; (6)
  down_vol_ratio_20x120 held 0.06 (frozen-name exposure capped at 0.5% by
  trader guardrail); (7) skew_20d_neg held 0.04 dir=-1 (tail-risk hedge in
  high vol). Defensive aggregate (beta_vix+vol_beta+down_vol)=0.50 at
  self-cap; momentum aggregate (mom_120+mom_10+sign_ewma)=0.46 short-horizon
  shifted. Excluded: beta_chi_60d (China momentum unreliable per trader 2x
  feedback), beta_cn10y_60d (CN10Y frozen/inert), vol_of_vol20x60
  (quarantined; whipsaw selector in high vol), low_vol_20d dir=-1 (risk-on
  premium premature; VIX 41.6 >>30, SPX 5040 below re-add trigger),
  vix_beta_cond_60x20 (redundant w/ anchor), corr_us10y_60d /
  vol_of_vol_chg_20d / xau_copper_cond_20d (low q, monitor).)

Trader-side guardrails:
  1. Regime detector FIXED (2026-12-17) to return-based cross-asset drift (was
     price-level drift -> always "bull"; caused the 2026-12-03 regime mismatch).
  2. Per-asset weight cap at 12% (limits single-name blowup e.g. BTC/WTI).
  3. Defensive floor on XAU/US10Y (0.05 bull / 0.12 sideways / 0.25 bear).
     2027-04-08: CN10Y removed from the defensive set - the series has been
     frozen since ~Nov-2026, floor weight there is wasted capital (screener
     flagged "flat names carry zero PnL"). Live defensive anchors: XAU, US10Y.
  4. FIX 2026-12-31: normalize the selected top-K raw weights to sum 1 BEFORE
     apply_floor/apply_cap so the cap (0.12) and defensive floor act on a
     proper probability vector.
  5. ADD 2027-02-25: inverse-20d-vol blend (60/40) inside the selected top-K
     set. Selection (which names enter top-K) stays purely factor-driven; only
     weighting inside the set is vol-tilted. Non-negative, sum-to-1.
  6. ADD 2027-04-22: minimum XAU weight 4% independent of factor scores.
  7. ADD 2027-06-17: crypto cap - ETH and BTC individually <= 8% and combined
     <= 15%. Three consecutive negative live blocks (2027-04-22/05-20/06-17)
     had crypto at 9-11% weights as the recurring largest loss contributor
     (vol_of_vol20x60 keeps scoring high-vol names). Cap is trader-side risk
     control; factor selection is unchanged.
  8. ADD 2027-11-18: WTI cap at 8%. WTI was the top loss contributor in two
     consecutive live blocks (2027-10-21 block -19.4% at ~9.2% weight;
     2027-11-04 block -18.4% at ~9.9% weight) while mom_120d_skip5 kept
     scoring it top (120d momentum still very positive after the summer rally
     even after a ~35% cumulative two-block crash). Same precedent as the
     crypto cap; excess weight redistributes proportionally to non-WTI names.
  9. ADD 2028-01-13: frozen-name cap at 0.5% each.
 10. FIX 2028-09-07: guardrail cap sequence iterated to a fixed
     point so redistribution from one cap cannot push another
     capped name over its limit (crypto 16.04% vs 15% overshoot
     observed in the 2028-08-24 block). HSI / 000688.SH / CN10Y
     have been completely flat (zero realized vol, zero 120d range) since
     ~Nov-2026; down_vol_ratio_20x120 keeps scoring these zero-vol series into
     top-K, wasting ~4-5% of capital as dead weight (confirmed in live blocks
     20271216, 20271230, 20280113). Cap is trader-side risk control; factor
     selection unchanged - excess weight redistributes to live names.
"""
from math import isfinite
import json
from pathlib import Path
import pandas as pd
from alphacrafter.sim.utils import (get_account_dict, get_stock_daily_data,
                                    get_index_daily_data, rebalance_to_weights,
                                    register_hook)

N_DAYS = 300            # covers 120d+ windows and buffers
BETA_WIN = 60           # beta windows (vix, cn10y): 60, min_periods 30
DEF = {"XAU", "US10Y"}  # live defensive tradable assets for bear tilt (CN10Y removed 2027-04-08: flat series)
CAP = 0.12              # trader-side per-asset weight cap
FLOOR = {"bull": 0.05, "sideways": 0.12, "bear": 0.25}
MIN_XAU = 0.04          # guardrail 6 (2027-04-22): hard minimum XAU weight
CRYPTO = {"ETH", "BTC"}  # guardrail 7 (2027-06-17)
CRYPTO_EACH = 0.08       # individual crypto weight cap
CRYPTO_TOTAL = 0.15      # combined crypto weight cap
WTI_CAP = 0.08          # guardrail 8 (2027-11-18): WTI single-name cap
FROZEN_VOL_EPS = 1e-9   # guardrail 9 (2028-01-13): realized vol below this => frozen series
FROZEN_CAP = 0.005      # guardrail 9: max weight per frozen (dead) name
VOL_BLEND = 0.4         # weight share given to inverse-20d-vol in top-K blend


def stock(a, n=N_DAYS):
    try:
        return get_stock_daily_data(a, days=n)
    except Exception:
        return None


def index(a, n=N_DAYS):
    try:
        return get_index_daily_data(a, days=n)
    except Exception:
        return None


def rank_series(values, assets):
    """Cross-sectional rank in [0,1]; missing values get neutral 0.5."""
    valid = sorted((float(v), a) for a, v in values.items()
                   if v is not None and isfinite(float(v)))
    out = {a: 0.5 for a in assets}
    for i, (_, a) in enumerate(valid):
        out[a] = i / max(1, len(valid) - 1)
    return out


def load_ensemble():
    for p in (Path(__file__).parent / "factor_ensemble.json",
              Path(__file__).parent / "factors" / "factor_ensemble.json"):
        try:
            ens = json.loads(p.read_text())
            factors = [dict(f) for f in ens.get("selected_factors", [])
                       if isinstance(f, dict) and f.get("factor_id")]
            if factors:
                return factors
        except (OSError, ValueError, TypeError):
            continue
    return []


def vol20_map(closes, assets):
    """{asset: 20d realized vol (std of daily returns)}; None when unavailable."""
    out = {}
    for a in assets:
        c = closes.get(a)
        if c is None or len(c) < 25:
            out[a] = None
            continue
        s = float(c.pct_change().tail(20).std())
        out[a] = s if isfinite(s) and s > 1e-12 else None
    return out


def frozen_set(closes, assets, eps=FROZEN_VOL_EPS):
    """Assets whose realized vol over the full lookback is ~0 (frozen series).

    These series stopped trading in the benchmark (HSI/000688.SH/CN10Y flat
    since ~Nov-2026). Weight on them is dead capital: zero expected PnL and no
    defensive value (guardrail 9, 2028-01-13).
    """
    out = set()
    for a in assets:
        c = closes.get(a)
        if c is None or len(c) < 30:
            continue
        s = float(c.pct_change().dropna().std())
        if isfinite(s) and s < eps:
            out.add(a)
    return out


def apply_frozen_cap(w, assets, frozen, cap=FROZEN_CAP):
    """Guardrail 9 (2028-01-13): cap each frozen (dead) name at `cap`.

    Redistributes the excess proportionally to all non-frozen assets. Keeps
    factor selection unchanged - only the weight of dead series is limited so
    the capital can work in live names.
    """
    w = dict(w)
    frozen = [a for a in frozen if a in assets]
    if not frozen:
        return w
    for _ in range(200):
        over = {a: max(w.get(a, 0.0) - cap, 0.0) for a in frozen}
        tot_over = sum(over.values())
        if tot_over < 1e-12:
            break
        live = [a for a in assets if a not in frozen]
        live_sum = sum(w.get(a, 0.0) for a in live)
        if live_sum <= 1e-12:
            break
        for a in frozen:
            w[a] = w.get(a, 0.0) - over[a]
        for a in live:
            w[a] = w.get(a, 0.0) + tot_over * (w.get(a, 0.0) / live_sum)
    return w


def compute_raw_factors(closes, vix_close, assets):
    """Return {factor_id: {asset: raw value}} on the last completed bar.

    Expressions per Miner/Screener (see miner_2_20280323_new_factor_batch.py):
      beta_vix_60d_neg    = -cov(ret, vix_ret, 60)/var(vix_ret, 60)  (min_periods 30)
      beta_cn10y_60d      = cov(ret, cn10y_pct_change, 60)/var(cn10y_pct_change, 60)
                            (dir=-1 per factor library; CN10Y frozen -> inert)
      beta_chi_60d        = cov(ret, csi300_pct_change, 60)/var(csi300_pct_change, 60)
                            (dir=-1 in old live ensemble: penalize high China-beta names)
      low_vol_20d         = -std(ret, 20)                            (min_periods 10)
      vol_of_vol20x60     = std(std(ret,20),60)  (library completeness only;
                            dropped from live ensemble 2028-03-23)
      mom_10d_skip5       = close.shift(5)/close.shift(15) - 1.0
      mom_120d_skip5      = close.shift(5)/close.shift(125) - 1.0
      down_vol_ratio_20x120 = -(std(max(-ret,0),20)/std(max(-ret,0),120))
      sign_ewma_60d       = ewm(span=60, adjust=False).mean() of (ret>0)  [NEW]
      vol_beta_spx_60d    = beta of asset 20d realized vol on SPX 20d
                            realized vol over 60d (min_periods 30)         [NEW]
      skew_20d_neg        = -skewness(ret, 20) (min_periods 10)           [NEW 2029-10-18]
                            Negative-skew (crash-prone) assets get high raw
                            values; ensemble dir=-1 penalizes them (tail-aversion).
    """
    fids = ["beta_vix_60d_neg", "beta_cn10y_60d", "beta_chi_60d",
            "vol_of_vol20x60", "low_vol_20d", "mom_10d_skip5",
            "mom_120d_skip5", "down_vol_ratio_20x120", "sign_ewma_60d",
            "vol_beta_spx_60d", "skew_20d_neg"]
    raw = {fid: {} for fid in fids}
    vix_ret = vix_close.pct_change() if vix_close is not None else None
    cn_close = closes.get("CN10Y")
    cn_ret = cn_close.pct_change() if cn_close is not None else None
    chi_close = closes.get("000300.SH")
    chi_ret = chi_close.pct_change() if chi_close is not None else None
    spx_close = closes.get("SPX")
    spx_ret = spx_close.pct_change() if spx_close is not None else None
    spx_vol20 = (spx_ret.rolling(20, min_periods=10).std()
                 if spx_ret is not None else None)
    for a in assets:
        c = closes.get(a)
        if c is None or len(c) < 140:
            for fid in fids:
                raw[fid][a] = None
            continue
        ret = c.pct_change()
        bv = None
        if vix_ret is not None:
            z = pd.concat([ret.rename("a"), vix_ret.rename("v")], axis=1).dropna().tail(BETA_WIN)
            varv = float(z["v"].var()) if len(z) else 0.0
            if len(z) >= 30 and varv > 1e-14:
                bv = -float(z["a"].cov(z["v"]) / varv)
        bc = None
        if cn_ret is not None:
            zc = pd.concat([ret.rename("a"), cn_ret.rename("c")], axis=1).dropna().tail(BETA_WIN)
            varc = float(zc["c"].var()) if len(zc) else 0.0
            if len(zc) >= 30 and varc > 1e-14:
                bc = float(zc["a"].cov(zc["c"]) / varc)
        bchi = None
        if chi_ret is not None:
            zq = pd.concat([ret.rename("a"), chi_ret.rename("q")], axis=1).dropna().tail(BETA_WIN)
            varq = float(zq["q"].var()) if len(zq) else 0.0
            if len(zq) >= 30 and varq > 1e-14:
                bchi = float(zq["a"].cov(zq["q"]) / varq)
        lv = -float(ret.rolling(20, min_periods=10).std().iloc[-1])
        vov = float(ret.rolling(20).std().rolling(60).std().iloc[-1])
        mom10 = float(c.shift(5).iloc[-1] / c.shift(15).iloc[-1] - 1.0) \
            if len(c) >= 16 else None
        mom = float(c.shift(5).iloc[-1] / c.shift(125).iloc[-1] - 1.0)
        down = -ret.clip(upper=0.0)  # max(-ret, 0)
        d20 = float(down.rolling(20, min_periods=10).std().iloc[-1])
        d120 = float(down.rolling(120, min_periods=60).std().iloc[-1])
        dvr = (-d20 / d120) if (isfinite(d20) and isfinite(d120) and d120 > 1e-12) else None
        # NEW sign_ewma_60d (miner_2 2028-03-23): EWMA(span=60) of (ret>0).
        up = (ret > 0).astype(float)
        sew = float(up.ewm(span=60, adjust=False).mean().iloc[-1])
        # NEW vol_beta_spx_60d (miner_2 2028-03-23): beta_of(vol20, spx vol20, 60).
        vb = None
        if spx_vol20 is not None:
            a_vol20 = ret.rolling(20, min_periods=10).std()
            zv = pd.concat([a_vol20.rename("a"), spx_vol20.rename("s")],
                           axis=1).dropna().tail(60)
            varsx = float(zv["s"].var()) if len(zv) else 0.0
            if len(zv) >= 30 and varsx > 1e-14:
                vb = float(zv["a"].cov(zv["s"]) / varsx)
        # NEW skew_20d_neg (miner_3 2027-06-17 / miner_1 2028-03-23):
        # -skewness of 20d returns (min_periods 10); negated so negative-skew
        # (crash-prone) assets carry high raw values.
        sk20 = ret.rolling(20, min_periods=10).skew().iloc[-1]
        sk = -float(sk20) if isfinite(sk20) else None
        raw["beta_vix_60d_neg"][a] = bv
        raw["beta_cn10y_60d"][a] = bc
        raw["beta_chi_60d"][a] = bchi
        raw["low_vol_20d"][a] = float(lv) if isfinite(lv) else None
        raw["vol_of_vol20x60"][a] = float(vov) if isfinite(vov) else None
        raw["mom_10d_skip5"][a] = float(mom10) if mom10 is not None and isfinite(mom10) else None
        raw["mom_120d_skip5"][a] = float(mom) if isfinite(mom) else None
        raw["down_vol_ratio_20x120"][a] = dvr
        raw["sign_ewma_60d"][a] = float(sew) if isfinite(sew) else None
        raw["vol_beta_spx_60d"][a] = vb
        raw["skew_20d_neg"][a] = sk
    return raw


def regime_from_market(panel):
    """Bull / sideways / bear from RETURN-based 20d cross-asset drift.

    Fixed 2026-12-17: previously used price levels (always 'bull'). Now uses
    daily cross-asset mean return; trend = t-stat of 20d drift scaled by sqrt(20).
    """
    if len(panel) < 30:
        return "sideways"
    rets = panel.pct_change().dropna()
    mkt = rets.mean(axis=1)
    r20 = float(mkt.tail(20).mean())
    v20 = float(mkt.tail(20).std())
    trend = r20 / v20 * (20.0 ** 0.5) if v20 and v20 > 1e-12 else 0.0
    if trend > 1.0:
        return "bull"
    if trend < -1.0:
        return "bear"
    return "sideways"


def apply_floor(w, assets, def_assets, floor):
    """Guarantee defensive assets collectively hold >= floor of portfolio."""
    cur = sum(w.get(a, 0.0) for a in def_assets)
    if cur >= floor - 1e-12:
        return w
    deficit = floor - cur
    nondef = [a for a in assets if a not in def_assets]
    nondef_sum = sum(w.get(a, 0.0) for a in nondef)
    if nondef_sum <= 1e-12:
        return w
    scale = (1.0 - floor) / nondef_sum
    for a in nondef:
        w[a] = w.get(a, 0.0) * scale
    def_cur = {a: w.get(a, 0.0) for a in def_assets}
    def_sum = sum(def_cur.values())
    for a in def_assets:
        w[a] = w.get(a, 0.0) + deficit * (
            def_cur[a] / def_sum if def_sum > 1e-12 else 1.0 / len(def_assets))
    return w


def apply_cap(w, assets, cap=CAP):
    """Iteratively redistribute weight above cap to assets with headroom."""
    w = dict(w)
    for _ in range(200):
        over = {a: max(w[a] - cap, 0.0) for a in assets}
        tot_over = sum(over.values())
        if tot_over < 1e-12:
            break
        under = {a: max(cap - w[a], 0.0) for a in assets}
        tot_under = sum(under.values())
        if tot_under < 1e-12:
            w = {a: min(w[a], cap) for a in assets}
            break
        for a in assets:
            w[a] = w[a] - over[a] + tot_over * (under[a] / tot_under)
    return w


def apply_min_xau(w, assets, min_xau=MIN_XAU):
    """Guardrail 6: hard minimum XAU weight (funded from all other assets)."""
    if "XAU" not in assets:
        return w
    xau = w.get("XAU", 0.0)
    if xau >= min_xau - 1e-12:
        return w
    deficit = min_xau - xau
    others = [a for a in assets if a != "XAU"]
    other_sum = sum(w.get(a, 0.0) for a in others)
    if other_sum <= 1e-12:
        return w
    scale = (1.0 - min_xau) / other_sum
    for a in others:
        w[a] = w.get(a, 0.0) * scale
    w["XAU"] = min_xau
    return w


def apply_crypto_cap(w, assets, each=CRYPTO_EACH, total=CRYPTO_TOTAL):
    """Guardrail 7 (2027-06-17): cap crypto individually (8%) and combined (15%).

    Three consecutive negative live blocks had ETH/BTC at 9-11% weights as the
    recurring largest loss contributors (vol_of_vol20x60 keeps scoring the
    high-vol crypto names). Cap is trader-side risk control; factor selection
    is unchanged. Excess weight is redistributed proportionally to non-crypto.
    """
    w = dict(w)
    crypto = [a for a in CRYPTO if a in assets]
    if not crypto:
        return w
    for _ in range(200):
        csum = sum(w.get(a, 0.0) for a in crypto)
        scale = 1.0
        if csum > 1e-12:
            scale = min(scale, total / csum)
        for a in crypto:
            wa = w.get(a, 0.0)
            if wa > 1e-12:
                scale = min(scale, each / wa)
        if scale >= 1.0 - 1e-12:
            break
        freed = csum * (1.0 - scale)
        for a in crypto:
            w[a] = w.get(a, 0.0) * scale
        noncrypto = [x for x in assets if x not in crypto]
        nsum = sum(w.get(x, 0.0) for x in noncrypto)
        if nsum > 1e-12:
            for x in noncrypto:
                w[x] = w.get(x, 0.0) + freed * (w.get(x, 0.0) / nsum)
        else:
            break
    return w


def apply_single_cap(w, assets, symbol, cap):
    """Guardrail 8 (2027-11-18): cap one named asset (WTI) at `cap`.

    Redistributes the excess proportionally to all other assets. Keeps factor
    selection unchanged - only the weight of the blowup-prone name is limited.
    """
    w = dict(w)
    if symbol not in assets:
        return w
    wa = w.get(symbol, 0.0)
    if wa <= cap + 1e-12:
        return w
    freed = wa - cap
    w[symbol] = cap
    others = [a for a in assets if a != symbol]
    osum = sum(w.get(a, 0.0) for a in others)
    if osum > 1e-12:
        for a in others:
            w[a] = w.get(a, 0.0) + freed * (w.get(a, 0.0) / osum)
    return w


@register_hook
def strategy_hook():
    assets = list(get_account_dict()["watch_list"])
    frames = {a: stock(a) for a in assets}
    closes = {a: (f.close.astype(float) if f is not None and "close" in f else None)
              for a, f in frames.items()}
    usable = [c.rename(a) for a, c in closes.items() if c is not None and len(c) >= 140]
    fallback = {a: 1.0 / len(assets) for a in assets}
    if len(usable) < 8:
        rebalance_to_weights(fallback, forecast_returns={a: 0.0 for a in assets},
                             horizon_days=10)
        return

    panel = pd.concat(usable, axis=1, join="inner")
    factors = load_ensemble()
    if not factors:
        rebalance_to_weights(fallback, forecast_returns={a: 0.0 for a in assets},
                             horizon_days=10)
        return
    factor_ids = [f["factor_id"] for f in factors][:10]

    vf = index("VIX")
    vix_close = vf.close.astype(float) if vf is not None and "close" in vf else None
    raw = compute_raw_factors(closes, vix_close, assets)

    # Composite score: sum of weight * direction * centered rank.
    score = {a: 0.0 for a in assets}
    for f in factors:
        fid, w, d = f["factor_id"], f.get("weight", 0.0), f.get("direction", 1)
        r = rank_series(raw.get(fid, {}), assets)
        for a in assets:
            score[a] += (w * d) * (r[a] - 0.5)

    # Regime-conditional concentration + defensive overlay.
    regime = regime_from_market(panel)
    K = {"bull": 12, "sideways": 10, "bear": 8}[regime]
    lo = min(score.values())
    span = max(max(score.values()) - lo, 1e-9)
    raw_w = {a: max((score[a] - lo) / span, 0.0) for a in assets}
    top = set(sorted(assets, key=lambda a: (raw_w[a], score[a]), reverse=True)[:K])
    w = {a: (raw_w[a] if a in top else 0.0) for a in assets}
    wsum = sum(w.values())
    if wsum < 1e-9:
        w = {a: (1.0 / K if a in top else 0.0) for a in assets}
    else:
        # Normalize the selected top-K weights to sum 1 BEFORE floor/cap so the
        # cap (0.12) and defensive floor act on a proper probability vector.
        w = {a: v / wsum for a, v in w.items()}

    # Guardrail 5 (2027-02-25): blend score weights with inverse-20d-vol
    # weights within the top-K set (60/40). Selection unchanged.
    vol20 = vol20_map(closes, assets)
    valid_vol = {a: v for a, v in vol20.items() if v is not None and v > 0}
    if valid_vol:
        vmin = min(valid_vol.values())
        inv = {a: (vmin / valid_vol[a] if a in valid_vol else 0.0) for a in assets}
        inv_top_sum = sum(inv.get(a, 0.0) for a in top)
        if inv_top_sum > 1e-12:
            blended = {a: ((1.0 - VOL_BLEND) * w.get(a, 0.0)
                           + VOL_BLEND * (inv.get(a, 0.0) / inv_top_sum if a in top else 0.0))
                       for a in assets}
            bsum = sum(blended.values())
            if bsum > 1e-12:
                w = {a: v / bsum for a, v in blended.items()}

    # Trader guardrails: defensive floor + per-asset cap + min XAU + crypto cap
    # + WTI cap (guardrail 8) + frozen-name cap (guardrail 9, 2028-01-13).
    frozen = frozen_set(closes, assets)
    w = apply_floor(w, assets, [a for a in DEF if a in assets], FLOOR[regime])
    w = apply_min_xau(w, assets)
    # FIX 2028-09-07 (guardrail ordering): iterate the cap sequence to a fixed
    # point so any cap's proportional redistribution (WTI cap, frozen cap,
    # crypto cap, per-asset cap) cannot push another capped name back over its
    # limit (2028-08-24 block: crypto combined 16.04% vs 15% cap after the WTI
    # cap re-fed crypto). Floor + min-XAU applied first; caps never decrease
    # XAU so the 4% minimum is preserved throughout.
    for _ in range(50):
        prev = dict(w)
        w = apply_cap(w, assets)
        w = apply_crypto_cap(w, assets)
        w = apply_single_cap(w, assets, "WTI", WTI_CAP)
        w = apply_frozen_cap(w, assets, frozen)
        w = apply_min_xau(w, assets)
        if sum(abs(w.get(a, 0.0) - prev.get(a, 0.0)) for a in assets) < 1e-11:
            break

    total = sum(w.values())
    weights = {a: (max(w.get(a, 0.0), 0.0) / total if total > 0 else 1.0 / len(assets))
               for a in assets}
    rem = 1.0 - sum(weights.values())
    weights[assets[0]] += rem  # floating-point exactness on sum-to-one

    # Deterministic forecast returns: z-scored composite * cross-sectional vol.
    score_mean = sum(score.values()) / len(assets)
    score_std = (sum((x - score_mean) ** 2 for x in score.values()) / len(assets)) ** 0.5
    _rets = panel.pct_change().dropna()
    ret_scale = float(_rets.tail(252).std(axis=1, ddof=0).median()) if len(_rets) else 0.01
    if not isfinite(ret_scale) or ret_scale <= 0:
        ret_scale = 0.01
    forecast_returns = {a: ((score[a] - score_mean) / max(score_std, 1e-12)) * ret_scale
                        for a in assets}

    rebalance_to_weights(weights, forecast_returns=forecast_returns,
                         factor_ids=factor_ids, horizon_days=10)
