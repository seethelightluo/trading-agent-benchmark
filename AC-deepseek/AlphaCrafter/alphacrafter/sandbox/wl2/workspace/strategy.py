"""Trader strategy v6 -- Screener quality_ic_tilt ensemble (10 factors) LIVE-computed.

Cross-sectional factor composite (CS rank -> z-score -> winsorize 3 sigma, dir
applied), fully-invested 15-asset long-only target, one atomic rebalance per
10-trading-day block. Risk-off regime tilts toward defensive tradable assets
(XAU/US10Y/CN10Y); never cash.

Ensemble (2031-03-20 refresh, quality_ic_tilt, 5f): max_consec_gain_20
.2821(+1), downbeta_spx_60 .2351(+1), mom_180d_skip5 .1920(+1), spx_corr60
.1865(+1), range_pos_252 .1043(+1). Same 5 factor IDs as the 2030-10-17
refresh; weights regime-updated (downbeta demoted .3263->.2351, mom_180d
promoted .1500->.1920, range_pos .0841->.1043). The 2026-11-19 10f momentum
set (mom20_volproxy60, gain_loss_20, vol_of_vol20x60,
usdjpy_beta_cond_120x60, mom30_vol60, days_since_high_60, max_consec_loss_20)
is DROPPED (miner_1 gate decay through 2030-10-16).

v6 changes (2026-11-19):
1. ENSEMBLE_PATH -> factors/factor_ensemble.json (screener persists there; the
   root copy was the stale 2026-08-13 9f ensemble).
2. LIVE_FIDS -> new 10-factor set; live computation added for vol_of_vol20x60
   (20d rv std over 60d), mom30_vol60 (30d skip-5 momentum / 60d vol),
   days_since_high_60 (days since last touch of trailing 60d high, dir -1),
   max_consec_loss_20 (longest 21d down-run, dir -1). Dropped mom_20d_skip5 /
   calmness_20 / volcluster_60 from live set (no longer selected).
3. Feed-artifact guard: SX5E/BTC/US10Y/CN10Y price series are perfectly flat
   since ~2026-11-03 (std of trailing 15d returns == 0). Such assets get NaN
   (neutral rank 0.5) on every live factor so the clamped feed cannot distort
   cross-sectional rankings; they still receive a portfolio weight via the
   sum-to-1 normalizer.
v5 heritage: live factor recompute (artifacts froze 2026-07-29); v4 trend-sanity
cap (20d return < -4% -> per-asset cap 0.09) retained; v8 China-specific gate
(000300.SH/000688.SH capped at 0.09 when 20d return < -2%) added cycle56; v9
momentum-add gate (SOX/N225 capped at 0.09 when 20d return < -2%) added cycle57; v10
commodity parabolic-move gate (WTI/COPPER capped at 0.09 when live 20d return > +15%)
added cycle78 - commodity momentum rewards then inverts (cycles 75-77: COPPER -7.7% /
WTI -2.9% add regrets after prior-block surges).
-- SOX/N225 momentum adds whipsawed 4+/2 times (cycles 52-56); evidence review
per cycle-55 plan completed after cycle-56 block.
"""
import json
import math
from datetime import date as _date
from pathlib import Path

from alphacrafter.sim.utils import (
    get_account_dict,
    get_index_daily_data,
    get_stock_daily_data,
    rebalance_to_weights,
    register_hook,
)

BASE = Path(__file__).parent
ENSEMBLE_PATH = BASE / "factors" / "factor_ensemble.json"
DATE_PATH = BASE.parent / "persistent" / "date.json"

ONLINE_START = "2026-07-16"
BLOCK = 10
CAP = 0.14
FLOOR = 0.012
TREND_CAP = 0.09        # per-asset cap when live 20d return < TREND_THRESH
MAX_TURNOVER = 0.30    # dampener: max one-way turnover per rebalance (reversal-tape risk control)
TREND_THRESH = -0.04    # 20d return threshold triggering the trend cap
CHINA = {"000300.SH", "000688.SH"}
CHINA_TREND_THRESH = -0.02   # China-specific 20d threshold (soft-China de-weight, cycle56)
CHINA_TREND_CAP = 0.09       # China cap when 20d return < CHINA_TREND_THRESH
MOMENTUM_ADD = {"SOX", "N225"}   # whipsaw-prone momentum adds (cycles 52-56 evidence)
MOM_TREND_THRESH = -0.02         # 20d threshold triggering the momentum-add cap
MOM_TREND_CAP = 0.09
COMMODITY_PARABOLIC = {"WTI", "COPPER"}   # commodity momentum inversion guard (cycles 75-77 evidence; v10 2030-07-11)
COMMODITY_PARABOLIC_THRESH = 0.15          # 20d return above which a commodity surge is 'parabolic'
COMMODITY_PARABOLIC_CAP = 0.09
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
AGGRESSIVE = {"SOX", "NDX", "ETH", "BTC", "000688.SH", "N225"}
EMBEDDED = {"spx_corr60"}
LIVE_FIDS = {
    "downbeta_spx_60", "max_consec_gain_20", "spx_corr60",
    "mom_180d_skip5", "range_pos_252",
}
ARTIFACT_START = "2020-01-01"
LIVE_MIN_FINITE = 10    # of 15 assets required to trust a live factor row
INERTIA = 1.0        # inertia disabled: pure live-factor target (v5.1 blend backtested worse)


def _load_ensemble():
    try:
        payload = json.loads(ENSEMBLE_PATH.read_text())
    except (OSError, ValueError, TypeError):
        return []
    out = []
    for item in payload.get("selected_factors", []):
        if not isinstance(item, dict):
            continue
        fid = item.get("factor_id")
        w = float(item.get("weight", 0.0) or 0.0)
        if not fid or not math.isfinite(w) or w <= 0.0:
            continue
        out.append({"factor_id": str(fid), "weight": w,
                    "direction": int(item.get("direction", 1) or 1)})
    return out


def _signal_row(fid, row_idx, n_assets):
    """Return the signal vector for a factor at the given artifact row index."""
    if fid in EMBEDDED:
        try:
            art = json.loads((BASE / "factors" / f"{fid}.json").read_text())["signal_artifact"]
            dates = art.get("dates", [])
            values = art.get("values", [])
            if not dates or not values:
                return None
            idx = min(row_idx, len(values) - 1)
            return [float(x) if x is not None else float("nan") for x in values[idx]]
        except (OSError, ValueError, KeyError, TypeError):
            return None
    try:
        import numpy as np
        arr = np.load(BASE / "factors" / f"{fid}.signal.npy", allow_pickle=True)
        if arr.ndim != 2 or arr.shape[1] < n_assets:
            return None
        row = arr[min(row_idx, arr.shape[0] - 1)]
        return [float(x) if x is not None and not (isinstance(x, float) and math.isnan(x))
                else float("nan") for x in row[:n_assets]]
    except (OSError, ValueError, TypeError):
        return None


def _rank_z(vals):
    """Cross-sectional rank -> [0,1] -> z-score -> winsorize 3 sigma."""
    n = len(vals)
    valid = sorted((v, i) for i, v in enumerate(vals) if v == v)
    ranks = [0.5] * n
    nv = len(valid)
    for k, (_, i) in enumerate(valid):
        ranks[i] = k / max(1, nv - 1)
    mean = sum(ranks) / n
    var = sum((x - mean) ** 2 for x in ranks) / n
    sd = math.sqrt(var) if var > 1e-14 else 1e-12
    return [max(-3.0, min(3.0, (x - mean) / sd)) for x in ranks]


def _closes(assets, n=130):
    closes = {}
    for a in assets:
        try:
            df = get_stock_daily_data(a, days=n)
        except Exception:
            df = None
        if df is not None and "close" in df and len(df) >= 62:
            closes[a] = df["close"].astype(float)
    return closes


def _regime(closes, assets):
    """Return risk score R in [0,1] using only data visible at decision date."""
    rets = {}
    for a, c in closes.items():
        s = c.pct_change().dropna()
        if len(s) >= 60:
            rets[a] = s
    if len(rets) < 8:
        return 0.5, 20.0, 0.0, 0.0
    panel = __import__("pandas").concat(rets, axis=1, join="inner").dropna()
    market = panel.mean(axis=1).tail(60)
    m20 = float(market.tail(20).mean()) if len(market) >= 20 else 0.0
    disp20 = float(panel.tail(20).std(axis=1).mean()) if len(panel) >= 20 else 0.0
    vix_level = 20.0
    try:
        vf = get_index_daily_data("VIX", days=40)
        if vf is not None and "close" in vf and len(vf) >= 2:
            vix_level = float(vf["close"].iloc[-1])
    except Exception:
        pass
    vix_comp = max(0.0, min(1.0, (vix_level - 15.0) / 20.0))
    trend_comp = max(0.0, min(1.0, -m20 / 0.04))
    r = 0.6 * vix_comp + 0.4 * trend_comp
    return r, vix_level, m20, disp20


def _current_weights(account, assets):
    """Current portfolio weights from the account (net-asset scaled), zeros elsewhere."""
    na = float(account.get("net_assets", 0.0) or 0.0)
    w = {a: 0.0 for a in assets}
    if na <= 0.0:
        return w
    for p in account.get("positions", []):
        s = p.get("symbol")
        if s in w:
            mv = float(p.get("market_value", 0.0) or 0.0)
            w[s] = max(0.0, mv) / na
    return w


def _live_factors(assets):
    """Recompute the 5 ensemble factor signals live from price data visible at
    the decision date (2031-03-20 ensemble: max_consec_gain_20, downbeta_spx_60, mom_180d_skip5,
    spx_corr60, range_pos_252). Formulas match the persisted
    factor JSONs (v2.0.0). Perfectly-flat trailing 15d series (feed artifact)
    -> NaN (neutral rank 0.5)."""
    import numpy as np
    import pandas as pd
    closes = {}
    for a in assets:
        try:
            df = get_stock_daily_data(a, days=300)
        except Exception:
            df = None
        if df is not None and "close" in df and len(df) >= 130:
            closes[a] = df.set_index(pd.to_datetime(df["date"]))["close"].astype(float)
    if len(closes) < 8:
        return {}
    spx = closes.get("SPX")
    spx_ret = spx.pct_change() if spx is not None else None

    def longest_run(x):
        m = 0.0
        cur = 0
        for v in x:
            if v == 1:
                cur += 1
                if cur > m:
                    m = cur
            else:
                cur = 0
        return m

    per = {a: {} for a in assets}
    for a, c in closes.items():
        ret = c.pct_change()
        f = per[a]
        flat = bool(len(ret) >= 15 and float(ret.tail(15).std()) < 1e-12)
        f["_flat"] = flat
        if flat:
            continue  # all live factor values stay NaN -> neutral rank
        pos = (ret > 0).astype(int)
        f["max_consec_gain_20"] = pos.rolling(21, min_periods=10).apply(longest_run, raw=True)
        f["mom_180d_skip5"] = c.shift(5) / c.shift(185) - 1.0
        rng_min = c.rolling(252, min_periods=30).min()
        rng_max = c.rolling(252, min_periods=30).max()
        f["range_pos_252"] = (c - rng_min) / (rng_max - rng_min).replace(0, np.nan)
        if spx_ret is not None:
            f["spx_corr60"] = ret.rolling(60, min_periods=15).corr(spx_ret)
            m2 = pd.concat([ret, spx_ret], axis=1, join="inner").dropna()
            m2.columns = ["a", "s"]

            def downbeta(x):
                sub = m2.loc[x.index]
                sub = sub[sub["s"] < 0]
                if len(sub) < 15:
                    return np.nan
                if sub["s"].var() < 1e-12:
                    return np.nan
                return float(sub["a"].cov(sub["s"]) / sub["s"].var())

            f["downbeta_spx_60"] = m2["a"].rolling(60, min_periods=20).apply(downbeta, raw=False)

    out = {}
    for fid in LIVE_FIDS:
        vals = []
        for a in assets:
            f = per.get(a, {})
            if f.get("_flat", False):
                vals.append(float("nan"))
                continue
            s = f.get(fid)
            if s is not None:
                if hasattr(s, "iloc"):
                    if len(s) > 0:
                        v = float(s.iloc[-1])
                        vals.append(v if v == v else float("nan"))
                    else:
                        vals.append(float("nan"))
                else:
                    v = float(s)
                    vals.append(v if v == v else float("nan"))
            else:
                vals.append(float("nan"))
        if sum(1 for v in vals if v == v) >= LIVE_MIN_FINITE:
            out[fid] = vals
    return out


def _fit_weights(pref, cap=CAP, floor=FLOOR, cap_map=None):
    """Iterative cap/floor normalization of a non-negative preference vector.

    cap_map optionally overrides the cap per asset (e.g., trend-failing assets).
    Preserves sum-to-1: excess above caps is taken from capped assets and
    redistributed to assets below cap (proportional to pref); assets below
    floor are raised by taking from donors above floor. Fixed 2026-09-24:
    the previous implementation never reduced capped assets, so it inflated
    the total and renormalized everything to ~equal weight.
    """
    total_pref = sum(max(0.0, float(x)) for x in pref.values())
    if total_pref <= 0.0:
        n = len(pref)
        return {a: 1.0 / n for a in pref}
    w = {a: max(0.0, float(x)) / total_pref for a, x in pref.items()}
    cap_a = {a: (cap_map.get(a, cap) if cap_map else cap) for a in w}
    n = len(w)
    for _ in range(200):
        excess = 0.0
        for a in w:
            if w[a] > cap_a[a]:
                excess += w[a] - cap_a[a]
                w[a] = cap_a[a]
        short = 0.0
        for a in w:
            if w[a] < floor:
                short += floor - w[a]
                w[a] = floor
        s = sum(w.values())
        need = 1.0 - s
        if need > 1e-12:
            room = [a for a in w if w[a] < cap_a[a] - 1e-12]
            remaining = need
            while remaining > 1e-12 and room:
                den = sum(max(0.0, pref.get(a, 0.0)) for a in room)
                if den <= 1e-12:
                    given = 0.0
                    for a in room:
                        add = min(remaining / len(room), cap_a[a] - w[a])
                        w[a] += add
                        given += add
                    remaining -= given
                    break
                given = 0.0
                for a in room:
                    share = remaining * pref.get(a, 0.0) / den
                    add = min(share, cap_a[a] - w[a])
                    if add > 1e-14:
                        w[a] += add
                        given += add
                remaining -= given
                room = [a for a in room if w[a] < cap_a[a] - 1e-12]
                if given <= 1e-12:
                    break
            if remaining > 1e-12:
                break
        elif need < -1e-12:
            donors = [a for a in w if w[a] > floor + 1e-12]
            remaining = -need
            while remaining > 1e-12 and donors:
                avail = sum(w[a] - floor for a in donors)
                if avail <= 1e-12:
                    break
                removed = 0.0
                for a in donors:
                    share = remaining * (w[a] - floor) / avail
                    cut = min(share, w[a] - floor)
                    if cut > 1e-14:
                        w[a] -= cut
                        removed += cut
                remaining -= removed
                donors = [a for a in w if w[a] > floor + 1e-12]
                if removed <= 1e-12:
                    break
            if remaining > 1e-12:
                break
        if excess <= 1e-12 and short <= 1e-12 and abs(need) <= 1e-12:
            break
    total = sum(w.values())
    if total <= 0.0:
        return {a: 1.0 / n for a in w}
    return {a: x / total for a, x in w.items()}


def build_target(assets, date_state, ensemble, current_weights=None):
    """Pure computation of (weights, forecast_returns, factor_ids, meta)."""
    trading_days = date_state.get("trading_days", [])
    visible = date_state.get("visible_through", date_state.get("current_date"))
    if ARTIFACT_START not in trading_days or visible not in trading_days:
        return None
    row_idx = trading_days.index(visible) - trading_days.index(ARTIFACT_START)
    if row_idx < 0:
        row_idx = 0

    n = len(assets)
    live = _live_factors(assets)
    z = [0.0] * n
    used = []
    for fac in ensemble:
        fid = fac["factor_id"]
        row = None
        lv = live.get(fid)
        if lv is not None and sum(1 for v in lv if v == v) >= LIVE_MIN_FINITE:
            row = lv
        if row is None:
            row = _signal_row(fid, row_idx, n)  # stale-artifact fallback
        if row is None:
            continue
        zz = _rank_z(row)
        z = [a + fac["weight"] * fac["direction"] * b for a, b in zip(z, zz)]
        used.append(fid)
    if not used:
        return None

    mean = sum(z) / n
    var = sum((x - mean) ** 2 for x in z) / n
    sd = math.sqrt(var) if var > 1e-14 else 1e-12
    z_std = [(x - mean) / sd for x in z]

    # regime overlay
    closes = _closes(assets)
    risk, vix, m20, disp = _regime(closes, assets)
    delta = 0.14 * risk

    # softmax base weights
    mx = max(z_std)
    exps = [math.exp(x - mx) for x in z_std]
    den = sum(exps)
    base = {a: exps[i] / den for i, a in enumerate(assets)}

    pref = {}
    for i, a in enumerate(assets):
        if a in DEFENSIVE:
            pref[a] = base[a] + delta / len(DEFENSIVE)
        else:
            pref[a] = base[a] * (1.0 - delta)

    # inertia disabled (v5.1 blend backtested -1.1% vs -0.06% pure; see 2026-09-24).
    lam = INERTIA
    if current_weights:
        pref = {a: lam * pref.get(a, 0.0) + (1.0 - lam) * max(0.0, current_weights.get(a, 0.0))
                for a in assets}

    # v4 trend-sanity overlay: deeply negative live 20d return -> tighter cap
    r20 = {}
    for a in assets:
        c = closes.get(a)
        r20[a] = float(c.iloc[-1] / c.iloc[-21] - 1.0) if (c is not None and len(c) >= 21) else 0.0
    cap_map = {a: TREND_CAP for a in assets if r20[a] < TREND_THRESH}
    for a in CHINA:
        if r20[a] < CHINA_TREND_THRESH:
            cap_map[a] = min(cap_map.get(a, CAP), CHINA_TREND_CAP)
    for a in MOMENTUM_ADD:
        if r20[a] < MOM_TREND_THRESH:
            cap_map[a] = min(cap_map.get(a, CAP), MOM_TREND_CAP)
    for a in COMMODITY_PARABOLIC:
        if r20[a] > COMMODITY_PARABOLIC_THRESH:
            cap_map[a] = min(cap_map.get(a, CAP), COMMODITY_PARABOLIC_CAP)
    weights = _fit_weights(pref, cap=CAP, floor=FLOOR, cap_map=cap_map or None)

    # v7 rotation dampener: cap one-way turnover so a single block's wrong bet
    # cannot migrate > MAX_TURNOVER of the book (whipsaw tape protection).
    if current_weights is not None:
        turn = sum(abs(weights[a] - max(0.0, current_weights.get(a, 0.0))) for a in assets)
        if turn > MAX_TURNOVER:
            s = MAX_TURNOVER / turn
            weights = {a: max(0.0, current_weights.get(a, 0.0)) + s * (weights[a] - max(0.0, current_weights.get(a, 0.0))) for a in assets}
            tot = sum(weights.values())
            if tot > 0:
                weights = {a: w / tot for a, w in weights.items()}

    # deterministic forecast returns (10-day horizon): z * daily vol * sqrt(10)
    sigma = {}
    for a in assets:
        c = closes.get(a)
        if c is not None and len(c) >= 21:
            s = c.pct_change().dropna().tail(20)
            v = float(s.std()) if len(s) >= 5 else 0.01
            sigma[a] = v if v > 1e-6 else 0.01
        else:
            sigma[a] = 0.01
    forecast = {a: z_std[i] * sigma[a] * math.sqrt(10.0) for i, a in enumerate(assets)}
    forecast = {a: (max(-0.25, min(0.25, v)) if v == v else 0.0) for a, v in forecast.items()}

    meta = {"risk": risk, "vix": vix, "m20": m20, "disp": disp, "lam": lam,
            "n_factors": len(used), "z": dict(zip(assets, z_std)),
            "r20": r20, "cap_map": cap_map}
    return weights, forecast, used, meta


@register_hook
def strategy_hook():
    account = get_account_dict()
    assets = list(account.get("watch_list", []))
    if len(assets) != 15:
        return
    try:
        date_state = json.loads(DATE_PATH.read_text())
    except (OSError, ValueError, TypeError):
        return
    current = date_state.get("current_date", "")
    if current < ONLINE_START:
        return  # warm-up: capital frozen, no holdings
    trading_days = date_state.get("trading_days", [])
    weekdays = [x for x in trading_days if _date.fromisoformat(x).weekday() < 5]
    if current not in weekdays:
        return
    k = weekdays.index(current) - weekdays.index(ONLINE_START)
    if k % BLOCK != 0:
        return  # not the first day of a 10-trading-day block

    ensemble = _load_ensemble()
    if not ensemble:
        return
    cur_w = _current_weights(account, assets)
    built = build_target(assets, date_state, ensemble, current_weights=cur_w)
    if built is None:
        return
    weights, forecast, used, meta = built
    total = sum(weights.values())
    if not (math.isfinite(total) and abs(total - 1.0) < 1e-6):
        return
    if any(not math.isfinite(weights[a]) or weights[a] < 0.0 for a in assets):
        return
    rebalance_to_weights(
        weights,
        forecast_returns=forecast,
        factor_ids=used[:10],
        horizon_days=BLOCK,
    )
