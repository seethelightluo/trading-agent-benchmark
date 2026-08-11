"""Trader strategy v5 -- Screener quality_ic_tilt ensemble (9 factors) LIVE-computed.

Cross-sectional factor composite (CS rank -> z-score -> winsorize 3 sigma, dir +1),
fully-invested 15-asset long-only target, one atomic rebalance per 10-trading-day block.
Risk-off regime tilts toward defensive tradable assets (XAU/US10Y/CN10Y); never cash.

Ensemble (2026-08-13, quality_ic_tilt, <=10 cap): max_consec_gain_20 .208,
mom20_volproxy60 .165, spx_corr60 .115, mom_20d_skip5 .110, gain_loss_20 .108,
downbeta_spx_60 .093, usdjpy_beta_cond_120x60 .083, volcluster_60 .059, calmness_20 .058.
All directions +1.

v5 change (2026-09-24): factor artifacts froze at 2026-07-29 (~40 td stale), giving
static z-ranks and negative gross edge at the 08-27 and 09-10 block starts (gate
correctly skipped both). Instead of waiting for a miner artifact refresh, compute the
9 ensemble factor signals LIVE from price data visible at each decision date (same
formulas/directions as the persisted artifacts; recomputation cross-checked in
scripts/trader_20260924_validate_livefactors.py). Persisted artifact rows are kept
only as a fallback when live data is insufficient. v4 trend-sanity cap retained.
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
ENSEMBLE_PATH = BASE / "factor_ensemble.json"
DATE_PATH = BASE.parent / "persistent" / "date.json"

ONLINE_START = "2026-07-16"
BLOCK = 10
CAP = 0.17
FLOOR = 0.012
TREND_CAP = 0.09        # per-asset cap when live 20d return < TREND_THRESH
TREND_THRESH = -0.04    # 20d return threshold triggering the trend cap
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
AGGRESSIVE = {"SOX", "NDX", "ETH", "BTC", "000688.SH", "N225"}
EMBEDDED = {"mom_20d_skip5", "range_pos_252", "spx_corr60"}
LIVE_FIDS = {
    "mom_20d_skip5", "mom20_volproxy60", "calmness_20", "volcluster_60",
    "max_consec_gain_20", "gain_loss_20", "spx_corr60", "downbeta_spx_60",
    "usdjpy_beta_cond_120x60",
}
ARTIFACT_START = "2020-01-01"
LIVE_MIN_FINITE = 10    # of 15 assets required to trust a live factor row
INERTIA = 0.5        # blend weight on the live-factor target (1-INERTIA on current holdings)


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
    """Recompute ensemble factor signals live from price data visible at the
    decision date. Returns {fid: [value per asset]} with NaN where unavailable.
    Formulas match the persisted artifacts (cross-checked 2026-09-24)."""
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
    try:
        uf = get_index_daily_data("USDJPY", days=300)
        usdjpy = uf.set_index(pd.to_datetime(uf["date"]))["close"].astype(float)
    except Exception:
        usdjpy = None
    spx = closes.get("SPX")
    spx_ret = spx.pct_change() if spx is not None else None
    usdjpy_ret = usdjpy.pct_change() if usdjpy is not None else None

    per = {a: {} for a in assets}
    for a, c in closes.items():
        ret = c.pct_change()
        f = per[a]
        f["mom_20d_skip5"] = c.shift(5) / c.shift(25) - 1.0
        raw20 = c.shift(5) / c.shift(25) - 1.0
        mom60 = c / c.shift(60) - 1.0
        f["mom20_volproxy60"] = raw20 / (1.0 + mom60.abs())
        std20 = ret.rolling(20, min_periods=10).std()
        f["calmness_20"] = (ret.abs() < 0.5 * std20).rolling(20, min_periods=10).mean()
        f["volcluster_60"] = ret.abs().rolling(60, min_periods=40).corr(ret.abs().shift(1))
        pos = (ret > 0).astype(int)

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

        f["max_consec_gain_20"] = pos.rolling(21, min_periods=10).apply(longest_run, raw=True)
        g = ret.clip(lower=0).rolling(20, min_periods=10).sum()
        l = ret.clip(upper=0).abs().rolling(20, min_periods=10).sum()
        f["gain_loss_20"] = g / l.replace(0, np.nan)
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
        if usdjpy_ret is not None:
            m3 = pd.concat([ret, usdjpy_ret], axis=1, join="inner").dropna()
            m3.columns = ["a", "u"]

            def jpybeta(x):
                sub = m3.loc[x.index]
                if len(sub) < 60 or sub["u"].var() < 1e-12:
                    return np.nan
                return float(sub["a"].cov(sub["u"]) / sub["u"].var())

            b = m3["a"].rolling(120, min_periods=60).apply(jpybeta, raw=False)
            mom60j = usdjpy / usdjpy.shift(60) - 1.0
            f["usdjpy_beta_cond_120x60"] = b * mom60j

    out = {}
    for fid in LIVE_FIDS:
        vals = []
        for a in assets:
            s = per[a].get(fid) if a in per else None
            if s is not None and len(s) > 0:
                v = float(s.iloc[-1])
                vals.append(v if v == v else float("nan"))
            else:
                vals.append(float("nan"))
        if sum(1 for v in vals if v == v) >= LIVE_MIN_FINITE:
            out[fid] = vals
    return out


def _fit_weights(pref, cap=CAP, floor=FLOOR, cap_map=None):
    """Iterative cap/floor normalization of a non-negative preference vector.

    cap_map optionally overrides the cap per asset (e.g., trend-failing assets).
    Preserves sum-to-1 (pref is normalized internally; excess above caps is
    redistributed only up to each asset's headroom).
    """
    total_pref = sum(max(0.0, float(x)) for x in pref.values())
    if total_pref <= 0.0:
        n = len(pref)
        return {a: 1.0 / n for a in pref}
    w = {a: max(0.0, float(x)) / total_pref for a, x in pref.items()}
    cap_a = {a: (cap_map.get(a, cap) if cap_map else cap) for a in w}
    n = len(w)
    for _ in range(500):
        excess = sum(max(0.0, w[a] - cap_a[a]) for a in w)
        if excess > 1e-12:
            room = [a for a in w if w[a] < cap_a[a] - 1e-12]
            if not room:
                break
            den = sum(max(0.0, pref.get(a, 0.0)) for a in room)
            moved = 0.0
            for a in room:
                add = excess * (max(0.0, pref.get(a, 0.0)) / den if den > 1e-12 else 1.0 / len(room))
                add = min(add, cap_a[a] - w[a])  # never exceed per-asset cap
                if add > 1e-14:
                    w[a] += add
                    moved += add
            if moved <= 1e-12:
                break  # headroom exhausted -> fall back to normalization
        short = sum(max(0.0, floor - x) for x in w.values())
        if short > 1e-12:
            donors = [a for a in w if w[a] > floor + 1e-12]
            avail = sum(w[a] - floor for a in donors)
            if avail > 1e-12:
                for a in donors:
                    w[a] -= short * (w[a] - floor) / avail
            for a in w:
                if w[a] < floor:
                    w[a] = floor
        if excess <= 1e-12 and short <= 1e-12:
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

    # v5.1 inertia blend: keep a share of current holdings so fresh ensemble
    # signals rotate the book gradually instead of churning winners away
    # (60d backtest of pure live factors was flat w/ 7.4% DD; blended book
    # preserves the accumulated momentum edge while adapting to new signals).
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
    weights = _fit_weights(pref, cap=CAP, floor=FLOOR, cap_map=cap_map or None)

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
