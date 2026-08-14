"""Screener ensemble strategy, trader refresh 2035-08-02 (current 10-factor ensemble).

Cross-sectional factor ensemble (quality_ic_tilt) drives a fully-invested
15-asset long-only target. One proposal per 10-trading-day block (first day
only); the rebalance helper applies the 3bp gross-edge gate.

Ensemble (2035-08-02, from factor_ensemble.json; weights non-negative sum 1):
  down_beta_60(+1,0.22) cn10y_beta_60(-1,0.16) spx_beta_60(+1,0.13)
  vol_adj_mom_20_60(+1,0.11) dxy_beta_cond_60x20(+1,0.07)
  hs300_beta_60(-1,0.07) intraday_ret_skew_20(+1,0.07)
  vol_of_vol20x60(+1,0.06) comm_basket_beta_60(+1,0.06)
  hilo_vol_ratio_20(+1,0.05).
Screener re-tilted the 12-21 mix slightly more defensive (down_beta 0.21->0.22,
cn10y 0.15->0.16, dxy 0.06->0.07, vov 0.05->0.06; spx 0.14->0.13,
vol_adj_mom 0.12->0.11, comm_basket 0.07->0.06, hilo 0.06->0.05) after
prolonged drawdowns (net -4.8% over 06-21..08-02 unlogged blocks,
mdd20 ~-10%). Defensive cluster (down_beta+cn10y+hs300) 0.45 vs risk-on
cluster (vol_adj_mom+spx+comm) 0.30. Same 10 factors, no swaps; strategy
reads factor_ensemble.json dynamically (in sync, docstring refresh only, no
logic rewrite).

Weighting: rank-linear tilt * inverse-vol (sqrt dampened), defensive floor,
water-fill cap at 0.18. Sum-to-1, cash 0, fractional quantities.
Trader guards: frozen-5 pin (0.5% floor), equity-stress trim (eq<=0.40,
ETH<=0.06), commodity guard (WTI<=0.04, COPPER<=0.10, XAU+COPPER+WTI<=0.33,
ETH<=0.04), tech guard (NDX+SOX+000688.SH<=0.24, since 2034-04-27),
SPX cap 0.12 (2035-03-29), XAU cap 0.16 (2035-06-07).
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from alphacrafter.sim.utils import (
    get_account_dict,
    get_stock_daily_data,
    get_index_daily_data,
    rebalance_to_weights,
    register_hook,
)

OBS_ONLY = {"DXY", "VIX", "USDCNY", "USDJPY", "EURUSD"}
DEF = {"XAU", "US10Y", "CN10Y"}
EQ_ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX"]
CAP = 0.18
EQ_CAP = 0.40            # live equity complex max combined weight under stress
ETH_CAP = 0.06           # max ETH weight under stress
WTI_CAP = 0.04           # 2032-04-29 trader re-tune: 3 consecutive post-raise loss blocks since 03-18 raise (pullback, -10.5%, -15.4%); cap 0.06->0.04
COMM_CAP = 0.33          # XAU+COPPER+WTI combined cap (trimmed 0.36->0.33 on 2031-12-11)
COPPER_CAP = 0.10          # 2032-08-05 trader re-tune: COPPER whipsaw (-1.7%, +2.4%, -9.5% over 3 blocks; -9.5% = dominant drag); cap 0.12->0.10 per 07-22 plan trigger
ETH_CAP_ALL = 0.04         # 2033-07-07 trader re-tune: ETH -22.1% block at ~5.6% w (near 6% cap), crash ongoing (-25% 1M thru 07-06); plan trigger met -> cap 0.06->0.04
TECH_CAP = 0.24          # 2034-04-27 trader: 3 consecutive loss blocks (02-02 -3.04%, 04-13 -2.13%, 04-27 -1.79%) driven by NDX/SOX/000688 tech complex; combined cap NDX+SOX+000688.SH
TECH_ASSETS = ["NDX", "SOX", "000688.SH"]   # live US/China tech complex
SPX_CAP = 0.12           # 2035-03-29 trader re-tune: SPX 3rd consecutive negative block (-4.44%, -9.71%, -2.46%); r21 -11.9%, r60 -23.8%; spx_beta_60 kept pushing SPX to ~16% largest weight -> hard cap 0.12 applied after guard stack
XAU_CAP = 0.16           # 2035-06-07 trader re-tune: XAU 3rd consecutive negative block at max weight (-0.37%, -1.76%, -3.54%, -6.88%); recurring biggest drag -> hard cap 0.18->0.16 applied after guard stack
VIX_STRESS = 30.0        # VIX level that flags equity stress
EQ_RET21_STRESS = -0.05  # live-equity mean 21d return threshold for stress
FROZEN_FLOOR = 0.005          # 0.5% per frozen (zero-return) asset
FROZEN_LOOKBACK = 120         # trading days used to detect frozen assets
ONLINE_START = "2026-07-16"
HORIZON = 10
FC_K_MULT = 2.0               # implied-alpha forecast scale multiplier


def get_df(symbol, days=260):
    try:
        if symbol in OBS_ONLY:
            return get_index_daily_data(symbol, days=days)
        return get_stock_daily_data(symbol, days=days)
    except Exception:
        return None


def series(df, col="close"):
    if df is None or col not in df or len(df) < 40:
        return None
    s = df[col].astype(float)
    try:
        s.index = pd.to_datetime(df["date"])
    except Exception:
        s.index = pd.RangeIndex(len(s))
    return s


def beta_last(y, x, win=60, min_obs=20):
    """Rolling-window beta of y on x; last window value."""
    q = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna().tail(win)
    if len(q) < min_obs:
        return None
    vx = float(q.x.var())
    if vx <= 1e-14:
        return None
    return float(q.y.cov(q.x) / vx)


def down_beta(y, x, win=60):
    q = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    q = q[q.x < 0].tail(win)
    if len(q) < 20:
        return None
    vx = float(q.x.var())
    if vx <= 1e-14:
        return None
    return float(q.y.cov(q.x) / vx)


def dd_duration_resid(c, r, r_spx):
    """log1p(calendar days since 120d high) - spx_beta * zscore(mom120 skip5).

    Canonical form from factor library dd_duration_120_resid (direction -1).
    """
    try:
        hi = c.rolling(120).max()
        if isinstance(c.index, pd.DatetimeIndex):
            last_high = c.index.to_series().where(c == hi).ffill()
            # pandas>=2: (DatetimeIndex - Series) -> Series of Timedelta; use .dt.days
            dur = np.log1p((c.index - last_high).dt.days.fillna(0).astype(float))
        else:
            pos = pd.Series(np.arange(len(c)), index=c.index)
            dur = np.log1p((pos - pos.where(c == hi).ffill()).fillna(0).astype(float))
        mom = c.shift(5) / c.shift(125) - 1.0
        zmom = (mom - mom.rolling(250).mean()) / mom.rolling(250).std()
        b = beta_last(r, r_spx)
        v = float(dur.iloc[-1]) - (b * float(zmom.iloc[-1]) if b is not None else 0.0)
        return v if np.isfinite(v) else None
    except Exception:
        return None


def cs_rank(values, assets):
    """Cross-sectional rank in [0,1]; missing -> 0.5."""
    valid = sorted((float(v), a) for a, v in values.items()
                   if v is not None and np.isfinite(float(v)))
    out = {a: 0.5 for a in assets}
    n = max(1, len(valid) - 1)
    for i, (_, a) in enumerate(valid):
        out[a] = i / n
    return out


def detect_frozen(close, lookback=FROZEN_LOOKBACK):
    """Assets whose price has not moved (<=2 unique closes) over the window."""
    out = set()
    for a, c in close.items():
        if c is None:
            continue
        q = c.dropna().tail(lookback)
        if len(q) >= 20 and q.nunique() <= 2:
            out.add(a)
    return out


def apply_frozen_override(w, assets, frozen, cap=CAP, floor=FROZEN_FLOOR):
    """Floor frozen assets at `floor`, redistribute the rest among live assets."""
    if not frozen or len(frozen) >= len(assets) - 1:
        return w
    w = dict(w)
    live = [a for a in assets if a not in frozen]
    for a in frozen:
        w[a] = floor
    tot = sum(w.values())
    if tot <= 0:
        return {a: 1.0 / len(assets) for a in assets}
    w = {a: x / tot for a, x in w.items()}
    for _ in range(200):
        excess = sum(max(0.0, w[a] - cap) for a in live)
        if excess < 1e-12:
            break
        for a in live:
            w[a] = min(cap, w[a])
        room = [a for a in live if w[a] < cap - 1e-9]
        if not room:
            break
        p = {a: max(w[a], 1e-9) for a in room}
        den = sum(p.values())
        for a in room:
            w[a] += excess * p[a] / den
    tot = sum(w.values())
    if tot <= 0:
        return {a: 1.0 / len(assets) for a in assets}
    w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())  # float guard
    return {a: max(0.0, float(x)) for a, x in w.items()}


def risk_trim(w, assets, live, stress, eq_cap=EQ_CAP, eth_cap=ETH_CAP, cap=CAP):
    """Stress de-risking: cap live equity complex and ETH, water-fill the rest.

    Applied AFTER the frozen override so the caps act on the true final
    weights (before the override the frozen assets' phantom weights dominate).
    """
    if not stress:
        return w
    w = dict(w)
    eq = [a for a in EQ_ASSETS if a in live]
    for _ in range(300):
        excess = 0.0
        for a in assets:
            c = cap
            if a == "ETH" and a in live:
                c = min(c, eth_cap)
            if w[a] > c:
                excess += w[a] - c
                w[a] = c
        s_eq = sum(w[a] for a in eq)
        if s_eq > eq_cap:
            excess += s_eq - eq_cap
            for a in eq:
                w[a] *= eq_cap / max(s_eq, 1e-12)
        if excess < 1e-12:
            break
        room = []
        for a in assets:
            if a not in live:
                continue
            c = cap
            if a == "ETH":
                c = min(c, eth_cap)
            if a in eq:
                if sum(w[x] for x in eq) < eq_cap - 1e-9 and w[a] < c - 1e-9:
                    room.append(a)
            elif w[a] < c - 1e-9:
                room.append(a)
        if not room:
            break
        p = {a: max(w[a], 1e-9) for a in room}
        den = sum(p.values())
        if den <= 0:
            break
        for a in room:
            w[a] += excess * p[a] / den
    tot = sum(w.values())
    if tot <= 0:
        w = {a: 1.0 / len(assets) for a in assets}
    else:
        w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())  # float guard
    return {a: max(0.0, float(x)) for a, x in w.items()}


def commodity_guard(w, assets, live, cap=CAP, wti_cap=WTI_CAP,
                       comm_cap=COMM_CAP, copper_cap=COPPER_CAP, eth_cap=ETH_CAP_ALL):
    """Trader guard (2031-04-03): cap WTI and the live commodity complex.

    Fired after WTI -14% in the 03-20..04-03 block and a third consecutive
    negative attribution from the commodity-beta cluster (comm_basket_beta_60,
    dxy_beta_cond_60x20) while the Screener ensemble stayed unchanged. Caps
    WTI at 6% and XAU+COPPER+WTI combined at 36%; freed weight water-fills to
    remaining live assets (per-asset cap preserved). Applied AFTER the frozen
    override and risk_trim so it acts on the true final weights.
    """
    w = dict(w)
    comm = [a for a in ("XAU", "COPPER", "WTI") if a in live]
    for _ in range(300):
        excess = 0.0
        for a in assets:
            c = cap
            if a == "WTI" and a in live:
                c = min(c, wti_cap)
            if a == "COPPER" and a in live:
                c = min(c, copper_cap)
            if a == "ETH" and a in live:
                c = min(c, eth_cap)
            if w[a] > c:
                excess += w[a] - c
                w[a] = c
        s_comm = sum(w[a] for a in comm)
        if s_comm > comm_cap:
            excess += s_comm - comm_cap
            for a in comm:
                w[a] *= comm_cap / max(s_comm, 1e-12)
        if excess < 1e-12:
            break
        room = []
        for a in assets:
            if a not in live:
                continue
            c = cap
            if a == "WTI":
                c = min(c, wti_cap)
            if a == "COPPER":
                c = min(c, copper_cap)
            if a == "ETH":
                c = min(c, eth_cap)
            if w[a] < c - 1e-9:
                room.append(a)
        if not room:
            break
        p = {a: max(w[a], 1e-9) for a in room}
        den = sum(p.values())
        if den <= 0:
            break
        for a in room:
            w[a] += excess * p[a] / den
    tot = sum(w.values())
    if tot <= 0:
        w = {a: 1.0 / len(assets) for a in assets}
    else:
        w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())  # float guard
    return {a: max(0.0, float(x)) for a, x in w.items()}



def tech_guard(w, assets, live, cap=CAP, tech_cap=TECH_CAP):
    """Trader guard (2034-04-27): cap the live tech complex NDX+SOX+000688.SH.

    Fired after 3 consecutive negative blocks (02-02 -3.04%, 04-13 -2.13%,
    04-27 -1.79%) where NDX/SOX/000688 were the dominant drags while the
    equity-stress flag (VIX>=30 or eq21d<-5%) was NOT tripped (VIX 22.7), so
    risk_trim left the tech complex near ~33%. Caps NDX+SOX+000688.SH
    combined at TECH_CAP; freed weight water-fills to remaining live assets
    (per-asset cap preserved). Applied AFTER commodity_guard.
    """
    w = dict(w)
    tech = [a for a in TECH_ASSETS if a in live]
    for _ in range(300):
        excess = 0.0
        for a in assets:
            if w[a] > cap:
                excess += w[a] - cap
                w[a] = cap
        s_tech = sum(w[a] for a in tech)
        if s_tech > tech_cap:
            excess += s_tech - tech_cap
            for a in tech:
                w[a] *= tech_cap / max(s_tech, 1e-12)
        if excess < 1e-12:
            break
        room = [a for a in assets if a in live and w[a] < cap - 1e-9]
        if not room:
            break
        p = {a: max(w[a], 1e-9) for a in room}
        den = sum(p.values())
        if den <= 0:
            break
        for a in room:
            w[a] += excess * p[a] / den
    tot = sum(w.values())
    if tot <= 0:
        w = {a: 1.0 / len(assets) for a in assets}
    else:
        w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())  # float guard
    return {a: max(0.0, float(x)) for a, x in w.items()}


def apply_all_caps(w, assets, live, stress=False, cap=CAP, spx_cap=SPX_CAP,
                     xau_cap=XAU_CAP, wti_cap=WTI_CAP, copper_cap=COPPER_CAP,
                     eth_cap=ETH_CAP_ALL, tech_cap=TECH_CAP, comm_cap=COMM_CAP,
                     eq_cap=EQ_CAP):
    """Comprehensive final cap guard (2035-04-12).

    Replaces the sequential commodity_guard -> tech_guard -> spx_guard tail.
    The sequential stack let each guard's water-fill redistribute freed weight
    to assets already at their sub-caps (the last guard only knew per-asset
    0.18 + SPX 0.12), so the 03-29 proposal breached COPPER/ETH/WTI/tech/comm
    caps (COPPER 12.9%, ETH 5.2%, WTI 5.2%, tech 26.4%, comm 35%). This guard
    enforces every cap in ONE water-fill loop and does NOT destructively
    renormalize (which had also drifted the frozen floor 0.5% -> ~1%): the
    frozen floor stays at FROZEN_FLOOR and sum-to-1 is restored by filling
    remaining room, not by scaling capped assets above their limits.
    """
    w = dict(w)
    tech = [a for a in TECH_ASSETS if a in live]
    comm = [a for a in ("XAU", "COPPER", "WTI") if a in live]
    eq = [a for a in EQ_ASSETS if a in live]

    def cfor(a):
        c = cap
        if a == "SPX":
            c = min(c, spx_cap)
        if a == "XAU":
            c = min(c, xau_cap)
        if a == "WTI":
            c = min(c, wti_cap)
        if a == "COPPER":
            c = min(c, copper_cap)
        if a == "ETH":
            c = min(c, eth_cap)
        return c

    def room_ok(a, w):
        if w[a] >= cfor(a) - 1e-9:
            return False
        if a in tech and sum(w[x] for x in tech) >= tech_cap - 1e-9:
            return False
        if a in comm and sum(w[x] for x in comm) >= comm_cap - 1e-9:
            return False
        if stress and a in eq and sum(w[x] for x in eq) >= eq_cap - 1e-9:
            return False
        return True

    for _ in range(500):
        excess = 0.0
        for a in assets:
            c = cfor(a)
            if w[a] > c:
                excess += w[a] - c
                w[a] = c
        if stress:
            s_eq = sum(w[a] for a in eq)
            if s_eq > eq_cap:
                excess += s_eq - eq_cap
                for a in eq:
                    w[a] *= eq_cap / max(s_eq, 1e-12)
        s_tech = sum(w[a] for a in tech)
        if s_tech > tech_cap:
            excess += s_tech - tech_cap
            for a in tech:
                w[a] *= tech_cap / max(s_tech, 1e-12)
        s_comm = sum(w[a] for a in comm)
        if s_comm > comm_cap:
            excess += s_comm - comm_cap
            for a in comm:
                w[a] *= comm_cap / max(s_comm, 1e-12)
        if excess < 1e-12:
            break
        room = [a for a in assets if a in live and room_ok(a, w)]
        if not room:
            break
        p = {a: max(w[a], 1e-9) for a in room}
        den = sum(p.values())
        if den <= 0:
            break
        for a in room:
            w[a] += excess * p[a] / den

    # restore exact sum-to-1 by filling remaining room (no destructive scaling)
    tot = sum(w.values())
    diff = 1.0 - tot
    if abs(diff) > 1e-9:
        room = [a for a in assets if a in live and room_ok(a, w)]
        if room and diff > 0:
            p = {a: max(w[a], 1e-9) for a in room}
            den = sum(p.values())
            if den > 0:
                for a in room:
                    w[a] += diff * p[a] / den
        tot = sum(w.values())
        w[assets[-1]] += 1.0 - tot  # float guard (<=1e-12)
    return {a: max(0.0, float(x)) for a, x in w.items()}


def is_block_start():
    try:
        d = json.load(open("../persistent/date.json"))
        tds = d.get("trading_days", [])
        cur = d.get("current_date")
        if ONLINE_START in tds and cur in tds:
            return (tds.index(cur) - tds.index(ONLINE_START)) % HORIZON == 0
    except Exception:
        pass
    return True


def build_weights(score, assets, panel, def_floor, spread, cap=CAP):
    """Rank-linear tilt * inverse-vol (sqrt), defensive floor, water-fill cap."""
    order = sorted(assets, key=lambda a: (-score[a], a))
    lin = {a: 1.0 - i / max(1, len(order) - 1) for i, a in enumerate(order)}
    vols = {a: max(float(panel[a].tail(20).std()), 0.003) for a in assets}
    vmed = float(np.median([vols[a] for a in assets]))
    pref = {a: (1.0 + spread * lin[a]) * math.sqrt(vmed / vols[a]) for a in assets}

    total = sum(max(0.0, float(x)) for x in pref.values())
    w = {a: max(0.0, float(pref[a])) / total for a in assets}

    # defensive floor (risk posture), then renormalize
    for a in DEF:
        w[a] = max(w[a], def_floor)
    tot = sum(w.values())
    if tot > 0:
        w = {a: x / tot for a, x in w.items()}

    # water-fill cap: cap at `cap`, redistribute excess proportionally to pref
    for _ in range(200):
        excess = sum(max(0.0, x - cap) for x in w.values())
        if excess < 1e-12:
            break
        w = {a: min(cap, x) for a, x in w.items()}
        room = [a for a, x in w.items() if x < cap - 1e-9]
        if not room:
            break
        p = {a: max(0.0, pref.get(a, 0.0)) for a in room}
        den = sum(p.values())
        if den <= 0:
            p = {a: 1.0 for a in room}
            den = len(room)
        for a in room:
            w[a] += excess * p[a] / den

    tot = sum(w.values())
    if tot <= 0:
        w = {a: 1.0 / len(assets) for a in assets}
    else:
        w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())  # float guard
    return {a: max(0.0, float(x)) for a, x in w.items()}


def load_ensemble():
    for path in ("factors/factor_ensemble.json", "factor_ensemble.json"):
        p = Path(__file__).parent / path
        try:
            ens = json.loads(p.read_text())
            sel = ens.get("selected_factors", [])
            if sel:
                return [(str(s["factor_id"]), float(s["weight"]), int(s["direction"]))
                        for s in sel if isinstance(s, dict) and s.get("factor_id")]
        except (OSError, ValueError, TypeError):
            continue
    return []


@register_hook
def strategy_hook():
    if not is_block_start():
        return  # mid-block: no new target; sim marks/processes orders

    assets = list(get_account_dict()["watch_list"])
    frames = {a: get_df(a) for a in assets}
    close = {a: series(frames[a]) for a in assets}
    open_ = {a: series(frames[a], "open") for a in assets}
    if any(c is None for c in close.values()):
        return

    frozen = detect_frozen(close)
    live = [a for a in assets if a not in frozen]

    ret = {a: close[a].pct_change() for a in assets}
    panel = pd.concat([ret[a].rename(a) for a in assets], axis=1, join="inner").dropna()
    if len(panel) < 70:
        return

    ens = load_ensemble()
    ens_ids = {fid for fid, _, _ in ens}

    r_spx = ret["SPX"]
    r_300 = ret["000300.SH"]
    d_cn = close["CN10Y"].pct_change()
    dxy = series(get_df("DXY"))
    r_dxy = dxy.pct_change() if dxy is not None else None
    vix = series(get_df("VIX"))
    r_vix = vix.pct_change() if vix is not None else None
    comm_basket = panel[["XAU", "COPPER", "WTI"]].mean(axis=1)  # ew commodity basket

    # ---- factor signals (only those in the active ensemble) --------------
    sig = {fid: {} for fid in ens_ids}
    for a in assets:
        c, o, r = close[a], open_[a], ret[a]
        if "down_beta_60" in ens_ids:
            sig["down_beta_60"][a] = down_beta(r, r_spx)
        if "spx_beta_60" in ens_ids:
            sig["spx_beta_60"][a] = beta_last(r, r_spx)
        if "hs300_beta_60" in ens_ids:
            sig["hs300_beta_60"][a] = beta_last(r, r_300)
        if "cn10y_beta_60" in ens_ids:
            sig["cn10y_beta_60"][a] = beta_last(r, d_cn)
        if "vol_adj_mom_20_60" in ens_ids:
            sig["vol_adj_mom_20_60"][a] = (
                (c.iloc[-6] / c.iloc[-26] - 1.0) / max(float(r.tail(60).std()), 1e-6)
                if len(c) >= 30 else None)
        if "dxy_beta_cond_60x20" in ens_ids:
            if r_dxy is not None:
                b = beta_last(r, r_dxy)
                sig["dxy_beta_cond_60x20"][a] = (
                    b * (dxy.iloc[-1] / dxy.iloc[-21] - 1.0) if b is not None else None)
            else:
                sig["dxy_beta_cond_60x20"][a] = None
        if "vix_beta_cond_60x20" in ens_ids:
            if r_vix is not None and len(vix) >= 25:
                b = beta_last(r, r_vix)
                sig["vix_beta_cond_60x20"][a] = (
                    b * (vix.iloc[-1] / vix.iloc[-21] - 1.0) if b is not None else None)
            else:
                sig["vix_beta_cond_60x20"][a] = None
        if "hilo_vol_ratio_20" in ens_ids:
            # (max20-min20)/close / std(ret,20)
            if len(c) >= 25:
                rng = (c.rolling(20).max() - c.rolling(20).min()) / c
                rv = r.rolling(20).std()
                q = (rng / rv).dropna()
                sig["hilo_vol_ratio_20"][a] = float(q.iloc[-1]) if len(q) else None
            else:
                sig["hilo_vol_ratio_20"][a] = None
        if "intraday_ret_skew_20" in ens_ids:
            if o is not None:
                ir = (c / o - 1.0).dropna().tail(20)
                sig["intraday_ret_skew_20"][a] = float(ir.skew()) if len(ir) >= 5 else None
            else:
                sig["intraday_ret_skew_20"][a] = None
        if "comm_basket_beta_60" in ens_ids:
            sig["comm_basket_beta_60"][a] = beta_last(r, comm_basket)
        if "vol_of_vol20x60" in ens_ids:
            rv20 = r.rolling(20).std()
            sig["vol_of_vol20x60"][a] = (
                float(rv20.tail(60).std()) if len(rv20.dropna()) >= 40 else None)
        if "vol_regime_switch_20x60" in ens_ids:
            rv20 = r.rolling(20).std()
            above = (rv20 > rv20.rolling(60).median()).astype(float)
            flips = above.diff().abs().rolling(60).mean().dropna()
            sig["vol_regime_switch_20x60"][a] = float(flips.iloc[-1]) if len(flips) else None
        if "dd_duration_120_resid" in ens_ids:
            sig["dd_duration_120_resid"][a] = dd_duration_resid(c, r, r_spx)

    # ---- composite score (direction preserved) --------------------------
    score = {a: 0.0 for a in assets}
    for fid, w, d in ens:
        rk = cs_rank(sig.get(fid, {}), assets)
        for a in assets:
            score[a] += w * d * rk[a]

    # ---- regime posture (live assets only; frozen zeros would dilute) ----
    lp = panel[live] if live else panel
    market = lp.mean(axis=1)
    wealth = (1.0 + market).cumprod()
    mdd = float((wealth / wealth.rolling(60).max() - 1.0).tail(20).min())
    mkt20 = float(market.tail(20).mean())
    vol20 = float(lp.tail(20).std().mean())
    vol_med = float(lp.tail(120).std().median(axis=0))
    risk_off = (mkt20 < 0.0 and mdd < -0.025) or (vol20 > 1.25 * max(vol_med, 1e-6))
    risk_on = mkt20 > 0.0 and mdd > -0.015
    def_floor = 0.18 if risk_off else (0.11 if risk_on else 0.13)
    spread = 2.0 if risk_off else (3.0 if risk_on else 2.0)

    # ---- equity stress flag (VIX or live-equity 21d breadth) -------------
    vix_level = float(vix.iloc[-1]) if vix is not None and len(vix) else None
    eq_live = [a for a in EQ_ASSETS if a in live]
    eq_ret21 = (float(np.mean([close[a].iloc[-1] / close[a].iloc[-22] - 1.0
                              for a in eq_live])) if eq_live else 0.0)
    stress = risk_off and ((vix_level is not None and vix_level >= VIX_STRESS)
                           or eq_ret21 < EQ_RET21_STRESS)
    if stress:
        print(f"[trader] EQUITY STRESS: VIX={vix_level:.1f} "
              f"live-eq 21d mean={eq_ret21 * 100:.2f}% -> eq<={EQ_CAP:.2f}, "
              f"ETH<={ETH_CAP:.2f}")

    # ---- target weights: full 15-asset, sum 1, cash 0 --------------------
    weights = build_weights(score, assets, panel, def_floor, spread)
    weights = apply_frozen_override(weights, assets, frozen)
    weights = risk_trim(weights, assets, live, stress)
    weights = apply_all_caps(weights, assets, live, stress=stress)
    print(f"[trader] commodity guard: XAU={weights['XAU'] * 100:.1f}% "
          f"COPPER={weights['COPPER'] * 100:.1f}% WTI={weights['WTI'] * 100:.1f}% "
          f"complex={sum(weights[a] for a in ('XAU','COPPER','WTI')) * 100:.1f}% "
          f"ETH={weights['ETH'] * 100:.1f}% "
          f"(WTI cap {WTI_CAP * 100:.0f}%, COPPER cap {COPPER_CAP * 100:.0f}%, "
          f"complex cap {COMM_CAP * 100:.0f}%, ETH cap {ETH_CAP_ALL * 100:.0f}%)")
    techw = sum(weights[a] for a in TECH_ASSETS if a in live)
    print(f"[trader] SPX cap: SPX={weights['SPX'] * 100:.1f}% (cap {SPX_CAP * 100:.0f}%)")
    print(f"[trader] tech guard: NDX={weights['NDX'] * 100:.1f}% "
          f"SOX={weights['SOX'] * 100:.1f}% 000688={weights['000688.SH'] * 100:.1f}% "
          f"complex={techw * 100:.1f}% (cap {TECH_CAP * 100:.0f}%)")
    if len(frozen):
        print(f"[trader] frozen assets floored at {FROZEN_FLOOR:.3f}: "
              f"{sorted(frozen)}; live={sorted(live)}")
        print(f"[trader] frozen total weight={sum(weights[a] for a in frozen)*100:.2f}%")
    if stress:
        eqw = sum(weights[a] for a in EQ_ASSETS)
        print(f"[trader] final live-equity weight={eqw * 100:.1f}% "
              f"XAU={weights['XAU'] * 100:.1f}% COPPER={weights['COPPER'] * 100:.1f}% "
              f"WTI={weights['WTI'] * 100:.1f}% ETH={weights['ETH'] * 100:.1f}%")

    # ---- forecast returns: migration-implied edge ------------------------
    # forecast_i = k * (w_target,i - w_current,i), k = FC_K_MULT * daily vol.
    # Conviction: moving toward the FINAL risk-adjusted target has positive
    # expected value proportional to the migration size, so the helper's
    # signed gross edge (sum of delta_w * forecast) is positive for any
    # meaningful rebalance while near-no-op proposals still get skipped.
    # 2028-09-07: replaced the old implied-alpha forecast k*(w-1/N), which
    # produced NEGATIVE edge whenever a defensive rebalance trimmed an
    # overweight-but-still-above-equal-weight asset (XAU 25.5%->18% cut with
    # alpha +11.3%*k), so every proposal since 2028-04-06 was rejected by the
    # 3bp gate and the portfolio froze in a stale commodity-heavy allocation.
    scale = float(lp.tail(252).std(axis=1, ddof=0).median()) if len(lp) >= 30 else 0.01
    if not math.isfinite(scale) or scale <= 0:
        scale = 0.01
    k = FC_K_MULT * scale
    cur_w = {a: 1.0 / len(assets) for a in assets}
    try:
        acct_now = get_account_dict()
        mv = {p["symbol"]: float(p.get("market_value", 0.0))
              for p in acct_now.get("positions", [])}
        nav = float(acct_now.get("net_assets", 0.0))
        if nav > 0 and sum(mv.values()) > 0:
            cur_w = {a: mv.get(a, 0.0) / nav for a in assets}
    except Exception:
        pass
    forecast = {a: float(k * (weights[a] - cur_w[a])) for a in assets}

    rebalance_to_weights(
        weights,
        forecast_returns=forecast,
        factor_ids=[fid for fid, _, _ in ens][:10],
        horizon_days=HORIZON,
    )
