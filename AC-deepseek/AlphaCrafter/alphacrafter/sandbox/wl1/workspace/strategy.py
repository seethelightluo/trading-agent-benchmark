"""Trader strategy v19 - Screener 6-factor quality_ic_tilt ensemble.

Ensemble (2034-06-16 refresh): miner2_20260715_nclv_1d (.20,+)
| miner2_20260715_rev_5d (.18,+) | miner2_20260715_rev_2d (.17,+)
| vix_beta_cond_60x20 (.23,-) | vol_of_vol20x60 (.13,+)
| mom_120d_skip5 (.09,+). MODEST refresh: ADD rev_5d (best fresh 5d-horizon
reversal quality; complements 1d/2d and fits the 10-td block cadence), TRIM
mom_120d .13->.09 (2nd straight momentum-top whipsaw block), RAISE vix_beta
.21->.23 (crisis guard; VIX 66.3), trim nclv_1d .24->.20, rev_2d .21->.17,
vol_of_vol .21->.13. Reversal family combined 0.55, defensive/quality 0.36,
momentum 0.09. Regime: SIDEWAYS-to-mild-bull on 20d with HIGH vol (mean ann
27.1%) + HIGH dispersion (1.97% daily) - strong short-horizon reversal regime.

Live weights are loaded from factor_ensemble.json at import, so this header
is documentation only.

Cross-sectional rank composite over the 15-name tradable panel; fully
invested, non-negative weights sum to 1, no cash sleeve. One atomic rebalance
proposal per 10-trading-day block via rebalance_to_weights with aligned
forecast returns so the execution gate (gross edge > one-way turnover * 3bp)
decides. Bear regime adds a modest defensive tilt (XAU/US10Y/CN10Y).

Portfolio guards (all factor- and trend-agnostic where noted):
v7/v10/v12 momentum top-pick caps (GUARD_CAP 6%), v8 composite MA guard (8%)
+ value-trap de-rank, v9 crypto pair cap (12%), v11 cyclical-commodity pair
cap (12%), v13 composite top-2 cap (9%) + v17 re-apply after pair caps,
v14 China-equity pair cap (12%), CAP_W 16% single-name hard cap.
v18 (2034-10-06): GLOBAL_CAP 10% single-name hard cap applied LAST after the
pair-cap/v13 convergence loop - ends pair-cap redistribution concentration
(XAU 10.28 in 0922, 10.55 in 0908, 10.50 in 0507, NDX 10.90 in 0729, NDX 9.99
in 0128-0211; excess redistributed proportionally to the remaining names).

Cadence: proposals fire on grid days (idx % 10 == 0 from ONLINE_START) or on
any hook call >= 10 trading days after the last proposal (drift fallback),
tracked via trader_state.json.
"""
import json
import math
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import (get_account_dict, get_stock_daily_data,
                                    rebalance_to_weights, register_hook)

ONLINE_START = "2026-07-16"
DATE_FILE = "../persistent/date.json"
VIX_FILE = "../persistent/index_data/VIX.csv"
STATE_FILE = "trader_state.json"
DATA_DAYS = 170          # enough for mom_120d_skip5 (shift(125)) + buffers
MIN_ROWS = 140
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
CRYPTO = {"BTC", "ETH"}
CYCLICAL_COMMOD = {"WTI", "COPPER"}
CHINA_EQ = {"000300.SH", "000688.SH"}   # v14: China equity pair (HSI/CN10Y excluded)
CAP_W = 0.16
GUARD_CAP = 0.06         # v7 cap for momentum top-picks below 20d MA
COMP_GUARD_CAP = 0.08    # v8 cap for ANY below-20d-MA asset with weight > 8%
CRYPTO_CAP = 0.12        # v9 combined BTC+ETH weight cap
COMMOD_CAP = 0.12        # v11 combined WTI+COPPER weight cap (14->12 on 2032-05-21: WTI 3rd down block in 4, COPPER 2nd straight, pair -1.19pp dominant drag last block)
CHINA_CAP = 0.12         # v14 combined 000300+000688 weight cap
COMP_TOP2_CAP = 0.090     # v13 composite-rank top-2 cap (9.5->9.0 on 2032-08-13 v16: 2nd straight SOX top-weight drag block)
GLOBAL_CAP = 0.10        # v18 single-name hard cap applied AFTER all pair caps
MOM_TOP_RANK = 0.60      # momentum rank threshold for the v7 guard
MOM_TOP2_RANK = 0.86  # v12: top-2 momentum names (rank >= .86 of 15)
TRAP_PENALTY = 0.50      # v8 value-trap score penalty as fraction of mom weight

_VIX_CACHE = {}


def _load_ensemble():
    with open("factor_ensemble.json") as f:
        ens = json.load(f)
    return [(x["factor_id"], float(x["weight"]), int(x["direction"]))
            for x in ens["selected_factors"]]


FACTORS = _load_ensemble()


def _mom_weight():
    for fid, w, _ in FACTORS:
        if "mom_120d_skip5" in fid:
            return w
    return 0.30


def _today_and_calendar():
    with open(DATE_FILE) as f:
        d = json.load(f)
    return str(d["current_date"]), d.get("trading_days", [])


def _is_rebalance_day(cur, tds):
    if cur < ONLINE_START or cur not in tds or ONLINE_START not in tds:
        return False
    return (tds.index(cur) - tds.index(ONLINE_START)) % 10 == 0


def _last_proposal_date(tds):
    """Last proposal date: state file first, else account last executed rebal."""
    try:
        with open(STATE_FILE) as f:
            last = json.load(f).get("last_proposal_date")
            if last and last in tds:
                return last
    except Exception:
        pass
    try:
        acc = get_account_dict()
        last = acc.get("last_rebalance_date")
        if last and last in tds:
            return last
    except Exception:
        pass
    return None


def _should_propose(cur, tds):
    if cur < ONLINE_START or cur not in tds:
        return False
    if _is_rebalance_day(cur, tds):           # fixed grid still honoured
        return True
    last = _last_proposal_date(tds)           # drift-tolerant fallback
    if last is None:
        return True                           # first online proposal
    return (tds.index(cur) - tds.index(last)) >= 10


def _persist_proposal(cur):
    try:
        with open(STATE_FILE, "w") as f:
            json.dump({"last_proposal_date": cur}, f)
    except Exception:
        pass


def _fetch(assets):
    frames = {}
    for a in assets:
        try:
            df = get_stock_daily_data(symbol=a, days=DATA_DAYS)
            if df is None or len(df) < MIN_ROWS:
                frames[a] = None
                continue
            df = df.copy()
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
            frames[a] = df
        except Exception:
            frames[a] = None
    return frames


def _vix_close(cur):
    if cur in _VIX_CACHE:
        return _VIX_CACHE[cur]
    try:
        vix = pd.read_csv(VIX_FILE)
        vix["date"] = pd.to_datetime(vix["date"])
        vix = vix[vix["date"] <= pd.Timestamp(cur)].sort_values("date")
        s = vix.set_index("date")["close"].astype(float)
        _VIX_CACHE[cur] = s
        return s
    except Exception:
        _VIX_CACHE[cur] = None
        return None


def _asset_factor(df, fid, cur):
    """Return the factor Series on df's date index (or None if unsupported)."""
    o = df["open"].astype(float)
    h = df["high"].astype(float)
    l = df["low"].astype(float)
    c = df["close"].astype(float)
    if fid.endswith("nclv_1d"):
        return -(c - l) / (h - l)
    if fid.endswith("nclv_2d"):
        return -(c - l.rolling(2).min()) / (h.rolling(2).max() - l.rolling(2).min())
    if fid.endswith("nclv_3d"):
        return -(c - l.rolling(3).min()) / (h.rolling(3).max() - l.rolling(3).min())
    if fid.endswith("rev_1d"):
        return -np.log(c / c.shift(1))
    if fid.endswith("rev_2d"):
        return -np.log(c / c.shift(2))
    if fid.endswith("rev_5d"):
        return -np.log(c / c.shift(5))
    if fid.endswith("nbody_1d"):
        return -(c - o) / (h - l)
    if "mom_120d_skip5" in fid:
        return c.shift(5) / c.shift(125) - 1.0
    if fid == "vol_of_vol20x60":
        return c.pct_change().rolling(20).std().rolling(60).std()
    if fid == "vix_beta_cond_60x20":
        vixc = _vix_close(cur)
        if vixc is None or len(vixc) < 90:
            return None
        v = vixc.reindex(df.index).ffill()
        ar = c.pct_change()
        vr = v.pct_change()
        beta = ar.rolling(60).cov(vr) / vr.rolling(60).var()
        vm = v / v.shift(20) - 1.0
        return -beta * vm
    return None


def _factor_values(frames, fid, cur):
    out = {}
    for a, df in frames.items():
        if df is None:
            out[a] = None
            continue
        try:
            s = _asset_factor(df, fid, cur)
            if s is None:
                out[a] = None
                continue
            s = s.replace([np.inf, -np.inf], np.nan)
            v = float(s.iloc[-1])
            out[a] = v if math.isfinite(v) else None
        except Exception:
            out[a] = None
    return out


def _ranks(values, assets):
    valid = sorted((float(v), a) for a, v in values.items()
                   if v is not None and math.isfinite(float(v)))
    out = {a: 0.5 for a in assets}
    n = len(valid)
    for i, (_, a) in enumerate(valid):
        out[a] = i / max(1, n - 1)
    return out


def _scores(frames, assets, cur):
    """Composite score with dead-factor renormalization.

    Factors with fewer than 8 valid cross-sectional values (e.g. the
    vix_beta_cond_60x20 risk guard when VIX is pinned flat at 9.0 since
    2031-02-13, which zeroes its 60d return variance) are dropped and the
    surviving factor weights are renormalized to sum to 1, preserving the
    Screener ensemble intent instead of degrading to equal weights.
    """
    score = {a: 0.0 for a in assets}
    used = 0
    used_w = 0.0
    for fid, w, direction in FACTORS:
        vals = _factor_values(frames, fid, cur)
        if sum(1 for v in vals.values() if v is not None) < 8:
            continue
        r = _ranks(vals, assets)
        for a in assets:
            score[a] += w * (r[a] if direction > 0 else 1.0 - r[a])
        used += 1
        used_w += w
    if used_w > 0:                      # renormalize to ensemble weights sum 1
        for a in assets:
            score[a] /= used_w
    return score, used


def _regime(frames, assets):
    rets = []
    for a in assets:
        df = frames.get(a)
        if df is not None and len(df) >= 25:
            rets.append(float(df["close"].pct_change().tail(20).mean()))
    if not rets:
        return "side"
    m = float(np.mean(rets))
    return "bull" if m > 0.010 else ("bear" if m < -0.010 else "side")


def _below_ma(frames, assets):
    """Assets whose last close is below their 20d simple moving average."""
    out = set()
    for a in assets:
        df = frames.get(a)
        if df is not None and len(df) >= 25:
            close = float(df["close"].iloc[-1])
            ma20 = float(df["close"].rolling(20).mean().iloc[-1])
            if math.isfinite(ma20) and close < ma20:
                out.add(a)
    return out


def _weights(scores, assets, regime):
    order = sorted(assets, key=lambda a: (scores[a], a))
    n = len(assets)
    raw = {}
    for i, a in enumerate(order):
        r = i / max(1, n - 1)
        raw[a] = 0.02 + 0.10 * r          # rank-linear 2%..12% pre-normalization
    if regime == "bear":
        tilt = 0.045
        defs = [a for a in DEFENSIVE if a in assets]
        nd = [a for a in assets if a not in DEFENSIVE]
        dsum = sum(raw[a] for a in defs)
        nsum = sum(raw[a] for a in nd)
        for a in defs:
            raw[a] += tilt * (raw[a] / dsum if dsum > 0 else 1.0 / len(defs))
        for a in nd:
            raw[a] -= tilt * (raw[a] / nsum if nsum > 0 else 1.0 / len(nd))
    tot = sum(raw.values())
    w = {a: max(0.0, x / tot) for a, x in raw.items()}
    for _ in range(80):                   # cap-and-redistribute
        excess = sum(max(0.0, x - CAP_W) for x in w.values())
        if excess < 1e-12:
            break
        w = {a: min(CAP_W, x) for a, x in w.items()}
        room = [a for a in w if w[a] < CAP_W - 1e-12]
        if not room:
            break
        den = sum(w[a] for a in room) + 1e-12
        for a in room:
            w[a] += excess * w[a] / den
    tot = sum(w.values())
    w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())   # exact sum-to-1 fix
    return w


def _composite_top2_cap(w, assets, scores):
    """v13: cap the top-2 composite-score names at COMP_TOP2_CAP (9.5%).

    Block 0330-0413: NDX 11.5% / N225 10.8% were top-2 COMPOSITE weights but
    not top-2 momentum names, so the v12 momentum cap missed them. With 10 of
    the last 11 blocks showing a top-pick whipsaw, composite-score leaders
    carry fat-tailed reversal risk regardless of factor origin (momentum OR
    reversal lifted). Factor-agnostic like v8/v9/v11; excess is redistributed
    proportionally to the remaining names.
    """
    order = sorted(assets, key=lambda a: (scores[a], a))
    top2 = set(order[-2:])
    for _ in range(80):                    # iterate until cap invariant holds
        penalized = {a for a in top2 if w[a] > COMP_TOP2_CAP + 1e-9}
        if not penalized:
            break
        excess = sum(w[a] - COMP_TOP2_CAP for a in penalized)
        for a in penalized:
            w[a] = COMP_TOP2_CAP
        room = [a for a in assets if a not in penalized]
        if not room:
            break
        den = sum(w[a] for a in room) + 1e-12
        for a in room:
            w[a] += excess * w[a] / den
    tot = sum(w.values())
    w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())
    return w


def _de_rank_value_traps(scores, frames, assets, cur):
    """v8: penalize below-20d-MA assets that also have negative 120d momentum.

    Reversal/nclv factors can lift broken-trend names (ETH had momentum rank
    14/15 yet carried 9.4% weight). A score penalty proportional to the
    momentum factor weight de-ranks them before weighting.
    """
    mom_vals = _factor_values(frames, "mom_120d_skip5", cur)
    below = _below_ma(frames, assets)
    pen = TRAP_PENALTY * _mom_weight()
    for a in assets:
        mv = mom_vals.get(a)
        if a in below and mv is not None and mv < 0:
            scores[a] -= pen
    return scores


def _composite_ma_guard(w, frames, assets):
    """v8: cap ANY asset with weight > 8% that trades below its 20d MA.

    Factor-origin agnostic: catches reversal/nclv-lifted names the momentum
    -only v7 guard misses (BTC 10.1%, ETH 9.4% both below MA20 last block).
    Excess is redistributed proportionally to the remaining names.
    """
    below = _below_ma(frames, assets)
    for _ in range(80):                    # iterate until cap invariant holds
        penalized = {a for a in assets if w[a] > COMP_GUARD_CAP + 1e-9 and a in below}
        if not penalized:
            break
        excess = sum(w[a] - COMP_GUARD_CAP for a in penalized)
        for a in penalized:
            w[a] = COMP_GUARD_CAP
        room = [a for a in assets if w[a] < COMP_GUARD_CAP - 1e-12 and a not in penalized]
        if not room:                       # no room: spread over all non-penalized
            room = [a for a in assets if a not in penalized]
        den = sum(w[a] for a in room) + 1e-12
        for a in room:
            w[a] += excess * w[a] / den
    tot = sum(w.values())
    w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())
    return w


def _ma_guard(w, frames, assets, cur):
    """v7+v10: cap momentum top-picks.

    v7 (Screener guard): extended momentum names below their short-term MA
    (e.g. COPPER) are the main post-rebalance whipsaw drag - cap at GUARD_CAP.
    v10/v12: the top-2 momentum names (rank >= MOM_TOP2_RANK) are capped at
    GUARD_CAP regardless of MA state. v10 closed the above-MA20 hole that WTI
    exploited in block 0331-0414 (top momentum name, above MA20, -21% crash on
    8.9% weight); v12 (2028-10-27) extended to rank-2 after SOX delivered
    -12.5% on 7.4% weight (9th top-pick whipsaw block in 10). Excess is
    redistributed proportionally to remaining names.
    """
    mom_vals = _factor_values(frames, "mom_120d_skip5", cur)
    mom_rank = _ranks(mom_vals, assets)
    below = _below_ma(frames, assets)
    for _ in range(80):                    # iterate until cap invariant holds
        penalized = {a for a in assets if w[a] > GUARD_CAP + 1e-9 and (
            (mom_rank[a] >= MOM_TOP_RANK and a in below) or
            (mom_rank[a] >= MOM_TOP2_RANK)  # v12: top-2, MA-agnostic
        )}
        if not penalized:
            break
        excess = sum(w[a] - GUARD_CAP for a in penalized)
        for a in penalized:
            w[a] = GUARD_CAP
        room = [a for a in assets if w[a] < GUARD_CAP - 1e-12 and a not in penalized]
        if not room:                       # no room: spread over all non-penalized
            room = [a for a in assets if a not in penalized]
        den = sum(w[a] for a in room) + 1e-12
        for a in room:
            w[a] += excess * w[a] / den
    tot = sum(w.values())
    w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())
    return w


def _crypto_cap(w, assets):
    """v9: cap combined BTC+ETH weight at 12%.

    Three consecutive blocks ended with a crypto top-weight drag (block
    0317-0331: ETH -25.9% on ~8% weight = entire block loss). ETH sat above
    its 20d MA with positive 120d momentum at the decision, so the v8 MA
    guards could not cap it. This guard is factor- and trend-agnostic: the
    combined crypto weight can never exceed CRYPTO_CAP; excess is
    redistributed proportionally to the 13 non-crypto names.
    """
    crypto = [a for a in assets if a in CRYPTO and a in w]
    csum = sum(w[a] for a in crypto)
    if csum <= CRYPTO_CAP + 1e-12:
        return w
    scale = CRYPTO_CAP / csum
    for a in crypto:
        w[a] *= scale
    excess = csum - CRYPTO_CAP
    room = [a for a in assets if a not in crypto]
    if room:
        den = sum(w[a] for a in room) + 1e-12
        for a in room:
            w[a] += excess * w[a] / den
    tot = sum(w.values())
    w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())
    return w


def _commod_cap(w, assets):
    """v11: cap combined cyclical-commodity weight (WTI+COPPER) at 14%.

    Block 0929-1013: WTI 9.4% + COPPER 8.0% (17.4% combined) both fell ~-7%
    (est. -1.29 combined contribution) - the block's dominant drag, the 2nd
    block the commodity pair carried ~17% into a synchronized drawdown (WTI
    also 2nd time as momentum top-pick whipsaw). Commodities remain below
    their 20d MA with negative 20d momentum at this decision. This guard is
    factor- and trend-agnostic: combined WTI+COPPER weight can never exceed
    COMMOD_CAP; excess is redistributed proportionally to the remaining
    names. XAU is deliberately excluded (defensive sleeve).
    """
    comm = [a for a in assets if a in CYCLICAL_COMMOD and a in w]
    csum = sum(w[a] for a in comm)
    if csum <= COMMOD_CAP + 1e-12:
        return w
    scale = COMMOD_CAP / csum
    for a in comm:
        w[a] *= scale
    excess = csum - COMMOD_CAP
    room = [a for a in assets if a not in comm]
    if room:
        den = sum(w[a] for a in room) + 1e-12
        for a in room:
            w[a] += excess * w[a] / den
    tot = sum(w.values())
    w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())
    return w


def _china_cap(w, assets):
    """v14: cap combined China-equity weight (000300+000688) at 12%.

    Block 0411-0425: 000688 sat just ABOVE its 20d MA at the 0410 decision
    (8.19% weight evaded the v8 8% below-MA cap) and collapsed -12.4% - the
    dominant block drag; 000300 was -3.57 in 1025-1108 on 8.4%. Both are
    China equity indices correlated in China risk-off. This guard is factor-
    and trend-agnostic: combined 000300+000688 weight can never exceed
    CHINA_CAP; excess is redistributed proportionally to the remaining names.
    HSI/CN10Y are deliberately excluded (HSI is HK with a flat-data artifact,
    CN10Y is a flat defensive rate).
    """
    cn = [a for a in assets if a in CHINA_EQ and a in w]
    csum = sum(w[a] for a in cn)
    if csum <= CHINA_CAP + 1e-12:
        return w
    scale = CHINA_CAP / csum
    for a in cn:
        w[a] *= scale
    excess = csum - CHINA_CAP
    room = [a for a in assets if a not in cn]
    if room:
        den = sum(w[a] for a in room) + 1e-12
        for a in room:
            w[a] += excess * w[a] / den
    tot = sum(w.values())
    w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())
    return w


def _global_cap(w, assets):
    """v18: single-name hard cap at GLOBAL_CAP (10%) applied LAST.

    Closes the residual concentration hole after pair-cap/v13 redistribution:
    multiple pair caps binding simultaneously push excess into uncapped names
    (XAU 10.28 in 0922 block, XAU 10.55 in 0908, NDX 10.90 in 0729, XAU 10.50
    in 0507, NDX 9.99 in 0128-0211). Factor-agnostic; excess is redistributed
    proportionally to the remaining names.
    """
    for _ in range(80):                    # iterate until cap invariant holds
        penalized = {a for a in assets if w[a] > GLOBAL_CAP + 1e-9}
        if not penalized:
            break
        excess = sum(w[a] - GLOBAL_CAP for a in penalized)
        for a in penalized:
            w[a] = GLOBAL_CAP
        room = [a for a in assets if a not in penalized]
        if not room:
            break
        den = sum(w[a] for a in room) + 1e-12
        for a in room:
            w[a] += excess * w[a] / den
    tot = sum(w.values())
    w = {a: x / tot for a, x in w.items()}
    w[assets[-1]] += 1.0 - sum(w.values())
    return w


def _forecasts(scores, assets):
    vals = [scores[a] for a in assets]
    mean = float(np.mean(vals))
    half = max(1e-9, (max(vals) - min(vals)) / 2.0)
    f = {}
    for a in assets:
        z = (scores[a] - mean) / half
        f[a] = float(np.clip(0.04 * z, -0.05, 0.05))
    return f


@register_hook
def strategy_hook():
    cur, tds = _today_and_calendar()
    if not _should_propose(cur, tds):
        return   # non-decision day: simulator marks positions / processes orders
    if not FACTORS:
        return   # no Screener ensemble -> skip this cycle
    account = get_account_dict()
    assets = list(account.get("watch_list", []))
    if len(assets) != 15:
        return
    frames = _fetch(assets)
    scores, used = _scores(frames, assets, cur)
    if used == 0:                           # fully degraded: equal weight
        w = {a: 1.0 / len(assets) for a in assets}
        w[assets[-1]] += 1.0 - sum(w.values())
        rebalance_to_weights(w)
        _persist_proposal(cur)
        return
    scores = _de_rank_value_traps(scores, frames, assets, cur)  # v8
    regime = _regime(frames, assets)
    w = _weights(scores, assets, regime)
    w = _composite_top2_cap(w, assets, scores)                 # v13 (9.5% top-2)
    w = _composite_ma_guard(w, frames, assets)                  # v8 (8% cap)
    w = _ma_guard(w, frames, assets, cur)                       # v7 (6% cap)
    for _ in range(6):                                           # v9/v11/v14 + v13 cap convergence
        w = _commod_cap(w, assets)                              # v11 (12% comm)
        w = _crypto_cap(w, assets)                              # v9 (12% crypto)
        w = _china_cap(w, assets)                               # v14 (12% China eq)
        w = _composite_top2_cap(w, assets, scores)              # v17: v13 re-applied AFTER pair caps
                                                                # (closes cap-leak where pair-cap
                                                                # redistribution pushed top-2 composite
                                                                # names back above 9.0%: XAU 10.50 in
                                                                # 0507, NDX 9.99 in 0128-0211 block)
    w = _global_cap(w, assets)                                   # v18: 10% single-name hard cap LAST
    f = _forecasts(scores, assets)
    rebalance_to_weights(
        w,
        forecast_returns=f,
        factor_ids=[fid for fid, _, _ in FACTORS],
        horizon_days=10,
    )
    _persist_proposal(cur)
