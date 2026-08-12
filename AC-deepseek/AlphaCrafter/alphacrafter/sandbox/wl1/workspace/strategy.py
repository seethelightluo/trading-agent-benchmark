"""Trader strategy v12 - Screener 5-factor quality_ic_tilt ensemble.

Ensemble (2028-11-10): mom_120d_skip5 (.26,+) | vol_of_vol20x60 (.23,+)
| vix_beta_cond_60x20 (.20,-) | miner2_20260715_rev_2d (.16,+)
| miner2_20260715_nclv_1d (.15,+). Screener re-tilted momentum up (.22->.26)
on a regime shift toward persistent momentum leaders with extreme dispersion
while trimming the vol/defensive block (vol_of_vol .25->.23, vix_beta .24
->.20); reversal pair reweighted (rev_2d .13->.16, nclv_1d .16->.15).

Momentum anchor (trimmed from .42 per COPPER whipsaw) + two decorrelated
reversal members + vol-of-vol regime + VIX-beta risk guard. Cross-sectional
rank composite over the 15-name tradable panel; fully invested, non-negative
weights sum to 1, no cash sleeve. One atomic rebalance proposal per
10-trading-day block via rebalance_to_weights with aligned forecast returns so
the execution gate (gross edge > one-way turnover * 3bp) decides. Bear regime
adds a modest defensive tilt (XAU/US10Y/CN10Y). Factors are loaded from
factor_ensemble.json at import.

v4 cadence fix (2027-01-22): the harness invokes cycles at idx%10==4 while the
fixed grid (idx%10==8 from ONLINE_START) is never hit, which froze proposals
since 2026-09-24. Proposals now fire on grid days OR on any hook call >=10
trading days after the last proposal (one proposal per 10-day block either
way), tracked via trader_state.json (fallback to account last_rebalance_date).

v7 (2027-07-23): Screener refreshed weights - momentum trimmed .42->.30
(COPPER 7-block whipsaw), vol_of_vol raised .16->.20 (strongest recent IC),
reversal pair raised (.15/.14 -> .19/.17), vix_beta .13->.14. Added Screener
recommended portfolio-level guard: momentum top-picks trading below their 20d
MA are weight-capped (extended names that broke short-term MA, e.g. COPPER),
excess redistributed to remaining names.

v8 (2028-01-07): two portfolio-construction fixes after 2 consecutive blocks
of top-3 crypto drag (BTC/ETH below 20d MA with 9-10% weights):
  1. Composite-score MA guard: ANY asset whose weight exceeds 8% while its
     price is below its 20d MA is capped at 8% regardless of which factor
     lifted it (reversal/nclv can no longer bypass the momentum-only v7 cap).
  2. Value-trap de-rank: assets below 20d MA with negative 120d momentum are
     penalized in the composite score before ranking (reversal-family lifts of
     broken-trend names, e.g. ETH rank 14/15 momentum with 9.4% weight).
Both keep the v7 momentum-top-pick 6% cap as the stricter inner guard.

v9 (2028-03-31): combined crypto cap after 3 consecutive blocks of crypto
top-weight drag (block 0317-0331: ETH -25.9% on ~8% weight accounted for the
entire block loss; ETH was above its 20d MA with positive 120d momentum at the
decision so the v8 guards could not catch it). BTC+ETH combined weight is
capped at 12% regardless of factor scores or trend state; excess is
redistributed proportionally to the 13 non-crypto names. This is a
portfolio-level risk guard (like v7/v8), not a factor change, and keeps
crypto slightly below its 2/15 fair share while preserving upside in strong
rallies.

v10 (2028-04-14): extend the v7 momentum guard to close the above-MA20 hole.
Block 0331-0414: WTI was the top 120d-momentum name, above its 20d MA (so v7
did not cap it) and carried 8.9% weight into a -21% crash - the dominant block
drag (7th distinct momentum top-pick whipsaw block). Empirical replay over 45
decision dates shows the top momentum name is a coin flip with fat tails
(avg fwd 10d +0.7% vs universe median -0.2%, underperformed in 22/45 blocks).
v10/v12 cap the top-2 momentum names (rank >= MOM_TOP2_RANK) at
GUARD_CAP regardless of MA state; the v7 below-MA20 top-6 rule and v8/v9
guards are unchanged.

v11 (2028-10-13): combined cyclical-commodity cap. Block 0929-1013: WTI 9.4% +
COPPER 8.0% (17.4% combined) both fell ~-7% (est. -1.29 combined contribution)
- the block's dominant drag, the 2nd block the commodity pair carried ~17%
into a synchronized drawdown (WTI also 2nd time as momentum top-pick whipsaw).
Commodities remain below their 20d MA with negative 20d momentum. WTI+COPPER
combined weight is capped at 14% regardless of factor scores or trend state;
excess is redistributed proportionally to the remaining names. XAU is
deliberately excluded (defensive sleeve). Factor- and trend-agnostic like v9.

v12 (2028-10-27): extend the v10 momentum cap to the top-2 momentum names.
Block 1013-1027: SOX was the rank-2 momentum name and delivered -12.5% on
7.4% weight - the 9th distinct momentum top-pick whipsaw block in 10
(Screener also flags XAU/BTC as top-4 trap names in 20d downtrends). Any of
the top-2 momentum names (rank >= .86 of 15) is now capped at GUARD_CAP
regardless of MA state, matching the empirical 9/10 top-pick reversal rate.

v13 (2029-06-08): composite-rank top-2 weight cap (Screener-recommended).
Block 0330-0413: NDX 11.5% / N225 10.8% were top-2 COMPOSITE weights but not
top-2 momentum names, so the v12 momentum cap missed them; 2029-04-13
feedback + 2029-06-08 screener both recommend a composite-rank cap beyond the
momentum-rank guard. With 10 of the last 11 blocks showing a top-pick
whipsaw (SOX -12.5%, WTI -18.4% twice, ETH -25.9%...), composite-score
leaders carry fat-tailed reversal risk regardless of factor origin. The
top-2 composite-score names are capped at COMP_TOP2_CAP (9.5%); excess is
redistributed proportionally to remaining names. Factor-agnostic like
v8/v9/v11; applied before the MA guards.
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
CAP_W = 0.16
GUARD_CAP = 0.06         # v7 cap for momentum top-picks below 20d MA
COMP_GUARD_CAP = 0.08    # v8 cap for ANY below-20d-MA asset with weight > 8%
CRYPTO_CAP = 0.12        # v9 combined BTC+ETH weight cap
COMMOD_CAP = 0.14        # v11 combined WTI+COPPER weight cap
COMP_TOP2_CAP = 0.095     # v13 composite-rank top-2 weight cap
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
    score = {a: 0.0 for a in assets}
    used = 0
    for fid, w, direction in FACTORS:
        vals = _factor_values(frames, fid, cur)
        if sum(1 for v in vals.values() if v is not None) < 8:
            continue
        r = _ranks(vals, assets)
        for a in assets:
            score[a] += w * (r[a] if direction > 0 else 1.0 - r[a])
        used += 1
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
    if used < 5:                            # degraded fallback: equal weight
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
    w = _crypto_cap(w, assets)                                  # v9 (12% crypto)
    w = _commod_cap(w, assets)                                  # v11 (14% comm)
    f = _forecasts(scores, assets)
    rebalance_to_weights(
        w,
        forecast_returns=f,
        factor_ids=[fid for fid, _, _ in FACTORS],
        horizon_days=10,
    )
    _persist_proposal(cur)
