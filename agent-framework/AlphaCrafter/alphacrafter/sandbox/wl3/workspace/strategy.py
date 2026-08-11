import numpy as np
from alphacrafter.sim.utils import register_hook, get_stock_daily_data, rebalance_to_weights

UNIVERSE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
DEFENSIVE = {"XAU", "US10Y", "CN10Y"}
# Screener's current ten-factor ensemble; directions are positive reversal ranks.
FACTORS = {"disp": .15, "vix": .13, "volshock": .12, "ranges": .10,
           "volconfirm": .10, "rev10": .10, "shock3": .06,
           "intraday": .08, "rev15": .08, "smooth5": .08}
CADENCE = 10
MIN_W, MAX_W = .035, .16
_day = 0
_previous = None


def rank(x):
    out = {s: .5 for s in UNIVERSE}
    good = sorted((s, float(v)) for s, v in x.items() if np.isfinite(v))
    n = len(good)
    for i, (s, _) in enumerate(good):
        out[s] = (i + 1.) / n
    return out


def bounded(raw):
    # Water filling enforces a complete, long-only, full-investment vector.
    w = {s: max(float(raw.get(s, 1.)), 1e-12) for s in UNIVERSE}
    fixed = set()
    for _ in range(40):
        free = [s for s in UNIVERSE if s not in fixed]
        if not free:
            break
        rem = 1. - sum(w[s] for s in fixed)
        z = sum(w[s] for s in free)
        for s in free:
            w[s] = rem * w[s] / z
        hit = {s for s in free if w[s] < MIN_W or w[s] > MAX_W}
        if not hit:
            break
        for s in hit:
            w[s] = MIN_W if w[s] < MIN_W else MAX_W
        fixed |= hit
    z = sum(w.values())
    return {s: w[s] / z for s in UNIVERSE}


@register_hook
def cross_asset_strategy():
    global _day, _previous
    _day += 1
    if _day != 1 and (_day - 1) % CADENCE != 0:
        return
    d = {}
    for s in UNIVERSE:
        df = get_stock_daily_data(symbol=s, days=270)
        if df is None or len(df) < 160:
            continue
        df = df.sort_values("date")
        c = np.asarray(df["close"], float)
        o = np.asarray(df["open"], float)
        hi = np.asarray(df["high"], float)
        lo = np.asarray(df["low"], float)
        if np.any(~np.isfinite(c)) or np.any(c <= 0):
            continue
        r = c[1:] / c[:-1] - 1.
        vol5 = max(float(np.std(r[-5:])), .006)
        vol20 = max(float(np.std(r[-20:])), .006)
        vol60 = max(float(np.std(r[-60:])), .006)
        v = np.asarray(df["volume"], float) if "volume" in df else np.ones(len(c))
        v = np.nan_to_num(v, nan=0.)
        d[s] = dict(h3=np.prod(1+r[-3:])-1, h5=np.prod(1+r[-5:])-1,
                    h10=np.prod(1+r[-10:])-1, h15=np.prod(1+r[-15:])-1,
                    h20=np.prod(1+r[-20:])-1, h120=np.prod(1+r[-120:])-1,
                    vol5=vol5, vol20=vol20, vol60=vol60,
                    vr5=np.mean(v[-5:]) / max(np.mean(v[-20:]), 1e-9),
                    intraday=np.nanmean((c[-5:]-o[-5:]) / np.maximum(o[-5:], 1e-12)),
                    ranges=np.nanmean((hi[-5:]-lo[-5:]) / c[-5:]))
    if len(d) < 10:
        return
    syms = list(d)
    med = {k: np.median([d[s][k] for s in syms]) for k in ("h3","h5","h10","h15","h20")}
    f = {k: {} for k in FACTORS}
    for s in syms:
        x = d[s]
        # All signals are cross-sectional and use only completed trailing bars.
        f["disp"][s] = (med["h20"]-x["h20"])/(x["vol20"]*np.sqrt(20)+.015)
        f["vix"][s] = (med["h20"]-x["h20"])/(max(x["vol20"], .006)*np.sqrt(20)+.02)
        f["volshock"][s] = (med["h3"]-x["h3"])/(x["vol5"]+.01) * np.clip(x["vol5"]/x["vol60"], .5, 2.)
        f["ranges"][s] = (med["h5"]-x["h5"])/(x["ranges"]+.01)
        f["volconfirm"][s] = (med["h5"]-x["h5"])/(x["vol20"]*np.sqrt(5)+.01) * (.75+.25*np.clip(x["vr5"], .5, 1.5))
        f["rev10"][s] = (med["h10"]-x["h10"])/(x["vol20"]*np.sqrt(10)+.01)
        f["shock3"][s] = f["volshock"][s]
        f["intraday"][s] = -x["intraday"]/(x["vol5"]+.01)
        f["rev15"][s] = (med["h15"]-x["h15"])/(x["vol20"]*np.sqrt(15)+.012)
        f["smooth5"][s] = .6*f["rev10"][s] + .4*f["rev15"][s]
    ranks = {k: rank(v) for k, v in f.items()}
    score = {s: sum(FACTORS[k]*ranks[k].get(s, .5) for k in FACTORS) for s in UNIVERSE}
    if _previous is not None:
        score = {s: .35*score[s] + .65*_previous[s] for s in UNIVERSE}
    _previous = score.copy()
    breadth = np.mean([d[s]["h120"] > 0 for s in syms])
    high_risk = np.median([d[s]["vol20"] for s in syms]) > .018 or breadth < .50
    invmean = np.mean([1./d[s]["vol20"] for s in syms])
    raw = {}
    for s in UNIVERSE:
        x = d.get(s, {"vol20": .02, "h120": 0.})
        inv = (1./x["vol20"]) / max(invmean, 1e-9) if s in d else 1.
        raw[s] = max(score[s], .10) * (.88 + .12*min(inv, 1.25))
        if high_risk and s in DEFENSIVE:
            raw[s] *= 2.5
        if high_risk and x["h120"] < -.10:
            raw[s] *= .65
    target = bounded(raw)
    if abs(sum(target.values())-1.) < 1e-8 and all(np.isfinite(v) and v >= 0 for v in target.values()):
        rebalance_to_weights(target)
