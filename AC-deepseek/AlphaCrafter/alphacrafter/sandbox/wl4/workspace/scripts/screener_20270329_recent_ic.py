"""Screener analytics: recent rank IC for the 4 active factors.

Decision date 2027-03-29; uses only data visible through the last completed
trading day (2027-03-26). Pure factor analytics on persisted CSVs; no
live-account backtest / step, no account or date mutation.
"""
import csv
import math
from datetime import datetime, date

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
CUTOFF = date(2027, 3, 26)   # last completed trading day before 2027-03-29
START = date(2026, 7, 16)    # online start


def load_close(sym):
    px, vol = {}, {}
    with open(f"../persistent/stock_data/{sym}.csv") as f:
        for row in csv.DictReader(f):
            d = datetime.strptime(row["date"], "%Y-%m-%d").date()
            if d > CUTOFF:
                continue
            try:
                px[d] = float(row["close"])
            except (TypeError, ValueError):
                continue
            try:
                v = float(row["volume"])
                vol[d] = v
            except (TypeError, ValueError):
                vol[d] = 0.0
    return px, vol


def load_obs(sym):
    px = {}
    with open(f"../persistent/index_data/{sym}.csv") as f:
        for row in csv.DictReader(f):
            d = datetime.strptime(row["date"], "%Y-%m-%d").date()
            if d > CUTOFF:
                continue
            try:
                px[d] = float(row["close"])
            except (TypeError, ValueError):
                continue
    return px


def daily_ret(px):
    out = {}
    ds = sorted(px)
    for a, b in zip(ds, ds[1:]):
        if px[a] and px[b]:
            out[b] = px[b] / px[a] - 1.0
    return out


def corr(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    if sxx <= 0 or syy <= 0:
        return float("nan")
    return sxy / math.sqrt(sxx * syy)


def rolling_beta(asset_r, ref_r, win=60, min_obs=40, down_only=False):
    out = {}
    ds = sorted(set(asset_r) & set(ref_r))
    for i in range(len(ds)):
        d = ds[i]
        window = ds[max(0, i - win + 1): i + 1]
        xs, ys = [], []
        for dd in window:
            x = ref_r[dd]
            if down_only and x >= 0:
                continue
            xs.append(x)
            ys.append(asset_r[dd])
        if len(xs) < min_obs:
            continue
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        if den == 0:
            continue
        out[d] = num / den
    return out


closes = {}
vols = {}
for a in ASSETS:
    px, v = load_close(a)
    closes[a] = px
    vols[a] = v

rets = {a: daily_ret(closes[a]) for a in ASSETS}
eurusd = load_obs("EURUSD")
eurusd_ret = daily_ret(eurusd)
cn10y_ret = daily_ret(closes["CN10Y"])

all_dates = sorted(set().union(*[set(r) for r in rets.values()]))
mkt_ret = {}
for d in all_dates:
    vals = [rets[a][d] for a in ASSETS if d in rets[a]]
    if len(vals) >= 8:
        mkt_ret[d] = sum(vals) / len(vals)


def factor_vol_price_corr(a, win=20, min_obs=10):
    out = {}
    px, v = closes[a], vols[a]
    ds = sorted(px)
    for i in range(len(ds)):
        d = ds[i]
        window = ds[max(0, i - win + 1): i + 1]
        rts, vv = [], []
        for dd in window:
            if dd in rets[a] and v.get(dd, 0) and v[dd] > 0:
                rts.append(rets[a][dd])
                vv.append(v[dd])
        if len(rts) < min_obs:
            continue
        out[d] = corr(rts, vv)
    return out


sig = {f: {} for f in ["vol_price_corr_20", "dn_mkt_beta_60d", "eurusd_beta_60d", "rate_beta_cn10y_60d"]}
for a in ASSETS:
    sig["vol_price_corr_20"][a] = factor_vol_price_corr(a)
    sig["dn_mkt_beta_60d"][a] = rolling_beta(rets[a], mkt_ret, 60, 40, down_only=True)
    sig["eurusd_beta_60d"][a] = rolling_beta(rets[a], eurusd_ret, 60, 40)
    sig["rate_beta_cn10y_60d"][a] = rolling_beta(rets[a], cn10y_ret, 60, 40)


def forward_ret(px, h=10):
    ds = sorted(px)
    out = {}
    for i, d in enumerate(ds):
        j = i + h
        if j < len(ds):
            out[d] = px[ds[j]] / px[d] - 1.0
    return out


fwd = {a: forward_ret(closes[a], 10) for a in ASSETS}


def rank_ic(factor_vals, fwd_vals, d):
    xs, ys = [], []
    for a in ASSETS:
        if d in factor_vals.get(a, {}) and d in fwd_vals.get(a, {}):
            fv = factor_vals[a][d]
            rv = fwd_vals[a][d]
            if fv == fv and rv == rv:
                xs.append(fv)
                ys.append(rv)
    if len(xs) < 8:
        return None

    def ranks(v):
        idx = sorted(range(len(v)), key=lambda k: v[k])
        r = [0] * len(v)
        for rank, pos in enumerate(idx):
            r[pos] = rank
        return r

    rx, ry = ranks(xs), ranks(ys)
    n = len(rx)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def ic_series(factor_name):
    out = []
    for d in all_dates:
        if d < START or d > CUTOFF:
            continue
        ic = rank_ic(sig[factor_name], fwd, d)
        if ic is not None:
            out.append((d, ic))
    out.sort()
    return out


print(f"=== Recent rank IC (h=10, {START}..{CUTOFF}) ===")
results = {}
for f in ["vol_price_corr_20", "dn_mkt_beta_60d", "eurusd_beta_60d", "rate_beta_cn10y_60d"]:
    s = ic_series(f)
    if not s:
        print(f"{f}: no IC dates")
        continue
    n = len(s)
    print(f"\n{f}: n={n}")
    for lab, k in [("online-all", n), ("last120", min(120, n)), ("last60", min(60, n))]:
        sub = s[-k:]
        vals = [x[1] for x in sub]
        m = sum(vals) / len(vals)
        sd = math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals)) if len(vals) > 1 else 0
        icir = m / sd if sd else 0
        hit = sum(1 for x in vals if x > 0) / len(vals)
        print(f"  [{lab}]: meanIC={m:+.4f} ICIR={icir:+.3f} hit={hit:.2f} last_date={sub[-1][0]}")
    # last 60-day stats used for tilt
    sub = s[-min(60, n):]
    vals = [x[1] for x in sub]
    m = sum(vals) / len(vals)
    sd = math.sqrt(sum((x - m) ** 2 for x in vals) / len(vals)) if len(vals) > 1 else 0
    icir = m / sd if sd else 0
    results[f] = {"mean_ic": m, "icir": icir, "n": len(vals)}

print("\n=== Quality tilt (q=|IC|*|ICIR|, last-60d, sign preserved) ===")
qs = {}
for f, r in results.items():
    q = abs(r["mean_ic"]) * abs(r["icir"])
    qs[f] = q
    print(f"{f}: IC={r['mean_ic']:+.4f} ICIR={r['icir']:+.3f} q={q:.6f}")
tot = sum(qs.values())
if tot > 0:
    for f, q in qs.items():
        print(f"  weight {f}: {q/tot:.4f}")
