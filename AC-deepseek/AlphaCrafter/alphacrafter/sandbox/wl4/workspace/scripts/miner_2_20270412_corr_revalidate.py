"""miner_2 2027-04-12: pairwise correlations among passing candidates + revalidation
of the 4 currently effective library factors (recent window). Cutoff 2027-04-09.
"""
import csv
import math
from datetime import datetime, date

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
CUTOFF = date(2027, 4, 9)


def load_ohlcv(sym):
    px, hi, lo, vo = {}, {}, {}, {}
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
                hi[d] = float(row["high"])
            except (TypeError, ValueError):
                hi[d] = float("nan")
            try:
                lo[d] = float(row["low"])
            except (TypeError, ValueError):
                lo[d] = float("nan")
            try:
                vo[d] = float(row["volume"])
            except (TypeError, ValueError):
                vo[d] = 0.0
    return px, hi, lo, vo


closes, highs, lows, vols = {}, {}, {}, {}
for a in ASSETS:
    closes[a], highs[a], lows[a], vols[a] = load_ohlcv(a)


def daily_ret(px):
    out = {}
    ds = sorted(px)
    for a, b in zip(ds, ds[1:]):
        if px[a] and px[b]:
            out[b] = px[b] / px[a] - 1.0
    return out


rets = {a: daily_ret(closes[a]) for a in ASSETS}


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def std(xs):
    n = len(xs)
    if n < 3:
        return float("nan")
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    rk = [0.0] * len(xs)
    for r, i in enumerate(order):
        rk[i] = r
    return rk


def spearman(x, y):
    n = len(x)
    if n < 4:
        return float("nan")
    rx, ry = rank(x), rank(y)
    mx, my = mean(rx), mean(ry)
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    den = math.sqrt(sum((v - mx) ** 2 for v in rx) * sum((v - my) ** 2 for v in ry))
    return num / den if den > 0 else float("nan")


# factor builders
def f_zscore_close(a, win=60):
    out = {}
    ds = sorted(closes[a])
    for i in range(len(ds)):
        d = ds[i]
        window = ds[max(0, i - win + 1): i + 1]
        vals = [closes[a][dd] for dd in window if closes[a].get(dd)]
        if len(vals) >= 20:
            s = std(vals)
            if s > 0:
                out[d] = (vals[-1] - mean(vals)) / s
    return out


def f_range_pos(a, win=20):
    out = {}
    ds = sorted(closes[a])
    for i in range(len(ds)):
        d = ds[i]
        window = ds[max(0, i - win + 1): i + 1]
        vals = []
        for dd in window:
            h, l, c = highs[a].get(dd), lows[a].get(dd), closes[a].get(dd)
            if h and l and c and h > l:
                vals.append((c - l) / (h - l))
        if len(vals) >= 10:
            out[d] = mean(vals)
    return out


def f_autocorr(a, lag=5, win=60):
    out = {}
    ds = sorted(closes[a])
    for i in range(len(ds)):
        d = ds[i]
        window = ds[max(0, i - win + 1): i + 1]
        rts = [rets[a][dd] for dd in window if dd in rets[a]]
        if len(rts) < 30:
            continue
        x, y = rts[:-lag], rts[lag:]
        n = len(x)
        mx, my = mean(x), mean(y)
        num = sum((x[i] - mx) * (y[i] - my) for i in range(n))
        den = math.sqrt(sum((v - mx) ** 2 for v in x) * sum((v - my) ** 2 for v in y))
        if den > 0:
            out[d] = num / den
    return out


def f_vol_price_corr(a, win=20, min_obs=10):
    out = {}
    ds = sorted(closes[a])
    for i in range(len(ds)):
        d = ds[i]
        window = ds[max(0, i - win + 1): i + 1]
        rts, vv = [], []
        for dd in window:
            if dd in rets[a] and vols[a].get(dd, 0) > 0:
                rts.append(rets[a][dd])
                vv.append(vols[a][dd])
        if len(rts) >= min_obs:
            n = len(rts)
            mr, mv = mean(rts), mean(vv)
            num = sum((rts[i] - mr) * (vv[i] - mv) for i in range(n))
            den = math.sqrt(sum((v - mr) ** 2 for v in rts) * sum((v - mv) ** 2 for v in vv))
            if den > 0:
                out[d] = num / den
    return out


def rolling_beta(asset_r, ref_r, win=60, min_obs=40):
    out = {}
    ds = sorted(set(asset_r) & set(ref_r))
    for i in range(len(ds)):
        d = ds[i]
        window = ds[max(0, i - win + 1): i + 1]
        xs, ys = [], []
        for dd in window:
            xs.append(ref_r[dd])
            ys.append(asset_r[dd])
        if len(xs) < min_obs:
            continue
        mx, my = mean(xs), mean(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = sum((x - mx) ** 2 for x in xs)
        if den > 0:
            out[d] = num / den
    return out


cands = {
    "zscore_close_60": {a: f_zscore_close(a) for a in ASSETS},
    "range_pos_20": {a: f_range_pos(a) for a in ASSETS},
    "autocorr_10_60": {a: f_autocorr(a, lag=10, win=60) for a in ASSETS},
    "autocorr_3_40": {a: f_autocorr(a, lag=3, win=40) for a in ASSETS},
}


def panel_corr(fa, fb):
    xs, ys = [], []
    for a in ASSETS:
        common = sorted(set(fa[a]) & set(fb[a]))
        for d in common:
            x, y = fa[a][d], fb[a][d]
            if math.isfinite(x) and math.isfinite(y):
                xs.append(x)
                ys.append(y)
    if len(xs) < 30:
        return float("nan")
    return spearman(xs, ys)


print("=== pairwise correlation among passing candidates ===")
names = list(cands)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        rho = panel_corr(cands[names[i]], cands[names[j]])
        print(f"  {names[i]:18s} vs {names[j]:18s}: rho={rho:+.4f}")

# library corr for autocorr variants
all_dates = sorted(set().union(*[set(closes[a]) for a in ASSETS]))
mkt_ret = {}
for d in all_dates:
    vals = [rets[a][d] for a in ASSETS if d in rets[a]]
    if len(vals) >= 8:
        mkt_ret[d] = mean(vals)
eur_ret = daily_ret({datetime.strptime(r["date"], "%Y-%m-%d").date(): float(r["close"])
                     for r in csv.DictReader(open("../persistent/index_data/EURUSD.csv"))
                     if datetime.strptime(r["date"], "%Y-%m-%d").date() <= CUTOFF})
cn10y_ret = rets["CN10Y"]
lib = {
    "vol_price_corr_20": {a: f_vol_price_corr(a) for a in ASSETS},
    "dn_mkt_beta_60d": {a: rolling_beta(rets[a], mkt_ret) for a in ASSETS},
    "eurusd_beta_60d": {a: rolling_beta(rets[a], eur_ret) for a in ASSETS},
    "rate_beta_cn10y_60d": {a: rolling_beta(rets[a], cn10y_ret) for a in ASSETS},
}
for cname in ["autocorr_10_60", "autocorr_3_40"]:
    print(f"--- library corr for {cname} ---")
    for lname, lf in lib.items():
        print(f"  vs {lname:22s}: rho={panel_corr(cands[cname], lf):+.4f}")

# ---- revalidation of currently effective factors (recent 1y h=10) ----
def ic_series(fvals, h=10):
    rows = []
    all_dates = sorted(set().union(*[set(v) for v in fvals.values()]))
    for d in all_dates:
        xs, ys = [], []
        for a in ASSETS:
            fv = fvals[a].get(d)
            if fv is None or not math.isfinite(fv):
                continue
            ds = sorted(closes[a])
            idx = [i for i, dd in enumerate(ds) if dd == d]
            if not idx:
                continue
            j = idx[0] + h
            if j >= len(ds):
                continue
            xs.append(fv)
            ys.append(closes[a][ds[j]] / closes[a][d] - 1.0)
        if len(xs) >= 8:
            ic = spearman(xs, ys)
            if math.isfinite(ic):
                rows.append((d, ic))
    return rows


eff = {
    "vol_price_corr_20": {a: f_vol_price_corr(a) for a in ASSETS},
    "dn_mkt_beta_60d": {a: rolling_beta(rets[a], mkt_ret) for a in ASSETS},
    "eurusd_beta_60d": {a: rolling_beta(rets[a], eur_ret) for a in ASSETS},
    "rate_beta_cn10y_60d": {a: rolling_beta(rets[a], cn10y_ret) for a in ASSETS},
}
print("\n=== revalidation of effective library factors (h=10, full vs recent 250d) ===")
for name, fv in eff.items():
    rows = ic_series(fv, 10)
    ics = [r[1] for r in rows]
    m, s = mean(ics), std(ics)
    full_icir = m / s if s > 0 else 0
    recent = rows[-250:]
    rics = [r[1] for r in recent]
    rm, rs = mean(rics), std(rics)
    print(f"  {name:22s} full IC={m:+.4f} ICIR={full_icir:+.4f} (n={len(rows)}) | "
          f"recent250 IC={rm:+.4f} ICIR={rm/rs if rs>0 else 0:+.4f} (n={len(recent)})")
