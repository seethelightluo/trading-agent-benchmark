"""miner_2 deep validation 2027-04-12 for shortlisted factors.

Factors: zscore_close_60, range_pos_20, autocorr_5_60 (+ variants).
Full metrics: IC/ICIR/hit/coverage, decay by horizon, turnover, sub-period
stability, library correlation vs the 4 effective factors. Cutoff 2027-04-09.
"""
import csv
import math
import zlib
import base64
import json
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


# ---------------- factor definitions ----------------
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


# ---------------- IC machinery ----------------
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


def summarize(name, fvals, horizons=(1, 2, 3, 5, 10, 20)):
    print(f"\n===== {name} =====")
    out = {}
    for h in horizons:
        rows = ic_series(fvals, h=h)
        if len(rows) < 60:
            print(f"  h={h}: insufficient ({len(rows)})")
            continue
        ics = [r[1] for r in rows]
        m, s = mean(ics), std(ics)
        hit = sum(1 for v in ics if (m >= 0 and v > 0) or (m < 0 and v < 0)) / len(ics)
        out[h] = {"ic": m, "icir": m / s if s > 0 else 0, "hit": hit, "n": len(rows)}
        print(f"  h={h:2d}: IC={m:+.4f} ICIR={m/s if s>0 else 0:+.4f} hit={hit:.3f} n={len(rows)}")
    return out


# zscore_close_60
fz = {a: f_zscore_close(a) for a in ASSETS}
summarize("zscore_close_60", fz)

# range_pos_20
fr = {a: f_range_pos(a) for a in ASSETS}
summarize("range_pos_20", fr)

# autocorr variants
for lag, win in [(5, 40), (5, 60), (10, 60), (3, 40)]:
    fa = {a: f_autocorr(a, lag=lag, win=win) for a in ASSETS}
    summarize(f"autocorr_{lag}_{win}", fa)

# ---------------- sub-period stability (h=10) ----------------
def sub_periods(fvals, name):
    rows = ic_series(fvals, h=10)
    bounds = [(date(2020, 1, 1), date(2022, 12, 31), "2020-2022"),
              (date(2023, 1, 1), date(2025, 12, 31), "2023-2025"),
              (date(2026, 1, 1), date(2027, 4, 9), "2026-2027")]
    print(f"\n--- sub-periods {name} (h=10) ---")
    for b0, b1, label in bounds:
        sub = [r for r in rows if b0 <= r[0] <= b1]
        if len(sub) < 40:
            print(f"  {label}: n={len(sub)} (skip)")
            continue
        ics = [r[1] for r in sub]
        m, s = mean(ics), std(ics)
        print(f"  {label}: n={len(sub):4d} IC={m:+.4f} ICIR={m/s if s>0 else 0:+.4f} hit={sum(1 for v in ics if (m>=0 and v>0) or (m<0 and v<0))/len(ics):.3f}")


sub_periods(fz, "zscore_close_60")
sub_periods(fr, "range_pos_20")

# ---------------- turnover & coverage ----------------
def turnover_cov(fvals, name):
    all_dates = sorted(set().union(*[set(v) for v in fvals.values()]))
    rank_changes = []
    prev = {}
    n_obs = 0
    n_cells = 0
    for d in all_dates:
        cur = {}
        for a in ASSETS:
            fv = fvals[a].get(d)
            if fv is not None and math.isfinite(fv):
                cur[a] = fv
                n_obs += 1
            n_cells += 1
        if len(cur) >= 8:
            rk = {a: r for r, a in enumerate(sorted(cur, key=lambda k: cur[k]))}
            if prev:
                common = set(rk) & set(prev)
                if len(common) >= 8:
                    delta = sum(abs(rk[a] - prev[a]) for a in common) / len(common)
                    rank_changes.append(delta / (len(common) - 1))
            prev = rk
    cov = n_obs / n_cells if n_cells else 0
    print(f"\n--- {name}: coverage={cov:.3f} n_dates={len(all_dates)} "
          f"turnover_10d_rank_avg={mean(rank_changes) if rank_changes else float('nan'):.3f} "
          f"({len(rank_changes)} obs)")


turnover_cov(fz, "zscore_close_60")
turnover_cov(fr, "range_pos_20")

# ---------------- library correlation vs effective factors ----------------
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


all_dates = sorted(set().union(*[set(closes[a]) for a in ASSETS]))
mkt_ret = {}
for d in all_dates:
    vals = [rets[a][d] for a in ASSETS if d in rets[a]]
    if len(vals) >= 8:
        mkt_ret[d] = mean(vals)

eur_ret = daily_ret({datetime.strptime(r["date"], "%Y-%m-%d").date(): float(r["close"])
                     for r in csv.DictReader(open("../persistent/index_data/EURUSD.csv"))
                     if datetime.strptime(r["date"], "%Y-%m-%d").date() <= CUTOFF})

lib = {
    "vol_price_corr_20": {a: f_vol_price_corr(a) for a in ASSETS},
    "dn_mkt_beta_60d": {a: rolling_beta(rets[a], mkt_ret) for a in ASSETS},
    "eurusd_beta_60d": {a: rolling_beta(rets[a], eur_ret) for a in ASSETS},
}
# rate beta needs CN10Y as reference
cn10y_ret = rets["CN10Y"]
lib["rate_beta_cn10y_60d"] = {a: rolling_beta(rets[a], cn10y_ret) for a in ASSETS}


def panel_corr(fa, fb):
    """pairwise correlation across all (asset,date) valid cells."""
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


for cand_name, cand in [("zscore_close_60", fz), ("range_pos_20", fr)]:
    print(f"\n--- library correlation for {cand_name} ---")
    for lname, lf in lib.items():
        rho = panel_corr(cand, lf)
        print(f"  vs {lname:22s}: rho={rho:+.4f}")
