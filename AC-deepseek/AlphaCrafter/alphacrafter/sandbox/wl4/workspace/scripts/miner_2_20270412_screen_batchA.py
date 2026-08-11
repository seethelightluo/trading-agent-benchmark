"""miner_2 exploration 2027-04-12: screen novel factor families.

Decision date 2027-04-12; cutoff = last completed trading day 2027-04-09.
Pure factor analytics on persisted CSVs; no account/date mutation, no backtest.

Screens one family per candidate (momentum/volatility/asymmetry/volume/
range-position) with h=10 forward rank IC on the 15-asset cross-section.
Admission gates: |IC|>=0.0070 and |ICIR|>=0.0840 (paper, daily, abs).
"""
import csv
import math
from datetime import datetime, date

ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
          "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
CUTOFF = date(2027, 4, 9)

# ---------------- data loading ----------------
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

# ---------------- factor implementations ----------------
def std(xs):
    n = len(xs)
    if n < 3:
        return float("nan")
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def skew(xs):
    n = len(xs)
    if n < 8:
        return float("nan")
    m = sum(xs) / n
    s = std(xs)
    if s == 0:
        return float("nan")
    return sum((x - m) ** 3 for x in xs) / (n * s ** 3)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def factor_skew_60(a, win=60):
    out = {}
    ds = sorted(closes[a])
    for i in range(len(ds)):
        d = ds[i]
        window = ds[max(0, i - win + 1): i + 1]
        rts = [rets[a][dd] for dd in window if dd in rets[a]]
        if len(rts) >= 20:
            out[d] = skew(rts)
    return out


def factor_semi_dev_ratio_60(a, win=60):
    """downside semi-deviation / total std over window (asymmetry of risk)."""
    out = {}
    ds = sorted(closes[a])
    for i in range(len(ds)):
        d = ds[i]
        window = ds[max(0, i - win + 1): i + 1]
        rts = [rets[a][dd] for dd in window if dd in rets[a]]
        if len(rts) < 20:
            continue
        s = std(rts)
        if s == 0:
            continue
        m = mean(rts)
        down = [r for r in rts if r < m]
        sd = math.sqrt(sum((r - m) ** 2 for r in down) / len(rts)) if down else 0.0
        out[d] = sd / s
    return out


def factor_dd_from_high_60(a, win=60):
    out = {}
    ds = sorted(closes[a])
    for i in range(len(ds)):
        d = ds[i]
        window = ds[max(0, i - win + 1): i + 1]
        vals = [closes[a][dd] for dd in window if closes[a].get(dd)]
        if len(vals) < 20 or vals[-1] == 0:
            continue
        out[d] = vals[-1] / max(vals) - 1.0
    return out


def factor_vol_ratio_10x60(a, short=10, long_=60):
    out = {}
    ds = sorted(closes[a])
    for i in range(len(ds)):
        d = ds[i]
        s_win = ds[max(0, i - short + 1): i + 1]
        l_win = ds[max(0, i - long_ + 1): i + 1]
        rs = [rets[a][dd] for dd in s_win if dd in rets[a]]
        rl = [rets[a][dd] for dd in l_win if dd in rets[a]]
        if len(rs) < 5 or len(rl) < 20:
            continue
        s10, s60 = std(rs), std(rl)
        if s60 == 0:
            continue
        out[d] = s10 / s60 - 1.0
    return out


def factor_autocorr_5_60(a, lag=5, win=60):
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
        out[d] = num / den if den > 0 else float("nan")
    return out


def factor_pos_day_ratio_60(a, win=60):
    out = {}
    ds = sorted(closes[a])
    for i in range(len(ds)):
        d = ds[i]
        window = ds[max(0, i - win + 1): i + 1]
        rts = [rets[a][dd] for dd in window if dd in rets[a]]
        if len(rts) >= 30:
            out[d] = sum(1 for r in rts if r > 0) / len(rts)
    return out


def factor_range_pos_20(a, win=20):
    """avg (close-low)/(high-low) over window: closing strength / location."""
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


def factor_vol_trend_20x60(a, short=20, long_=60):
    out = {}
    ds = sorted(closes[a])
    for i in range(len(ds)):
        d = ds[i]
        s_win = ds[max(0, i - short + 1): i + 1]
        l_win = ds[max(0, i - long_ + 1): i + 1]
        vs = [vols[a][dd] for dd in s_win if vols[a].get(dd, 0) > 0]
        vl = [vols[a][dd] for dd in l_win if vols[a].get(dd, 0) > 0]
        if len(vs) >= 10 and len(vl) >= 30:
            ms, ml = mean(vs), mean(vl)
            if ml > 0:
                out[d] = ms / ml - 1.0
    return out


def factor_gain_loss_ratio_20(a, win=20):
    out = {}
    ds = sorted(closes[a])
    for i in range(len(ds)):
        d = ds[i]
        window = ds[max(0, i - win + 1): i + 1]
        rts = [rets[a][dd] for dd in window if dd in rets[a]]
        if len(rts) < 10:
            continue
        gains = [r for r in rts if r > 0]
        losses = [-r for r in rts if r < 0]
        ag = mean(gains) if gains else 0.0
        al = mean(losses) if losses else 0.0
        out[d] = ag / al if al > 0 else (float("inf") if ag > 0 else 0.0)
    return out


def factor_zscore_close_60(a, win=60):
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


CANDIDATES = {
    "skew_60d": factor_skew_60,
    "semi_dev_ratio_60": factor_semi_dev_ratio_60,
    "dd_from_high_60": factor_dd_from_high_60,
    "vol_ratio_10x60": factor_vol_ratio_10x60,
    "autocorr_5_60": factor_autocorr_5_60,
    "pos_day_ratio_60": factor_pos_day_ratio_60,
    "range_pos_20": factor_range_pos_20,
    "vol_trend_20x60": factor_vol_trend_20x60,
    "gain_loss_ratio_20": factor_gain_loss_ratio_20,
    "zscore_close_60": factor_zscore_close_60,
}

# ---------------- IC evaluation ----------------
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


def eval_factor(name, fvals, h=10):
    """fvals: dict asset -> {date: value}. Compute daily cross-sectional rank IC."""
    all_dates = sorted(set().union(*[set(v) for v in fvals.values()]))
    ic_rows = []  # (date, ic)
    for d in all_dates:
        xs, ys = [], []
        for a in ASSETS:
            fv = fvals[a].get(d)
            if fv is None or not math.isfinite(fv):
                continue
            # forward h-day return
            ds = sorted(closes[a])
            idx = [i for i, dd in enumerate(ds) if dd == d]
            if not idx:
                continue
            j = idx[0] + h
            if j >= len(ds):
                continue
            fwd = closes[a][ds[j]] / closes[a][d] - 1.0
            xs.append(fv)
            ys.append(fwd)
        if len(xs) >= 8:
            ic = spearman(xs, ys)
            if math.isfinite(ic):
                ic_rows.append((d, ic))
    if len(ic_rows) < 60:
        return None
    ics = [r[1] for r in ic_rows]
    ic_mean = mean(ics)
    ic_std = std(ics)
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = sum(1 for v in ics if (ic_mean >= 0 and v > 0) or (ic_mean < 0 and v < 0)) / len(ics)
    # coverage
    tot_asset_days = sum(len(v) for v in fvals.values())
    cov_ad = tot_asset_days / (len(ASSETS) * len(all_dates)) if all_dates else 0
    # recent window (last ~250 trading dates)
    recent = ic_rows[-250:]
    rics = [r[1] for r in recent]
    ric_mean = mean(rics)
    ric_std = std(rics)
    ricir = ric_mean / ric_std if ric_std > 0 else 0.0
    return {
        "ic": ic_mean, "icir": icir, "hit": hit, "n_dates": len(ic_rows),
        "cov_ad": cov_ad, "ic_recent250": ric_mean, "icir_recent250": ricir,
        "first_date": ic_rows[0][0].isoformat(), "last_date": ic_rows[-1][0].isoformat(),
    }


print(f"cutoff={CUTOFF}  assets={len(ASSETS)}")
for name, fn in CANDIDATES.items():
    fvals = {a: fn(a) for a in ASSETS}
    res = eval_factor(name, fvals, h=10)
    if res is None:
        print(f"{name:22s} INSUFFICIENT DATA")
        continue
    flag = "PASS" if (abs(res['ic']) >= 0.0070 and abs(res['icir']) >= 0.0840) else ""
    print(f"{name:22s} IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} hit={res['hit']:.3f} "
          f"n={res['n_dates']:4d} cov={res['cov_ad']:.2f} | recent250 IC={res['ic_recent250']:+.4f} "
          f"ICIR={res['icir_recent250']:+.4f} | {res['first_date']}..{res['last_date']} {flag}")
