"""Screener regime assessment - data through prev completed trading day (2027-07-28)."""
import csv, math, statistics, datetime

CUTOFF = "2027-07-28"
SYMS = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
BASE = "../persistent/stock_data/"

def load(sym):
    rows = list(csv.reader(open(BASE + sym + ".csv")))
    hdr = rows[0]
    idx = {c: i for i, c in enumerate(hdr)}
    out = []
    for r in rows[1:]:
        d = r[idx["date"]]
        if d > CUTOFF:
            continue
        try:
            px = float(r[idx["close"]])
        except (ValueError, IndexError):
            continue
        out.append((d, px))
    return out

data = {}
for s in SYMS:
    data[s] = load(s)
    print(s, "obs:", len(data[s]), "first:", data[s][0][0], "last:", data[s][-1][0])

def ret(series, n):
    if len(series) < n + 1:
        return None
    return series[-1][1] / series[-1 - n][1] - 1.0

def ann_vol(series, n=20):
    if len(series) < n + 1:
        return None
    rs = [series[i][1] / series[i - 1][1] - 1.0 for i in range(len(series) - n, len(series))]
    m = sum(rs) / len(rs)
    var = sum((x - m) ** 2 for x in rs) / (len(rs) - 1)
    return math.sqrt(var) * math.sqrt(252)

print("\n=== Return matrix (to 2027-07-28) ===")
print(f"{'asset':10s} {'r5':>8s} {'r10':>8s} {'r20':>8s} {'r60':>8s} {'r120':>8s} {'vol20a':>8s}")
for s in SYMS:
    ser = data[s]
    print(f"{s:10s} {ret(ser,5)*100:8.2f} {ret(ser,10)*100:8.2f} {ret(ser,20)*100:8.2f} {ret(ser,60)*100:8.2f} {ret(ser,120)*100:8.2f} {ann_vol(ser,20)*100:8.1f}")

# SPX trend / MA
spx = data["SPX"]
pxs = [p for _, p in spx]
ma20 = sum(pxs[-20:]) / 20
ma60 = sum(pxs[-60:]) / 60
ma20_prev = sum(pxs[-21:-1]) / 20
ma60_prev = sum(pxs[-61:-1]) / 60
print(f"\nSPX close={pxs[-1]:.1f} MA20={ma20:.1f} MA60={ma60:.1f} MA20/MA60={ma20/ma60:.4f}")
print(f"MA20 slope={ma20-ma20_prev:.2f} MA60 slope={ma60-ma60_prev:.2f}")

# cross-asset avg pairwise corr of daily returns over last 60d
import itertools
def daily_ret_series(series, n):
    rs = []
    for i in range(len(series) - n, len(series)):
        rs.append(series[i][1] / series[i - 1][1] - 1.0)
    return rs

def corr(a, b):
    n = min(len(a), len(b))
    a, b = a[-n:], b[-n:]
    ma, mb = sum(a)/n, sum(b)/n
    cov = sum((x-ma)*(y-mb) for x, y in zip(a, b)) / (n-1)
    sa = math.sqrt(sum((x-ma)**2 for x in a)/(n-1))
    sb = math.sqrt(sum((y-mb)**2 for y in b)/(n-1))
    if sa == 0 or sb == 0:
        return 0.0
    return cov / (sa * sb)

r60 = {s: daily_ret_series(data[s], 60) for s in SYMS}
pairs = list(itertools.combinations(SYMS, 2))
cs = [corr(r60[a], r60[b]) for a, b in pairs]
print(f"\nAvg pairwise corr (60d, all 15): {sum(cs)/len(cs):.3f}  median: {statistics.median(cs):.3f}  n={len(cs)}")

# macro signals
print("\n=== Macro observation-only ===")
for m in ["DXY", "USDJPY", "EURUSD", "USDCNY", "VIX"]:
    rows = list(csv.reader(open(f"../persistent/index_data/{m}.csv")))
    hdr = rows[0]
    idx = {c: i for i, c in enumerate(hdr)}
    vals = []
    for r in rows[1:]:
        d = r[idx["date"]]
        if d > CUTOFF:
            continue
        try:
            px = float(r[idx["close"]])
        except (ValueError, IndexError):
            continue
        vals.append((d, px))
    if vals:
        def rr(n):
            return vals[-1][1] / vals[-1 - n][1] - 1.0 if len(vals) > n else None
        print(f"{m:8s} last={vals[-1][1]:10.3f} r5={rr(5)*100:6.2f}% r20={rr(20)*100:6.2f}% r60={rr(60)*100:6.2f}%")
