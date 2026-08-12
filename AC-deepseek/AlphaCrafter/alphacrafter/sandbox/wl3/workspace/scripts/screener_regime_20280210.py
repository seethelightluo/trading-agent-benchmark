"""Screener regime assessment - data through prev completed trading day (2028-02-09)."""
import csv, math, statistics, itertools

CUTOFF = "2028-02-09"
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

def max_dd(series, n=120):
    seg = series[-n:]
    peak = -1e18
    mdd = 0.0
    for _, p in seg:
        peak = max(peak, p)
        mdd = min(mdd, p / peak - 1.0)
    return mdd

print("\n=== Return matrix (to 2028-02-09) ===")
print(f"{'asset':10s} {'r5':>8s} {'r10':>8s} {'r20':>8s} {'r60':>8s} {'r120':>8s} {'vol20a':>8s} {'mdd120':>8s}")
for s in SYMS:
    ser = data[s]
    print(f"{s:10s} {ret(ser,5)*100:8.2f} {ret(ser,10)*100:8.2f} {ret(ser,20)*100:8.2f} {ret(ser,60)*100:8.2f} {ret(ser,120)*100:8.2f} {ann_vol(ser,20)*100:8.1f} {max_dd(ser,120)*100:8.1f}")

# SPX trend / MA
spx = data["SPX"]
pxs = [p for _, p in spx]
ma20 = sum(pxs[-20:]) / 20
ma60 = sum(pxs[-60:]) / 60
ma20_prev = sum(pxs[-21:-1]) / 20
ma60_prev = sum(pxs[-61:-1]) / 60
print(f"\nSPX close={pxs[-1]:.1f} MA20={ma20:.1f} MA60={ma60:.1f} MA20/MA60={ma20/ma60:.4f}")
print(f"MA20 slope={ma20-ma20_prev:.2f} MA60 slope={ma60-ma60_prev:.2f}")
print(f"SPX 60d ret={ret(spx,60)*100:.2f}%  120d ret={ret(spx,120)*100:.2f}%")

# cross-asset avg pairwise corr of daily returns over last 60d
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

# dispersion: cross-sectional std of 20d returns
import statistics as st
r20_all = [ret(data[s], 20) for s in SYMS]
r20_all = [x for x in r20_all if x is not None]
print(f"Cross-sectional dispersion of 20d returns: mean={st.mean(r20_all)*100:.2f}% std={st.pstdev(r20_all)*100:.2f}%")

# macro signals
print("\n=== Macro observation-only (to 2028-02-09) ===")
for m in ["DXY", "USDJPY", "EURUSD", "USDCNY", "VIX"]:
    rows = list(csv.reader(open(f"../persistent/index_data/{m}.csv")))
    hdr = rows[0]
    idx = {c: i for i, c in enumerate(hdr)}
    ser = []
    for r in rows[1:]:
        d = r[idx["date"]]
        if d > CUTOFF:
            continue
        try:
            v = float(r[idx["close"]])
        except (ValueError, IndexError):
            continue
        ser.append((d, v))
    if ser:
        last5 = ser[-5:]
        chg20 = ser[-1][1] / ser[-21][1] - 1 if len(ser) > 21 else float('nan')
        print(f"{m:8s} last={ser[-1][1]:.2f} prev5={[round(v,2) for _, v in last5]}  chg20d={chg20*100:+.1f}%")

# defensive vs risk-on spread
defensive = ["XAU", "US10Y", "CN10Y"]
risky = ["SPX", "NDX", "BTC", "ETH", "SOX", "000688.SH", "COPPER"]
print("\n=== Defensive vs risky composite (60d ret) ===")
print("defensive avg 60d:", sum(ret(data[s], 60) for s in defensive)/len(defensive)*100, "%")
print("risky avg 60d:    ", sum(ret(data[s], 60) for s in risky)/len(risky)*100, "%")
