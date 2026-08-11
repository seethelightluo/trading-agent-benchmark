import csv, os, math
from datetime import datetime

assets = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
obs = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]

def load(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def close_series(rows):
    out = []
    for row in rows:
        try:
            c = float(row['close'])
            d = row['date']
            out.append((d, c))
        except Exception:
            continue
    return out

print("=== TRADABLE 15: recent stats (as of last data date) ===")
all_last = {}
for a in assets:
    p = os.path.join('../persistent/stock_data', a + '.csv')
    rows = load(p)
    cs = close_series(rows)
    if len(cs) < 30:
        print(a, 'insufficient', len(cs)); continue
    dates = [d for d,_ in cs]
    closes = [c for _,c in cs]
    last = closes[-1]
    # returns over windows
    def ret(n):
        if len(closes) > n:
            return closes[-1]/closes[-1-n] - 1
        return None
    r5, r21, r63 = ret(5), ret(21), ret(63)
    # realized vol (21d, annualized approx by daily std * sqrt(252))
    import statistics
    daily = [closes[i]/closes[i-1]-1 for i in range(1, len(closes))]
    vol21 = statistics.pstdev(daily[-21:]) * math.sqrt(252) if len(daily) >= 21 else None
    all_last[a] = (dates[-1], last)
    print(f"{a:12s} last={dates[-1]} px={last:12.4f} r5={r5*100:7.2f}% r21={r21*100:7.2f}% r63={r63*100:7.2f}% vol21={vol21*100:5.1f}%" if r5 is not None else f"{a:12s} last={dates[-1]}")

print()
print("=== OBSERVATION ONLY ===")
for a in obs:
    p = os.path.join('../persistent/index_data', a + '.csv')
    rows = load(p)
    cs = close_series(rows)
    if len(cs) < 30:
        print(a, 'insufficient', len(cs)); continue
    closes = [c for _,c in cs]
    def ret(n):
        if len(closes) > n:
            return closes[-1]/closes[-1-n] - 1
        return None
    print(f"{a:8s} last={cs[-1][0]} px={closes[-1]:10.4f} r5={ret(5)*100:7.2f}% r21={ret(21)*100:7.2f}% r63={ret(63)*100:7.2f}%")

print()
print("=== cross-sectional 21d dispersion (tradable 15) ===")
rets = {}
for a in assets:
    p = os.path.join('../persistent/stock_data', a + '.csv')
    rows = load(p)
    cs = close_series(rows)
    if len(cs) >= 22:
        rets[a] = cs[-1][1]/cs[-22][1] - 1
import statistics
if rets:
    vals = list(rets.values())
    print('mean 21d ret: %.2f%%  median: %.2f%%  min: %.2f%%  max: %.2f%%  std(dispersion): %.2f%%' % (
        statistics.mean(vals)*100, statistics.median(vals)*100, min(vals)*100, max(vals)*100, statistics.pstdev(vals)*100))
    for a, v in sorted(rets.items(), key=lambda x: x[1], reverse=True):
        print(f"   {a:12s} {v*100:7.2f}%")
