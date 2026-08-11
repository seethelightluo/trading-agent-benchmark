import csv, os, math, statistics
from datetime import datetime

CUTOFF = '2027-01-15'  # last completed trading day before current date 2027-01-18
assets = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
obs = ["DXY","USDCNY","USDJPY","EURUSD","VIX"]

def load(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            d = row['date']
            if d <= CUTOFF:
                rows.append(row)
    return rows

def close_series(rows):
    return [(row['date'], float(row['close'])) for row in rows if row.get('close') not in (None, '')]

print("=== TRADABLE 15 as of", CUTOFF, "===")
last_dates = {}
for a in assets:
    p = os.path.join('../persistent/stock_data', a + '.csv')
    cs = close_series(load(p))
    if len(cs) < 70:
        print(a, 'insufficient', len(cs)); continue
    closes = [c for _, c in cs]
    def ret(n):
        return closes[-1]/closes[-1-n] - 1 if len(closes) > n else None
    daily = [closes[i]/closes[i-1]-1 for i in range(1, len(closes))]
    vol21 = statistics.pstdev(daily[-21:]) * math.sqrt(252) if len(daily) >= 21 else None
    vol63 = statistics.pstdev(daily[-63:]) * math.sqrt(252) if len(daily) >= 63 else None
    last_dates[a] = cs[-1][0]
    print(f"{a:12s} last={cs[-1][0]} px={closes[-1]:12.4f} r5={ret(5)*100:7.2f}% r21={ret(21)*100:7.2f}% r63={ret(63)*100:7.2f}% vol21={vol21*100:5.1f}% vol63={vol63*100:5.1f}%")

print()
print("=== OBSERVATION ONLY as of", CUTOFF, "===")
for a in obs:
    p = os.path.join('../persistent/index_data', a + '.csv')
    cs = close_series(load(p))
    if len(cs) < 70:
        print(a, 'insufficient', len(cs)); continue
    closes = [c for _, c in cs]
    def ret(n):
        return closes[-1]/closes[-1-n] - 1 if len(closes) > n else None
    print(f"{a:8s} last={cs[-1][0]} px={closes[-1]:10.4f} r5={ret(5)*100:7.2f}% r21={ret(21)*100:7.2f}% r63={ret(63)*100:7.2f}%")

print()
print("=== 21d cross-sectional dispersion (tradable 15) ===")
rets = {}
for a in assets:
    p = os.path.join('../persistent/stock_data', a + '.csv')
    cs = close_series(load(p))
    if len(cs) >= 22 and cs[-1][0] == CUTOFF:
        rets[a] = cs[-1][1]/cs[-22][1] - 1
vals = list(rets.values())
print('n=%d mean=%.2f%% median=%.2f%% min=%.2f%% max=%.2f%% dispersion(std)=%.2f%%' % (
    len(vals), statistics.mean(vals)*100, statistics.median(vals)*100, min(vals)*100, max(vals)*100, statistics.pstdev(vals)*100))
for a, v in sorted(rets.items(), key=lambda x: x[1], reverse=True):
    print(f"   {a:12s} {v*100:7.2f}%")

print()
print("=== 63d (quarter) cross-sectional ===")
rets63 = {}
for a in assets:
    p = os.path.join('../persistent/stock_data', a + '.csv')
    cs = close_series(load(p))
    if len(cs) >= 64 and cs[-1][0] == CUTOFF:
        rets63[a] = cs[-1][1]/cs[-64][1] - 1
vals = list(rets63.values())
print('n=%d mean=%.2f%% median=%.2f%% min=%.2f%% max=%.2f%% dispersion(std)=%.2f%%' % (
    len(vals), statistics.mean(vals)*100, statistics.median(vals)*100, min(vals)*100, max(vals)*100, statistics.pstdev(vals)*100))
for a, v in sorted(rets63.items(), key=lambda x: x[1], reverse=True):
    print(f"   {a:12s} {v*100:7.2f}%")

# MA trend for SPX as broad equity proxy
p = os.path.join('../persistent/stock_data', 'SPX.csv')
cs = close_series(load(p))
closes = [c for _, c in cs]
ma20 = statistics.mean(closes[-20:]); ma60 = statistics.mean(closes[-60:])
print()
print('SPX px=%.1f ma20=%.1f ma60=%.1f slope60=(ma20-ma60)/ma60=%.2f%%' % (closes[-1], ma20, ma60, (ma20/ma60-1)*100))
p = os.path.join('../persistent/stock_data', '000300.SH.csv')
cs = close_series(load(p))
closes = [c for _, c in cs]
ma20 = statistics.mean(closes[-20:]); ma60 = statistics.mean(closes[-60:])
print('000300 px=%.1f ma20=%.1f ma60=%.1f slope60=%.2f%%' % (closes[-1], ma20, ma60, (ma20/ma60-1)*100))
