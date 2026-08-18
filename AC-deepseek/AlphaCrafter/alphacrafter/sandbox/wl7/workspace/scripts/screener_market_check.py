"""Screener market-regime assessment through visible date (2028-08-28)."""
import csv, json, math, datetime

VIS = '2028-08-28'
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
MACRO = {'DXY':'../persistent/index_data/DXY.csv','VIX':'../persistent/index_data/VIX.csv',
         'USDJPY':'../persistent/index_data/USDJPY.csv','EURUSD':'../persistent/index_data/EURUSD.csv'}

def load(fp):
    rows = list(csv.reader(open(fp)))
    hdr = rows[0]
    idx = {c:i for i,c in enumerate(hdr)}
    out = []
    for r in rows[1:]:
        d = r[idx['date']]
        if d > VIS: continue
        try:
            c = float(r[idx['close']])
        except (ValueError, IndexError):
            continue
        out.append((d, c))
    out.sort()
    return out

prices = {}
for a in ASSETS:
    prices[a] = load(f'../persistent/stock_data/{a}.csv')

mac = {k: load(v) for k,v in MACRO.items()}

def rets(series, n):
    if len(series) <= n: return None
    return series[-1][1]/series[-1-n][1] - 1.0

def last(series):
    return series[-1][1]

# returns
print("=== ASSET RETURNS thru", VIS, "===")
names_ret = {}
for a in ASSETS:
    s = prices[a]
    r5, r20, r60 = rets(s,5), rets(s,20), rets(s,60)
    names_ret[a] = (r5, r20, r60)
    # MA status
    closes = [x[1] for x in s]
    ma20 = sum(closes[-20:])/20 if len(closes)>=20 else None
    ma60 = sum(closes[-60:])/60 if len(closes)>=60 else None
    above20 = closes[-1] > ma20 if ma20 else None
    above60 = closes[-1] > ma60 if ma60 else None
    print(f"{a:10s} r5 {r5*100:7.2f}%  r20 {r20*100:8.2f}%  r60 {r60*100:8.2f}%  >MA20 {above20}  >MA60 {above60}")

print("\n=== MACRO ===")
for k,v in mac.items():
    print(f"{k:8s} last {last(v):9.3f}  r5 {rets(v,5)*100:7.2f}%  r20 {rets(v,20)*100:7.2f}%  r60 {rets(v,60)*100:7.2f}%")

# vol regime: realized vol of SPX 20d annualized
def rvol(series, n=20):
    closes=[x[1] for x in series]
    rets=[math.log(closes[i]/closes[i-1]) for i in range(1,len(closes))]
    seg=rets[-n:]
    m=sum(seg)/len(seg)
    v=sum((x-m)**2 for x in seg)/(len(seg)-1)
    return math.sqrt(v)*math.sqrt(252)

print("\n=== REALIZED VOL (20d ann) ===")
for a in ASSETS:
    print(f"{a:10s} {rvol(prices[a])*100:6.1f}%")

# mean pairwise corr of 20d returns across live names
import numpy as np
def ret_series(series, n=20):
    closes=[x[1] for x in series]
    return [math.log(closes[i]/closes[i-1]) for i in range(1,len(closes))][-n:]

# align by date
common_dates = set(prices['SPX'][i][0] for i in range(len(prices['SPX'])))
for a in ASSETS:
    common_dates &= set(x[0] for x in prices[a])
common_dates = sorted(common_dates)
print("\ncommon dates:", len(common_dates), common_dates[-3:] if common_dates else None)

def rets_by_date(a):
    d = {x[0]: x[1] for x in prices[a]}
    out = []
    for i in range(1, len(common_dates)):
        d0, d1 = common_dates[i-1], common_dates[i]
        if d0 in d and d1 in d and d[d0] > 0:
            out.append(math.log(d[d1]/d[d0]))
        else:
            out.append(0.0)
    return out

R = np.array([rets_by_date(a) for a in ASSETS])
# 20d trailing correlation matrix (last 20 daily returns)
C = np.corrcoef(R[:, -20:])
mask = ~np.eye(C.shape[0], dtype=bool)
print(f"mean pairwise 20d corr (last 20d): {C[mask].mean():.3f}")
C60 = np.corrcoef(R[:, -60:])
mask60 = ~np.eye(C60.shape[0], dtype=bool)
print(f"mean pairwise 20d corr (last 60d): {C60[mask60].mean():.3f}")

# market breadth
above20_count = 0; above60_count=0
for a in ASSETS:
    closes=[x[1] for x in prices[a]]
    above20_count += closes[-1] > sum(closes[-20:])/20
    above60_count += closes[-1] > sum(closes[-60:])/60
print(f"\nbreadth >MA20: {above20_count}/15, >MA60: {above60_count}/15")

# EW market returns
ew = []
for i in range(len(common_dates)):
    day = 0.0
    for a in ASSETS:
        d = {x[0]: x[1] for x in prices[a]}
        if common_dates[i-1] in d and common_dates[i] in d and d[common_dates[i-1]]>0:
            day += math.log(d[common_dates[i]]/d[common_dates[i-1]])
    ew.append(day/len(ASSETS))
print(f"EW mkt 5d {sum(ew[-5:])*100:.2f}%  20d {sum(ew[-20:])*100:.2f}%  60d {sum(ew[-60:])*100:.2f}%")
