"""Screener regime assessment as of 2029-03-13 (data thru 2029-03-12)."""
import pandas as pd
import numpy as np

AS_OF = '2029-03-12'
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU',
          'COPPER','WTI','BTC','ETH','US10Y','CN10Y']
OBS = ['DXY','USDCNY','USDJPY','EURUSD','VIX']

closes = {}
for a in ASSETS:
    df = pd.read_csv(f'../persistent/stock_data/{a}.csv')
    df.columns = [c.strip() for c in df.columns]
    dcol = [c for c in df.columns if 'date' in c.lower()][0]
    ccol = [c for c in df.columns if c.lower() in ('close','收盘')]
    if not ccol:
        ccol = [c for c in df.columns if 'close' in c.lower()]
    df = df[[dcol, ccol[0]]].rename(columns={dcol:'date', ccol[0]:'close'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates('date')
    closes[a] = df.set_index('date')['close']

px = pd.DataFrame(closes).sort_index()
px = px[px.index <= AS_OF]
ret = px.pct_change()

print("=== TRADABLE ASSET DATA (thru %s, last row) ===" % px.index[-1].date())
print(px.tail(1).T.round(4).to_string())

# equal-weight market index
mkt = ret.mean(axis=1)
def ret_over(days):
    return (1 + mkt).rolling(days).apply(np.prod, raw=True) - 1

print("\n=== MARKET REGIME (equal-weight 15 assets) ===")
print("mkt_5  :", f"{(1+mkt).tail(5).prod()-1:+.2%}")
print("mkt_20 :", f"{(1+mkt).tail(20).prod()-1:+.2%}")
print("mkt_60 :", f"{(1+mkt).tail(60).prod()-1:+.2%}")
print("mkt_120:", f"{(1+mkt).tail(120).prod()-1:+.2%}")
print("mkt_252:", f"{(1+mkt).tail(252).prod()-1:+.2%}")

# realized vol of mkt (annualized, 20d)
vol20 = mkt.tail(20).std() * np.sqrt(252)
vol60 = mkt.tail(60).std() * np.sqrt(252)
print(f"mkt ann vol 20d: {vol20:.1%}  60d: {vol60:.1%}")

# asset-level 20d/60d returns
print("\n=== ASSET 5/20/60d RETURNS ===")
r5  = (1+ret.tail(5)).prod()-1
r20 = (1+ret.tail(20)).prod()-1
r60 = (1+ret.tail(60)).prod()-1
tbl = pd.DataFrame({'r5':r5,'r20':r20,'r60':r60}).sort_values('r20', ascending=False)
print(tbl.round(4).to_string())

# individual asset 20d vol
vols = ret.tail(20).std()*np.sqrt(252)
print("\n=== ASSET 20d ANN VOL ===")
print(vols.sort_values(ascending=False).round(4).to_string())

# pairwise correlation regime (60d mean pairwise)
c60 = ret.tail(60).corr()
vals = c60.values[np.triu_indices_from(c60.values, k=1)]
print(f"\nmean pairwise corr 60d: {vals.mean():.3f}  median: {np.median(vals):.3f}")

# observation signals
print("\n=== OBSERVATION SIGNALS ===")
for s in OBS:
    df = pd.read_csv(f'../persistent/index_data/{s}.csv')
    df.columns = [c.strip() for c in df.columns]
    dcol = [c for c in df.columns if 'date' in c.lower()][0]
    ccol = [c for c in df.columns if c.lower() in ('close','收盘','close_price')]
    if not ccol:
        ccol = [c for c in df.columns if 'close' in c.lower() or c.lower().startswith('c')]
    df = df[[dcol, ccol[0]]].rename(columns={dcol:'date', ccol[0]:'close'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').drop_duplicates('date').set_index('date')['close']
    df = df[df.index <= AS_OF]
    last = df.iloc[-1]
    r5 = df.iloc[-1]/df.iloc[-6]-1
    r20 = df.iloc[-1]/df.iloc[-21]-1
    r60 = df.iloc[-1]/df.iloc[-61]-1
    r252 = df.iloc[-1]/df.iloc[-253]-1 if len(df)>253 else np.nan
    print(f"{s:8s} last={last:10.3f}  5d={r5:+7.2%}  20d={r20:+7.2%}  60d={r60:+7.2%}  252d={r252:+7.2%}")

# risk-off trigger check per strategy
vix = pd.read_csv('../persistent/index_data/VIX.csv')
vix.columns = [c.strip() for c in vix.columns]
dcol = [c for c in vix.columns if 'date' in c.lower()][0]
ccol = [c for c in vix.columns if c.lower() in ('close','收盘') or 'close' in c.lower()]
vix = vix[[dcol, ccol[0]]].rename(columns={dcol:'date', ccol[0]:'close'})
vix['date'] = pd.to_datetime(vix['date'])
vix = vix.sort_values('date').set_index('date')['close']
vix = vix[vix.index <= AS_OF]
vix_last = vix.iloc[-1]
vix_20 = vix.iloc[-1]/vix.iloc[-21]-1
mkt20 = (1+mkt).tail(20).prod()-1
print(f"\n=== RISK-OFF TRIGGER ===")
print(f"VIX last={vix_last:.2f}  20d chg={vix_20:+.1%}  (>30? {vix_last>30})")
print(f"mkt_20={mkt20:+.2%}  (<0? {mkt20<0})")
print(f"trigger HIT (mkt_20<0 AND VIX>30): {mkt20<0 and vix_last>30}")

# frozen feeds check: recent flat prices
print("\n=== FROZEN FEED CHECK (last 5 closes) ===")
for a in ['NDX','SOX','000688.SH','CN10Y']:
    s = px[a].dropna()
    print(a, s.tail(5).round(4).tolist())
