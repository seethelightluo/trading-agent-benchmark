import pandas as pd, numpy as np, glob, os

TRADABLE = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUTOFF = '2027-02-10'

closes = {}
for s in TRADABLE:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv', parse_dates=['date'])
    df = df[df['date'] <= CUTOFF].sort_values('date').reset_index(drop=True)
    closes[s] = df.set_index('date')['close']

px = pd.DataFrame(closes).dropna(how='all')
print("data range:", px.index.min().date(), "->", px.index.max().date(), "rows:", len(px))
ret = px.pct_change()

last = px.iloc[-1]
print("\n=== LEVELS (close %s) ===" % px.index[-1].date())
for s in TRADABLE:
    print(f"{s:10s} {last[s]:12.2f}")

print("\n=== RETURNS (through %s) ===" % CUTOFF)
out = {}
for s in TRADABLE:
    r5  = px[s].iloc[-1]/px[s].iloc[-6]-1 if len(px)>=6 else np.nan
    r20 = px[s].iloc[-1]/px[s].iloc[-21]-1 if len(px)>=21 else np.nan
    r60 = px[s].iloc[-1]/px[s].iloc[-61]-1 if len(px)>=61 else np.nan
    vol20 = ret[s].iloc[-20:].std()*np.sqrt(252) if len(ret)>=20 else np.nan
    vol60 = ret[s].iloc[-60:].std()*np.sqrt(252) if len(ret)>=60 else np.nan
    ma20 = px[s].iloc[-20:].mean(); ma60 = px[s].iloc[-60:].mean()
    slope20 = (px[s].iloc[-1]/px[s].iloc[-21]-1) if len(px)>=21 else np.nan
    out[s] = dict(r5=r5, r20=r20, r60=r60, vol20=vol20, vol60=vol60,
                  above_ma20 = px[s].iloc[-1] > ma20, above_ma60 = px[s].iloc[-1] > ma60)
    print(f"{s:10s} r5={r5*100:7.2f}% r20={r20*100:7.2f}% r60={r60*100:7.2f}% vol20={vol20*100:5.1f}% vol60={vol60*100:5.1f}% aboveMA20={px[s].iloc[-1]>ma20} aboveMA60={px[s].iloc[-1]>ma60}")

print("\n=== CROSS-SECTION ===")
r20x = np.array([out[s]['r20'] for s in TRADABLE])
r60x = np.array([out[s]['r60'] for s in TRADABLE])
print("20d cross-section: mean %.2f%%  median %.2f%%  std %.2f%%  min %.2f%%  max %.2f%%" % (
    r20x.mean()*100, np.median(r20x)*100, r20x.std()*100, r20x.min()*100, r20x.max()*100))
print("60d cross-section: mean %.2f%%  median %.2f%%  std %.2f%%  min %.2f%%  max %.2f%%" % (
    r60x.mean()*100, np.median(r60x)*100, r60x.std()*100, r60x.min()*100, r60x.max()*100))

# correlation regime: mean pairwise corr of 20d returns over last 60d
w = ret.iloc[-60:]
corr = w.corr()
mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
print("\nmean pairwise corr (60d window, 20d ret): %.3f" % corr.values[mask].mean())
w2 = ret.iloc[-20:]
corr2 = w2.corr()
print("mean pairwise corr (20d window, 20d ret): %.3f" % corr2.values[mask].mean())

# dispersion: std of 20d returns across assets over time
disp = ret[TRADABLE].std(axis=1)
print("recent 20d dispersion mean: %.3f%%  last: %.3f%%" % (disp.iloc[-20:].mean()*100, disp.iloc[-1]*100))

# momentum leaders/laggards
rank20 = pd.Series({s: out[s]['r20'] for s in TRADABLE}).sort_values()
print("\n20d leaders:", [(s, round(v*100,1)) for s,v in rank20.tail(5).items()])
print("20d laggards:", [(s, round(v*100,1)) for s,v in rank20.head(5).items()])
