import pandas as pd, numpy as np

ASSETS = ['000300.SH','000688.SH','SPX','HSI','N225','SX5E','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
VIS = '2029-11-02'

def load(sym, col='close'):
    df = pd.read_csv(f'../persistent/stock_data/{sym}.csv', parse_dates=['date'])
    df = df[df['date'] <= VIS].set_index('date')[col]
    return df

closes = {s: load(s) for s in ASSETS}
px = pd.DataFrame(closes).dropna(how='all')
ret = px.pct_change()

last = px.iloc[-1]
print('=== LEVELS (as of %s) ===' % VIS)
print(last.round(4).to_string())

for w in [5,10,20,60]:
    r = px.iloc[-1] / px.iloc[-1-w] - 1
    print(f'\n=== {w}d returns (%) ===')
    print((r*100).round(2).sort_values(ascending=False).to_string())

vol20 = ret.tail(20).std()*np.sqrt(252)
vol60 = ret.tail(60).std()*np.sqrt(252)
print('\n=== annualized vol % (20d / 60d) ===')
print(pd.DataFrame({'vol20':vol20*100,'vol60':vol60*100}).round(1).to_string())

ma60 = px.rolling(60).mean()
above = (last > ma60.iloc[-1])
print('\n=== above 60d MA (1=yes) ===')
print(above.astype(int).to_string())

corr = ret.tail(60).corr().values.copy()
np.fill_diagonal(corr, np.nan)
print('\n=== avg pairwise corr 60d: %.3f ===' % np.nanmean(corr))

vix = pd.read_csv('../persistent/index_data/VIX.csv', parse_dates=['date'])
vix = vix[vix['date'] <= VIS].set_index('date')['close']
dxy = pd.read_csv('../persistent/index_data/DXY.csv', parse_dates=['date'])
dxy = dxy[dxy['date'] <= VIS].set_index('date')['close']
print('\n=== VIX: last %.1f | 20d ago %.1f | 60d ago %.1f ===' % (vix.iloc[-1], vix.iloc[-21], vix.iloc[-61]))
print('=== DXY: last %.2f | 20d chg %.2f%% | 60d chg %.2f%% ===' % (dxy.iloc[-1], (dxy.iloc[-1]/dxy.iloc[-21]-1)*100, (dxy.iloc[-1]/dxy.iloc[-61]-1)*100))

ew = ret.mean(axis=1)
ew_cum = (1+ew).cumprod()
print('\n=== EW basket 60d ret: %.2f%% | 20d: %.2f%% | 10d: %.2f%% ===' % ((ew_cum.iloc[-1]/ew_cum.iloc[-61]-1)*100, (ew_cum.iloc[-1]/ew_cum.iloc[-21]-1)*100, (ew_cum.iloc[-1]/ew_cum.iloc[-11]-1)*100))
print('=== EW basket vs 60d MA: %.3f%% ===' % ((ew_cum.iloc[-1]/ew_cum.rolling(60).mean().iloc[-1]-1)*100))
