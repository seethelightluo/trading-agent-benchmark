"""Screener regime + factor feedback assessment. Data through visible_through date only."""
import pandas as pd, numpy as np, json, os

VD = '2032-06-25'  # visible through (previous completed trading day)

ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load(sym):
    df = pd.read_csv(f'../persistent/stock_data/{sym}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= VD].sort_values('date').reset_index(drop=True)
    return df

px = {}
for s in ASSETS:
    df = load(s)
    px[s] = df.set_index('date')['close']

px = pd.DataFrame(px).dropna(how='all').ffill()
rets = px.pct_change().dropna(how='all')

# ---- Market regime ----
mkt = rets.mean(axis=1)  # equal-weight cross-asset market
mkt_idx = (1 + mkt).cumprod()

def ann_vol(r, w=20):
    return r.rolling(w).std() * np.sqrt(252)

print('=== MARKET REGIME (through %s) ===' % VD)
print('mkt ret 10d: %+.2f%%  20d: %+.2f%%  60d: %+.2f%%  120d: %+.2f%%' % (
    mkt.tail(10).sum()*100, mkt.tail(20).sum()*100, mkt.tail(60).sum()*100, mkt.tail(120).sum()*100))

ma20 = mkt_idx.rolling(20).mean(); ma60 = mkt_idx.rolling(60).mean(); ma120 = mkt_idx.rolling(120).mean()
last = mkt_idx.iloc[-1]
print('mkt_idx last %.3f | vs MA20 %+.2f%% | MA20 vs MA60 %+.2f%% | MA60 vs MA120 %+.2f%%' % (
    last, (last/ma20.iloc[-1]-1)*100, (ma20.iloc[-1]/ma60.iloc[-1]-1)*100, (ma60.iloc[-1]/ma120.iloc[-1]-1)*100))

vol20 = ann_vol(mkt, 20); vol60 = ann_vol(mkt, 60)
print('mkt ann vol 20d: %.1f%%  60d: %.1f%%  vol20/vol60: %.2f' % (vol20.iloc[-1]*100, vol60.iloc[-1]*100, vol20.iloc[-1]/vol60.iloc[-1]))

# 120d vol percentile vs own history since 2020
vol120 = ann_vol(mkt, 120).dropna()
pct = (vol120.iloc[-1] <= vol120).mean() * 100
print('120d ann vol %.1f%% at percentile %.0f of own history' % (vol120.iloc[-1]*100, pct))

# cross-sectional dispersion & correlation
disp20 = rets.tail(20).std(axis=1).mean() * np.sqrt(252) * 100
corr = rets.tail(60).corr()
avg_corr = (corr.values[np.triu_indices(len(corr),1)]).mean()
print('cross-sectional dispersion (20d, ann): %.1f%% | avg pairwise corr (60d): %.2f' % (disp20, avg_corr))

# asset-level 60d momentum ranks (for factor context)
mom60 = (px.iloc[-1]/px.iloc[-61] - 1)
mom20 = (px.iloc[-1]/px.iloc[-21] - 1)
print('\n=== ASSET MOMENTUM (60d / 20d) ===')
for s in ASSETS:
    print('%-10s 60d %+7.2f%%  20d %+7.2f%%  px %.4f' % (s, mom60[s]*100, mom20[s]*100, px[s].iloc[-1]))

# observation-only signals
for s in ['VIX','DXY','USDJPY','USDCNY','EURUSD']:
    try:
        d = pd.read_csv(f'../persistent/index_data/{s}.csv')
        d['date'] = pd.to_datetime(d['date'])
        d = d[d['date'] <= VD].sort_values('date')
        c = d['close'] if 'close' in d.columns else d.iloc[:,1]
        print('%-7s last %.3f  20d %+6.2f%%  60d %+6.2f%%' % (s, c.iloc[-1], (c.iloc[-1]/c.iloc[-21]-1)*100, (c.iloc[-1]/c.iloc[-61]-1)*100))
    except Exception as e:
        print(s, 'ERR', e)
