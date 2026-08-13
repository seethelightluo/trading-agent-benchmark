import pandas as pd, numpy as np, os

assets = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
obs = ['DXY','USDCNY','USDJPY','EURUSD','VIX']
base = '../persistent'
cutoff = '2031-05-02'

def load(p):
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    dcol = df.columns[0]
    df[dcol] = pd.to_datetime(df[dcol])
    df = df[df[dcol] <= cutoff].set_index(dcol).sort_index()
    return df

print("=== TRADABLE ASSETS ===")
rets = {}
for a in assets:
    p = os.path.join(base,'stock_data', a+'.csv')
    if not os.path.exists(p):
        print(a, 'MISSING'); continue
    df = load(p)
    c = [c for c in df.columns if c.lower() in ('close','c')]
    c = c[0] if c else df.columns[1]
    close = df[c].astype(float)
    ret = close.pct_change()
    rets[a] = close
    last = close.iloc[-1]
    r10 = close.iloc[-1]/close.iloc[-11]-1 if len(close)>11 else np.nan
    r20 = close.iloc[-1]/close.iloc[-21]-1 if len(close)>21 else np.nan
    r60 = close.iloc[-1]/close.iloc[-61]-1 if len(close)>61 else np.nan
    vol20 = ret.tail(20).std()*np.sqrt(252)
    ma20 = close.tail(20).mean(); ma60 = close.tail(60).mean() if len(close)>=60 else np.nan
    trend = 'UP' if close.iloc[-1]>ma20>ma60 else ('DOWN' if close.iloc[-1]<ma20<ma60 else 'FLAT')
    print(f"{a:12s} last={last:12.4f} r10={r10:8.2%} r20={r20:8.2%} r60={r60:8.2%} vol20={vol20:6.2%} trend={trend}")

print("\n=== OBSERVATION ONLY ===")
for a in obs:
    p = os.path.join(base,'index_data', a+'.csv')
    df = load(p)
    c = df.columns[1]
    close = df[c].astype(float)
    last = close.iloc[-1]
    r10 = close.iloc[-1]/close.iloc[-11]-1 if len(close)>11 else np.nan
    r20 = close.iloc[-1]/close.iloc[-21]-1 if len(close)>21 else np.nan
    r60 = close.iloc[-1]/close.iloc[-61]-1 if len(close)>61 else np.nan
    ma20 = close.tail(20).mean(); ma60 = close.tail(60).mean() if len(close)>=60 else np.nan
    trend = 'UP' if close.iloc[-1]>ma20>ma60 else ('DOWN' if close.iloc[-1]<ma20<ma60 else 'FLAT')
    print(f"{a:8s} last={last:10.4f} r10={r10:8.2%} r20={r20:8.2%} r60={r60:8.2%} trend={trend}")

print("\n=== CROSS-SECTIONAL DISPERSION (20d avg |cross-sectional z|) ===")
px = pd.DataFrame(rets).dropna(how='all')
xs_ret = px.pct_change()
disp = xs_ret.sub(xs_ret.mean(axis=1), axis=0).abs().mean(axis=1)
print("avg dispersion last 10d:", round(disp.tail(10).mean(),5))
print("avg dispersion last 60d:", round(disp.tail(60).mean(),5))
print("avg dispersion last 120d vs prior120:", round(disp.tail(120).mean(),5), round(disp.iloc[-240:-120].mean(),5))

# pairwise correlation last 60d among liquid assets
print("\n=== 60d pairwise corr (mean) ===")
r = xs_ret.tail(60)
cormat = r.corr()
vals = cormat.values[np.triu_indices(len(cormat), k=1)]
print("mean pairwise corr:", round(np.nanmean(vals),3), "n valid:", int(np.sum(~np.isnan(vals))))
