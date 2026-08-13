"""Screener regime assessment - data through 2033-06-24 (last completed trading day visible)."""
import pandas as pd, numpy as np

ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END = '2033-06-24'

px = {}
for s in ASSETS:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= END].set_index('date').sort_index()
    px[s] = df['close']

PX = pd.DataFrame(px).dropna(how='all')
PX = PX.ffill()
print('Rows through', END, ':', len(PX))

ret = PX.pct_change()

def ann_vol(r, win=20):
    return r.rolling(win).std() * np.sqrt(252)

def maxdd(px, win=60):
    return (px / px.rolling(win, min_periods=1).max() - 1).min()

rows = {}
for s in ASSETS:
    p = PX[s]
    r = ret[s]
    rows[s] = {
        'close': p.iloc[-1],
        'ret_10d': p.iloc[-1]/p.iloc[-11]-1 if len(p) > 11 else np.nan,
        'ret_20d': p.iloc[-1]/p.iloc[-21]-1 if len(p) > 21 else np.nan,
        'ret_60d': p.iloc[-1]/p.iloc[-61]-1 if len(p) > 61 else np.nan,
        'ret_120d': p.iloc[-1]/p.iloc[-121]-1 if len(p) > 121 else np.nan,
        'ma20': p.rolling(20).mean().iloc[-1],
        'ma60': p.rolling(60).mean().iloc[-1],
        'ma20_slope_10d': (p.rolling(20).mean().iloc[-1] / p.rolling(20).mean().iloc[-11] - 1) if len(p) > 30 else np.nan,
        'vol20_ann': ann_vol(r).iloc[-1],
        'vol60_ann': ann_vol(r, 60).iloc[-1],
        'maxdd60': maxdd(p, 60),
        'maxdd120': maxdd(p, 120),
        'above_ma20': p.iloc[-1] > p.rolling(20).mean().iloc[-1],
        'above_ma60': p.iloc[-1] > p.rolling(60).mean().iloc[-1],
    }

R = pd.DataFrame(rows).T
print('\n=== Per-asset snapshot (through 2033-06-24) ===')
print(R[['ret_10d','ret_20d','ret_60d','ret_120d','vol20_ann','maxdd60','above_ma20','above_ma60']].round(4).to_string())

print('\n=== Cross-sectional stats (ret_20d / ret_60d) ===')
for col in ['ret_10d','ret_20d','ret_60d','ret_120d']:
    v = R[col].dropna()
    print(f'{col}: mean={v.mean():.4f} std={v.std():.4f} min={v.min():.4f} max={v.max():.4f} spread={v.max()-v.min():.4f}')

print('\n=== MA20/MA60 alignment ===')
print((R['above_ma20'] & R['above_ma60']).sum(), 'assets above both MAs;',
      (~R['above_ma20'] & ~R['above_ma60']).sum(), 'assets below both')

print('\n=== Macro observation-only (through 2033-06-24) ===')
for m in ['VIX','DXY','EURUSD','USDJPY','USDCNY']:
    df = pd.read_csv(f'../persistent/index_data/{m}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] <= END].set_index('date').sort_index()
    c = df['close']
    print(f"{m}: last={c.iloc[-1]:.3f} ret_10d={c.iloc[-1]/c.iloc[-11]-1:+.3%} ret_20d={c.iloc[-1]/c.iloc[-21]-1:+.3%} ret_60d={c.iloc[-1]/c.iloc[-61]-1:+.3%} vol20={c.pct_change().rolling(20).std().iloc[-1]*np.sqrt(252):.2%}")

# Equal-weight basket regime
basket = PX.mean(axis=1)
bret = basket.pct_change()
print(f'\n=== Equal-weight basket ===')
print(f'basket ret_20d={basket.iloc[-1]/basket.iloc[-21]-1:+.3%} ret_60d={basket.iloc[-1]/basket.iloc[-61]-1:+.3%} vol20={bret.rolling(20).std().iloc[-1]*np.sqrt(252):.2%}')
print(f'basket maxdd60={(basket/basket.rolling(60).max()-1).min():.2%}')

# pairwise correlation regime (20d rolling)
c20 = ret.iloc[-21:].corr()
print(f'\n=== 20d pairwise corr === mean={c20.values[np.triu_indices_from(c20.values,1)].mean():.2f} max={c20.values[np.triu_indices_from(c20.values,1)].max():.2f}')
c60 = ret.iloc[-61:].corr()
print(f'60d pairwise corr mean={c60.values[np.triu_indices_from(c60.values,1)].mean():.2f} max={c60.values[np.triu_indices_from(c60.values,1)].max():.2f}')
