import pandas as pd, numpy as np
assets = ['SPX','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','US10Y']
px = {}
for a in assets:
    df = pd.read_csv(f'../persistent/stock_data/{a}.csv')
    df.columns = [c.strip() for c in df.columns]
    datecol = df.columns[0]
    pxcol = 'close' if 'close' in df.columns else df.columns[1]
    df = df.rename(columns={datecol:'date', pxcol:'close'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    px[a] = pd.to_numeric(df['close'], errors='coerce')
px = pd.DataFrame(px).loc[:'2031-04-04']
rets = px.pct_change().dropna(how='all')

print("=== TREND (last px vs 20d/60d MA) ===")
for a in assets:
    s = px[a].dropna()
    if len(s) < 70: continue
    ma20 = s.rolling(20).mean().iloc[-1]
    ma60 = s.rolling(60).mean().iloc[-1]
    px_now = s.iloc[-1]
    slope = (ma20 - ma60)/ma60
    state = 'UP' if px_now>ma20>ma60 else 'DOWN' if px_now<ma20<ma60 else 'MIX'
    print(f"{a:<10} px={px_now:>8.1f} slope20v60={slope:>+7.1%} {state}")

print("\n=== REALIZED VOL (annualized) ===")
vol20 = rets.tail(20).std()*np.sqrt(252)
vol60 = rets.tail(60).std()*np.sqrt(252)
for a in assets:
    print(f"{a:<10} vol20={vol20[a]:>7.1%}  vol60={vol60[a]:>7.1%}")

mkt = rets.mean(axis=1)
disp = (rets.sub(mkt, axis=0)).abs().mean(axis=1)
print(f"\ncross-sectional disp last20 mean: {disp.tail(20).mean():.4f}  last5: {disp.tail(5).mean():.4f}")

c = rets.tail(60).corr().copy()
np.fill_diagonal(c.values, np.nan)
print(f"avg pairwise corr (60d): {c.stack().mean():.3f}")

print("\n=== LAST 10d BLOCK RETURNS ===")
blk = (px.iloc[-1]/px.iloc[-11]-1)
print(blk.sort_values(ascending=False).round(4).to_string())
