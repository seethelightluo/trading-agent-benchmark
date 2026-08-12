"""miner_1 2029-01-15: data availability probe + recent market snapshot."""
import pandas as pd, numpy as np, glob, os

CUT = '2029-01-12'

files = sorted(glob.glob('../persistent/stock_data/*.csv'))
px = {}
for f in files:
    sym = os.path.basename(f).replace('.csv', '')
    df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    px[sym] = df['close'].astype(float)
px = pd.DataFrame(px).sort_index()
px = px[px.index <= CUT]
print('visible range:', px.index.min().date(), '->', px.index.max().date(), '| rows', len(px), '| cols', len(px.columns))

# rows per asset and last 5 closes
for c in px.columns:
    s = px[c].dropna()
    print(f'{c:10s} rows={len(s):5d} last={s.index[-1].date()} close={s.iloc[-1]:.4f}')

# density: fraction of non-null in recent windows
for w in [250, 500, 1000]:
    sub = px.iloc[-w:]
    print(f'last {w}d non-null frac: {sub.notna().mean().mean():.3f}')

# recent 10d / 60d returns by asset
rets = px.pct_change()
for label, n in [('10d', 10), ('60d', 60), ('250d', 250)]:
    r = px.iloc[-1] / px.iloc[-1 - n] - 1
    print(f'--- {label} returns (to {px.index[-1].date()}) ---')
    print(r.dropna().sort_values().round(4).to_string())

# volume availability
vol_null = 0
for f in files:
    sym = os.path.basename(f).replace('.csv', '')
    df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    df = df[df.index <= CUT]
    vol_null += float(df['volume'].isna().mean())
print(f'avg volume null frac per asset: {vol_null/len(files):.4f}')
