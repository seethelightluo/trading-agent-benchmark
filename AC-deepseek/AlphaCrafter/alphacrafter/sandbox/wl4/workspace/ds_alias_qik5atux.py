import pandas as pd, numpy as np, json

assets = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
data = {}
for a in assets:
    df = pd.read_csv(f'../persistent/stock_data/{a}.csv')
    df.columns = [c.strip() for c in df.columns]
    # find date column
    datecol = df.columns[0]
    pxcol = 'close' if 'close' in df.columns else df.columns[1]
    df = df.rename(columns={datecol:'date', pxcol:'close'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    data[a] = df['close']

px = pd.DataFrame(data)
px = px.loc[:'2031-04-04']  # only visible data
px = px.dropna(how='all')
print('rows:', len(px), '| last date:', px.index[-1].date())

rets = px.pct_change()
def ret_over(periods):
    return (px.iloc[-1] / px.iloc[-1-periods] - 1) if len(px) > periods else np.nan

print(f"\n{'asset':<10} {'1m':>8} {'3m':>8} {'6m':>8} {'12m':>9} {'YTD':>8} {'last_px':>10}")
for a in assets:
    try:
        px_a = px[a].dropna()
        if len(px_a) < 30:
            print(f"{a:<10} insufficient data ({len(px_a)})")
            continue
        r1 = px_a.iloc[-1]/px_a.iloc[-22]-1 if len(px_a)>22 else np.nan
        r3 = px_a.iloc[-1]/px_a.iloc[-66]-1 if len(px_a)>66 else np.nan
        r6 = px_a.iloc[-1]/px_a.iloc[-132]-1 if len(px_a)>132 else np.nan
        r12 = px_a.iloc[-1]/px_a.iloc[-264]-1 if len(px_a)>264 else np.nan
        ytd_base = px_a.loc['2030-12-31':'2031-01-01'].iloc[0] if (px_a.index>='2030-12-31').any() else np.nan
        rytd = px_a.iloc[-1]/ytd_base-1 if ytd_base==ytd_base else np.nan
        print(f"{a:<10} {r1:>8.1%} {r3:>8.1%} {r6:>8.1%} {r12:>9.1%} {rytd:>8.1%} {px_a.iloc[-1]:>10.1f}")
    except Exception as e:
        print(a, 'ERR', e)
