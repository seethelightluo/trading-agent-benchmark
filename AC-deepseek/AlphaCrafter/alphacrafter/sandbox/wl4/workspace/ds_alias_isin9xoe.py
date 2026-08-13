import pandas as pd, numpy as np
assets = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
for a in assets:
    df = pd.read_csv(f'../persistent/stock_data/{a}.csv')
    df.columns = [c.strip() for c in df.columns]
    datecol = df.columns[0]
    pxcol = 'close' if 'close' in df.columns else df.columns[1]
    df = df.rename(columns={datecol:'date', pxcol:'close'})
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    df['close'] = pd.to_numeric(df['close'], errors='coerce')
    d = df.loc[:'2031-04-04','close'].dropna()
    # count non-zero changes in last 60 visible rows
    last60 = d.iloc[-60:]
    nz = (last60.diff().dropna() != 0).sum()
    last_date = d.index[-1].date()
    print(f"{a:<10} last_update={last_date}  nz_changes_last60={nz}  px={d.iloc[-1]:.1f}")
