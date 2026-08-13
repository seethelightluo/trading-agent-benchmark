import pandas as pd, os
names = ['000300.SH','HSI','BTC','ETH','CN10Y','SPX','XAU','US10Y','WTI']
for n in names:
    p = os.path.join('../persistent/stock_data', n+'.csv')
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    dcol = df.columns[0]
    df['_d'] = pd.to_datetime(df[dcol])
    sub = df[(df['_d']>='2031-01-01') & (df['_d']<='2031-05-02')]
    if len(sub)==0:
        sub = df[(df['_d']>='2031-01-01') & (df['_d']<='2031-12-31')]
    last = sub.iloc[-1]
    close_col = 'close' if 'close' in df.columns else df.columns[1]
    chg_col = 'change' if 'change' in df.columns else (df.columns[5] if len(df.columns)>5 else None)
    print(n, '| rows_in_window:', len(sub),
          '| last_date:', str(last['_d'].date()),
          '| last_close:', round(float(last[close_col]),4),
          '| last_change:', last[chg_col] if chg_col else None)
    # count non-zero changes in the window
    if chg_col and len(sub)>0:
        nz = (sub[chg_col].fillna(0).astype(float)!=0).sum()
        print('   nonzero changes in window:', nz, 'of', len(sub))
