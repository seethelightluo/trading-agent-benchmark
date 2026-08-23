import pandas as pd
for s in ['000688.SH','SOX','NDX','CN10Y']:
    d = pd.read_csv(f'../persistent/stock_data/{s}.csv')
    d['date']=pd.to_datetime(d['date'])
    sub = d[d['date']<='2030-04-17'].tail(80)
    print(s, "last-80d close nunique:", sub['close'].nunique(), "first close:", float(sub['close'].iloc[0]), "last close:", float(sub['close'].iloc[-1]))
    print(sub[['date','close']].head(3).to_string(index=False), "...", sub[['date','close']].tail(3).to_string(index=False))