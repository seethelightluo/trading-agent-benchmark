import pandas as pd
cut='2031-12-10'
for s in ['SPX','NDX','SOX','WTI','COPPER','XAU','ETH','BTC','000300.SH','000688.SH','HSI','N225','SX5E','US10Y','CN10Y']:
    df=pd.read_csv(f'../persistent/stock_data/{s}.csv')
    df['date']=pd.to_datetime(df['date'])
    df=df[df['date']<=cut].sort_values('date')
    last=df.tail(1).iloc[0]
    d10=df.tail(11).iloc[0]
    d21=df.tail(22).iloc[0] if len(df)>=22 else df.iloc[0]
    d60=df.tail(61).iloc[0] if len(df)>=61 else df.iloc[0]
    r10=(last['close']/d10['close']-1)*100
    r21=(last['close']/d21['close']-1)*100
    r60=(last['close']/d60['close']-1)*100
    print(f'{s:10s} last={last["date"].date()} close={last["close"]:.2f} r10d={r10:+6.1f}% r21d={r21:+6.1f}% r60d={r60:+6.1f}%')
