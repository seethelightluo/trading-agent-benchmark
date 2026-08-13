import pandas as pd
for s in ['DXY','VIX','USDJPY','EURUSD']:
    df=pd.read_csv(f'../persistent/index_data/{s}.csv')
    df['date']=pd.to_datetime(df['date'])
    df=df[df['date']<='2031-12-10'].sort_values('date')
    last=df.tail(1).iloc[0]
    d10=df.tail(11).iloc[0]
    d21=df.tail(22).iloc[0] if len(df)>=22 else df.iloc[0]
    d60=df.tail(61).iloc[0] if len(df)>=61 else df.iloc[0]
    print(f'{s:7s} last={last["date"].date()} close={last["close"]:.2f} r10d={(last["close"]/d10["close"]-1)*100:+.1f}% r21d={(last["close"]/d21["close"]-1)*100:+.1f}% r60d={(last["close"]/d60["close"]-1)*100:+.1f}%')
