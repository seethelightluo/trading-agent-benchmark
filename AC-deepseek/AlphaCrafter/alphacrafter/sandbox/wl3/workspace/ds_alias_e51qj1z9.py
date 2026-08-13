import pandas as pd
for s in ['BTC','HSI','SX5E','US10Y','CN10Y','SPX']:
    df=pd.read_csv(f'../persistent/stock_data/{s}.csv')
    df['date']=pd.to_datetime(df['date'])
    df=df[df['date']<='2031-12-10']
    print(s, 'rows through 2031-12-10:', len(df), 'first:', df['date'].iloc[0].date(), 'last:', df['date'].iloc[-1].date())
    print(df.tail(3)[['date','close']].to_string(index=False))
