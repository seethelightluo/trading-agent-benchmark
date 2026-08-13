import pandas as pd
for s in ['SPX','BTC','CN10Y','XAU','000300.SH']:
    df=pd.read_csv(f'../persistent/stock_data/{s}.csv')
    print(s, df.columns.tolist(), 'max date:', df['date'].max(), 'rows:', len(df))
    print(df.head(1).to_dict('records'))
    print(df.tail(1).to_dict('records'))
    print()
