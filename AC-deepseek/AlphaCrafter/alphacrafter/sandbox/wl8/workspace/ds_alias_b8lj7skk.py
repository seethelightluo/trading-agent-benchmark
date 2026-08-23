import pandas as pd
for s in ['SPX','ETH','US10Y','WTI','CN10Y','000300.SH']:
    d = pd.read_csv(f'../persistent/stock_data/{s}.csv')
    d['date']=pd.to_datetime(d['date'])
    sub = d[d['date']<='2030-04-17']
    v = sub['volume']
    print(s, "rows:", len(sub), "vol nunique:", v.nunique(), "vol min/max:", float(v.min()), float(v.max()), "vol last5:", list(v.tail(5).round(0)))
print("--- CN10Y vol flat tail ---")
d = pd.read_csv('../persistent/stock_data/CN10Y.csv'); d['date']=pd.to_datetime(d['date'])
sub = d[d['date']<='2030-04-17']
print("CN10Y vol nunique last 120:", sub['volume'].tail(120).nunique())