import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query("date <= '2026-07-15'").set_index('date').sort_index()
 prev=x.close.shift(1); intra=-(x.close/x.open-1); clv=-(2*x.close-x.high-x.low)/(x.high-x.low).replace(0,np.nan)
 rev=-(x.close/x.close.shift(5)-1); mom=x.close/x.close.shift(20)-1
 rows.append(pd.DataFrame({'intra':intra,'clv':clv,'rev':rev,'mom':mom}))
a=pd.concat(rows).dropna(); print('rows',len(a)); print(a.corr().round(5).to_string())
# year IC for intra 1d
for yr,g in a.groupby(a.index.year): print('year',yr,'n',len(g))
