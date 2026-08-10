import pandas as pd,numpy as np
END=pd.Timestamp('2026-12-17'); syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in syms:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=END];r=d.close.pct_change();rows.append(pd.DataFrame({'date':d.date,'symbol':s,'r2':d.close.pct_change(2),'vol20':r.rolling(20,min_periods=15).std()}))
x=pd.concat(rows,ignore_index=True); med=x.pivot(index='date',columns='symbol',values='r2').median(axis=1);x['factor']=-(x.r2-x.date.map(med))/x.vol20;x.dropna(subset=['factor'])[['date','symbol','factor']].to_csv('scripts/miner_2_20261217_residual2_signal.csv',index=False)
print(len(x.dropna(subset=['factor'])))
