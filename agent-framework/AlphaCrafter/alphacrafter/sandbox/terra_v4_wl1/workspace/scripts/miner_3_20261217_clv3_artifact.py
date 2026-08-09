import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-12-16')
out=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d.date); d=d.sort_values('date'); d=d[d.date<=END]
 rg=(d.high-d.low).replace(0,np.nan)
 # Negative 3-day close-location value: low close location is positive reversal signal.
 d['signal']=(-(2*(d.close-d.low)/rg-1)).rolling(3,min_periods=3).mean()
 out.append(d[['date']].assign(symbol=s,value=d.signal))
a=pd.concat(out,ignore_index=True).dropna()
a.to_csv('scripts/miner_3_20261217_clv3_signal.csv',index=False)
print(len(a),a.date.min(),a.date.max(),a.value.notna().mean())
