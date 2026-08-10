import pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}).sort_index().ffill().loc[:'2027-02-24']
r=P.pct_change(); S=r.rolling(20).sum()-0.5*r.rolling(5).sum()
out=S.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/factor_signals_miner_2_20270225_pullback_trend_10d.csv',index=False)
print(len(out),out.date.min(),out.date.max())
