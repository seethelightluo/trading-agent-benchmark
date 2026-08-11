import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end]['close'] for s in U}
# Contrarian 5-day return scaled by recent realized volatility; signal lagged one session.
F=pd.DataFrame({s:(-(D[s].pct_change(5))/(D[s].pct_change().rolling(20,min_periods=12).std()+1e-8)).shift(1) for s in U}).sort_index()
print('idea=volatility-scaled 5d reversal, lag1')
for h in [1,5,10]:
 Y=pd.DataFrame({s:D[s].shift(-h)/D[s]-1 for s in U}).sort_index(); a=[]; ns=[]
 for dt in F.index:
  z=pd.DataFrame({'f':F.loc[dt],'y':Y.loc[dt]}).dropna()
  if len(z)>=8: a.append(z.f.corr(z.y));ns.append(len(z))
 a=np.array(a); print('horizon',h,'dates',len(a),'avg_names',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4))
print('turnover',round(F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
F.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_2_20260730_volscaled_reversal_signals.csv',index=False)
