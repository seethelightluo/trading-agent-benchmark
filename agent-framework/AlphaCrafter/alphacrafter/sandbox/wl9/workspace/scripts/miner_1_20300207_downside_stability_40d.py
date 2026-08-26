import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-02-07'); D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}; p=pd.concat(D,axis=1).sort_index().loc[:end]; r=p.pct_change()
# downside deviation uses mean squared negative returns, with min periods for asynchronous calendars
neg=r.clip(upper=0); dd=np.sqrt((neg**2).rolling(40,min_periods=15).mean()); trend=r.rolling(40,min_periods=20).sum(); f=trend/(dd+1e-8); sig=f.shift(1)
for h in [10,20,40]:
 y=p.shift(-h)/p-1; a=[]
 for dt in p.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): a.append((dt,q,len(z)))
 a=pd.DataFrame(a,columns=['date','ic','n']).set_index('date'); m=a.ic.mean(); ir=m/(a.ic.std(ddof=1)+1e-12)
 print('H',h,'dates',len(a),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(m,6),'ICIR',round(ir,6),'hit',round((a.ic>0).mean(),4))
 for nm,sl in [('online',a.loc['2026-07-16':]),('recent',a.loc['2029-01-01':])]:
  print(nm,'dates',len(sl),'IC',round(sl.ic.mean(),6),'ICIR',round(sl.ic.mean()/(sl.ic.std(ddof=1)+1e-12),6))
 if h==40:a.to_csv('scripts/miner_1_20300207_downside_stability_40d_signal.csv')
