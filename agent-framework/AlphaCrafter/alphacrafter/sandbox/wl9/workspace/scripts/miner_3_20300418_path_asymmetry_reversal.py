import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2030-04-18')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:end]
r=p.pct_change()
# Path asymmetry: fraction of positive sessions minus fraction of negative sessions,
# scaled by absolute cumulative return; contrarian signal, lagged one day.
pos=(r>0).rolling(20).mean(); neg=(r<0).rolling(20).mean()
r20=p.pct_change(20)
raw=-(pos-neg)*(r20.abs()+0.01)
sig=raw.shift(1)
rows=[]
for h in [10,20,40,60]:
 y=p.shift(-h)/p-1; vals=[]
 for dt in p.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append((dt,q,len(z)))
 a=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date')
 print('H',h,'dates',len(a),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/(a.ic.std(ddof=1)+1e-12),6),'hit',round((a.ic>0).mean(),4))
 for name,sl in [('early',a.loc[:'2023-12-31']),('middle',a.loc['2024-01-01':'2026-12-31']),('late',a.loc['2027-01-01':])]:
  print(' ',name,len(sl),round(sl.ic.mean(),6) if len(sl) else None,round(sl.ic.mean()/(sl.ic.std(ddof=1)+1e-12),6) if len(sl)>1 else None)
# turnover proxy via rank changes
rank=sig.rank(axis=1,pct=True); turn=(rank.diff().abs().mean(axis=1)/2).mean()
print('turnover_proxy',round(turn,6))
sig.index.name='date'; sig.to_csv('scripts/miner_3_20300418_path_asymmetry_reversal_signal.csv')
