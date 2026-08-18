import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').close for a in A}).sort_index(); r=p.pct_change(); m=r.median(axis=1)
for win in [20,60]:
 out=[]
 for i in range(win,len(r)-10):
  h=r.iloc[i-win:i]; bm=m.iloc[i-win:i]; up=bm>0; dn=bm<0
  if up.sum()<4 or dn.sum()<4:continue
  # downside asymmetry: performance on up days minus performance on down days
  f=h.where(np.repeat(up.values[:,None],15,1)).mean()-h.where(np.repeat(dn.values[:,None],15,1)).mean()
  z=pd.concat([f,r.iloc[i+1:i+2].sum()],axis=1).dropna()
  if len(z)>=8:out.append((r.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 ic=pd.Series(dict(out));print('win',win,'N',len(ic),'names',r.loc[ic.index].notna().sum(1).mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean())
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  x=ic[(ic.index.year>=lo)&(ic.index.year<=hi)]; print(lo,round(x.mean(),4),len(x))
