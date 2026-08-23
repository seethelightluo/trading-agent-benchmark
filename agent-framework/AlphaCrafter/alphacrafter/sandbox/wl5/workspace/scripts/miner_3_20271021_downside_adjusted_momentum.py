import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2027-10-20')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@E').drop_duplicates('date').set_index('date').close
p=pd.concat({s:load(s) for s in U},axis=1).sort_index(); r=p.pct_change(); mom=p.pct_change(20); down=r.where(r<0).rolling(30,min_periods=20).std()*np.sqrt(20); f=mom/(down+1e-12)
for h in [1,5,10,20]:
 y=p.pct_change(h).shift(-h); a=[]; ns=[]; ds=[]
 for d in f.index:
  z=pd.DataFrame({'f':f.loc[d],'y':y.loc[d]}).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   a.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); ds.append(d)
 a=np.array(a); print('h',h,'dates',len(a),'N',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2027)]:
  q=a[[lo<=d.year<=hi for d in ds]]; print('reg',lo,hi,len(q),round(q.mean(),6) if len(q) else None)
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
f.stack().rename('signal').rename_axis(['date','symbol']).reset_index().to_csv('scripts/miner_3_20271021_downside_adjusted_momentum_signal.csv',index=False)
