import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT=pd.Timestamp('2027-12-31')
p=pd.DataFrame({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'] for s in U}).sort_index().loc[:CUT]
r=p.pct_change(); f=pd.DataFrame(index=p.index,columns=U,dtype=float)
# Cross-asset breadth-conditioned momentum: asset 20d return, gated by the
# contemporaneous cross-sectional breadth (median 5d return), avoiding future data.
for i,dt in enumerate(p.index):
 if i<25: continue
 mom=p.iloc[i]/p.iloc[i-20]-1
 breadth=r.iloc[i-5:i].median(axis=1).mean()
 # in positive breadth, reward momentum; in negative breadth, prefer relative strength
 f.loc[dt]=mom*(1 if breadth>=0 else -1)

def run(h):
 y=p.pct_change(h).shift(-h); z=[]; ns=[]
 for dt in f.index:
  a=pd.DataFrame({'f':f.loc[dt],'y':y.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.y.nunique()>1:
   z.append(spearmanr(a.f,a.y).statistic); ns.append(len(a))
 z=np.asarray(z); print('horizon',h,'dates',len(z),'meanN',round(np.mean(ns),2),'IC',round(z.mean(),8),'ICIR',round(z.mean()/z.std(ddof=1),8),'hit',round((z>0).mean(),6))
 for lo,hi in [(2020,2022),(2023,2025),(2026,2027)]:
  q=[]; # recompute regime via aligned date list
  for dt in f.index:
   if lo<=dt.year<=hi:
    a=pd.DataFrame({'f':f.loc[dt],'y':y.loc[dt]}).dropna()
    if len(a)>=8 and a.f.nunique()>1 and a.y.nunique()>1:q.append(spearmanr(a.f,a.y).statistic)
  print('regime',lo,hi,'IC',round(np.mean(q),8) if q else None,'n',len(q))
 return z
run(10)
print('coverage',round(f.notna().sum().sum()/f.size,6),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6),'instruments',len(U),'period',p.index.min().date(),p.index.max().date())
out=f.copy();out.index.name='date';out.reset_index().to_csv('scripts/miner_1_20271231_breadth_conditioned_momentum_signal.csv',index=False)
