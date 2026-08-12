import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUT='2030-04-17'
def load(p):
 x=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index(); return x.loc[:CUT]
D={s:load('../persistent/stock_data/'+s+'.csv') for s in U}
m=load('../persistent/index_data/DXY.csv')['close'].pct_change()
def factor(x):
 r=np.log(x.close).diff(); mm=m.reindex(x.index).ffill()
 beta=r.rolling(60,min_periods=30).cov(mm)/mm.rolling(60,min_periods=30).var()
 e=r-beta*mm
 return -e.rolling(3,min_periods=3).sum()/(e.rolling(20,min_periods=15).std()*np.sqrt(3)+1e-12)
rows=[]
for s,x in D.items():
 f=factor(x); r=np.log(x.close).diff()
 for h in [1,3,5,10]:
  y=sum(r.shift(-k) for k in range(1,h+1))
  rows += [pd.DataFrame({'date':x.index,'f':f.values,'y':y.values,'s':s,'h':h})]
a=pd.concat(rows).dropna(); out=[]
for (dt,h),g in a.groupby(['date','h']):
 if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: out.append((dt,h,spearmanr(g.f,g.y).statistic,len(g)))
z=pd.DataFrame(out,columns=['date','h','ic','n']).set_index('date')
for h in [1,3,5,10]:
 q=z[z.h==h].ic; print('H',h,'dates',len(q),'avgN',z[z.h==h].n.mean(),'IC',q.mean(),'ICIR_ann',q.mean()/q.std(ddof=1)*np.sqrt(252),'hit',(q>0).mean())
 for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030')]:
  w=q.loc[lo:hi];
  if len(w): print(' ',lo,hi,len(w),round(w.mean(),6),round(w.mean()/w.std(ddof=1)*np.sqrt(252),4))
f=pd.concat([factor(x).rename(s) for s,x in D.items()],axis=1); ranks=f.rank(axis=1,pct=True)
print('coverage',f.notna().sum().sum()/(f.shape[0]*15),'turnover',ranks.diff().abs().mean(axis=1).dropna().mean(),'dates',len(f),'assets',len(f.columns),'last',f.index.max())
f.index.name='date'; f.to_csv('scripts/miner_2_20300418_dxy_residual_reversal_signal.csv')
