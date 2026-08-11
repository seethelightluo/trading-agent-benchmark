import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15')
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close.loc[:cut]
p=pd.concat({s:load(s) for s in U},axis=1).sort_index(); r=p.pct_change()
# Trend persistence blend: 20d signed return weighted by path efficiency; rank cross-sectionally.
w=20
ret=p.pct_change(w); path=r.abs().rolling(w,min_periods=10).sum(); eff=(ret/path.replace(0,np.nan))
f=(ret*eff).rank(axis=1,pct=True)
for h in [1,5,10]:
  vals=[]; ns=[]; dates=[]
  for i in range(len(p)-h):
    q=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
    if len(q)>=8 and q.f.nunique()>1: vals.append(spearmanr(q.f,q.y).statistic); ns.append(len(q)); dates.append(p.index[i])
  x=np.array(vals); print('horizon',h,'dates',len(x),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
  if h==10:
   z=pd.Series(x,index=dates); print('annual10d',{int(y):round(z[z.index.year==y].mean(),6) for y in sorted(z.index.year.unique())})
turn=[]
for i in range(1,len(f)):
 a=f.iloc[i]; b=f.iloc[i-1]; ix=a.dropna().index.intersection(b.dropna().index)
 if len(ix)>=8: turn.append(np.abs(a[ix]-b[ix]).mean())
print('turnover',round(float(np.mean(turn)),6),'rows',len(p),'assets',len(U),'cutoff',p.index[-1].date())
