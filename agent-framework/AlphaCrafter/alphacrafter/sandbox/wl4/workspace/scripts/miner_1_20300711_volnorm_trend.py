import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).sort_values('date').set_index('date')
 px[s]=d['close'].astype(float)
P=pd.DataFrame(px).sort_index()
# signal uses completed close at t: medium trend divided by recent annualized risk
ret=P.pct_change()
f=P.pct_change(60)/(ret.rolling(20).std()*np.sqrt(20))
# test forward horizons; rank IC, dates >=8 names
for h in [5,10,20]:
 vals=[]
 for i in range(len(P)-h):
  dt=P.index[i]; x=f.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1
  z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:
   vals.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
 a=pd.DataFrame(vals,columns=['date','n','ic'])
 recent=a[a.date>=a.date.max()-pd.Timedelta(days=365)]
 ic=a.ic.mean(); sd=a.ic.std(ddof=1); ric=recent.ic.mean(); rsd=recent.ic.std(ddof=1)
 print('H',h,'dates',len(a),'avgN',round(a.n.mean(),2),'IC',round(ic,6),'ICIR',round(ic/sd*np.sqrt(252/h),4),'hit',round((a.ic>0).mean(),4),'recent_dates',len(recent),'recentIC',round(ric,6),'recentICIR',round(ric/rsd*np.sqrt(252/h),4))
# turnover: average rank top/bottom signal changes over adjacent valid dates
r=f.rank(axis=1,pct=True); ch=(r.diff().abs().mean(axis=1)).dropna()
print('coverage',round(f.notna().mean().mean(),4),'turnover_rank',round(ch.mean(),6),'last',P.index[-1].date())
# regime by thirds h10
h=10; vals=[]
for i in range(len(P)-h):
 z=pd.concat([f.iloc[i],P.iloc[i+h]/P.iloc[i]-1],axis=1).dropna()
 if len(z)>=8: vals.append((P.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
a=pd.DataFrame(vals,columns=['date','ic'])
for k,g in a.groupby(pd.qcut(np.arange(len(a)),3,labels=False)):
 print('regime',k,'dates',len(g),'IC',round(g.ic.mean(),6),'ICIR',round(g.ic.mean()/g.ic.std(ddof=1)*np.sqrt(252/10),4))
