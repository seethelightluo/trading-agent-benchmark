import pandas as pd,numpy as np
from scipy.stats import rankdata
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 q=pd.read_csv('../persistent/stock_data/'+s+'.csv'); q.date=pd.to_datetime(q.date); D[s]=q.sort_values('date').set_index('date').close.rename(s)
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
# Asset-specific participation: reward trends whose daily path has few down days,
# while scaling by downside risk. All inputs lagged one session before scoring.
down=r.clip(upper=0)
downrisk=np.sqrt((down**2).rolling(40,min_periods=25).mean())
particip=(r>0).rolling(20,min_periods=15).mean()
mom=np.log(p/p.shift(20))
f=(mom/(downrisk*np.sqrt(20)+1e-8)*particip).shift(1)
y=np.log(p).shift(-10)-np.log(p)
o=[];ns=[]
for i in range(len(p)):
 a=f.iloc[i].values;b=y.iloc[i].values;ok=np.isfinite(a)&np.isfinite(b)
 if ok.sum()>=8:
  o.append(np.corrcoef(rankdata(a[ok]),rankdata(b[ok]))[0,1]);ns.append(ok.sum())
 else:o.append(np.nan);ns.append(0)
x=pd.DataFrame({'ic':o,'n':ns},index=p.index).loc['2024-01-01':'2033-12-20'].dropna()
print('dates',len(x),'avgN',x.n.mean(),'coverage',x.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f'%(x.ic.mean(),x.ic.mean()/x.ic.std(),(x.ic>0).mean()))
print('turnover %.6f'%f.rank(axis=1,pct=True).diff().abs().mean(axis=1).loc[x.index].mean())
for a,b in [('2024','2026'),('2027','2029'),('2030','2032'),('2033','2033')]:
 q=x.loc[a:b];print(a,b,len(q),'IC %.6f ICIR %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std()))
for h in [5,10,20]:
 yy=np.log(p).shift(-h)-np.log(p); z=[]
 for i in range(len(p)):
  a=f.iloc[i].values;b=yy.iloc[i].values;ok=np.isfinite(a)&np.isfinite(b)
  z.append(np.corrcoef(rankdata(a[ok]),rankdata(b[ok]))[0,1] if ok.sum()>=8 else np.nan)
 print('horizon',h,'IC',pd.Series(z,index=p.index).loc[x.index].mean())
