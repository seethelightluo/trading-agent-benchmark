import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'; x=pd.read_csv(p); x['date']=pd.to_datetime(x['date']); D[s]=x.set_index('date')['close'].sort_index()
px=pd.DataFrame(D).sort_index(); ret=px.pct_change()
# trend quality: trailing return multiplied by fraction of positive daily observations, scaled by downside volatility
for h in [5,10]:
 vals=[]; dates=[]
 for i in range(30,len(px)-h):
  r=ret.iloc[i-20:i]; trend=px.iloc[i]/px.iloc[i-20]-1
  pos=(r>0).sum()/r.notna().sum()
  down=r.where(r<0).std()
  fac=(trend*pos/down.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan)
  fwd=px.iloc[i+h]/px.iloc[i]-1
  z=pd.concat([fac,fwd],axis=1).dropna();
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(px.index[i])
 a=np.array(vals); print('h',h,'dates',len(a),'mean_names',round(np.nanmean([len(pd.concat([((px.iloc[i]/px.iloc[i-20]-1)*(ret.iloc[i-20:i]>0).sum()/ret.iloc[i-20:i].notna().sum()/ret.iloc[i-20:i].where(ret.iloc[i-20:i]<0).std().replace(0,np.nan)),px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()) for i in range(30,len(px)-h)]),2),'IC',a.mean(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean())
 # online and regimes
 for label,lo in [('pre',pd.Timestamp('2020-01-01')),('online',pd.Timestamp('2026-07-16')),('recent',pd.Timestamp('2027-01-01'))]:
  q=a[np.array(dates)>=lo]; print(label,len(q),q.mean() if len(q) else np.nan,q.mean()/q.std() if len(q)>1 else np.nan)
