import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    for f in (get_stock_daily_data,get_index_daily_data):
        try:
            x=f(s,days=5000)
            if x is not None and len(x)>100: return x
        except Exception: pass
    return None
D={}
for s in U:
 x=get(s)
 if x is not None:
  x=x.copy(); x['date']=pd.to_datetime(x.date); x=x.set_index('date').sort_index(); D[s]=x.close.astype(float)
px=pd.DataFrame(D).sort_index(); r=np.log(px).diff()
# relaxed compression: medium trend normalized by vol, activated if recent vol <= 1.10 long vol
sig=(np.log(px/px.shift(20))/r.rolling(40).std()).where(r.rolling(10).std() <= 1.10*r.rolling(60).std()).shift(1)
rows=[]
for h in [1,5,10,20]:
 vals=[]
 for i in range(len(px)-h):
  a=sig.iloc[i]; fwd=np.log(px.iloc[i+h]/px.iloc[i])
  z=pd.concat([a,fwd],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1]))
 v=np.array(vals); print(h,'dates',len(v),'avg_n',round(float(np.nanmean([pd.concat([sig.iloc[i],np.log(px.iloc[i+h]/px.iloc[i])],axis=1).dropna().shape[0] for i in range(len(px)-h)]),),2),'IC',round(float(np.nanmean(v)),6),'ICIR',round(float(np.nanmean(v)/np.nanstd(v,ddof=1)),6),'hit',round(float(np.mean(v>0)),4))
# coverage and turnover based valid cross-sectional ranks
valid=sig.notna().sum(axis=1); print('rows',len(px),'assets',len(D),'coverage',round(float(sig.notna().sum().sum()/(len(px)*len(D))),4),'dates',len(px),'mean_valid',round(float(valid.mean()),2),'turnover',round(float((sig.rank(axis=1,pct=True).diff().abs().mean(axis=1)).mean()),4),'end',px.index[-1])
# thirds for 20
v=[]
for i in range(len(px)-20):
 z=pd.concat([sig.iloc[i],np.log(px.iloc[i+20]/px.iloc[i])],axis=1).dropna()
 if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1]))
print('thirds',*[round(float(np.mean(a)),6) for a in np.array_split(v,3)])
