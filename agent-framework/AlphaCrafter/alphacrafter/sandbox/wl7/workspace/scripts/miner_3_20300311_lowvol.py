import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data,get_account_dict
TODAY='2030-03-11'; H=10
u=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in u:
 d=None
 try:d=get_index_daily_data(s,days=4000)
 except Exception:pass
 if d is None or len(d)<150:
  try:d=get_stock_daily_data(s,days=4000)
  except Exception:d=None
 if d is not None and len(d):px[s]=d.set_index('date').close.astype(float)
P=pd.DataFrame(px).sort_index().loc[:TODAY]
rows=[]
for i in range(100,len(P)-H):
 z=[]
 for s in u:
  if s not in P:continue
  x=P[s].iloc[:i+1].dropna()
  if len(x)<100 or i+H>=len(P) or pd.isna(P[s].iloc[i+H]):continue
  r=np.log(x).diff(); v=r.iloc[-21:-1].std(); down=r.iloc[-61:-1].where(r.iloc[-61:-1]<0).std()
  if np.isfinite(v) and v>0 and np.isfinite(down): z.append((s,-0.7*v-0.3*down,P[s].iloc[i+H]/x.iloc[-1]-1))
 if len(z)>=8:
  med=np.median([q[1] for q in z]);rows += [(P.index[i],s,f-med,fw) for s,f,fw in z]
df=pd.DataFrame(rows,columns=['date','symbol','factor','fwd']).dropna();ic=df.groupby('date').apply(lambda z:z.factor.corr(z.fwd),include_groups=False).dropna(); ranks=df.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('universe',len(u),'available',len(px),'dates',len(ic),'avg_names',df.groupby('date').size().mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(),'hit',(ic>0).mean(),'turnover',ranks.diff().abs().mean(axis=1).dropna().mean())
for a,b in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030')]:
 q=ic.loc[a:b];print('regime',a,b,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std() if len(q)>1 else np.nan)
for h in [1,5,10,20,40]:
 aa=[]
 for i in range(100,len(P)-h):
  z=[]
  for s in u:
   if s not in P:continue
   x=P[s].iloc[:i+1].dropna();
   if len(x)<100 or i+h>=len(P) or pd.isna(P[s].iloc[i+h]):continue
   r=np.log(x).diff();v=r.iloc[-21:-1].std();d=r.iloc[-61:-1].where(r.iloc[-61:-1]<0).std()
   if np.isfinite(v) and v>0 and np.isfinite(d):z.append((-0.7*v-0.3*d,P[s].iloc[i+h]/x.iloc[-1]-1))
  if len(z)>=8: 
   m=np.median([q[0] for q in z]);aa.append(pd.Series([q[0]-m for q in z]).corr(pd.Series([q[1] for q in z])))
 print('horizon',h,'dates',len(aa),'IC',np.nanmean(aa))
df.to_csv('scripts/miner_3_20300311_lowvol_signal.csv',index=False);ic.rename('ic').to_csv('scripts/miner_3_20300311_lowvol_ic.csv')
