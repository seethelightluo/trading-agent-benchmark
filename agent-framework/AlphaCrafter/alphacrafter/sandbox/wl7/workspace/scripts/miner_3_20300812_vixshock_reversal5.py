import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=3000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').drop_duplicates('date');F[s]=d
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v.sort_values('date');col=[x for x in v.columns if x.lower() in ('close','adj_close')][0]
v['vix']=pd.to_numeric(v[col],errors='coerce'); v['shock']=v.vix.pct_change(5); v['gate']=(v.shock>0.08).astype(float); v=v[['date','gate']]
rows=[]
for s,d in F.items():
 c=pd.to_numeric(d.close,errors='coerce'); r=c.pct_change(); vol=r.rolling(20,min_periods=10).std()
 sig=(-c.pct_change(5)/vol.replace(0,np.nan)).shift(1).clip(-5,5)
 x=pd.DataFrame({'date':d.date,'asset':s,'raw':sig,'close':c}).merge(v,on='date',how='left'); x['signal']=x.raw*x.gate.shift(1); rows.append(x)
a=pd.concat(rows).sort_values(['date','asset'])
for H in [1,5,10,20]:
 a['fwd']=a.groupby('asset').close.shift(-H)/a.close-1; vals=[]
 for dt,g in a.groupby('date'):
  z=g.dropna(subset=['signal','fwd'])
  if len(z)>=8: vals.append((dt,len(z),z.signal.corr(z.fwd,method='spearman')))
 q=pd.DataFrame(vals,columns=['date','n','ic']).dropna(); m=q.ic.mean(); ir=m/q.ic.std(ddof=1)*np.sqrt(252)
 print('H',H,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC %.8f ICIR %.8f hit %.4f'%(m,ir,(q.ic>0).mean()))
 if H==1: q.to_csv('scripts/miner_3_20300812_vixshock_reversal5_ic.csv',index=False)
coverage=a.signal.notna().groupby(a.date).mean().mean(); rr=a.dropna(subset=['signal']).pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True)
print('coverage %.4f turnover %.6f'%(coverage,(rr.diff().abs().mean(axis=1)/2).dropna().mean()))
a.to_csv('scripts/miner_3_20300812_vixshock_reversal5_signal.csv',index=False)
