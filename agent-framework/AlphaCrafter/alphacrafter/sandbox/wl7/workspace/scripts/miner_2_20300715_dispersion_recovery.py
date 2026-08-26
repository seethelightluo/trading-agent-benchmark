import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=3000)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date').drop_duplicates('date'); F[s]=d
print('assets',len(F),'avg_history',round(np.mean([len(x) for x in F.values()]),1))
rows=[]
for s,d in F.items():
 c=pd.to_numeric(d.close,errors='coerce'); r=np.log(c).diff(); vol=r.rolling(20,min_periods=10).std()
 ret5=np.log(c/c.shift(5)); raw=(-ret5/(vol*np.sqrt(5))).clip(-8,8)
 rows.append(pd.DataFrame({'date':d.date,'asset':s,'signal':raw.shift(1),'close':c}))
a=pd.concat(rows).sort_values(['date','asset'])
wide=a.pivot(index='date',columns='asset',values='close').pct_change(5)
disp=wide.std(axis=1); disp[wide.count(axis=1)<8]=np.nan
threshold=disp.rolling(60,min_periods=30).quantile(.65)
gate=(disp>threshold).shift(1).fillna(False)
a['signal']=a['signal']*a['date'].map(gate).astype(float)
for H in [1,5,10,20,40]:
 a['fwd']=a.groupby('asset').close.shift(-H)/a.close-1; vals=[]
 for dt,g in a.groupby('date'):
  z=g.dropna(subset=['signal','fwd'])
  if len(z)>=8: vals.append((dt,len(z),z.signal.corr(z.fwd,method='spearman')))
 q=pd.DataFrame(vals,columns=['date','n','ic']).dropna(); m=q.ic.mean(); ir=m/q.ic.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan
 print('H',H,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC %.8f ICIR %.8f hit %.4f'%(m,ir,(q.ic>0).mean()))
 if H==10:
  for nm,sub in [('early',q.iloc[:len(q)//3]),('middle',q.iloc[len(q)//3:2*len(q)//3]),('late',q.iloc[2*len(q)//3:])]: print(nm,len(sub),'IC %.8f ICIR %.8f'%(sub.ic.mean(),sub.ic.mean()/sub.ic.std(ddof=1)*np.sqrt(252)) if len(sub)>1 else (nm,len(sub)))
  q.to_csv('scripts/miner_2_20300715_dispersion_recovery_ic.csv',index=False)
r=a.dropna(subset=['signal']).pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True)
print('coverage %.4f turnover %.6f active_dates %.4f'%(a.signal.notna().groupby(a.date).mean().mean(),(r.diff().abs().mean(axis=1)/2).dropna().mean(),gate.mean()))
a.to_csv('scripts/miner_2_20300715_dispersion_recovery_signal.csv',index=False)
