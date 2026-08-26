import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=3000)
 if d is not None and len(d):
  d=d.copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').drop_duplicates('date');d['close']=pd.to_numeric(d.close,errors='coerce');F[s]=d[['date','close']]
rows=[]
for s,d in F.items():
 c=d.close; x=pd.DataFrame({'date':d.date,'asset':s,'ret20':c.pct_change(20),'vol20':c.pct_change().rolling(20,min_periods=15).std(),'close':c});rows.append(x)
a=pd.concat(rows,ignore_index=True).sort_values(['date','asset'])
med=a.groupby('date').ret20.transform('median');raw=(a.ret20-med)/a.vol20.replace(0,np.nan);a['signal']=raw.groupby(a.asset).shift(1).clip(-5,5)
for H in [1,5,10,20]:
 a['fwd']=a.groupby('asset').close.shift(-H)/a.close-1; vals=[]
 for dt,g in a.groupby('date'):
  z=g.dropna(subset=['signal','fwd'])
  if len(z)>=8: vals.append((dt,len(z),z.signal.corr(z.fwd,method='spearman')))
 q=pd.DataFrame(vals,columns=['date','n','ic']).dropna();m=q.ic.mean();ir=m/q.ic.std(ddof=1)*np.sqrt(252);cuts=np.array_split(q.ic,3)
 print('H',H,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC %.8f ICIR %.8f hit %.4f'%(m,ir,(q.ic>0).mean()));print('regime_IC',*[round(x.mean(),8) for x in cuts])
 if H==1:q.to_csv('scripts/miner_3_20300826_residual_trend20_ic.csv',index=False)
coverage=a.signal.notna().groupby(a.date).mean().mean();rr=a.dropna(subset=['signal']).pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True)
print('dates_total',a.date.nunique(),'assets',len(F),'coverage %.4f turnover %.6f'%(coverage,(rr.diff().abs().mean(axis=1)/2).dropna().mean()));a.to_csv('scripts/miner_3_20300826_residual_trend20_signal.csv',index=False)
