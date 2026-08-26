import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=3000)
 if d is not None and len(d):
  d=d.copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date').drop_duplicates('date'); d.close=pd.to_numeric(d.close,errors='coerce'); F[s]=d[['date','close']]
rows=[]
for s,d in F.items():
 c=d.close; r=c.pct_change(); rows.append(pd.DataFrame({'date':d.date,'asset':s,'r3':c.pct_change(3),'vol20':r.rolling(20,min_periods=10).std(),'close':c}))
a=pd.concat(rows,ignore_index=True).sort_values(['date','asset'])
disp=a.groupby('date').r3.std(); x=pd.DataFrame({'disp':disp}); x['med']=disp.rolling(90,min_periods=45).median(); x['gate']=(x.disp>x.med).astype(float); x.index.name='date'
a=a.join(x.gate.rename('gate'),on='date'); a['raw']=(-a.r3/a.vol20.replace(0,np.nan))*a.gate; a['signal']=a.groupby('asset').raw.shift(1).clip(-5,5)
for H in [1,5,10,20]:
 a['fwd']=a.groupby('asset').close.shift(-H)/a.close-1; out=[]
 for dt,g in a.groupby('date'):
  z=g.dropna(subset=['signal','fwd'])
  if len(z)>=8:
   ic=z.signal.corr(z.fwd,method='spearman')
   if pd.notna(ic): out.append((dt,len(z),ic))
 q=pd.DataFrame(out,columns=['date','n','ic']); m=q.ic.mean(); ir=m/q.ic.std(ddof=1)*np.sqrt(252)
 print('H',H,'dates',len(q),'avg_n',round(q.n.mean(),2),'IC %.8f ICIR %.8f hit %.4f'%(m,ir,(q.ic>0).mean()),'regime',*[round(z.mean(),8) for z in np.array_split(q.ic,3)])
 if H==1:q.to_csv('scripts/miner_3_20300923_dispersion_reversal3_ic.csv',index=False)
coverage=a.signal.notna().groupby(a.date).mean().mean(); rank=a.dropna(subset=['signal']).pivot(index='date',columns='asset',values='signal').rank(axis=1,pct=True)
print('dates_total',a.date.nunique(),'assets',len(F),'coverage %.4f turnover %.6f'%(coverage,(rank.diff().abs().mean(axis=1)/2).dropna().mean())); a.to_csv('scripts/miner_3_20300923_dispersion_reversal3_signal.csv',index=False)
