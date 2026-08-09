import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C=pd.Timestamp('2026-09-23')
d=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').sort_index().loc[:C]; dr=d.close.pct_change()
rows=[]; sig=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').sort_values('date').set_index('date').loc[:C]
 r=x.close.pct_change(); cov=r.rolling(60,min_periods=45).cov(dr); var=dr.rolling(60,min_periods=45).var(); f=-(cov/var) # assets benefiting from weaker DXY score high
 y=x.close.shift(-1)/x.close-1
 for dt,q in pd.DataFrame({'f':f,'y':y}).dropna().iterrows(): rows.append((dt,s,q.f,q.y))
 for dt,v in f.dropna().items(): sig.append({'date':dt.strftime('%Y-%m-%d'),'symbol':s,'signal':float(v)})
a=pd.DataFrame(rows,columns=['date','symbol','f','y']); obs=[]
for dt,g in a.groupby('date'):
 if len(g)>=8:
  q=g.f.corr(g.y,method='spearman')
  if pd.notna(q): obs.append((dt,q,len(g)))
z=pd.DataFrame(obs,columns=['date','ic','n']); print('dates',len(z),'avgN',round(z.n.mean(),2),'coverage',round(len(a)/(15*len(z)),4)); print('IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),4))
for h in [5,10]:
 vals=[]
 for s in U:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').sort_values('date').set_index('date').loc[:C]; r=x.close.pct_change(); f=-(r.rolling(60,min_periods=45).cov(dr)/dr.rolling(60,min_periods=45).var()); y=x.close.shift(-h)/x.close-1; vals += [(dt,v.f,v.y) for dt,v in pd.DataFrame({'f':f,'y':y}).dropna().iterrows()]
 b=pd.DataFrame(vals,columns=['date','f','y']); oo=[g.f.corr(g.y,method='spearman') for _,g in b.groupby('date') if len(g)>=8]; oo=np.array([q for q in oo if pd.notna(q)]); print('horizon',h,'dates',len(oo),'IC',round(oo.mean(),6),'ICIR',round(oo.mean()/oo.std(ddof=1),6))
for yr,g in z.groupby(z.date.dt.year): print('regime',yr,len(g),round(g.ic.mean(),5),round(g.ic.mean()/g.ic.std(ddof=1),4))
print('turnover_proxy',round(a.sort_values(['symbol','date']).groupby('symbol').f.diff().abs().mean(),5)); pd.DataFrame(sig).to_csv('scripts/miner_2_20260924_dxy_beta_signal.csv',index=False); print('signal_rows',len(sig))
