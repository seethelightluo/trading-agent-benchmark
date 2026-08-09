import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C=pd.Timestamp('2026-10-08')
rows=[]; sig=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').sort_values('date').set_index('date').loc[:C]
 # prior-close to open gap; mean-reversion signal for next close-to-close return
 gap=x.open/x.close.shift(1)-1
 f=-gap
 y=x.close.shift(-1)/x.close-1
 q=pd.DataFrame({'f':f,'y':y}).dropna()
 for d,r in q.iterrows(): rows.append((d,s,r.f,r.y))
 for d,v in f.dropna().items(): sig.append({'date':d.strftime('%Y-%m-%d'),'symbol':s,'signal':float(v)})
a=pd.DataFrame(rows,columns=['date','symbol','f','y']); obs=[]
for d,g in a.groupby('date'):
 if len(g)>=8:
  ic=g.f.corr(g.y,method='spearman')
  if pd.notna(ic): obs.append((d,ic,len(g)))
z=pd.DataFrame(obs,columns=['date','ic','n']);
print('cutoff',C.date(),'dates',len(z),'avgN',round(z.n.mean(),2),'coverage',round(len(a)/(len(z)*15),4))
print('IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),4))
for yr,g in z.groupby(z.date.dt.year): print('regime',yr,'n',len(g),'IC',round(g.ic.mean(),6),'ICIR',round(g.ic.mean()/g.ic.std(ddof=1),5))
for h in [5,10]:
 oo=[]
 for s in U:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').sort_values('date').set_index('date').loc[:C]; f=-(x.open/x.close.shift(1)-1); y=x.close.shift(-h)/x.close-1; q=pd.DataFrame({'f':f,'y':y}).dropna()
  for d,r in q.iterrows(): oo.append((d,s,r.f,r.y))
 b=pd.DataFrame(oo,columns=['date','s','f','y']); vv=[]
 for d,g in b.groupby('date'):
  if len(g)>=8:
   c=g.f.corr(g.y,method='spearman')
   if pd.notna(c): vv.append(c)
 vv=np.array(vv); print('horizon',h,'dates',len(vv),'IC',round(vv.mean(),6),'ICIR',round(vv.mean()/vv.std(ddof=1),6))
turn=a.sort_values(['symbol','date']).groupby('symbol').f.diff().abs().mean(); print('turnover_proxy',round(turn,6))
pd.DataFrame(sig).to_csv('scripts/miner_3_20261008_gap_reversal_signal.csv',index=False); print('signal_rows',len(sig),'symbols',a.symbol.nunique())
