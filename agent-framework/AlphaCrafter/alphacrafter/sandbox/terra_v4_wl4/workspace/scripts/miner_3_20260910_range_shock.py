import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C=pd.Timestamp('2026-09-09')
rows=[]; sig=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').sort_values('date').set_index('date').loc[:C]
 prev=x.close.shift(1); tr=pd.concat([x.high-x.low,(x.high-prev).abs(),(x.low-prev).abs()],axis=1).max(axis=1)
 atr=tr.rolling(20,min_periods=15).mean(); f=-(tr/(atr+1e-12)-1); y=x.close.shift(-1)/x.close-1
 z=pd.DataFrame({'f':f,'y':y}).dropna()
 rows += [(d,s,float(q.f),float(q.y)) for d,q in z.iterrows()]
 sig += [{'date':d.strftime('%Y-%m-%d'),'symbol':s,'signal':float(v)} for d,v in f.dropna().items()]
a=pd.DataFrame(rows,columns=['date','symbol','f','y']); out=[]
for d,g in a.groupby('date'):
 if len(g)>=8:
  q=g.f.corr(g.y,method='spearman')
  if pd.notna(q): out.append((d,q,len(g)))
z=pd.DataFrame(out,columns=['date','ic','n']); print('cutoff',C.date(),'dates',len(z),'avgN',round(z.n.mean(),2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),4))
for h in [5,10]:
 rr=[]
 for s in U:
  x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').sort_values('date').set_index('date').loc[:C]; p=x.close.shift(1); tr=pd.concat([x.high-x.low,(x.high-p).abs(),(x.low-p).abs()],axis=1).max(axis=1); f=-(tr/tr.rolling(20,min_periods=15).mean()-1); y=x.close.shift(-h)/x.close-1; q=pd.DataFrame({'f':f,'y':y}).dropna(); rr += [(d,float(v.f),float(v.y)) for d,v in q.iterrows()]
 b=pd.DataFrame(rr,columns=['date','f','y']); oo=[]
 for d,g in b.groupby('date'):
  if len(g)>=8:
   q=g.f.corr(g.y,method='spearman')
   if pd.notna(q): oo.append(q)
 oo=np.array(oo); print('horizon',h,'dates',len(oo),'IC',round(oo.mean(),6),'ICIR',round(oo.mean()/oo.std(ddof=1),6))
for yr,g in z.groupby(z.date.dt.year): print('regime',yr,len(g),round(g.ic.mean(),5),round(g.ic.mean()/g.ic.std(ddof=1),4))
print('turnover_proxy',round(a.sort_values(['symbol','date']).groupby('symbol').f.diff().abs().mean(),5)); pd.DataFrame(sig).to_csv('scripts/miner_3_20260910_range_shock_signal.csv',index=False); print('signal_rows',len(sig))
