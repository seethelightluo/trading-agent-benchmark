import pandas as pd, numpy as np, json
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C=pd.Timestamp('2026-07-15')
rows=[]; signals=[]
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); x=x.loc[:C]
 f=-(x.close/x.open-1); y=x.close.shift(-1)/x.close-1
 signals += [{'date':d.strftime('%Y-%m-%d'),'symbol':s,'signal':float(v)} for d,v in f.dropna().items()]
 z=pd.DataFrame({'f':f,'y':y}).dropna()
 for d,r in z.iterrows(): rows.append((d,s,float(r.f),float(r.y)))
a=pd.DataFrame(rows,columns=['date','symbol','f','y']); out=[]
for d,g in a.groupby('date'):
 if len(g)>=8:
  ic=g.f.corr(g.y,method='spearman')
  if pd.notna(ic): out.append((d,ic,len(g)))
z=pd.DataFrame(out,columns=['date','ic','n']); ic=z.ic.mean(); icir=ic/z.ic.std(ddof=1)
print('dates',len(z),'avgN',z.n.mean(),'coverage',a.f.notna().mean(),'IC',ic,'ICIR',icir,'hit',(z.ic>0).mean())
for yr,g in z.groupby(z.date.dt.year): print('regime',yr,len(g),g.ic.mean(),g.ic.mean()/g.ic.std(ddof=1))
print('turnover_proxy',a.sort_values(['symbol','date']).groupby('symbol').f.rank(pct=True).groupby(a.sort_values(['symbol','date']).symbol).diff().abs().mean())
pd.DataFrame(signals).to_csv('scripts/miner_3_20260827_intraday_reversal_signal.csv',index=False)
print('signal_rows',len(signals),'symbols',a.symbol.nunique())
