import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2034-07-09'); xs={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').loc[:cut]; px=d.close.astype(float); vol=px.pct_change().rolling(20,min_periods=15).std()*np.sqrt(252); v=d.volume.astype(float).replace(0,np.nan); vr=v.rolling(5,min_periods=3).mean()/v.rolling(30,min_periods=15).mean(); xs[s]=pd.DataFrame({'sig':(-px.pct_change(5)/vol*vr.pow(.25)).shift(1),'fwd':px.shift(-10)/px-1})
rows=[]
for dt in sorted(set.intersection(*[set(x.index) for x in xs.values()])):
 z=pd.DataFrame({'s':{s:xs[s].loc[dt,'sig'] for s in U},'f':{s:xs[s].loc[dt,'fwd'] for s in U}}).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.s,z.f).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for name,start in [('all',None),('recent180',cut-pd.Timedelta(days=270)),('recent500',cut-pd.Timedelta(days=730)),('recent750',cut-pd.Timedelta(days=1100))]:
 q=r if start is None else r.loc[start:]; print(name,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
print('coverage',round(r.n.sum()/(len(r)*15),4))
for h in [1,5,10,20]:
 vals=[]
 for s in U:
  d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').loc[:cut]; px=d.close.astype(float); vol=px.pct_change().rolling(20,min_periods=15).std()*np.sqrt(252); v=d.volume.astype(float).replace(0,np.nan); vr=v.rolling(5,min_periods=3).mean()/v.rolling(30,min_periods=15).mean(); vals.append(pd.DataFrame({'s':(-px.pct_change(5)/vol*vr.pow(.25)).shift(1),'f':px.shift(-h)/px-1}))
 rr=[]
 for dt in sorted(set.intersection(*[set(x.index) for x in vals])):
  z=pd.DataFrame({s:vals[i].loc[dt] for i,s in enumerate(U)}).T.dropna()
  if len(z)>=8: rr.append(spearmanr(z.s,z.f).statistic)
 print('H',h,'IC',round(np.nanmean(rr),6),'n',len(rr))
pd.concat([pd.DataFrame({'date':xs[s].index,'symbol':s,'signal':xs[s].sig.values}) for s in U]).dropna().to_csv('scripts/miner_1_20340710_volume_confirmed_reversal_signal.csv',index=False)
