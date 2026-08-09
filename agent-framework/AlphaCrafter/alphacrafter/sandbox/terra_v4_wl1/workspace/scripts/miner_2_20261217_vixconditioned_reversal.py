import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-12-17')
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date');v=v[v.date<=end].set_index('date').close;reg=(v/v.rolling(60,min_periods=30).median()-1).clip(-2,2)
rows=[]
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date');d=d[d.date<=end].set_index('date');r=d.close.pct_change();vol=r.rolling(20,min_periods=10).std();base=-r.rolling(5,min_periods=5).sum()/(vol*np.sqrt(5)+1e-12);d['factor']=base*(1+0.75*(reg.reindex(d.index).fillna(0)>0).astype(float));d['y1']=d.close.shift(-1)/d.close-1;rows.append(d[['factor','y1']].assign(symbol=s).reset_index())
x=pd.concat(rows);out=[]
for dt,g in x.groupby('date'):
 g=g.dropna(subset=['factor','y1'])
 if len(g)>=8 and g.factor.nunique()>1 and g.y1.nunique()>1:out.append((dt,spearmanr(g.factor,g.y1).statistic,len(g)))
a=pd.DataFrame(out,columns=['date','ic','n']);q=a.ic;print('dates',len(q),'avgN',round(a.n.mean(),2),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'coverage',x.factor.notna().mean())
for lo,hi in [(2019,2022),(2022,2024),(2024,2026),(2026,2027)]:
 z=a[(a.date.dt.year>lo)&(a.date.dt.year<=hi)].ic;print('regime',lo,hi,'n',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1) if len(z)>1 else np.nan)
r=x.dropna(subset=['factor']).pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True);print('turnover',r.diff().abs().mean(axis=1).mean());x.to_csv('scripts/miner_2_20261217_vixconditioned_reversal_signal.csv',index=False);print('period',x.date.min(),x.date.max(),'symbols',x.symbol.nunique())
