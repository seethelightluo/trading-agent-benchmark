import pandas as pd,numpy as np
from scipy.stats import spearmanr
END=pd.Timestamp('2026-12-17'); U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
S={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=END].set_index('date'); S[s]=d.close
p=pd.concat(S,axis=1).sort_index(); p=p[~p.index.duplicated(keep='last')]; r=p.pct_change()
rr=r.rolling(5,min_periods=5).sum(); res=rr.sub(rr.median(axis=1),axis=0)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date').set_index('date'); vix=v[[c for c in v if c.lower() in ('close','value','vix')][0]]
vp=vix.reindex(p.index).ffill().shift(1); q=vp.rolling(60,min_periods=30).rank(pct=True); high=(q>0.7).astype(float)
f=(-res.shift(1)*(1+0.75*high.values[:,None])).replace([np.inf,-np.inf],np.nan)
out=[]
for s in U: out.append(pd.DataFrame({'date':p.index,'symbol':s,'factor':f[s],'y1':p[s].shift(-1)/p[s]-1,'y5':p[s].shift(-5)/p[s]-1,'y10':p[s].shift(-10)/p[s]-1}))
x=pd.concat(out,ignore_index=True)
for h in [1,5,10]:
 a=[]
 for dt,g in x.groupby('date'):
  g=g.dropna(subset=['factor',f'y{h}'])
  if len(g)>=8 and g.factor.nunique()>1 and g[f'y{h}'].nunique()>1:
   z=spearmanr(g.factor,g[f'y{h}']).statistic
   if np.isfinite(z): a.append((dt,z,len(g)))
 z=pd.DataFrame(a,columns=['date','ic','n']); z.date=pd.to_datetime(z.date); qic=z.ic
 print('H',h,'dates',len(z),'avgN',round(z.n.mean(),2),'IC',round(qic.mean(),6),'ICIR',round(qic.mean()/qic.std(ddof=1),6),'hit',round((qic>0).mean(),4))
 if h==1:
  for yr,g in z.groupby(z.date.dt.year): print('YR',yr,len(g),round(g.ic.mean(),5),round(g.ic.mean()/g.ic.std(ddof=1),4))
vv=x.dropna(subset=['factor']); rank=vv.pivot(index='date',columns='symbol',values='factor').rank(axis=1,pct=True)
print('coverage',round(len(vv)/len(x),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4),'symbols',x.symbol.nunique(),'period',x.date.min(),x.date.max())
x.to_csv('scripts/miner_1_20261217_macro_residual_reversal_signal.csv',index=False)
