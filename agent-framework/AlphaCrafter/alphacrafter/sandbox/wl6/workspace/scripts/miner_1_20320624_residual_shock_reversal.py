import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2032-06-23')
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x['date']=pd.to_datetime(x.date); D[s]=x.loc[x.date<=cutoff].set_index('date').close
px=pd.concat(D,axis=1).sort_index(); r=np.log(px).diff(); bench=r.mean(axis=1)
cov=r.rolling(60,min_periods=40).cov(bench); var=bench.rolling(60,min_periods=40).var(); beta=cov.div(var,axis=0)
resid=r.sub(beta.mul(bench,axis=0)); factor=-resid.rolling(3,min_periods=3).sum().div(resid.rolling(20,min_periods=15).std()); fwd=px.shift(-10).div(px)-1
ics=[]; dates=[]; turns=[]; ns=[]; prev=None
for i,d in enumerate(px.index):
 if d>cutoff or i+10>=len(px): continue
 z=pd.concat([factor.loc[d],fwd.loc[d]],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(d); ns.append(len(z)); ranks=factor.loc[d].rank(pct=True); turns.append(np.nan if prev is None else (ranks-prev).abs().mean()); prev=ranks
v=pd.Series(ics,index=dates).dropna(); print('cutoff',cutoff.date(),'dates',len(v),'avgN',np.mean(ns),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'hit',(v>0).mean(),'turn',np.nanmean(turns),'assets',len(D),'last_data',px.index[-1].date())
for y,g in v.groupby(v.index.year): print(y,round(g.mean(),5),len(g))
