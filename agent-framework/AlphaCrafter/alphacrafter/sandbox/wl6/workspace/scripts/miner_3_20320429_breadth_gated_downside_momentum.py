import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for a in assets}).sort_index().loc[:'2032-04-28']
r=P.pct_change(); ret20=P/P.shift(20)-1
neg=r.clip(upper=0); down=np.sqrt((neg**2).rolling(40,min_periods=20).mean())*np.sqrt(20)
breadth=(ret20>0).mean(axis=1); gate=(0.65+0.7*breadth).clip(0.65,1.35)
f=ret20.div(down).mul(gate,axis=0)
print('cutoff',P.index.max().date(),'dates',len(P),'assets',P.shape[1],'factor_coverage',round(f.notna().stack().mean(),4))
for h in [5,10,20]:
  ics=[]; ns=[]; turns=[]; prev=None; dates=[]; fr=P.shift(-h)/P-1
  for dt in P.index:
    z=pd.concat([f.loc[dt].rename('x'),fr.loc[dt].rename('y')],axis=1).dropna()
    if len(z)>=8: ics.append(spearmanr(z.x,z.y).statistic);ns.append(len(z));dates.append(dt)
    rr=f.loc[dt].rank(pct=True)
    if prev is not None:
      q=pd.concat([rr,prev],axis=1).dropna()
      if len(q): turns.append(np.mean(abs(q.iloc[:,0]-q.iloc[:,1])))
    prev=rr
  a=np.array(ics)
  print({'horizon':h,'valid_dates':len(a),'avg_instruments':round(np.mean(ns),3),'ic':round(float(np.mean(a)),6),'icir':round(float(np.mean(a)/np.std(a,ddof=1)),6),'hit_ratio':round(float(np.mean(a>0)),4),'turnover':round(float(np.mean(turns)),6)})
  if h==20:
    q=pd.DataFrame({'ic':a},index=dates)
    print('regimes',q.groupby(q.index.year).ic.agg(['mean','count']).round(6).to_dict('index'))
