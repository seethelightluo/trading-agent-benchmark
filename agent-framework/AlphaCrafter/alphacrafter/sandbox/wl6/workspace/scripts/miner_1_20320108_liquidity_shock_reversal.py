import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-01-07')
px={}; vol={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').loc[:cut]
 px[s]=d.close; vol[s]=d.volume.replace(0,np.nan)
p=pd.concat(px,axis=1).sort_index(); v=pd.concat(vol,axis=1).reindex(p.index); r=p.pct_change()
# Reversal after unusually large move, normalized by its own recent volatility and conditioned on volume shock
rv=r.rolling(20).std(); move=r.rolling(3).sum(); vs=(v/v.rolling(20).mean()).clip(0.25,4)
f=-move/rv.replace(0,np.nan)*np.log(vs)
# with missing volumes, fallback pure shock reversal
f=f.fillna(-move/rv.replace(0,np.nan))
fr=p.shift(-5)/p-1
ics=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
a=np.array(ics);print('IC %.8f ICIR %.5f hit %.4f dates %d avgN %.3f coverage %.6f turnover %.6f'%(np.mean(a),np.mean(a)/np.std(a,ddof=1),np.mean(a>0),len(a),np.mean(ns),f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for h in [1,5,10,20]:
 fr=p.shift(-h)/p-1; aa=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('H',h,'%.8f'%np.mean(aa))
