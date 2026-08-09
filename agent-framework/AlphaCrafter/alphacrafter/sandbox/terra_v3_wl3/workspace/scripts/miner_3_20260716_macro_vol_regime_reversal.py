import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U}
p=pd.DataFrame(px).sort_index(); rets=p.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
# Macro interaction: reversal in high/rising volatility, momentum in calm/falling vol.
# signal is -5d asset return times signed macro state; state = +1 if VIX 5d change positive, -1 otherwise
state=np.where(vix.pct_change(5)>0,1.,-1.)
factor=-rets.rolling(5).sum().mul(state,axis=0)
fwd=rets.shift(-1)
ics=[]; dates=[]; vals=[]; fwds=[]
for d in factor.index:
 x=factor.loc[d]; y=fwd.loc[d]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(d); vals.append(z.iloc[:,0]); fwds.append(z.iloc[:,1])
ic=np.array(ics); print('dates',len(ic),'meanIC',ic.mean(),'std',ic.std(ddof=1),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean(),'coverage',np.mean([len(x)/15 for x in vals]))
for h in [5,10]:
 yy=rets.shift(-h); ii=[]
 for d in factor.index:
  z=pd.concat([factor.loc[d],yy.loc[d]],axis=1).dropna()
  if len(z)>=8: ii.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.mean(ii),len(ii))
# rank turnover
r=factor.rank(axis=1,pct=True); turn=(r.diff().abs().mean(axis=1)).dropna().mean(); print('turnover',turn)
# correlations with existing 3/5 reversal, across all asset-date values
base=-rets.rolling(5).sum(); short=-rets.rolling(3).sum()
a=factor.stack(); print('corr5',a.corr(base.stack()),'corr3',a.corr(short.stack()))
print('recent',ic[-250:].mean(),ic[-250:].std(ddof=1),len(ic[-250:]))
