import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Macro-conditioned residual reversal: negate 5d residual return, scaled by 20d vol;
# activate more strongly when trailing cross-asset median 20d trend is negative (risk-off)
med20=r.rolling(20).sum().median(axis=1)
vol=r.rolling(20).std().mean(axis=1)
trend_z=(med20/vol.replace(0,np.nan)).clip(-3,3)
activation=1+0.35*(-trend_z).clip(-1,1)
raw=r.rolling(5).sum().sub(r.rolling(5).sum().median(axis=1),axis=0)
f=-raw.div(r.rolling(20).std().replace(0,np.nan))*activation.values[:,None]
# forward 10d arithmetic return
fr=p.shift(-10).div(p)-1
rows=[]
for d in f.index:
 x=f.loc[d]; y=fr.loc[d]
 ok=x.notna()&y.notna()
 if ok.sum()>=8:
  rows.append((d,spearmanr(x[ok],y[ok]).statistic,ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
# exclude last unavailable and require realistic start
z=z.loc['2020-04-01':'2034-01-04']
mean=z.ic.mean(); sd=z.ic.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
# daily paper ICIR convention annualized
# turnover as rank signal changes
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).mean()
print('dates',len(z),'meanN',z.n.mean(),'coverage',f.notna().stack().mean(),'IC10',mean,'ICIR',icir,'hit', (z.ic>0).mean(),'turnover',turn)
for h in [5,10,20]:
 yy=p.shift(-h).div(p)-1; vals=[]
 for d in f.index:
  ok=f.loc[d].notna()&yy.loc[d].notna()
  if ok.sum()>=8: vals.append(spearmanr(f.loc[d,ok],yy.loc[d,ok]).statistic)
 print('decay',h,np.nanmean(vals),len(vals))
for s,e in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2029-12-31'),('2030','2034')]: print('regime',s,e,z.loc[s:e].ic.mean())
out=pd.DataFrame(f.loc[z.index].stack(),columns=['signal']); out.index.names=['date','symbol']; out.reset_index().to_csv('scripts/miner_1_20340105_macro_residual_reversal_signal.csv',index=False)
print('artifact rows',len(out))
