import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2026-07-15')
px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d[d.index<=end]
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# candidates: medium momentum, short reversal, low vol, trend/vol
fac={
 'mom_20':p.pct_change(20),
 'mom_60':p.pct_change(60),
 'reversal_5':-p.pct_change(5),
 'lowvol_20':-r.rolling(20).std(),
 'trend_eff_20':p.pct_change(20)/r.rolling(20).std(),
 'mom_20_lowvol':p.pct_change(20)/r.rolling(20).std(),
}
fwd=p.shift(-1).div(p)-1
for name,x in fac.items():
 vals=[]; dates=[]; cov=[]
 for dt in p.index:
  if dt>end: continue
  a=x.loc[dt]; b=fwd.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); cov.append(len(z)/15)
 ic=np.array(vals); mean=ic.mean(); sd=ic.std(ddof=1); icir=mean/sd*np.sqrt(252) if sd else np.nan
 # 10d forward decay
 for h in [1,5,10]:
  ff=p.shift(-h).div(p)-1; vv=[]
  for dt in p.index:
   if dt>end: continue
   z=pd.concat([x.loc[dt],ff.loc[dt]],axis=1).dropna()
   if len(z)>=8: vv.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  if h==1: pass
  print(name,'h',h,'IC',round(np.mean(vv),5),'n',len(vv))
 print('SUMMARY',name,'mean',round(mean,6),'icir',round(icir,4),'hit',round((ic>0).mean(),3),'dates',len(ic),'avgcov',round(np.mean(cov),3),'turnover_proxy',round(np.mean(np.abs(x.diff()).mean(axis=1).dropna()),6))
