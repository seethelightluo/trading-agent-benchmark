import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for a in assets:
 x=pd.read_csv('../persistent/stock_data/'+a+'.csv'); x['date']=pd.to_datetime(x.date); d[a]=x.set_index('date').close
px=pd.DataFrame(d).sort_index(); rets=px.pct_change(); dates=px.index
# 40-observation participation/trend, lagged one completed day
mom=px.pct_change(40)/rets.rolling(40).std()
up= (rets>0).rolling(40).mean()-0.5
raw=mom*up
# residualize cross-sectionally against ordinary risk adjusted momentum each date
sig=pd.DataFrame(index=dates,columns=assets,dtype=float)
for t in dates:
 y=raw.loc[t]; x=mom.loc[t]; z=pd.concat([y,x],axis=1).dropna()
 if len(z)>=8:
  xx=np.column_stack([np.ones(len(z)),z.iloc[:,1].values]); b=np.linalg.lstsq(xx,z.iloc[:,0].values,rcond=None)[0]
  sig.loc[t,z.index]=z.iloc[:,0]-xx@b
sig=sig.shift(1)
for h in [1,5,10,20]:
 fwd=px.shift(-h)/px-1; ics=[]; n=[]
 for t in dates:
  z=pd.concat([sig.loc[t],fwd.loc[t]],axis=1).dropna()
  if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); n.append(len(z))
  
 a=np.array(ics); print('H',h,'dates',len(a),'meanN',np.mean(n),'IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)))
# turnover and coverage
ranks=sig.rank(axis=1,pct=True); print('coverage',sig.notna().mean().mean(),'turn10',np.nanmean((ranks-ranks.shift(10)).abs().mean(axis=1)))
# regime
for label,lo,hi in [('2020-24','2020','2024-12-31'),('2025-27','2025','2027-12-31'),('2028-29','2028','2029-12-31')]:
 a=[]
 for t in dates[(dates>=lo)&(dates<=hi)]:
  z=pd.concat([sig.loc[t],(px.shift(-10)/px-1).loc[t]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print(label,len(a),np.mean(a),np.mean(a)/np.std(a,ddof=1))
