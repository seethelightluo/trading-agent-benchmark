import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}; vol={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')
 px[a]=d['close']; vol[a]=d['volume'] if 'volume' in d else pd.Series(index=d.index,dtype=float)
close=pd.DataFrame(px).sort_index(); volume=pd.DataFrame(vol).reindex(close.index)
# Idea: liquidity-adjusted intermediate momentum. Reward 20d trend, penalize
# abnormal turnover/volume volatility; all inputs lagged one day by evaluation construction.
r=close.pct_change(); mom=close.pct_change(20)
logv=np.log(volume.replace(0,np.nan)); vvol=logv.rolling(20,min_periods=10).std()
f=mom/(1+vvol) # interpretable, broad coverage
for h in [1,5,10,20]:
 vals=[]; dates=[]
 for i in range(len(close)-h):
  s=f.iloc[i]; y=close.pct_change(h).iloc[i+1] # forward from next close
  z=pd.concat([s,y],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(close.index[i])
 x=np.array(vals); print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d meanN %.2f'%(np.nanmean(x),np.nanmean(x)/np.nanstd(x,ddof=1),np.mean(x>0),len(x), np.nanmean([len(pd.concat([f.loc[d],close.pct_change(h).loc[d]],axis=1).dropna()) for d in dates])))
# 10d rank turnover and coverage
ranks=f.rank(axis=1,pct=True); turn=(ranks.diff(10).abs().mean(axis=1)).dropna()
print('coverage',f.notna().mean().mean(),'turnover10',turn.mean(),'n_assets',len(assets),'dates',len(close))
for lo,hi in [('2020','2024'),('2025','2028'),('2029','2031')]:
 x=[]
 for i in range(len(close)-1):
  if str(close.index[i].year)[:4] in []: pass
  if close.index[i].strftime('%Y')>=lo and close.index[i].strftime('%Y')<=hi:
   z=pd.concat([f.iloc[i],close.pct_change(1).iloc[i+1]],axis=1).dropna()
   if len(z)>=8:x.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('regime',lo,hi,'IC %.6f ICIR %.6f n %d'%(np.mean(x),np.mean(x)/np.std(x,ddof=1),len(x)))
# pooled signal saved for possible audit
out=f.stack().rename('signal').reset_index();out.columns=['date','asset','signal'];out.to_csv('/tmp/candidate_signal.csv',index=False)
