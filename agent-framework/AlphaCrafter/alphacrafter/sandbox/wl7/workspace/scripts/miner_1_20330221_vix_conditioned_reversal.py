import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={a:pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close']
allidx=sorted(set.intersection(*[set(x.index) for x in px.values()]) & set(vix.index))
idx=pd.DatetimeIndex(allidx); P=pd.DataFrame({a:px[a].reindex(idx) for a in assets}); V=vix.reindex(idx)
ret=P.pct_change()
# lagged 10-session reversal, scaled by vol, amplified in high-VIX regime; all inputs end t-1
raw=-(P.shift(10)/P.shift(20)-1)/(ret.rolling(40).std().shift(10)+1e-12)
vz=(V-V.rolling(60).mean())/(V.rolling(60).std()+1e-12)
mult=(1+0.8*vz.clip(-1,2)).clip(0.25,2.6)
f=raw.mul(mult,axis=0)
res={h:[] for h in [5,10,20]}; nins=[]; turns=[]; dates=[]
for j in range(0,len(idx)-20):
    # forward returns t through t+h-1; factor known at t based on t-1 or earlier
    x=f.iloc[j]; y=P.iloc[j+1+j*0] # placeholder
    for h in res:
        if j+h>=len(idx): continue
        z=pd.concat([x,P.iloc[j+h]/P.iloc[j]-1],axis=1).dropna()
        if len(z)>=8: res[h].append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
    if x.notna().sum()>=8: nins.append(x.notna().sum()); dates.append(idx[j])
for h,a in res.items():
 a=np.array(a); print('H',h,'obs',len(a),'IC %.6f'%np.nanmean(a),'ICIR %.6f'%(np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12)),'hit %.4f'%np.mean(a>0))
print('dates',len(dates),'avg_valid',np.mean(nins),'coverage',np.mean(P.notna().values))
# regime diagnostics
for q,name in [(vz>1,'high_vix'),(vz<=1,'normal')]:
 vals=[]
 for j in range(len(idx)-10):
  if not q.iloc[j] or j+10>=len(idx): continue
  z=pd.concat([f.iloc[j],P.iloc[j+10]/P.iloc[j]-1],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print(name,len(vals),np.mean(vals) if vals else np.nan, (np.mean(vals)/(np.std(vals,ddof=1)+1e-12)) if len(vals)>1 else np.nan)
# rank turnover on consecutive valid dates
r=f.rank(axis=1,pct=True); d=[]
for j in range(1,len(r)):
 z=pd.concat([r.iloc[j-1],r.iloc[j]],axis=1).dropna()
 if len(z)>=8:d.append(np.mean(abs(z.iloc[:,0]-z.iloc[:,1])))
print('rank_turnover_proxy',np.mean(d))
out=pd.DataFrame(f,index=idx,columns=assets); out.to_csv('scripts/miner_1_20330221_vix_conditioned_reversal_signal.csv')
