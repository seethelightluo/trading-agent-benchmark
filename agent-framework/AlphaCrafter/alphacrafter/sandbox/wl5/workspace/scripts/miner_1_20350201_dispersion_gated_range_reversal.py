import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).sort_values('date').set_index('date')
 px[s]=d['close'].astype(float)
close=pd.DataFrame(px).sort_index(); ret=close.pct_change()
# shock reversal: recent 5d loss normalized by 20d realized vol, only in high cross-asset dispersion
r5=close.pct_change(5); vol20=ret.rolling(20).std(); disp=ret.rolling(20).std().mean(axis=1)
# percentile threshold is causal rolling cross-date threshold
thr=disp.rolling(252,min_periods=126).quantile(.60)
f=(-r5/(vol20*np.sqrt(5))).where(disp.gt(thr), -r5/(vol20*np.sqrt(5))*0.35)
f=f.replace([np.inf,-np.inf],np.nan)
rows=[]
for i in range(len(close)-10):
 dt=close.index[i]; f0=f.iloc[i]; fr=close.iloc[i+1:i+11].iloc[-1]/close.iloc[i]/1 # unused
 # forward 10-session return from close t to close t+10
 fw=close.iloc[i+10]/close.iloc[i]-1
 ok=f0.notna()&fw.notna()
 if ok.sum()>=8:
  ic=spearmanr(f0[ok],fw[ok]).statistic
  rows.append((dt,ic,ok.sum()))
out=pd.DataFrame(rows,columns=['date','ic','n']).dropna()
# recent validation begins after ample warmup
out=out[out.date>='2023-06-30']
mean=out.ic.mean(); sd=out.ic.std(ddof=1); icir=mean/sd if sd else np.nan
# turnover rank changes, cross-sectional coverage
ranks=f.rank(axis=1,pct=True); turnover=(ranks.diff().abs().mean(axis=1)/2).loc[out.date].mean()
coverage=np.mean([n/15 for n in out.n])
print('dates',len(out),'mean_n',out.n.mean(),'coverage',coverage)
print('ic10',mean,'icir',icir,'hit',np.mean(out.ic>0),'turnover',turnover)
for h in [5,10,20]:
 z=[]
 for i in range(len(close)-h):
  if close.index[i]<pd.Timestamp('2023-06-30'): continue
  a=f.iloc[i]; b=close.iloc[i+h]/close.iloc[i]-1; ok=a.notna()&b.notna()
  if ok.sum()>=8:z.append(spearmanr(a[ok],b[ok]).statistic)
 print('decay',h,np.nanmean(z),len(z))
out.to_csv('scripts/miner_1_20350201_dispersion_gated_range_reversal_signal.csv',index=False)
