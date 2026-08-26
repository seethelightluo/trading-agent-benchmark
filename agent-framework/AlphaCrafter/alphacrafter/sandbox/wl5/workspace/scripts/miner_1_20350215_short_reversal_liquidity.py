import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];P='../persistent/stock_data'
d={s:pd.read_csv(f'{P}/{s}.csv',parse_dates=['date']).sort_values('date').set_index('date') for s in U}
close=pd.DataFrame({s:x.close.astype(float) for s,x in d.items()}).sort_index(); ret=close.pct_change()
# Very short reversal, scaled by recent volatility and conditioned on an unusually large recent move.
r3=close.pct_change(3); v20=ret.rolling(20,min_periods=15).std(); z=(-r3/(v20*np.sqrt(3)))
# emphasize shocks only, but retain a small signal otherwise
f=z*(1+0.75*(r3.abs()>r3.abs().rolling(126,min_periods=63).quantile(.75)))
f=f.replace([np.inf,-np.inf],np.nan)
rows=[]
for i in range(len(close)-20):
 if close.index[i]<pd.Timestamp('2023-06-30'):continue
 a=f.iloc[i]; b=close.iloc[i+10]/close.iloc[i]-1; ok=a.notna()&b.notna()
 if ok.sum()>=8:rows.append((close.index[i],spearmanr(a[ok],b[ok]).statistic,ok.sum()))
out=pd.DataFrame(rows,columns=['date','ic','n']).dropna();m=out.ic.mean();sd=out.ic.std(ddof=1)
rank=f.rank(axis=1,pct=True);to=(rank.diff().abs().mean(axis=1)/2).reindex(out.date).mean()
print('dates',len(out),'mean_n',out.n.mean(),'coverage',out.n.mean()/15);print('ic10',m,'icir',m/sd,'hit',np.mean(out.ic>0),'turnover',to)
for h in [5,10,20]:
 z=[]
 for i in range(len(close)-h):
  if close.index[i]<pd.Timestamp('2023-06-30'):continue
  a=f.iloc[i];b=close.iloc[i+h]/close.iloc[i]-1;ok=a.notna()&b.notna()
  if ok.sum()>=8:z.append(spearmanr(a[ok],b[ok]).statistic)
 print('decay',h,np.mean(z),len(z))
out.to_csv('scripts/miner_1_20350215_short_reversal_liquidity_signal.csv',index=False)
