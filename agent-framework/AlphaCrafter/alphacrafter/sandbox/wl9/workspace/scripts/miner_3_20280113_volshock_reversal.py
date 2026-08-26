import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
from pathlib import Path

symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-01-13')
prices={}
for s in symbols:
    f=Path('../persistent/stock_data')/(s+'.csv')
    d=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date')
    prices[s]=d['close'].loc[:end]
px=pd.DataFrame(prices).sort_index()
ret=px.pct_change()
# Volatility-shock weighted reversal: reversal is amplified only when short vol exceeds its medium baseline.
r5=px.pct_change(5)
v20=ret.rolling(20,min_periods=15).std()*np.sqrt(5)
v5=ret.rolling(5,min_periods=4).std()
v60=ret.rolling(60,min_periods=40).std()
shock=(v5/(v60+1e-12)-1).clip(lower=0)
factor=-r5/(v20+1e-12)*(1+shock)
# forward one common observation/date return
fwd=px.shift(-1)/px-1
rows=[]; dates=sorted(set(factor.index)&set(fwd.index))
for dt in dates:
    x=factor.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
        rows.append((dt,ic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for label,sub in [('all',r),('pre_online',r.loc[:'2026-07-15']),('online',r.loc['2026-07-16':]),('2027',r.loc['2027-01-01':])]:
    if len(sub): print(label,'dates',len(sub),'mean_ic',sub.ic.mean(),'icir',sub.ic.mean()/(sub.ic.std(ddof=1)/np.sqrt(len(sub))) if len(sub)>1 else np.nan,'hit', (sub.ic>0).mean(),'avg_n',sub.n.mean())
print('coverage',factor.notna().mean().mean(),'dates',len(r),'avg instruments',r.n.mean())
# rank turnover: mean fraction rankings changed from prior common date
rank=factor.rank(axis=1,pct=True); turn=(rank.diff().abs().mean(axis=1)).dropna().mean(); print('turnover_rank_change',turn)
# decay horizons using shift h and per-asset forward observations via date panel
for h in [1,5,10]:
    fy=px.shift(-h)/px-1; rr=[]
    for dt in factor.index:
      z=pd.concat([factor.loc[dt],fy.loc[dt]],axis=1).dropna()
      if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
    print('decay',h,'n',len(rr),'mean_ic',np.nanmean(rr),'icir',np.nanmean(rr)/(np.nanstd(rr,ddof=1)/np.sqrt(len(rr))))
