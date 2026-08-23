import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in watch:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p): px[s]=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].astype(float)
prices=pd.DataFrame(px).sort_index(); ret20=prices.pct_change(20); vol60=prices.pct_change().rolling(60).std()*np.sqrt(20)
raw=ret20/vol60.replace(0,np.nan)
risk=[x for x in ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','COPPER','WTI','BTC','ETH'] if x in prices]
breadth=ret20[risk].gt(0).mean(axis=1)
signal=raw.mul(0.5+breadth,axis=0)
fwd=prices.shift(-10)/prices-1
rows=[]; dates=[]; ns=[]
for dt in prices.index:
 a=signal.loc[dt]; b=fwd.loc[dt]; ok=a.notna()&b.notna()
 if ok.sum()>=8: rows.append(spearmanr(a[ok],b[ok]).statistic); dates.append(dt); ns.append(ok.sum())
ic=np.array(rows); mean=float(np.nanmean(ic)); sd=float(np.nanstd(ic,ddof=1)); icir=mean/sd*np.sqrt(252/10)
turn=[]
for i in range(1,len(dates)):
 x=signal.loc[dates[i-1]]; y=signal.loc[dates[i]]; ok=x.notna()&y.notna()
 if ok.sum()>=8: turn.append(np.mean((x[ok].rank()-y[ok].rank()).abs())/ok.sum())
reg={}
for name,lo,hi in [('2020-24','2020-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027-28','2027-01-01','2028-12-31')]:
 mask=(pd.Series(dates)>=lo)&(pd.Series(dates)<=hi); z=ic[mask.values]; reg[name]=(len(z),float(np.mean(z)),float(np.mean(z))/float(np.std(z,ddof=1))*np.sqrt(252/10) if len(z)>1 else None)
print('factor=breadth_gated_voladj_momentum_20d dates',len(ic),'range',dates[0],dates[-1],'avg_n',float(np.mean(ns)))
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f coverage %.4f'%(mean,icir,float(np.mean(ic>0)),float(np.mean(turn)),float(signal.notna().sum().sum()/signal.size)))
print('regimes',reg)
pd.DataFrame({'date':dates,'ic':ic}).to_csv('scripts/miner_2_20280824_breadth_gated_momentum_ic.csv',index=False)
signal.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').to_csv('scripts/miner_2_20280824_breadth_gated_momentum_signal.csv',index=False)
