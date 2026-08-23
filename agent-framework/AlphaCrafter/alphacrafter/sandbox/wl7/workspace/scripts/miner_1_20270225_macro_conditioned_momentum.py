import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data, get_account_dict

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Macro-conditioned medium momentum: risk-adjusted 20d momentum, attenuated when VIX rises sharply.
prices={}
for s in U:
    d=get_stock_daily_data(s, days=3000)
    if d is not None and len(d): prices[s]=d.assign(date=pd.to_datetime(d['date'])).set_index('date')['close'].astype(float)
p=pd.DataFrame(prices).sort_index()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(p.index).ffill()
r=p.pct_change()
base=(p/p.shift(20)-1)/(r.rolling(60).std()*np.sqrt(20)+1e-12)
# use only information through t: shift signal one session; VIX shock regime is also lagged
vixshock=vix.pct_change(5)
regime=(1-0.7*np.clip(vixshock/0.25,-1,1)) # rising VIX suppresses momentum; falling VIX boosts it
fac=base.mul(regime, axis=0).shift(1)
fwd=p.shift(-1)/p-1
rows=[]
for dt in fac.index:
 x=fac.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
out=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('cutoff',out.index.max(),'dates',len(out),'avg_n',out.n.mean(),'coverage',fac.notna().sum().sum()/(len(fac)*len(U)))
print('IC %.6f ICIR %.6f hit %.4f turnover %.4f'%(out.ic.mean(),out.ic.mean()/(out.ic.std(ddof=1)+1e-12), (out.ic>0).mean(), fac.rank(axis=1).diff().abs().mean().mean()/len(U)))
for name,sl in [('2020-22',out.loc['2020':'2022']),('2023-24',out.loc['2023':'2024']),('2025-27',out.loc['2025':'2027']),('online',out.loc['2026-07-16':'2027-02-24'])]:
 print(name,len(sl),sl.ic.mean() if len(sl) else np.nan,sl.ic.mean()/(sl.ic.std(ddof=1)+1e-12) if len(sl)>1 else np.nan)
for h in [2,5,10,20]:
 yy=p.shift(-h)/p-1; rr=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,np.nanmean(rr),len(rr))
# save exact signal artifact for provenance
fac.loc[:'2027-02-24'].to_csv('scripts/miner_1_20270225_macro_conditioned_momentum_signal.csv')
