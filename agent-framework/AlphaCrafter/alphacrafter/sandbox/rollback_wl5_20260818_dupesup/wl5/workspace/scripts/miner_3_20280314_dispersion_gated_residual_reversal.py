import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-03-13'); b=Path('../persistent/stock_data')
P=pd.DataFrame({s:pd.read_csv(b/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}).sort_index().loc[:end].ffill(); R=P.pct_change(); bench=R.mean(axis=1)
cov=R.rolling(60,min_periods=30).cov(bench); var=bench.rolling(60,min_periods=30).var(); beta=cov.div(var,axis=0)
resid=P.pct_change(5)-beta.mul(bench.rolling(5).sum(),axis=0)
base=-resid.sub(resid.median(axis=1),axis=0)
# dispersion-conditioned: emphasize reversal when cross-sectional 1d dispersion is above its trailing median
csdisp=R.std(axis=1); gate=(csdisp>csdisp.rolling(60,min_periods=30).median()).astype(float)
f=base.mul(gate,axis=0); y=P.shift(-10)/P-1
def calc(lo=None,hi=None):
 a=[];ns=[]
 for dt in f.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 a=np.asarray(a); return len(a),float(np.mean(ns)),float(a.mean()),float(a.mean()/a.std(ddof=1)),float(np.mean(a>0))
print('dispersion-gated beta residual reversal',end.date()); print('ALL',calc())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-03-13')]:print('REG',lo,hi,calc(lo,hi))
rk=f.rank(axis=1,pct=True);print('coverage_dates',float(f.notna().sum(axis=1).ge(8).mean()),'turnover',float((rk-rk.shift()).abs().mean(axis=1).dropna().mean()),'avg_valid',float(f.notna().mean().mean()))
f.to_csv('scripts/miner_3_20280314_dispersion_gated_residual_signal.csv')
