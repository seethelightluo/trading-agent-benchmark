import pandas as pd, numpy as np
from scipy.stats import spearmanr
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2028-01-10'); root=Path('../persistent/stock_data')
P=pd.DataFrame({s:pd.read_csv(root/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index() for s in U}).sort_index().loc[:end].ffill()
R=P.pct_change(); down=R.where(R<0,0.0).rolling(20,min_periods=15).std(); mom=P/P.shift(20)-1
# downside-risk-adjusted medium momentum; cross-sectional demean keeps level effects out
f=mom/(down*np.sqrt(252)+1e-8); f=f.sub(f.median(axis=1),axis=0)
y=P.shift(-10)/P-1

def calc(lo=None,hi=None):
 vals=[]; ns=[]
 for dt in f.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 a=np.asarray(vals)
 return len(a), round(float(np.mean(ns)),2), round(float(a.mean()),6), round(float(a.mean()/a.std(ddof=1)),6), round(float(np.mean(a>0)),4)
print('downside-risk-adjusted 20d momentum, end',end.date())
print('ALL',calc())
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-12-31'),('2027','2028-01-10')]: print('REG',lo,hi,calc(lo,hi))
for h in [1,5,10,20]:
 yy=P.shift(-h)/P-1; old=y;y=yy; print('H',h,calc());y=old
rk=f.rank(axis=1,pct=True)
print('coverage',round(float(f.notna().mean().mean()),4),'dates>=8',round(float(f.notna().sum(axis=1).ge(8).mean()),4),'turnover',round(float((rk-rk.shift()).abs().mean(axis=1).dropna().mean()),6),'avgN',round(float(f.notna().sum(axis=1).mean()),2))
f.to_csv('scripts/miner_2_20280111_downside_risk_momentum_signal.csv')
