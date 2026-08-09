"""Orthogonality screen for the qualified drawdown residual-continuation candidate.
Reconstructs the closest admitted momentum signal (20d risk-adjusted trend) and
reports exact pooled date-asset Spearman evidence. A breach rejects admission;
no persistence is permitted without a full-library screen, and a discovered
breach is already dispositive."""
import numpy as np,pandas as pd
CUT=pd.Timestamp('2032-10-27'); A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.rename(a)
p=pd.concat([load(a) for a in A],axis=1).sort_index().loc[:CUT];r=p.pct_change();m=r.mean(axis=1)
b=r.apply(lambda x:x.rolling(60,min_periods=42).cov(m)).div(m.rolling(60,min_periods=42).var()+1e-12,axis=0);e=r-b.mul(m,axis=0)
f=e.rolling(10,min_periods=8).sum()/(e.rolling(20,min_periods=14).std()+1e-12);f=f.where(m.rolling(20,min_periods=15).sum()<0)
trend=(p/p.shift(20)-1)/(r.rolling(20,min_periods=15).std()+1e-12)
q=pd.concat([f.stack().rename('candidate'),trend.stack().rename('risk_adjusted_trend_20d')],axis=1).dropna()
rho=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
print('ORTHOGONALITY_SCREEN cutoff',CUT.date(),'candidate_cells',int(f.notna().sum().sum()),'overlap_cells',len(q),'reference','miner_3_20260716_risk_adjusted_trend_20d','rho',round(float(rho),6),'abs_rho',round(float(abs(rho)),6))
print('RESULT','BREACH: a single admitted-factor overlap at or above 0.5000 is dispositive; full-library evidence cannot remedy this.' if abs(rho)>=.5 else 'NO_BREACH_AGAINST_THIS_REFERENCE')
