"""Reconstruct exact historical signal for admitted market-beta contraction factor.
The calculation and cutoff reproduce the original miner_1 producer definition."""
import pickle
import numpy as np
import pandas as pd

A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2027-03-10')
prices={}
for a in A:
    d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
    prices[a]=d['close'].astype(float)
p=pd.DataFrame(prices)
r=p.pct_change()
m=r.mean(axis=1)
beta60=pd.DataFrame({a:r[a].rolling(60,min_periods=40).cov(m)/m.rolling(60,min_periods=40).var() for a in A})
beta20=pd.DataFrame({a:r[a].rolling(20,min_periods=15).cov(m)/m.rolling(20,min_periods=15).var() for a in A})
f=beta60-beta20
factor_id='miner_1_market_beta_contraction_60_20'
out='scripts/'+factor_id+'_signal.pkl'
with open(out,'wb') as h:
    pickle.dump({'factor_id':factor_id,'producer':'miner_1_20270311_market_beta_contraction_60_20.py','end':str(END.date()),'symbols':A,'signal':f},h)
print('SERIALIZED',out,'rows',len(f),'cols',len(f.columns),'start',f.index.min().date(),'end',f.index.max().date(),'coverage',round(float(f.notna().mean().mean()),6))
print('FACTOR_EXPRESSION beta_60(i, equal_weight_daily_return)-beta_20(i, equal_weight_daily_return); beta_w=rolling_cov(r_i,M,w)/rolling_var(M,w); min_periods 40 and 15 respectively')
print('NONNULL_DATES',int(f.notna().any(axis=1).sum()),'FULL_CROSS_SECTION_DATES',int((f.notna().sum(axis=1)==len(A)).sum()))
