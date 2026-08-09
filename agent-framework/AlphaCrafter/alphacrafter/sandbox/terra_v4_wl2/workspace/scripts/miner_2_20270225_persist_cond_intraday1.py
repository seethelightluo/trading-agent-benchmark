import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];C={};O={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index();C[a]=d.close;O[a]=d.open
p=pd.concat(C,axis=1).sort_index();op=pd.concat(O,axis=1).reindex(p.index); intr=p/op-1
f=(-intr).where(intr.abs()>=.01);f.reset_index().melt(id_vars='date',var_name='asset',value_name='signal').to_csv('../persistent/factor_signals_miner_2_20270225_cond_intraday1.csv',index=False)
