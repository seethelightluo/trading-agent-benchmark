import pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:d['close'] for s,d in D.items()}); q=P.pct_change(5); f=-(q.sub(q.mean(axis=1),axis=0))
f.to_csv('scripts/miner_2_20261217_market_neutral_reversal_signal.csv',index_label='date')
print('saved',f.shape)
