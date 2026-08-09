import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-12-17'); base='../persistent/stock_data'
D={s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:D[s].close for s in U}).sort_index().loc[:END]
O=pd.DataFrame({s:D[s].open for s in U}).reindex(P.index)
H=pd.DataFrame({s:D[s].high for s in U}).reindex(P.index)
L=pd.DataFrame({s:D[s].low for s in U}).reindex(P.index)
tr=(H-L).combine((H-P.shift(1)).abs(),np.maximum).combine((L-P.shift(1)).abs(),np.maximum)
atr=tr.shift(1).rolling(20,min_periods=15).mean()
signal=-(P-O)/atr
out=signal.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20261217_intraday_shock_signal.csv',index=False)
print('artifact rows',len(out),'coverage',signal.notna().sum().sum()/signal.size)
