import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P='../persistent/stock_data/'; CUT=pd.Timestamp('2027-02-25')
C=pd.concat({s:pd.read_csv(P+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index().loc[:CUT]
r=C.pct_change(); b=(r<0).sum(axis=1)/r.notna().sum(axis=1); h=b.shift(1).rolling(252,min_periods=60); flag=b.shift(1)>=np.maximum(.60,h.median()); active=flag.rolling(2,min_periods=2).sum().eq(2)
rr=C.pct_change(3).shift(1); f=-rr.sub(rr.median(axis=1),axis=0)
out=f.where(active).stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('../persistent/factor_signals_miner_1_20270225_breadth_persistence_p2_med.csv',index=False); print(len(out),out.date.min(),out.date.max())
