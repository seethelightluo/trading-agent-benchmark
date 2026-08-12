import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}; v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end].close
F=[]
for s in U:
 r=D[s].close.pct_change(); z=pd.concat([r.rename('r'),v.pct_change().rename('v')],axis=1).dropna(); F.append((-(z.r.rolling(60,min_periods=45).cov(z.v)/z.v.rolling(60,min_periods=45).var())).rename(s))
F=pd.concat(F,axis=1).sort_index()
# compare pooled common dates to library definitions
R=pd.concat({s:D[s].close.pct_change() for s in U},axis=1)
libs={'short_term_reversal_5d':-R.rolling(5).sum(),'peer_median_leadlag_5d':pd.DataFrame({s:R.drop(columns=s).rolling(5).sum().median(axis=1) for s in U}),'risk_adjusted_momentum_20d':R.rolling(20).sum()/R.rolling(20).std()}
# CLV
clv=pd.DataFrame({s:((D[s].close-D[s].low)-(D[s].high-D[s].close))/(D[s].high-D[s].low).replace(0,np.nan) for s in U})
libs['miner_3_clv_1d']=clv
for n,x in libs.items(): print(n,F.stack().corr(x.stack()))
print('valid',F.notna().sum().sum()/(len(F)*15))
