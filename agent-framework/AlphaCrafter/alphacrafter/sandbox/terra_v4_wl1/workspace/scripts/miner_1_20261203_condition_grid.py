import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; macro='../persistent/index_data'; cut=pd.Timestamp('2026-12-03')
P=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}).sort_index();P=P[P.index<=cut]
v=pd.read_csv(f'{macro}/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(P.index).ffill(); sp=P.SPX
conds={'stress':((sp.pct_change(20)<0)&(v.pct_change(5)>0))|(v>v.rolling(60,min_periods=40).median()),'vix_rise':v.pct_change(5)>0,'highvix':v>v.rolling(120,min_periods=60).median()}
for cn,c in conds.items():
 for calm in ['rev5','mom20','rev10']:
  stress=-P.pct_change(5); other={'rev5':-P.pct_change(5),'mom20':P.pct_change(20),'rev10':-P.pct_change(10)}[calm]; mask=pd.DataFrame(np.repeat(c.values[:,None],15,axis=1),index=P.index,columns=P.columns); f=stress.where(mask,other); Y=P.shift(-1).div(P)-1; rows=[]
  for dt in P.index:
   q=pd.concat([f.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
   if len(q)>=8: rows.append(q.f.corr(q.y))
  x=pd.Series(rows).dropna(); print(cn,calm,len(x),round(x.mean(),5),round(x.mean()/x.std(ddof=1),5),(x>0).mean())
