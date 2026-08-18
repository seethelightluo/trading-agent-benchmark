import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2035-11-09')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').close.loc[:CUT] for s in U}).sort_index(); r=px.pct_change()
ret20=px.pct_change(20); rel=ret20.sub(ret20.median(axis=1),axis=0)
disp=ret20.std(axis=1); threshold=disp.rolling(60,min_periods=30).median(); active=(disp>threshold).shift(1)
dd=r.where(r<0).rolling(20,min_periods=10).std(); f=(-rel/dd).where(active, np.nan).shift(1)
fr=px.shift(-10)/px-1; rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
d=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); a=d.ic
print('N',len(a),'avgN',round(d.n.mean(),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'active_dates',int(active.sum()))
for label,x in [('2026_2028',a.loc['2026':'2028']),('2029_2032',a.loc['2029':'2032']),('2033_2035',a.loc['2033':])]: print(label,'N',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for h in [3,5,10,20]:
 fh=px.shift(-h)/px-1; q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fh.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,'N',len(q),'IC',round(np.mean(q),6),'ICIR',round(np.mean(q)/np.std(q,ddof=1),6))
f.to_csv('../persistent/miner_3_20351109_dispersion_gated_reversal_signal.csv'); d.to_csv('../persistent/miner_3_20351109_dispersion_gated_reversal_ic.csv')
