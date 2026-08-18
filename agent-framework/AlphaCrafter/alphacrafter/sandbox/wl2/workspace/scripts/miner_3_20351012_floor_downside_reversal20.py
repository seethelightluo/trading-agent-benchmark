import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2035-10-12')
px=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').close.loc[:CUT] for s in U}).sort_index()
r=px.pct_change(); ret=px.pct_change(20); rel=ret.sub(ret.median(axis=1),axis=0)
dd=r.where(r<0).rolling(20,min_periods=5).std(); floor=dd.median(axis=1).replace(0,np.nan); den=dd.clip(lower=floor,axis=0)
f=(-rel/den).shift(1)
fr=px.shift(-10)/px-1; rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
d=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); a=d.ic
print('N',len(a),'avgN',round(d.n.mean(),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'coverage',round(f.notna().mean().mean(),4),'turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.15).mean(),4))
for label,x in [('2026_2028',a.loc['2026':'2028']),('2029_2032',a.loc['2029':'2032']),('2033_2035',a.loc['2033':])]: print(label,'N',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
for h in [3,5,10,20]:
 fh=px.shift(-h)/px-1; rr=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fh.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(rr); print('DECAY',h,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
f.to_csv('../persistent/miner_3_20351012_floor_downside_reversal20_signal.csv'); d.to_csv('../persistent/miner_3_20351012_floor_downside_reversal20_ic.csv')
