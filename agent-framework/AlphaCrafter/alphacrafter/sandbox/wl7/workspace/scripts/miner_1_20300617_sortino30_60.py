import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is None or len(d)<150: d=get_index_daily_data(s,days=4000)
 if d is not None and len(d):
  x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date)
  frames[s]=x.drop_duplicates('date').set_index('date').close.astype(float)
px=pd.DataFrame(frames).sort_index().ffill(); ret=px.pct_change()
# Candidate: lagged 30-day trend divided by 60-day downside deviation.
r30=px.pct_change(30); down=ret.where(ret<0,0).rolling(60).std()
factor=(r30/(down*np.sqrt(60)+1e-8)).shift(1)
rows=[]
for dt in px.index:
 z=pd.concat([factor.loc[dt],(px.shift(-10)/px-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
res=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna()
for lab,rr in [('full',res),('early',res.iloc[:len(res)//3]),('mid',res.iloc[len(res)//3:2*len(res)//3]),('late',res.iloc[2*len(res)//3:])]:
 m=rr.ic.mean(); sd=rr.ic.std(ddof=1)
 print(lab,'dates',len(rr),'avg_n',round(rr.n.mean(),2),'IC',round(m,8),'ICIR',round(m/sd*np.sqrt(252),8) if sd else np.nan,'hit',round((rr.ic>0).mean(),4))
for h in [1,5,10,20,40]:
 vals=[]
 for dt in px.index:
  z=pd.concat([factor.loc[dt],(px.shift(-h)/px-1).loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,'IC',round(np.nanmean(vals),8),'dates',len(vals))
rank=factor.rank(axis=1,pct=True)
print('cutoff',px.index[-1],'assets',len(frames),'dates',len(px),'coverage',round(factor.notna().sum().sum()/(factor.shape[0]*len(U)),6),'turnover',round(rank.diff().abs().mean(axis=1).dropna().mean(),6))
res.to_csv('scripts/miner_1_20300617_sortino30_60_ic.csv'); factor.to_csv('scripts/miner_1_20300617_sortino30_60_signal.csv')
