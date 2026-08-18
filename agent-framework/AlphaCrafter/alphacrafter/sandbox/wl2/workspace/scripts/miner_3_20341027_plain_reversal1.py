import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv');d.date=pd.to_datetime(d.date);px[s]=d.set_index('date').close
p=pd.DataFrame(px).sort_index();v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date);v=v.set_index('date').close.reindex(p.index).ffill()
r=p.pct_change(1);vr=v.rolling(60,min_periods=30).rank(pct=True)
sig=(-r).shift(1); fwd=p.pct_change().shift(-1)
rows=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('dates',len(q),'meanN',q.n.mean(),'coverage',q.n.mean()/15);print('IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
for h in [2,5,10,20]:
 a=[];fw=p.pct_change(h).shift(-h)
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(a);print('h',h,'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean())
for lab,mask in [('early',q.index<'2028-01-01'),('mid',(q.index>='2028-01-01')&(q.index<'2031-01-01')),('recent',q.index>='2031-01-01'),('highvix',vr.reindex(q.index)>0.7),('lowvix',vr.reindex(q.index)<=0.7)]:
 a=q.loc[mask,'ic'];print(lab,len(a),a.mean(),a.mean()/a.std(ddof=1))
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean());sig.to_csv('../persistent/miner_3_20341027_plain_reversal1_signal.csv');q.to_csv('../persistent/miner_3_20341027_plain_reversal1_ic.csv')
