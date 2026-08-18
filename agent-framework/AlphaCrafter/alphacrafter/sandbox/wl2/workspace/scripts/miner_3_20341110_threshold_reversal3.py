import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={}
for s in U:
 d=pd.read_csv(f'../persistent/stock_data/{s}.csv'); d.date=pd.to_datetime(d.date); p[s]=d.set_index('date').close
p=pd.DataFrame(p).sort_index(); r=p.pct_change(3); fwd=p.pct_change().shift(-1)
print('range',p.index.min().date(),p.index.max().date(),'assets',len(p.columns))
for cut in [.5,.6,.7,.8]:
 threshold=r.abs().quantile(cut,axis=1)
 sig=(-r).where(r.abs().ge(threshold,axis=0),0).shift(1)
 rows=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=q.ic.mean(); ir=ic/q.ic.std(ddof=1)
 turn=sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()
 print('cut',cut,'dates',len(q),'meanN',q.n.mean(),'coverage',q.n.mean()/15,'IC',ic,'ICIR',ir,'hit',(q.ic>0).mean(),'turn',turn)
 for h in [2,5,10]:
  fw=p.pct_change(h).shift(-h); a=[]
  for dt in sig.index:
   z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
   if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
  a=np.array(a);print(' decay',h,a.mean(),a.mean()/a.std(ddof=1))
 for lab,mask in [('early',q.index<'2028-01-01'),('mid',(q.index>='2028-01-01')&(q.index<'2031-01-01')),('recent',q.index>='2031-01-01')]:
  a=q.loc[mask,'ic'];print(' ',lab,len(a),a.mean(),a.mean()/a.std(ddof=1))
 if cut==.7:
  sig.to_csv('../persistent/miner_3_20341110_threshold_reversal3_signal.csv');q.to_csv('../persistent/miner_3_20341110_threshold_reversal3_ic.csv')
