import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];D=['XAU','US10Y','CN10Y'];p={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):p[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').close
P=pd.DataFrame(p).sort_index();r=P.pct_change();db=r[D].mean(axis=1)
b=r.rolling(90,min_periods=60).cov(db).div(db.rolling(90,min_periods=60).var(),axis=0).shift(1);res=r-b.mul(db,axis=0);rv=res.rolling(60,min_periods=40).std().shift(1)
base=-res.rolling(30,min_periods=20).sum().shift(1)/(rv*np.sqrt(30)+1e-9); cond=db.rolling(60,min_periods=40).sum().shift(1)<=0;sig=base.where(cond);y=P.pct_change(40).shift(-40)
v=[];n=[];ds=[]
for d in sig.index:
 ok=sig.loc[d].notna()&y.loc[d].notna()
 if ok.sum()>=8:v.append(spearmanr(sig.loc[d][ok],y.loc[d][ok]).statistic);n.append(ok.sum());ds.append(d)
q=pd.Series(v,index=ds);print('negative condition dates',len(q),'avg_n',np.mean(n),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()))
for z,s in [('2026-2030',q.loc['2026':'2030']),('2031-2035',q.loc['2031':'2035']),('2034-2035',q.loc['2034':'2035'])]:print(z,len(s),'IC %.6f ICIR %.6f hit %.4f'%(s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
print('coverage %.4f turnover %.4f cond %.4f'%(sig.notna().mean().mean(),sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),cond.mean()))
