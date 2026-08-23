import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; q={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f): q[a]=pd.read_csv(f,parse_dates=['date']).set_index('date')['close'].sort_index()
p=pd.concat(q,axis=1).sort_index().ffill().loc[:'2027-03-08']; r=p.pct_change(); v5=r.rolling(5).std(); v60=r.rolling(60).std()
# compression-adjusted breakout: recent trend rewarded when short volatility is below long volatility
s=((p/p.shift(20)-1)/v60*(v5/v60).clip(0.2,3)).shift(1); y=r
ics=[]
for d in p.index:
 ok=s.loc[d].notna()&y.loc[d].notna()
 if ok.sum()>=8: ics.append((d,spearmanr(s.loc[d][ok],y.loc[d][ok]).statistic,ok.sum()))
z=pd.DataFrame(ics,columns=['date','ic','n']).set_index('date'); m=z.ic.mean(); ir=m/z.ic.std(ddof=1)*np.sqrt(252)
print('dates',len(z),'avg_n',z.n.mean(),'coverage',s.notna().mean().mean(),'IC',m,'ICIR',ir,'hit',(z.ic>0).mean(),'turnover',s.rank(pct=True).diff().abs().mean(axis=1).mean())
for x in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-08')]:
 a=z.loc[x[0]:x[1]].ic; print('regime',x,len(a),a.mean(),a.mean()/a.std()*np.sqrt(252))
for h in [5,10,20]:
 yy=p.shift(-h)/p-1; a=[]
 for d in p.index:
  ok=s.loc[d].notna()&yy.loc[d].notna()
  if ok.sum()>=8:a.append(spearmanr(s.loc[d][ok],yy.loc[d][ok]).statistic)
 print('decay',h,np.mean(a),len(a))
pd.DataFrame(s.stack(),columns=['signal']).to_csv('scripts/miner_2_20270308_compression_signal.csv')
