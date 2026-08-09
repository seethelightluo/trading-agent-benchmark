import pandas as pd, numpy as np, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
def load(s):
 d=pd.read_csv(os.path.join(base,s+'.csv')); d['date']=pd.to_datetime(d['date']); return d.set_index('date')['close'].astype(float)
px=pd.concat({s:load(s) for s in U},axis=1).sort_index(); r=px.pct_change()
d=pd.read_csv('../persistent/index_data/DXY.csv'); d['date']=pd.to_datetime(d.date); dx=d.set_index('date')['close'].astype(float).reindex(px.index).ffill().pct_change()
for w in [20,40,60,90]:
 minp=max(10,w//2)
 beta=r.rolling(w,min_periods=minp).cov(dx)/dx.rolling(w,min_periods=minp).var()
 mom=px.pct_change(w)
 f=mom.sub(beta.mul(dx.rolling(w,min_periods=minp).sum(),axis=0),axis=0)
 ic=[]; n=[]
 for i in range(len(px)-1):
  z=pd.concat([f.iloc[i],r.iloc[i+1]],axis=1).dropna()
  if len(z)>=8: ic.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); n.append(len(z))
 a=np.array(ic); ranks=f.rank(axis=1,pct=True); turn=ranks.diff().abs().mean(axis=1).mean()
 print(w,'obs',len(a),'names',round(np.mean(n),2),'IC',round(np.nanmean(a),5),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),5),'hit',round(np.mean(a>0),4),'turn',round(turn,4))
# end
