import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
D={os.path.basename(f)[:-4]:pd.read_csv(f,parse_dates=['date']).set_index('date') for f in glob.glob('../persistent/stock_data/*.csv')}
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];assets=[a for a in assets if a in D]
px=pd.concat({a:D[a]['close'] for a in assets},axis=1).sort_index(); r=px.pct_change()
# Tail-adjusted medium momentum: recent 10d cumulative return scaled by downside deviation over 30d.
# Both windows are lagged one session before the forward-return test.
neg=r.where(r<0); dd=neg.rolling(30,min_periods=20).std(); factor=(r.rolling(10,min_periods=10).sum()/(dd*np.sqrt(10)+1e-12)).shift(1)
print('assets',len(assets),'range',px.index.min().date(),px.index.max().date())
for h in [1,5,10,20]:
 fwd=px.pct_change(h).shift(-h); vals=[];ns=[];ds=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 a=np.asarray(vals); print('H',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/(a.std(ddof=1)+1e-12),(a>0).mean()))
fr=r.shift(-1);out=[]
for dt in factor.index:
 z=pd.concat([factor.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:out.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
y=pd.DataFrame(out,columns=['date','ic']).set_index('date');print('year',y.groupby(y.index.year).ic.mean().round(5).to_dict())
rank=factor.rank(axis=1,pct=True);print('coverage',factor.notna().sum().sum()/factor.size,'daily_dates',len(y),'valid_avg',factor.notna().sum(axis=1).mean(),'turnover',rank.diff().abs().sum(axis=1).div(len(assets)).dropna().mean())
for w in [250,500]:
 a=y.ic.tail(w);print('recent',w,'IC %.6f ICIR %.6f hit %.4f'%(a.mean(),a.mean()/(a.std(ddof=1)+1e-12),(a>0).mean()))
# library correlation evidence against admitted factor files if signal histories can be reconstructed approximately from factor definitions omitted; calculate date-wise ranks versus known signals where available impossible
