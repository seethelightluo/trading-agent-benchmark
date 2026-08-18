import pandas as pd, numpy as np
from scipy.stats import spearmanr
import os

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
    d=pd.read_csv(os.path.join(base,s+'.csv'))
    d['date']=pd.to_datetime(d['date'])
    px[s]=d.set_index('date')['close'].astype(float)
close=pd.DataFrame(px).sort_index()
ret=close.pct_change()
# signal at t uses through t, then forward return t+1; explicitly no lookahead
vol=ret.rolling(20,min_periods=15).std()
breadth=ret.lt(0).mean(axis=1)
# stronger reversal after unusually broad selloffs, mild bounded conditioning
baseline=breadth.rolling(120,min_periods=60).mean()
bstd=breadth.rolling(120,min_periods=60).std()
z=((breadth-baseline)/(bstd+1e-12)).clip(-2,2)
mult=(1+0.35*z).clip(0.55,1.45)
sig=((-ret/(vol*np.sqrt(20)+1e-12)).mul(mult,axis=0)).shift(0)
# evaluate dates with >=8 cross-section and forward next-day data
fwd=ret.shift(-1)
rows=[]; signals=[]
for dt in sig.index:
    x=sig.loc[dt]; y=fwd.loc[dt]; ok=x.notna()&y.notna()
    if ok.sum()>=8:
        ic=spearmanr(x[ok],y[ok]).statistic
        rows.append((dt,ic,ok.sum()))
        signals.append(x[ok].rank(pct=True))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
ics=r.ic.dropna(); mean=ics.mean(); sd=ics.std(ddof=1)
print('factor=breadth_conditioned_volscaled_reversal1d')
print('dates',len(r),'instruments',len(U),'avg_n',r.n.mean(),'coverage',r.n.mean()/len(U))
print('daily_ic %.8f daily_icir %.8f hit %.4f'%(mean,mean/(sd+1e-12)*np.sqrt(len(ics)),(ics>0).mean()))
for h in [1,5,10]:
    yy=close.pct_change(h).shift(-h)
    vals=[]
    for dt in sig.index:
      ok=sig.loc[dt].notna()&yy.loc[dt].notna()
      if ok.sum()>=8: vals.append(spearmanr(sig.loc[dt][ok],yy.loc[dt][ok]).statistic)
    print('horizon',h,'ic',np.nanmean(vals),'n',len(vals))
# blocks and recent
for name,a in [('early',ics.iloc[:len(ics)//3]),('middle',ics.iloc[len(ics)//3:2*len(ics)//3]),('recent',ics.iloc[2*len(ics)//3:]),('recent120',ics.tail(120))]:
 print(name,'n',len(a),'ic',a.mean(),'icir',a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),'hit',(a>0).mean())
# rank turnover
rr=sig.rank(axis=1,pct=True)
turn=rr.diff().abs().mean(axis=1).dropna().mean()
print('rank_turnover',turn)
print('period',r.index.min().date(),r.index.max().date())
