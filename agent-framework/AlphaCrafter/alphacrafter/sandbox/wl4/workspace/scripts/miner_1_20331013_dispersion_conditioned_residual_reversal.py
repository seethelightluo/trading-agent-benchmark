import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2033-10-12'); H=10
px={}
for s in U:
    d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index()
    px[s]=d[d.index<=cut]
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# residualized 20-session reversal, volatility normalized; only amplify when dispersion is elevated
ret20=p.pct_change(20); vol40=r.rolling(40).std(); disp=ret20.std(axis=1).rolling(20).mean()
# cross-sectional median dispersion threshold, computed expanding to avoid look-ahead
thr=disp.expanding(min_periods=100).median().shift(1)
gate=(disp.shift(1)>thr).astype(float)
base=-(ret20.sub(ret20.mean(axis=1),axis=0))/vol40
f=base.mul(gate,axis=0).shift(1)
# evaluate date-wise forward 10d returns; f at t uses through t-1, target t to t+10
fr=p.shift(-H)/p-1
rows=[]
for dt in p.index:
    if dt>cut-pd.Timedelta(days=H): continue
    x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8:
        rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
ics=a.ic.replace([np.inf,-np.inf],np.nan).dropna(); m=ics.mean(); sd=ics.std(ddof=1)
# turnover of daily cross-sectional ranks on valid dates
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean()
print('dates',len(ics),'avgN',a.n.mean(),'IC %.8f ICIR %.8f hit %.4f'%(m,m/sd*np.sqrt(len(ics)) if sd else np.nan,(ics>0).mean()))
print('coverage %.4f turnover %.6f active_rate %.4f'%(a.n.mean()/15,turn,gate.mean()))
for n in [120,260,520]:
 q=ics.tail(n); print('recent',n,'IC %.8f ICIR %.8f'%(q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q))))
for h in [5,10,20,30]:
 yy=p.shift(-h)/p-1; rr=[]
 for dt in f.index:
  if dt>cut-pd.Timedelta(days=h): continue
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('decay',h,'IC %.8f n %d'%(np.nanmean(rr),len(rr)))
