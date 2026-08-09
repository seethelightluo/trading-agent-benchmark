"""Miner_3: downside-volume asymmetry reversal, validated through completed 2033-03-16 bars.
One interpretable idea: unusually heavy-volume down days may reflect capitulation and
predict relative rebound. Score is negative, volume-standardized 20d downside return.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2033-03-16')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for a in A:
    x=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT]
    raw[a]=x[['close','volume']]
p=pd.DataFrame({a:x.close for a,x in raw.items()})
r=p.pct_change()
f={}
for a,x in raw.items():
    # Standardized log-volume, lagged baseline makes today's surprise implementable.
    lv=np.log(x.volume.replace(0,np.nan))
    z=(lv-lv.rolling(60,min_periods=42).mean().shift(1))/(lv.rolling(60,min_periods=42).std().shift(1)+1e-12)
    # Only losses receive a participation multiplier; bounded to prevent isolated prints.
    downside=(-r[a].clip(upper=0))* (1+z.clip(-1,2)).clip(0,3)
    f[a]=downside.rolling(20,min_periods=16).mean()
f=pd.DataFrame(f).reindex(p.index)
print('CANDIDATE downside_volume_asymmetry_reversal_20_60 cutoff',CUT.date(),'calendar_dates',len(p),'assets',len(A))
print('valid_dates',int(f.notna().any(axis=1).sum()),'valid_cells',int(f.notna().sum().sum()),'coverage',round(float(f.notna().mean().mean()),6))
ics={}
for h in (1,3,5,7,10,20):
    fw=p.shift(-h).div(p)-1; out=[]; nn=[]
    for d in f.index:
        q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
        if len(q)>=8 and q.f.nunique()>1:
            v=spearmanr(q.f,q.y).statistic
            if np.isfinite(v): out.append((d,v));nn.append(len(q))
    s=pd.Series(dict(out),dtype=float);ics[h]=s;sd=s.std(ddof=1)
    print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/sd,len(s),(s>0).mean(),np.mean(nn)))
    if h==10:
      for name,lo,hi in [('2020-2024','2020-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027+','2027-01-01',str(CUT.date()))]:
        z=s.loc[lo:hi]; print('REGIME10',name,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
rk=f.rank(axis=1,pct=True);tr=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: tr.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('RANK_TURNOVER',round(float(np.mean(tr)),6),'pairs',len(tr))
print('DECAY',{h:(round(float(s.mean()),6),round(float(s.mean()/s.std(ddof=1)),6),len(s)) for h,s in ics.items()})
f.to_pickle('scripts/miner_3_20330317_downside_volume_asymmetry_reversal_20_60_signal.pkl')
