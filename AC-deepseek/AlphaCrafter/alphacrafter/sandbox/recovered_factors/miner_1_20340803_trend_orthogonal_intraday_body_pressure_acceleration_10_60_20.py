import os, glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# One idea: trend-orthogonal intraday body-pressure acceleration.  It avoids the
# prior-close/open gap and uses only same-session OHLC geometry.
END = pd.Timestamp('2034-08-02')
assets = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
root='../persistent/stock_data'
def wide(field):
    d={}
    for a in assets:
        x=pd.read_csv(f'{root}/{a}.csv',parse_dates=['date']).set_index('date')
        d[a]=x.loc[x.index<=END,field]
    return pd.DataFrame(d).sort_index()
O,C,H,L=map(wide,['open','close','high','low'])
# Ensure no missing calendar alignment can introduce a fabricated session.
r=C.pct_change(); rng=(H-L).replace(0,np.nan)
body=((C-O)/rng).clip(-1,1)
# acceleration of signed, range-normalised within-day demand
raw=body.rolling(10,min_periods=8).mean()-body.rolling(60,min_periods=40).mean()
trend=(C/C.shift(20)-1)/r.rolling(20,min_periods=15).std()
# Each date: cross-sectionally residualise against completed-bar trend.
sig=pd.DataFrame(index=C.index,columns=assets,dtype=float)
for t in C.index:
    x=trend.loc[t]; y=raw.loc[t]; ok=x.notna()&y.notna()
    if ok.sum()>=8 and x[ok].std()>1e-12:
        b=np.cov(x[ok],y[ok],ddof=1)[0,1]/np.var(x[ok],ddof=1)
        sig.loc[t,ok]=y[ok]-(y[ok].mean()-b*x[ok].mean())-b*x[ok]

print('IDEA trend_orthogonal_intraday_body_pressure_acceleration_10_60_20obs')
print('endpoint',END.date(),'rows',len(C),'assets',len(assets),'signal_cells',int(sig.notna().sum().sum()),'coverage',round(sig.notna().mean().mean(),4))
# daily CS Spearman IC. A minimum of eight usable instruments is binding here.
def ics(h):
    f=C.shift(-h)/C-1; z=[]; ns=[]
    for t in C.index:
        q=pd.concat([sig.loc[t],f.loc[t]],axis=1).dropna()
        if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
            z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic); ns.append(len(q))
    return np.array(z),ns
for h in [1,5,10,20]:
    z,n=ics(h)
    ic=np.mean(z) if len(z) else np.nan; ir=ic/np.std(z,ddof=1) if len(z)>1 else np.nan
    print('H',h,'dates',len(z),'meanN',round(np.mean(n),2) if n else None,'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(np.mean(z>0),4) if len(z) else None)
    # broad chronological regime audit, retaining same-horizon observations
    for name,sel in [('early',C.index<'2025-01-01'),('mid',(C.index>='2025-01-01')&(C.index<'2030-01-01')),('late',C.index>='2030-01-01')]:
        zz=[]
        for t in C.index[sel]:
            q=pd.concat([sig.loc[t],(C.shift(-h)/C-1).loc[t]],axis=1).dropna()
            if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: zz.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
        print(' regime',name,'n',len(zz),'IC',round(float(np.mean(zz)),6) if zz else None)
# turnover / concentration
rank=sig.rank(axis=1,pct=True); dif=rank.diff().abs()
print('turnover_mean_abs_rank_change',round(dif.stack().mean(),6),'comparisons',int(dif.notna().sum().sum()))
print('median_cs_IQR',round(sig.quantile(.75,axis=1).sub(sig.quantile(.25,axis=1)).median(),8))
# Novelty diagnostic against the closest admitted signals that can be exactly
# reconstructed: trend and trend-orthogonal CLV acceleration.
clv=((C-L)/(H-L).replace(0,np.nan)).clip(0,1)
clvraw=clv.rolling(10,min_periods=8).mean()-clv.rolling(60,min_periods=40).mean()
clvs=pd.DataFrame(index=C.index,columns=assets,dtype=float)
for t in C.index:
 x=trend.loc[t];y=clvraw.loc[t];ok=x.notna()&y.notna()
 if ok.sum()>=8 and x[ok].std()>1e-12:
  b=np.cov(x[ok],y[ok],ddof=1)[0,1]/np.var(x[ok],ddof=1);clvs.loc[t,ok]=y[ok]-(y[ok].mean()-b*x[ok].mean())-b*x[ok]
for name,other in [('risk_adjusted_trend',trend),('closest_library_CLV_acceleration',clvs)]:
 q=pd.concat([sig.stack(),other.stack()],axis=1).dropna(); rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>2 else np.nan
 print('novelty_proxy',name,'paired_cells',len(q),'rho',round(rho,6))
print('NOTE Full-library novelty must be computed before admission; this candidate is not persisted by this script.')
