"""Screener 2034-12-07: compute recent rank-IC for all 17 active factors
on two recent windows (2022+ and 2033+) to assess current regime suitability."""
import pandas as pd, numpy as np, glob, os
from pathlib import Path

SD=Path('../persistent/stock_data'); ID=Path('../persistent/index_data')
order=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end='2034-12-06'
def load(a):
    f=SD/f'{a}.csv'
    if not f.exists(): f=ID/f'{a}.csv'
    df=pd.read_csv(f,parse_dates=['date']); df=df[df['date']<=pd.Timestamp(end)].sort_values('date').set_index('date')
    return df['close'].astype(float)
panel=pd.DataFrame({a:load(a) for a in order}).dropna(how='all')
R=panel.pct_change().replace([np.inf,-np.inf],np.nan)

vix=pd.read_csv(ID/'VIX.csv',parse_dates=['date']); vix['date']=pd.to_datetime(vix['date'])
vix=vix[vix['date']<=pd.Timestamp(end)].sort_values('date').set_index('date')['close']
rvix=vix.pct_change()
cny=pd.read_csv(ID/'USDCNY.csv',parse_dates=['date']); cny['date']=pd.to_datetime(cny['date'])
cny=cny[cny['date']<=pd.Timestamp(end)].sort_values('date').set_index('date')
rcny=cny.iloc[:,0].pct_change().reindex(panel.index)
dxy=pd.read_csv(ID/'DXY.csv',parse_dates=['date']); dxy['date']=pd.to_datetime(dxy['date'])
dxy=dxy[dxy['date']<=pd.Timestamp(end)].sort_values('date').set_index('date')['close']
rdxy=dxy.pct_change().reindex(panel.index)

def rolling_beta(y,x,n):
    out=pd.DataFrame(index=y.index,columns=y.columns,dtype=float)
    for c in y.columns:
        yy=pd.concat([y[c],x],axis=1).dropna()
        out[c]=yy.iloc[:,0].rolling(n).cov(yy.iloc[:,1])/yy.iloc[:,1].rolling(n).var()
    return out

F={}
F['mom_120d_skip5']=panel/panel.shift(126)-1
F['mom_10d_skip5']=panel/panel.shift(15)-1
F['mom_10_vixreg']=panel/panel.shift(15)-1
F['bb_width_20d']=R.rolling(20).std()*np.sqrt(252)
F['vol_z_20d']=R.rolling(20).std()
F['ac1_120d']=R.rolling(120).apply(lambda x: x.autocorr() if len(x)==120 and x.std()>0 else np.nan,raw=False)
F['skew_20d']=R.rolling(20).skew()
F['kurt_20d']=R.rolling(20).kurt()
F['rng_pos_20d']=(panel-panel.rolling(20).min())/(panel.rolling(20).max()-panel.rolling(20).min()+1e-12)
F['days_since_high_60']=panel.rolling(60).apply(lambda x:(x==x.max()).sum()-1 if len(x)==60 else np.nan,raw=False)
def kaufman(px,n=20):
    sig=px.diff(n).abs(); noise=px.diff().abs().rolling(n).sum(); return sig/noise
F['kaufman_eff_20d']=kaufman(panel,20)
F['streak_len_14']=panel.pct_change().rolling(14).apply(lambda x:(x>0).sum() if len(x)==14 else np.nan,raw=False)
F['beta_VIX_60']=rolling_beta(R,rvix.reindex(panel.index),60)
F['cny_beta_60']=rolling_beta(R,rcny,60)
F['vix_beta_cond_60x20']=rolling_beta(R.rolling(20).std(),rvix.reindex(panel.index),60)
F['vix_roc_20d']=vix/vix.shift(20)-1
# dxy corr change 20_60
cor20=rolling_beta(R,rdxy,20); cor60=rolling_beta(R,rdxy,60)
F['dxy_corr_change_20_60']=cor20-cor60

fwd=panel.shift(-10)/panel-1

def ric(fv,start):
    if fv.ndim==1 or fv.shape[1]==1:
        fv=pd.DataFrame({c:fv for c in panel.columns})
    f=fv.reindex(fwd.index); ii=fwd.index[fwd.index>=pd.Timestamp(start)]
    ics=[]
    for d in ii:
        x=f.loc[d]; y=fwd.loc[d]; m=x.notna()&y.notna()
        if m.sum()>=8 and np.std(x[m].rank())>0 and np.std(y[m].rank())>0:
            ics.append(np.corrcoef(x[m].rank(),y[m].rank())[0,1])
    ics=np.array(ics)
    if len(ics)<20: return dict(IC=0.,ICIR=0.,n=0,hit=0.)
    mu=ics.mean(); sd=ics.std()
    return dict(IC=float(mu),ICIR=float(mu/sd*np.sqrt(len(ics)) if sd>0 else 0),n=len(ics),hit=float((ics>0).mean()))

print("factor, dir, IC22, ICIR22, IC33, ICIR33, hit33")
dirs={'ac1_120d':-1,'beta_VIX_60':-1,'vix_beta_cond_60x20':-1,'days_since_high_60':-1}
for nm in F.keys():
    fv=F.get(nm)
    if fv is None: continue
    a=ric(fv,'2022-01-01'); b=ric(fv,'2033-01-01')
    print(f"{nm},{dirs.get(nm,1)},{a['IC']:.4f},{a['ICIR']:.4f},{b['IC']:.4f},{b['ICIR']:.4f},{b['hit']:.3f}")