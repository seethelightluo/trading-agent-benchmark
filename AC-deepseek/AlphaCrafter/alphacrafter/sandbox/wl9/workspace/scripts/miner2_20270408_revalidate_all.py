#!/usr/bin/env python
"""Re-validate all 17 existing factors as of 2027-04-08. Cross-asset 15-instrument pool. Gates: |IC|>=0.0070, |ICIR|>=0.0840"""
import numpy as np, pandas as pd, json, warnings
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
warnings.filterwarnings('ignore')

DT="2027-04-08"
WL=["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
ML=["DXY","USDCNY","USDJPY","EURUSD","VIX"]
print(f"=== REVALIDATION: {DT} ===")
data={}
for s in WL:
    df=get_stock_daily_data(s,750)
    if df is None or len(df)<200: df=get_index_daily_data(s,750)
    if df is not None and len(df)>=200: data[s]=df
for s in ML:
    df=get_index_daily_data(s,750)
    if df is not None and len(df)>=200: data[s]=df
print(f"Loaded {len(data)} symbols")
min_dt=max(d.index[0] for d in data.values() if len(d)>0)
aligned={s:d.loc[d.index>=min_dt] for s,d in data.items()}
common=aligned[WL[0]].index
for s in WL[1:]: common=common.intersection(aligned.get(s,pd.DataFrame()).index)
print(f"Common dates: {len(common)} from {common[0]} to {common[-1]}")
C=np.column_stack([aligned[s]['close'].reindex(common).values for s in WL])
R=np.column_stack([aligned[s]['pct_change'].reindex(common).fillna(0).values for s in WL])
Lo=np.column_stack([aligned[s]['low'].reindex(common).values for s in WL])
Hi=np.column_stack([aligned[s]['high'].reindex(common).values for s in WL])
dxy=aligned['DXY']['close'].values if 'DXY' in aligned else None
vix=aligned['VIX']['close'].values if 'VIX' in aligned else None
cny=aligned['USDCNY']['close'].values if 'USDCNY' in aligned else None
T,N=R.shape; print(f"Panel: {T}d x {N}a")
fwd1=np.full_like(R,np.nan)
for i in range(T-1): fwd1[i]=R[i+1]

# compute factors
mom120=np.full_like(C,np.nan)
for i in range(125,T): mom120[i]=C[i-5]/np.maximum(C[i-125],1e-10)-1
mom10=np.full_like(C,np.nan)
for i in range(15,T): mom10[i]=C[i-5]/np.maximum(C[i-15],1e-10)-1
mu20=pd.DataFrame(R).rolling(20,min_periods=20).mean().values
sd20=pd.DataFrame(R).rolling(20,min_periods=20).std(ddof=0).values
volz=(R-mu20)/np.maximum(sd20,1e-10)
kauf=np.full_like(C,np.nan)
for i in range(20,T):
    d=np.abs(C[i]-C[i-20]); vt=np.sum(np.abs(np.diff(C[i-20:i+1],axis=0)),axis=0)
    kauf[i]=d/np.maximum(vt,1e-10)
ma20=pd.DataFrame(C).rolling(20,min_periods=20).mean().values
sd20c=pd.DataFrame(C).rolling(20,min_periods=20).std(ddof=0).values
bbw=(4*sd20c)/np.maximum(ma20,1e-10)
rv=np.diff(vix,prepend=vix[0]) if vix is not None else None
bv=np.full_like(R,np.nan)
if rv is not None:
    for i in range(60,T):
        rx=R[i-60:i]; vx=rv[i-60:i]; vv=np.var(vx)
        if vv>1e-12: bv[i]=np.mean((rx-np.mean(rx,0))*(vx-np.mean(vx))[:,None],0)/vv
cr=np.diff(cny,prepend=cny[0]) if cny is not None else None
cb=np.full_like(R,np.nan)
if cr is not None:
    for i in range(60,T):
        rx=R[i-60:i]; cx=cr[i-60:i]; cv=np.var(cx)
        if cv>1e-12: cb[i]=np.mean((rx-np.mean(rx,0))*(cx-np.mean(cx))[:,None],0)/cv
ac1=np.full_like(R,np.nan)
for i in range(121,T):
    s=R[i-120:i]
    for j in range(N):
        a=s[:-1,j];b=s[1:,j]
        ac1[i,j]=np.corrcoef(a,b)[0,1] if np.std(a)>1e-10 and np.std(b)>1e-10 else 0
sk=np.full_like(R,np.nan)
for i in range(20,T):
    for j in range(N):
        s=R[i-20:i,j]
        sk[i,j]=pd.Series(s).skew() if np.std(s)>1e-10 else 0
rp=np.full_like(C,np.nan)
for i in range(20,T): rp[i]=(C[i]-np.min(Lo[i-20:i],0))/np.maximum(np.max(Hi[i-20:i],0)-np.min(Lo[i-20:i],0),1e-10)
kt=np.full_like(R,np.nan)
for i in range(20,T):
    for j in range(N):
        s=R[i-20:i,j]
        kt[i,j]=pd.Series(s).kurt() if np.std(s)>1e-10 else 0
st=np.zeros_like(R)
for j in range(N):
    c=0
    for i in range(1,T):
        if R[i,j]>0: c=c+1 if R[i-1,j]>0 else 1
        elif R[i,j]<0: c=c-1 if R[i-1,j]<0 else -1
        else: c=0
        st[i,j]=c
dsh=np.full_like(C,np.nan)
for i in range(60,T):
    rm=np.max(C[i-60:i],0)
    dsh[i]=C[i]/np.maximum(rm,1e-10)-1
dc=np.full_like(R,np.nan)
if dxy is not None:
    dr=np.diff(dxy,prepend=dxy[0])
    for i in range(60,T):
        for j in range(N):
            r=R[i-60:i,j];d_=dr[i-60:i]
            dc[i,j]=np.corrcoef(r,d_)[0,1] if np.std(r)>1e-10 and np.std(d_)>1e-10 else 0
mv=np.full_like(R,np.nan)
if vix is not None:
    v75=np.percentile(vix,75)
    for i in range(15,T):
        mv[i]=mom10[i]
        if vix[i]>v75: mv[i]=-mom10[i]
vc=np.full_like(R,np.nan)
if vix is not None:
    vx2=np.diff(vix,prepend=vix[0])
    b20=np.full_like(R,np.nan)
    for i in range(20,T):
        rx=R[i-20:i];vx_=vx2[i-20:i];vv=np.var(vx_)
        if vv>1e-12: b20[i]=np.mean((rx-np.mean(rx,0))*(vx_-np.mean(vx_))[:,None],0)/vv
    for i in range(60,T): vc[i]=b20[i] if vx2[i]>0 else bv[i]
vv20=np.full_like(R,np.nan)
for i in range(20,T): vv20[i]=np.std(R[i-20:i],0)
vov=np.full_like(R,np.nan)
for i in range(60,T): vov[i]=np.std(vv20[i-60:i],0