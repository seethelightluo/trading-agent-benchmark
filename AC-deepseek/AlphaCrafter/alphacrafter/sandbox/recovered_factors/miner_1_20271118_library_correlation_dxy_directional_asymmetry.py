# library novelty test: pooled Spearman of daily cross-sectional ranks, common valid cells
exec(open('scripts/miner_1_20271118_dxy_directional_return_asymmetry_60.py').read().split("print('candidate")[0])
V={}
med=R.median(axis=1); other={a:R.drop(columns=a).median(axis=1) for a in A}
V['ravmom']=P.pct_change(20)/R.rolling(20,min_periods=15).std(); V['volrev5']=-P.pct_change(5)/R.rolling(5,min_periods=4).std(); V['idio']=-R.sub(med,axis=0).rolling(20,min_periods=15).std(); V['downbeta']=pd.DataFrame({a:R[a].where(med<0).rolling(40,min_periods=20).cov(med.where(med<0))/med.where(med<0).rolling(40,min_periods=20).var() for a in A}); V['skew']=R.rolling(60,min_periods=40).skew(); V['lagauto']=-pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(R[a].shift(1)) for a in A}); V['common']=-pd.DataFrame({a:R[a].rolling(40,min_periods=25).corr(other[a]) for a in A});
vol5=R.rolling(5,min_periods=4).std();vol20=R.rolling(20,min_periods=15).std(); V['voltransition']=V['lagauto']*np.log(vol5/vol20).clip(-2,2); V['trend']=P.pct_change(20)/vol20; V['quiet']=P.pct_change(20).abs()/R.abs().rolling(20).sum()*(1-vol20.rank(axis=0,pct=True).rolling(60,min_periods=40).mean()); V['commonexp']=pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(other[a]) for a in A}).rolling(20).mean()-pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(other[a]) for a in A}).shift(20).rolling(20).mean(); V['tail']=-pd.DataFrame({a:(R[a]<R[a].rolling(60,min_periods=40).quantile(.2).shift(1)).rolling(40,min_periods=25).mean() for a in A})
# VIX and volumes
vd=get_index_daily_data('VIX',5000).copy();vd.date=pd.to_datetime(vd.date); vr=pd.to_numeric(vd.set_index('date').sort_index().close,errors='coerce').reindex(P.index).ffill().pct_change();
V['vixtrend']=V['trend'].mul(np.where((vr.rolling(20).sum()>0),-1,1),axis=0); V['vixbeta']=-pd.DataFrame({a:R[a].where(vr>0).rolling(40,min_periods=15).cov(vr.where(vr>0))/vr.where(vr>0).rolling(40,min_periods=15).var() for a in A})
vols={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date); vols[a]=pd.to_numeric(d.set_index('date').sort_index().get('volume'),errors='coerce').reindex(P.index)
Q=pd.DataFrame(vols).replace(0,np.nan);V['volume']=np.log(Q/Q.rolling(20).mean());V['stablevol']=-V['volume'].rolling(20,min_periods=12).std()
# downside excess and asymmetry approximate stated definitions
thr=med.rolling(60,min_periods=40).quantile(.35); V['downexcess']=R.sub(med,axis=0).where(med.shift(1)<thr.shift(1),axis=0).rolling(40,min_periods=10).median()
V['asym']=pd.DataFrame({a:R[a].where(med<0).rolling(60,min_periods=20).cov(med.where(med<0))/med.where(med<0).rolling(60,min_periods=20).var()-R[a].where(med>=0).rolling(60,min_periods=20).cov(med.where(med>=0))/med.where(med>=0).rolling(60,min_periods=20).var() for a in A})
res=[]
for n,x in V.items():
 z=pd.concat([F.rank(axis=1,pct=True).stack(),x.rank(axis=1,pct=True).stack()],axis=1).dropna();res.append((n,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
print('LIBRARY_CORRELATIONS');[print(n,round(c,6),k) for n,c,k in res];print('MAX',max(abs(c) for _,c,_ in res))
