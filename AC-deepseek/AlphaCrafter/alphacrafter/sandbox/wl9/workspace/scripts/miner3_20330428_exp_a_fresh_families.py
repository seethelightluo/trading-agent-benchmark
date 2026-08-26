def ev2(panel,fp,mv=8,horizon=None):
    ics=[]
    for t in panel.index:
        if t not in fp.index: continue
        f=np.asarray(panel.loc[t],dtype=float); r=np.asarray(fp.loc[t],dtype=float)
        v=~(np.isnan(f)|np.isnan(r))
        if v.sum()>=mv:
            rho,_=spearmanr(f[v],r[v])
            if not np.isnan(rho): ics.append(rho)
    ia=np.array(ics)
    if len(ia)<10: return dict(ic=0,icir=0,n=len(ia))
    ic=float(ia.mean()); s=float(ia.std(ddof=1))
    return dict(ic=ic,icir=float(ic/s if s>1e-10 else 0),n=len(ia))
def t10(panel):
    r=panel.rank(axis=1); return float(r.diff(10).abs().mean(axis=1).mean())
def sig(fname):
    d=json.load(open("factors/"+fname)); sa=d.get("validation",{}).get("signal_artifact",{})
    if not sa: return None
    rows=list(csv.reader(io.StringIO(zlib.decompress(base64.b64decode(sa["data"])).decode())))
    hd=rows[0]; dt=[r[0] for r in rows[1:]]
    M=np.array([[float(x) if x!="" else np.nan for x in r[1:]] for r in rows[1:]])
    return pd.DataFrame(M,index=dt,columns=hd[1:])
def mlc(panel,libs):
    b=0.0
    for f in libs:
        lp=sig(f)
        if lp is None: continue
        c=panel.index.intersection(pd.DatetimeIndex(lp.index))
        if len(c)<60: continue
        a=panel.loc[c].rank(axis=1); bb=lp.loc[c].rank(axis=1)
        rs=[spearmanr(a.iloc[i].values,bb.iloc[i].values)[0] for i in range(a.shape[0])]
        rs=[x for x in rs if not np.isnan(x)]
        if rs: b=max(b,abs(float(np.mean(rs))))
    return float(b)
libs=["ac1_120d.json","bb_width_20d.json","beta_VIX_60.json","cny_beta_60.json","days_since_high_60.json","dxy_corr_change_20_60.json","kaufman_eff_20d.json","kurt_20d.json","mom_10_vixreg.json","mom_10d_skip5.json","mom_120d_skip5.json","rng_pos_20d.json","skew_20d.json","streak_len_14.json","vix_beta_cond_60x20.json","vix_roc_20d.json","vol_z_20d.json"]
def main():
    np.seterr(all="ignore")
    df=build(); cp=pc(df); rv=rp_(df); vv=vp_(df)
    op=oc(df); hh=hi(df); ll=lo(df)
    m={x:df[f"{x}__close"] for x in MACRO}
    f5=fw(cp,5); f10=fw(cp,10); f20=fw(cp,20)
    vi=m["VIX"]; dx=m["DXY"]; cny=m["USDCNY"]; jpy=m["USDJPY"]; eur=m["EURUSD"]
    cand={}
    # 1) upside/downside capture ratio over 30d
    up=(rv.clip(lower=0)+1e-9).rolling(30).mean()
    dn=(-rv.clip(upper=0)+1e-9).rolling(30).mean()
    cand["updown_capture_30"]=up/(dn+1e-9)
    # 2) cross-asset equal-weight market beta (60d), cross-sectionally demeaned
    mkt=cp.mean(axis=1).pct_change().replace([np.inf,-np.inf],np.nan)
    beta=rv.rolling(60).cov(mkt)/(mkt.rolling(60).var()+1e-12)
    cand["beta_market_60_demean"]=beta.sub(beta.mean(axis=1),axis=0)
    # 3) vol ratio vs market vol (20d)
    mv=mkt.rolling(20).std()
    cand["vol_ratio_mkt_20"]=rv.rolling(20).std().div(mv,axis=0)
    # 4) intraday range position inside (high-low) band: (close-low)/(high-low), 20d avg
    rng=(hh-ll).replace(0,np.nan)
    pos=((cp-ll)/rng).rolling(20).mean()
    cand["range_pos_20_ll_hh"]=pos
    # 5) daily close-to-open gap mean 20d (overnight drift)
    gap=(cp/op-1).rolling(20).mean()
    cand["overnight_mom_20"]=gap
    # 6) CNY-linked return alignment: correlation of asset returns with USDCNY over 60d
    dcny=cny.pct_change()
    carr=rv.rolling(60).corr(dcny)
    cand["cny_ret_corr_60"]=carr
    # 7) EUR vs USD divergence: asset return corr differential USDJPY minus EURUSD (risk-on linkage)
    dj=jpy.pct_change(); de=eur.pct_change()
    cand["jx_ret_corr_diff_60"]=rv.rolling(60).corr(dj)-rv.rolling(60).corr(de)
    # 8) drawdown distance: (close - rolling_max 120)/rolling_max 120
    m120=cp.rolling(120).max()
    cand["dd_from_high_120"]=(cp-m120)/m120
    # 9) dispersion risk: 10d rolling std of cross-sectional returns (breadth/regime)
    rr=rv.sub(rv.mean(axis=1),axis=0)
    disp=rr.std(axis=1).rolling(10).mean()
    cand["cs_dispersion_10"]=disp
    print("n_dates",df.shape[0],"n_assets",len(ASSETS))
    print("factor                      h5_ic h5_icir h10_ic h10_icir h20_ic h20_icir  cov   mlc  to10")
    for name,p in cand.items():
        cov=float((~p.isna()).mean().mean())
        e5=ev2(p,f5); e10=ev2(p,f10); e20=ev2(p,f20)
        ml=mlc(p,libs); to=t10(p)
        print(f"{name:26s} {e5['ic']:6.4f} {e5['icir']:7.3f} {e10['ic']:6.4f} {e10['icir']:7.3f} {e20['ic']:6.4f} {e20['icir']:7.3f} {cov:5.2f} {ml:5.2f} {to:5.2f}")
main()