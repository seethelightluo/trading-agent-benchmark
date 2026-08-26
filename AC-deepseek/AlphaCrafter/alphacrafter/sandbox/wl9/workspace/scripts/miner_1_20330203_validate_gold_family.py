"""miner_1 cycle 2033-02-03: validate gold-linking factor candidates + library corr.
Visible history up to 2033-02-02. No lookahead.
Admission gates (benchmark, 15-asset): abs daily paper IC >= 0.0070, abs ICIR >= 0.0840.
"""
import numpy as np
import pandas as pd
from pathlib import Path

VISIBLE_END = '2033-02-02'
STOCK_DIR = Path('../persistent/stock_data')
INDEX_DIR = Path('../persistent/index_data')
ASSETS = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
          'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

def load_series(assets):
    out = {}
    for a in assets:
        f = STOCK_DIR/f'{a}.csv'
        if not f.exists(): f = INDEX_DIR/f'{a}.csv'
        df = pd.read_csv(f, parse_dates=['date'])
        df = df[df['date']<=VISIBLE_END].sort_values('date').set_index('date')
        s = df['close'].astype(float)
        s = s[~s.index.duplicated(keep='last')]
        out[a] = s
    return out

ser = load_series(ASSETS)
close = pd.DataFrame(ser).dropna()
rets = close.pct_change().dropna()
fwd10 = rets.shift(-10).rolling(10).mean()
fwd5 = rets.shift(-5).rolling(5).mean()
fwd20 = rets.shift(-20).rolling(20).mean()
print(f"Panel: {close.shape[0]} dates x {close.shape[1]} assets, "
      f"{close.index[0]:%Y-%m-%d}..{close.index[-1]:%Y-%m-%d}", flush=True)

def compute_ic(fv, fwd, min_dates=30, start=None):
    fv = fv.reindex(fwd.index)
    if start is not None: mask = fwd.index >= pd.Timestamp(start)
    else: mask = np.ones(len(fwd.index), dtype=bool)
    ics=[]; n_dates=0
    for d in fwd.index[mask]:
        f=fv.loc[d]; r=fwd.loc[d]
        m=f.notna()&r.notna()
        if m.sum()>=8:
            n_dates+=1
            fv_=f[m].rank().values; rv_=r[m].rank().values
            if fv_.std()>0 and rv_.std()>0: ics.append(np.corrcoef(fv_,rv_)[0,1])
    ics=np.array(ics)
    if len(ics)<min_dates: return {'IC':0.0,'ICIR':0.0,'n':len(ics),'hit':0.0,'cov':0.0}
    mu=ics.mean(); sd=ics.std()
    icir = mu/sd*np.sqrt(len(ics)) if sd>0 else 0.0
    hit=float((ics>0).mean()); cov=float(fv.notna().mean().mean())
    return {'IC':float(mu),'ICIR':float(icir),'n':len(ics),'hit':hit,'cov':cov}

def turnover(fv):
    s=np.sign(fv.rank(axis=1).sub(fv.shape[1]/2)).fillna(0)
    return float((s.diff()!=0).mean().mean())

def show(name, cand, fwd=fwd10):
    ic = compute_ic(cand, fwd)
    ic_r = compute_ic(cand, fwd10, start='2030-01-01')
    print(f"{name}: IC={ic['IC']:.4f} ICIR={ic['ICIR']:.4f} n={ic['n']} "
          f"hit={ic['hit']:.3f} cov={ic['cov']:.3f} tov={turnover(cand):.3f} | "
          f"recent30+ IC={ic_r['IC']:.4f} ICIR={ic_r['ICIR']:.4f} n={ic_r['n']}", flush=True)

xau10 = close.pct_change(10)['XAU']
cand_gold_corr = pd.DataFrame({a: close.pct_change(10)[a].rolling(60).corr(xau10) for a in ASSETS}).reindex(fwd10.index)
show("CORR_TO_GOLD_60", cand_gold_corr)

xau5 = close.pct_change(5)['XAU']
cand_gold_corr20 = pd.DataFrame({a: close.pct_change(5)[a].rolling(20).corr(xau5) for a in ASSETS}).reindex(fwd10.index)
show("CORR_TO_GOLD_20", cand_gold_corr20)

xau_mom = close.pct_change(10)['XAU']
show("XAU_TREND10_XSEC", pd.DataFrame({a: xau_mom for a in ASSETS}).reindex(fwd10.index))

print("\n== decay for CORR_TO_GOLD_60 ==")
for h,fd in [('5',fwd5),('10',fwd10),('20',fwd20)]:
    ic = compute_ic(cand_gold_corr, fd)
    print(f"  {h}d: IC={ic['IC']:.4f} ICIR={ic['ICIR']:.4f} n={ic['n']} hit={ic['hit']:.3f}")

print("\n== sub-regime IC (10d horizon) CORR_TO_GOLD_60 ==")
for s in ['2023-01-01','2025-01-01','2027-01-01','2029-01-01','2031-01-01','2032-01-01']:
    ic = compute_ic(cand_gold_corr, fwd10, start=s)
    print(f"  from {s}: IC={ic['IC']:.4f} ICIR={ic['ICIR']:.4f} n={ic['n']} hit={ic['hit']:.3f}")

print("\n== cross-asset 10d-return corr matrix (full sample) ==")
mtx = rets.corr()
for a in ['XAU','BTC','ETH','COPPER','WTI','SPX','NDX','US10Y','CN10Y']:
    print(f"  {a:7s}: " + " ".join(f"{mtx.loc[a,x]:+.3f}" for x in ['XAU','BTC','ETH','SPX','US10Y']))

# ---- library correlation for candidate (cross-sectional rank rho) ----
vix = load_series(['VIX'])['VIX']
dxy = load_series(['DXY'])['DXY']
c = close
def library_factor(fid):
    if fid=='kaufman_eff_20d':
        return c.diff().abs().rolling(20).mean()/c.pct_change().abs().rolling(20).sum()
    if fid=='mom_120d_skip5':
        return c.shift(5)/c.shift(125)-1
    if fid=='mom_10d_skip5':
        return c.shift(5)/c.shift(15)-1
    if fid=='bb_width_20d':
        return (c.rolling(20).max()-c.rolling(20).min())/c.rolling(20).mean()
    if fid=='vol_z_20d':
        v=c.pct_change().rolling(20).std(); return (v-v.rolling(120).mean())/v.rolling(120).std()
    if fid=='skew_20d':
        return c.pct_change().rolling(20).skew()
    if fid=='kurt_20d':
        return c.pct_change().rolling(20).kurt()
    if fid=='ac1_120d':
        r=c.pct_change(); return r.rolling(120).apply(lambda x: np.corrcoef(x[:-1],x[1:])[0,1],raw=True)
    if fid=='beta_VIX_60':
        vr=vix.pct_change().dropna(); rr=close.pct_change().dropna()
        return rr.rolling(60).cov(vr)/vr.rolling
# ---- library correlation for candidate (cross-sectional rank rho) ----
vix = load_series(['VIX'])['VIX']
dxy = load_series(['DXY'])['DXY']
c = close
def library_factor(fid):
    if fid=='kaufman_eff_20d':
        return c.diff().abs().rolling(20).mean()/c.pct_change().abs().rolling(20).sum()
    if fid=='mom_120d_skip5':
        return c.shift(5)/c.shift(125)-1
    if fid=='mom_10d_skip5':
        return c.shift(5)/c.shift(15)-1
    if fid=='bb_width_20d':
        return (c.rolling(20).max()-c.rolling(20).min())/c.rolling(20).mean()
    if fid=='vol_z_20d':
        v=c.pct_change().rolling(20).std(); return (v-v.rolling(120).mean())/v.rolling(120).std()
    if fid=='skew_20d':
        return c.pct_change().rolling(20).skew()
    if fid=='kurt_20d':
        return c.pct_change().rolling(20).kurt()
    if fid=='ac1_120d':
        r=c.pct_change(); return r.rolling(120).apply(lambda x: np.corrcoef(x[:-1],x[1:])[0,1],raw=True)
    if fid=='beta_VIX_60':
        vr=vix.pct_change().dropna(); rr=close.pct_change().dropna()
        return rr.rolling(60).cov(vr)/vr.rolling(60).var()
    if fid=='dxy_corr_change_20_60':
        dr=dxy.pct_change(10); r10=c.pct_change(10)
        return r10.rolling(20).corr(dr)-r10.rolling(60).corr(dr)
    if fid=='cny_beta_60':
        cn10=c['CN10Y']; cr=close.pct_change().dropna(); ref=cn10.pct_change().dropna()
        return cr.rolling(60).cov(ref)/ref.rolling(60).var()
    if fid=='rng_pos_20d':
        rng=(c.rolling(20).max()-c.rolling(20).min())/c
        return rng.rolling(120).rank(pct=True)
    if fid=='streak_len_14':
        r=(c.pct_change()>0).astype(int); out=pd.DataFrame(0.0,index=c.index,columns=c.columns)
        for col in c.columns:
            s=r[col]; grp=(s!=s.shift()).cumsum(); cnt=s.groupby(grp).cumsum()
            out[col]=cnt
        return out
    return None

lib_fids = ['beta_VIX_60','kaufman_eff_20d','mom_120d_skip5','bb_width_20d','cny_beta_60',
            'vol_z_20d','ac1_120d','mom_10d_skip5','dxy_corr_change_20_60','skew_20d',
            'kurt_20d','rng_pos_20d','streak_len_14']
lib={}
for fid in lib_fids:
    f_=library_factor(fid)
    if f_ is not None: lib[fid]=f_.reindex(fwd10.index)

def rank_sign(x):
    r=x.rank(axis=1, pct=True)
    return r.sub(0.5)

cand_r = rank_sign(cand_gold_corr.reindex(fwd10.index))
maxrho=0.0; arg=''
for fid,fv in lib.items():
    fr=rank_sign(fv)
    common = cand_r.notna() & fr.notna()
    vals=[]
    for d in fwd10.index:
        cm=common.loc[d]
        if cm.sum()>=5:
            a=cand_r.loc[d,cm].values; b=fr.loc[d,cm].values
            if a.std()>0 and b.std()>0:
                vals.append(np.corrcoef(a,b)[0,1])
    if vals:
        rho=float(np.mean(vals))
        if abs(rho)>abs(maxrho):
            maxrho=rho; arg=fid
        print(f"  libcorr {fid:20s}: mean_rank_rho={rho:+.4f} n={len(vals)}")
print(f"MAX_ABS_LIB_CORR={maxrho:.4f} (vs {arg})")
