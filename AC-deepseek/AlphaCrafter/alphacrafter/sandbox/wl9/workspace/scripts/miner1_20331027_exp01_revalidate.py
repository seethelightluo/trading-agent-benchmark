"""miner_1 cycle 2033-10-27. Visible through 2033-10-26.
Re-validate current effective library."""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner_1_lib import build_panel, load_macro, compute_ic, coverage, turnover, decay_ic
VIS = pd.Timestamp('2033-10-26')
closes, highs, lows, vols, rets = build_panel(end=VIS)
idx = closes.index
print(f"Panel: {closes.shape[0]} dates x {closes.shape[1]} assets, {idx[0]:%Y-%m-%d}..{idx[-1]:%Y-%m-%d}", flush=True)
vix=load_macro('VIX',VIS); dxy=load_macro('DXY',VIS); cny=load_macro('USDCNY',VIS)
dVIX=vix.pct_change(); dDXY=dxy.pct_change(); dCNY=cny.pct_change()
def fwd(h): return rets.shift(-h).rolling(h).mean()
fwd10=fwd(10)
def beta_win(x,y,w):
    out=pd.DataFrame(np.nan,index=x.index,columns=x.columns)
    for c in x.columns: out[c]=x[c].rolling(w).cov(y)
    return out.div(y.rolling(w).var())
def corr_win(x,y,w):
    out=pd.DataFrame(np.nan,index=x.index,columns=x.columns)
    for c in x.columns: out[c]=x[c].rolling(w).corr(y)
    return out
def ac_roll(ser,w):
    def _a(a):
        a=a[~np.isnan(a)]
        if len(a)<4 or np.std(a)==0: return np.nan
        return np.corrcoef(a[:-1],a[1:])[0,1]
    return ser.rolling(w).apply(_a,raw=True)
recent=pd.Timestamp('2033-01-01')
lib={
 'beta_VIX_60': -beta_win(rets,dVIX,60),
 'kaufman_eff_20d': (closes.diff().abs().rolling(20).sum()/(closes.diff(20).abs()).replace(0,np.nan)),
 'mom_120d_skip5': closes.pct_change(120),
 'bb_width_20d': (closes.rolling(20).std()*4)/closes.rolling(20).mean(),
 'cny_beta_60': beta_win(rets,dCNY,60),
 'vol_z_20d': -((rets.rolling(20).std()-rets.rolling(60).std().rolling(20).mean())/rets.rolling(60).std().rolling(20).std()),
 'ac1_120d': -ac_roll(rets,120),
 'mom_10d_skip5': closes.pct_change(10),
 'dxy_corr_chg_20_60': corr_win(rets,dDXY,20)-corr_win(rets,dDXY,60),
 'skew_20d': rets.rolling(20).skew(),
}
print('===== REVALIDATE EFFECTIVE LIBRARY (10d horizon) =====', flush=True)
for name,fv in lib.items():
    a=compute_ic(fv,fwd10); r=compute_ic(fv.loc[fv.index>=recent],fwd10.loc[fwd10.index>=recent])
    rok=abs(r['IC'])>=0.0070 and abs(r['ICIR'])>=0.084
    print(f"[rec{'OK' if rok else '--'}]{name:22s} FULL IC={a['IC']:+.4f} ICIR={a['ICIR']:+.4f} n={a['n']:4d} | rec(9m) IC={r['IC']:+.4f} ICIR={r['ICIR']:+.4f} n={r['n']:4d}",flush=True)
print('DONE',flush=True)