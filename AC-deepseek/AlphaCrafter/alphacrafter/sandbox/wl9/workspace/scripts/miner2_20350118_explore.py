"""miner_2 2035-01-18 fresh candidate factor exploration (full + recent window)."""
import pandas as pd, numpy as np, sys
sys.path.insert(0, 'scripts')
from miner2_20350118_toolkit import load_panel, build_frame, compute_forward_returns, rank_ic, ASSETS, VISIBLE

panel = load_panel()
frame = build_frame(panel)
rets = frame.pct_change()

def load_index(name):
    fp = f"../persistent/index_data/{name}.csv"
    df = pd.read_csv(fp); df['date']=pd.to_datetime(df['date'])
    df=df.set_index('date').sort_index(); df=df[df.index<=pd.Timestamp(VISIBLE)]
    return df.rename(columns={df.columns[1]:'close'})['close']

vix=load_index('VIX'); dxy=load_index('DXY')

# ---- constructors taking a single close series ----
def vol_ratio(p, w1=10, w2=60):
    r=p.pct_change(); return r.rolling(w1).std()/r.rolling(w2).std()
def range_pos(p, w=20):
    hi=p.rolling(w).max(); lo=p.rolling(w).min(); return (p-lo)/(hi-lo).replace(0,np.nan)
def rsi(p, w=14):
    r=p.diff(); g=r.clip(lower=0).rolling(w).mean(); l=(-r.clip(upper=0)).rolling(w).mean()
    return g/(g+l).replace(0,np.nan)
def updown21(p, w=21):
    x=p.pct_change(); up=x.clip(lower=0).sum(); dn=(-x.clip(upper=0)).sum()
    return up/dn.replace(0,np.nan)
def downside_share(p, w=20):
    r=p.pct_change(); return r.clip(upper=0).rolling(w).std()/r.rolling(w).std().replace(0,np.nan)
def dd120(p, w=120):
    return p/p.rolling(w).max()-1
def macd_ratio(p, fast=12, slow=26):
    ef=p.ewm(span=fast,adjust=False).mean(); es=p.ewm(span=slow,adjust=False).mean()
    return (ef-es)/p
def eff_ratio(p, w=20):
    return (p-p.shift(w)).abs()/p.diff().abs().rolling(w).sum().replace(0,np.nan)
def avg_price_zscore(p, w=40):
    return (p-p.rolling(w).mean())/p.rolling(w).std().replace(0,np.nan)
def up_vol_ratio(p, w=30):
    r=p.pct_change(); up=r.clip(lower=0); return up.rolling(w).sum()/(-r.clip(upper=0)).rolling(w).sum().replace(0,np.nan)

def gauss_order(p, w=30):
    r=p.pct_change()
    def _g(x):
        x=x[~np.isnan(x)]
        if len(x)<10 or np.std(x)<1e-12: return np.nan
        z=(x[-1]-np.mean(x))/np.std(x)
        from scipy.stats import norm
        return norm.cdf(z)
    return r.rolling(w).apply(_g, raw=True)

def ratio_recent_vol(p, w=5, bw=60):
    r=p.pct_change(); return r.rolling(w).std()/r.rolling(bw).std()

def build(fn, **kw):
    return pd.DataFrame({a: fn(frame[a], **kw) for a in frame.columns})

cands={}
cands['vol_ratio_10_60']=build(vol_ratio)
cands['range_pos_20']=build(range_pos)
cands['rsi_14']=build(rsi)
cands['updown21']=build(updown21)
cands['downside_vol_share_20']=build(downside_share)
cands['dd_120']=build(dd120)
cands['macd_ratio_12_26']=build(macd_ratio)
cands['eff_ratio_20']=build(eff_ratio)
cands['avg_price_z_40']=build(avg_price_zscore)
cands['upvol21_ratio']=build(up_vol_ratio)
cands['norm_cdf_30']=build(gauss_order)
cands['forecast_vol_5_60']=build(forecast_vol)

print("Exploring N factors:", len(cands), flush=True)
for h in (5,10,20):
    fwd = compute_forward_returns(frame, h)
    print(f"\n=== horizon {h} ===", flush=True)
    for name, f in cands.items():
        r = rank_ic(f, fwd, 8)
        ok = abs(r['ic'])>=0.0070 and abs(r['icir'])>=0.084
        print(f"[{'OK' if ok else '--'}] {name:22s} IC={r['ic']:+.4f} ICIR={r['icir']:+.4f} ndates={r['n_ic_dates']:5d} hit={r['ic_hit_ratio']:.3f}", flush=True)