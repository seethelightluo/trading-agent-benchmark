"""miner_2 exploration: candidate factors on 15-asset panel, cross-sectional rank IC validation."""
import pandas as pd, numpy as np, sys
sys.path.insert(0, 'scripts')
from miner2_20341221_toolkit import load_panel, build_frame, compute_forward_returns, rank_ic, ASSETS, VISIBLE

np.set_printoptions(suppress=True)
panel = load_panel()
frame = build_frame(panel)
rets = frame.pct_change()

def load_index(name):
    fp = f"../persistent/index_data/{name}.csv"
    df = pd.read_csv(fp); df['date']=pd.to_datetime(df['date'])
    df=df.set_index('date').sort_index(); df=df[df.index<=pd.Timestamp(VISIBLE)]
    return df.rename(columns={df.columns[1]:'close'})['close']

dxy = load_index('DXY'); vix = load_index('VIX')

def vol_ratio(p, w1=10, w2=60):
    r = p.pct_change()
    return r.rolling(w1).std() / r.rolling(w2).std()

def vol_zscore(p, w=20, lookback=120):
    r = p.pct_change(); mv = r.rolling(w).std()
    return (mv - mv.rolling(lookback).mean()) / mv.rolling(lookback).std()

def ac(p, w=10, lag=1):
    r = p.pct_change()
    return r.rolling(w).apply(lambda x: np.corrcoef(x[:-lag], x[lag:])[0,1] if len(x)>=lag+2 and x[:-lag].var()>0 and x[lag:].var()>0 else np.nan, raw=False)

def range_pos(p, w=20):
    hi=p.rolling(w).max(); lo=p.rolling(w).min()
    return (p-lo)/(hi-lo).replace(0,np.nan)

def winrate(p, w=20):
    return p.pct_change().rolling(w).apply(lambda x:(x>0).mean(), raw=True)

def drawdown(p, w=120):
    return p/p.rolling(w).max()-1.0

def macd_hist(p, fast=12, slow=26):
    ema= lambda s: p.ewm(span=s, adjust=False).mean()
    return ema(fast)-ema(slow)

def bs_force(p, w=14):
    r = p.diff()
    up = r.clip(lower=0).rolling(w).sum()
    dn = (-r.clip(upper=0)).rolling(w).sum()
    return (up-dn)/(up+dn).replace(0,np.nan)

def rsi(p, w=14):
    r = p.diff()
    g = r.clip(lower=0).rolling(w).mean()
    l = (-r.clip(upper=0)).rolling(w).mean()
    return g/(g+l).replace(0,np.nan)

def skew(p, w=20):
    return p.pct_change().rolling(w).skew()

def kurt(p, w=20):
    return p.pct_change().rolling(w).kurt()

def caic(p, w=20):
    r=p.pct_change()
    return r.rolling(w).apply(lambda x: x.autocorr() if (len(x.dropna())>=3 and x.std()>0) else np.nan, raw=False)

def width_fn(p, w=20):
    r=p.pct_change().rolling(w).std()
    return r / p.rolling(w).mean()

def best_pos(p, w=20):
    hi=p.rolling(w).max()
    return (p-hi)/hi

def up_down_21(p, w=21):
    x=p.pct_change()
    up=x.clip(lower=0).sum(); dn=(-x.clip(upper=0)).sum()
    return up/(dn).replace(0,np.nan)

# Build candidates
cands = {}
cands['vol_ratio_10_60'] = pd.DataFrame({a: vol_ratio(frame[a]) for a in frame.columns})
cands['vol_z_20d'] = pd.DataFrame({a: vol_zscore(frame[a],20) for a in frame.columns})
cands['autocorr_10'] = pd.DataFrame({a: ac(frame[a],10) for a in frame.columns})
cands['range_pos_20'] = pd.DataFrame({a: range_pos(frame[a],20) for a in frame.columns})
cands['winrate_20'] = pd.DataFrame({a: winrate(frame[a],20) for a in frame.columns})
cands['drawdown_120'] = pd.DataFrame({a: drawdown(frame[a],120) for a in frame.columns})
cands['mom_60_skip5'] = pd.DataFrame({a: (frame[a].shift(5)/frame[a].shift(65)-1) for a in frame.columns})
cands['rsi_14'] = pd.DataFrame({a: rsi(frame[a],14) for a in frame.columns})
cands['bsi_14'] = pd.DataFrame({a: best_pos(frame[a],14) for a in frame.columns})
# remove the broken ones (skip)

results=[]
for name, fdf in cands.items():
    for h in (5,10,20):
        fwd = compute_forward_returns(frame, horizon=h)
        res = rank_ic(fdf, fwd, 8)
        results.append(dict(factor=name,horizon=h,**res))
for r in results:
    print(f"{r['factor']:16s} h={r['horizon']:<3d} IC={r['ic']: .4f} ICIR={r['icir']: .4f} hit={r['ic_hit_ratio']:.3f} ndates={r['n_ic_dates']}")
print("N_FACTORS", len(cands))