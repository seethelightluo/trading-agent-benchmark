"""miner_2 2035-01-18: revalidate existing effective library factors on recent window + full history."""
import pandas as pd, numpy as np, sys, json
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

vix = load_index('VIX'); dxy = load_index('DXY'); cny = load_index('USDCNY')
jpy = load_index('USDJPY'); eur = load_index('EURUSD')
dVIX = vix.pct_change(); dDXY = dxy.pct_change(); dCNY=cny.pct_change()
dJPY=jpy.pct_change(); dEUR=eur.pct_change()

def beta_win(rd, mr, w):
    rd, mr = rd.align(mr, join='inner', axis=0)
    cov = rd.rolling(w).cov(mr); var = mr.rolling(w).var().replace(0,np.nan)
    return cov.div(var, axis=0)

def kaufman(c, w=20):
    num = (c-c.shift(w)).abs(); den = c.diff().abs().rolling(w).sum().replace(0,np.nan)
    return num/den

def corr_win(rd, m, w):
    rd, m = rd.align(m, join='inner', axis=0)
    out = pd.DataFrame(np.nan, index=rd.index, columns=rd.columns)
    for c in rd.columns:
        out[c] = rd[c].rolling(w).corr(m)
    return out

def acf(c, w=120):
    r = c.pct_change()
    def _ac(x):
        x=x[~np.isnan(x)]
        if len(x)<5 or np.std(x)<1e-12: return np.nan
        return np.corrcoef(x[:-1],x[1:])[0,1]
    return r.rolling(w).apply(_ac, raw=True)

lib = {}
lib['beta_VIX_60'] = -beta_win(rets, dVIX, 60)
lib['kaufman_eff_20d'] = kaufman(frame, 20)
lib['mom_120d_skip5'] = frame.shift(5)/frame.shift(125)-1
lib['mom_10d_skip5'] = frame.shift(5)/frame.shift(15)-1
lib['vol_z_20d'] = pd.DataFrame({a: (lambda p: (lambda r: (r.rolling(20).std()-r.rolling(20).std().rolling(120).mean())/r.rolling(20).std().rolling(120).std())(p.pct_change()))(frame[a]) for a in frame.columns})
lib['bb_width_20d'] = pd.DataFrame({a: (frame[a].rolling(20).std()/frame[a].rolling(20).mean()) for a in frame.columns})
lib['cny_beta_60'] = beta_win(rets, dCNY, 60)
lib['ac1_120d'] = pd.DataFrame({a: acf(frame[a],120) for a in frame.columns}) if False else pd.DataFrame({a: (lambda s: s.pct_change().rolling(120).apply(lambda x: np.corrcoef(x[:-1],x[1:])[0,1] if len(x)>4 and np.std(x)>1e-12 else np.nan, raw=True))(frame[a].pct_change()) for a in frame.columns})
lib['dxy_corr_change_20_60'] = corr_win(rets.rank(axis=1).pct_change(), dDXY.rank().diff(), 1)  # placeholder
lib['skew_20d'] = rets.rolling(20).skew()

def report_lib(name, fdf, fwd10):
    r = rank_ic(fdf, fwd10, 8)
    ok = abs(r['ic'])>=0.0070 and abs(r['icir'])>=0.084
    print(f"[{'OK' if ok else '--'}] {name:22s} IC={r['ic']:+.4f} ICIR={r['icir']:+.4f} ndates={r['n_ic_dates']:5d} hit={r['ic_hit_ratio']:.3f}")

fwd10 = compute_forward_returns(frame, 10)
print("=== Revalidate LIBRARY (h=10) ===", flush=True)
for name, f in lib.items():
    try:
        report_lib(name, f, fwd10)
    except Exception as e:
        print(f"  !! {name} ERROR {e}", flush=True)