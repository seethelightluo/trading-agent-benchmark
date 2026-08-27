"""miner_2 2035-06-21: re-validate all EFFECTIVE factors + explore fresh candidates.
Visible window through 2035-06-20. Cross-sectional rank IC vs 10d forward return on 15-asset universe."""
import pandas as pd, numpy as np, json
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH","SPX","HSI","N225","SX5E","000688.SH","SOX","NDX","XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]
VISIBLE = pd.Timestamp('2035-06-20')

uni = {}
for s in WATCH:
    df = get_stock_daily_data(symbol=s, days=4000)
    if df is None or len(df) < 300:
        df = get_index_daily_data(symbol=s, days=4000)
    if df is not None and len(df) >= 300:
        df = df.copy(); df['date'] = pd.to_datetime(df['date']); df = df.set_index('date').sort_index()
        df = df[df.index <= VISIBLE]
        uni[s] = df
    else:
        print("MISSING", s)

close = pd.DataFrame({s: uni[s]['close'] for s in uni}).sort_index()
vol = pd.DataFrame({s: uni[s].get('volume', pd.Series(index=close.index)) for s in uni}).sort_index() if all('volume' in uni[s] for s in uni) else None
ret = close.pct_change()
fwd_10 = ret.shift(-10)
print(f"close rows={len(close)} assets={close.shape[1]} range {close.index[0].date()}..{close.index[-1].date()}", flush=True)

# Macro signals
dxy = pd.read_csv('../persistent/index_data/DXY.csv'); dxy['date']=pd.to_datetime(dxy['date']); d_dxy=dxy.set_index('date').sort_index()['pct_change']
vix = pd.read_csv('../persistent/index_data/VIX.csv'); vix['date']=pd.to_datetime(vix['date']); d_vix=vix.set_index('date').sort_index()['close']
cny = pd.read_csv('../persistent/index_data/USDCNY.csv'); cny['date']=pd.to_datetime(cny['date']); d_cny=cny.set_index('date').sort_index()['pct_change']
jpy = pd.read_csv('../persistent/index_data/USDJPY.csv'); jpy['date']=pd.to_datetime(jpy['date']); d_jpy=jpy.set_index('date').sort_index()['pct_change']
eur = pd.read_csv('../persistent/index_data/EURUSD.csv'); eur['date']=pd.to_datetime(eur['date']); d_eur=eur.set_index('date').sort_index()['pct_change']

def rank_ic_series(factor_df, fwd_ret, min_valid=8):
    ics = {}
    for dt in factor_df.index:
        if dt not in fwd_ret.index: continue
        f=factor_df.loc[dt].astype(float); r=fwd_ret.loc[dt].astype(float)
        m=f.notna()&r.notna()
        if m.sum()<min_valid: continue
        ic=f[m].rank().corr(r[m].rank())
        if not np.isnan(ic): ics[dt]=ic
    return pd.Series(ics)

def summarize(name, s, window=520):
    if len(s)<20:
        print(f"{name:26s}: TOO FEW ({len(s)})"); return None
    ic=s.mean(); std=s.std(ddof=1); icir=ic/std if std>0 else 0; hit=(s>0).mean()
    extra=""
    if window and len(s)>=window:
        r=s.iloc[-window:]; ric=r.mean(); ricir=ric/r.std(ddof=1) if r.std(ddof=1)>0 else 0
        extra=f"  recent{window}: ic={ric:+.4f} icir={ricir:+.4f} hit={(r>0).mean():.3f}"
    ok = abs(ic)>=0.0070 and abs(icir)>=0.084
    tag = "OK" if ok else "--"
    print(f"[{tag}] {name:30s}: ic={ic:+.4f} icir={icir:+.4f} hit={hit:.3f} n={len(s):5d} [{s.index.min():%Y-%m-%d}~{s.index.max():%Y-%m-%d}]{extra}")
    out={'ic':float(ic),'icir':float(icir),'hit':float(hit),'n_dates':len(s),'first':str(s.index.min().date()),'last':str(s.index.max().date())}
    if window and len(s)>=window:
        r=s.iloc[-window:]; out['recent_ic']=float(r.mean()); out['recent_icir']=float(r.mean()/r.std(ddof=1) if r.std(ddof=1)>0 else 0)
    return out

# ---- ESTABLISHED LIBRARY FACTORS ----
F={}
F['mom_10d_skip5'] = close/close.shift(10)-1
F['mom_120d_skip5'] = close/close.shift(120)-1
ma20=close.rolling(20).mean(); sd20=close.rolling(20).std()
F['bb_width_20d'] = (2*sd20)/ma20
v20=ret.rolling(20).std(); v120m=v20.rolling(120).mean(); v120s=v20.rolling(120).std(ddof=0)
F['vol_z_20d'] = (v20-v120m)/v120s
F['skew_20d'] = ret.rolling(20).skew()
F['kurt_20d'] = ret.rolling(20).kurt()
def ac(s,w=120): return s.rolling(w,min_periods=60).apply(lambda x: pd.Series(x).autocorr(lag=1),raw=False)
F['ac1_120d'] = pd.DataFrame({s:acr(ret[s],120) for s in close.columns})
F['kaufman_eff_20d'] = pd.DataFrame({s:(close[s]-close[s].shift(20)).abs()/close[s].diff().abs().rolling(20).sum() for s in close.columns})
F['rng_pos_20d'] = (close-close.shift(20))/close.shift(20)
F['days_since_high_60'] = pd.DataFrame({k: (lambda s:(lambda ss: pd.Series(np.cumsum((ss>=ss.rolling(60,min_periods=1).max()).astype(int)).astype(int)) )(s))(close[k]) for k in close.columns})
def streak(s,w=14):
    out=pd.Series(np.nan,index=s.index)
    for i in range(w,len(s)):
        win=s.iloc[i-w+1:i+1]; c=0
        for j in range(len(win)-1,-1,-1):
            if win.iloc[j]>0: c+=1
            else: break
        out.iloc[i]=c
    return out
F['streak_len_14'] = pd.DataFrame({s:streak(ret[s],14) for s in close.columns})
def rbeta(ret_df, macro, w):
    si=pd.concat([ret_df,macro.rename('M')],axis=1,join='inner')