"""Screener cycle 2031-07-24: recompute factor signals on date-gated data and
measure recent cross-sectional rank IC (last 180/90 trading days)."""
import pandas as pd, numpy as np, glob, json, warnings
warnings.filterwarnings('ignore')

CUTOFF = pd.Timestamp('2031-07-23')
HORIZON = 10
WATCH = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

# ---- load price data (date-gated) ----
px = {}
for f in glob.glob('../persistent/stock_data/*.csv'):
    name = f.split('/')[-1].replace('.csv','')
    df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]
    dc = 'date' if 'date' in df.columns else df.columns[0]
    df[dc] = pd.to_datetime(df[dc])
    px[name] = df.set_index(dc)['close'].sort_index()
px = pd.DataFrame(px).sort_index()
px = px[px.index <= CUTOFF]
px = px[WATCH]

macro = {}
for f in glob.glob('../persistent/index_data/*.csv'):
    name = f.split('/')[-1].replace('.csv','')
    df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]
    dc = 'date' if 'date' in df.columns else df.columns[0]
    df[dc] = pd.to_datetime(df[dc])
    macro[name] = df.set_index(dc)['close'].sort_index()
macro = pd.DataFrame(macro).sort_index()
macro = macro[macro.index <= CUTOFF]

rets = px.pct_change()
fwd = px.shift(-HORIZON)/px - 1.0

def rank_ic(sig, fwd, min_assets=8):
    """Cross-sectional Spearman rank IC per date."""
    dates, ics = [], []
    for dt in sig.index:
        s = sig.loc[dt]; f = fwd.loc[dt]
        m = s.notna() & f.notna()
        if m.sum() < min_assets: continue
        ics.append(s[m].rank().corr(f[m].rank()))
        dates.append(dt)
    return pd.Series(ics, index=dates)

def summarize(ic):
    if len(ic) < 5: return None
    out = {}
    for lab, sl in [('all', ic), ('last180', ic.iloc[-180:]), ('last90', ic.iloc[-90:])]:
        if len(sl) < 5: continue
        out[lab] = dict(ic=sl.mean(), icir=sl.mean()/sl.std() if sl.std()>0 else np.nan, n=len(sl), hit=(sl>0).mean())
    return out

# ---- compute signals per documented expressions ----
S = {}
lp = np.log(px)
t = np.arange(len(px))
def trend_r2(col, w=30):
    x = col.values; out = np.full(len(x), np.nan)
    for i in range(w-1, len(x)):
        y = x[i-w+1:i+1]
        if not np.isfinite(y).all(): continue
        sl = np.polyfit(t[:w], y, 1)[0]
        r2 = np.corrcoef(t[:w], y)[0,1]**2
        out[i] = np.sign(sl)*r2
    return pd.Series(out, index=col.index)
S['trend_r2_30_signed'] = lp.apply(lambda c: trend_r2(c))

r = rets.clip(upper=0); ru = rets.clip(lower=0)
S['semi_down_ratio_20'] = np.sqrt((r**2).rolling(20, min_periods=8).mean())/np.sqrt((ru**2).rolling(20, min_periods=8).mean()) - 1.0

S['mom_120d_skip5'] = px.shift(5)/px.shift(125) - 1.0
S['mom_10d_skip5'] = px.shift(5)/px.shift(15) - 1.0

tuw = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
for c in px.columns:
    roll_max = px[c].rolling(120, min_periods=40).max()
    days = 0; vals = []
    for dt in px.index:
        if np.isnan(roll_max.loc[dt]):
            vals.append(np.nan); continue
        if px.loc[dt, c] >= roll_max.loc[dt] - 1e-12:
            days = 0
        else:
            days += 1
        vals.append(days)
    tuw[c] = vals
S['time_under_water_120'] = tuw

rv = rets.rolling(20, min_periods=10).std()
S['vol_of_vol20x60'] = rv.rolling(60, min_periods=30).std()

S['tail_ratio_20'] = rets.rolling(20, min_periods=10).quantile(0.95)/rets.rolling(20, min_periods=10).quantile(0.05).abs()

def kurt(col, w=20):
    m4 = (col**4).rolling(w, min_periods=8).mean()
    m2 = (col**2).rolling(w, min_periods=8).mean()
    return m4/m2**2 - 3.0
S['kurt_20'] = kurt(rets)

def beta_to(asset_rets, macro_ret, w=60, minp=30):
    out = pd.DataFrame(index=asset_rets.index, columns=asset_rets.columns, dtype=float)
    for c in asset_rets.columns:
        a = asset_rets[c]
        df = pd.concat([a, macro_ret.rename('m')], axis=1).dropna()
        if len(df) < minp: continue
        cov = df[c].rolling(w, min_periods=minp).cov(df['m'])
        var = df['m'].rolling(w, min_periods=minp).var()
        out[c] = cov/var
    return out

m_ret = {}
for k in ['USDCNY','DXY','WTI','VIX']:
    m_ret[k] = macro[k].pct_change()
S['cny_beta_60'] = beta_to(rets, m_ret['USDCNY'])
S['dxy_beta_60'] = beta_to(rets, m_ret['DXY'])
S['WTI_BETA_60'] = beta_to(rets, m_ret['WTI'])
S['vix_beta_cond_60x20'] = -beta_to(rets, m_ret['VIX']) * (macro['VIX']/macro['VIX'].shift(20) - 1.0).reindex(rets.index).ffill()

# ---- evaluate ----
print(f"{'factor':24s} {'ic_all':>8s} {'icir_all':>8s} {'ic_180':>8s} {'icir_180':>8s} {'hit_180':>7s} {'ic_90':>8s} {'icir_90':>8s} {'hit_90':>6s}")
results = {}
for fid, sig in S.items():
    ic = rank_ic(sig, fwd)
    res = summarize(ic)
    if not res: continue
    results[fid] = res
    def g(k, lab):
        v = res.get(lab, {})
        return v.get(k, np.nan)
    print(f"{fid:24s} {g('ic','all'):8.4f} {g('icir','all'):8.3f} {g('ic','last180'):8.4f} {g('icir','last180'):8.3f} {g('hit','last180'):7.2f} {g('ic','last90'):8.4f} {g('icir','last90'):8.3f} {g('hit','last90'):6.2f}")

json.dump({k: {kk: {kkk: (float(vvv) if vvv==vvv else None) for kkk,vvv in vv.items()} for kk,vv in v.items()} for k,v in results.items()}, open('scripts/screener_20310724_recent_ic.json','w'), indent=1)
print('saved scripts/screener_20310724_recent_ic.json')
