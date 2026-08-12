"""miner3 2030-05-24: revalidate effective library replicas + screen novel candidates (batch A)."""
import pandas as pd, numpy as np, json

panel = pd.read_pickle('scripts/panel_cache_20300524.pkl')
C = panel['close']; O = panel['open']; H = panel['high']; L = panel['low']
R = panel['ret']; V = panel['vol']; M = panel['macro']
gate_ic, gate_icir = 0.0070, 0.0840

def mom(nd, skip=1):
    return C.shift(skip) / C.shift(skip + nd) - 1.0

def vol(nd):
    return R.rolling(nd).std()

# ---------------- library replicas (exact definitions) ----------------
factors = {}
factors["lib_mom120"] = mom(120, 5)
factors["lib_vov20x60"] = vol(20).rolling(60).std()
vix = M['VIX']; vix_ret = vix.pct_change()
beta_vix60 = R.rolling(60).cov(vix_ret) / vix_ret.rolling(60).var()
factors["lib_vixbeta"] = -beta_vix60 * (vix / vix.shift(20) - 1.0).clip(-0.5, 0.5).reindex(C.index)
factors["lib_nclv1"] = -(C - L.rolling(1).min()) / (H.rolling(1).max() - L.rolling(1).min())
factors["lib_rev2"] = -(np.log(C) - np.log(C.shift(2)))

# ---------------- novel batch A ----------------
ma20 = C.rolling(20).mean(); ma60 = C.rolling(60).mean()
sd20 = R.rolling(20).std(); sd60 = R.rolling(60).std()

# A1 cross-sectional relative 5d return (contrarian dispersion)
r5 = mom(5, 0)
factors["xs_dev_5"] = r5.sub(r5.mean(axis=1), axis=0)
# A2 RSI(2)
up = R.clip(lower=0); dn = (-R).clip(lower=0)
au = up.rolling(2).mean(); ad = dn.rolling(2).mean()
factors["rsi_2"] = 100 - 100 / (1 + au / ad)
# A3 efficiency ratio 20d (trend quality)
gross = R.abs().rolling(20).sum()
factors["eff_ratio_20"] = (C / C.shift(20) - 1.0).abs() / gross
# A4 AR(1) autocorrelation of daily returns over 20d
def ar1(x):
    return x.autocorr()
factors["autocorr_1_20"] = R.rolling(20).apply(ar1, raw=False)
# A5 skewness 20d
factors["skew_20"] = R.rolling(20).skew()
# A6 downside vol ratio 20d
dnr = R.where(R < 0, 0.0)
factors["downside_ratio_20"] = np.sqrt((dnr ** 2).rolling(20).mean()) / sd20
# A7 vol trend 5x60
factors["vol_trend_5x60"] = vol(5) / vol(60) - 1.0
# A8 volume trend 5x60
vm5 = V.rolling(5).mean(); vm60 = V.rolling(60).mean()
factors["volume_trend_5x60"] = vm5 / vm60 - 1.0
# A9 beta to equal-weight cross-asset index (60d)
ew_ret = R.mean(axis=1)
beta_ew60 = R.rolling(60).cov(ew_ret) / ew_ret.rolling(60).var()
factors["beta_ew_60"] = beta_ew60
# A10 correlation to EW index (60d)
factors["corr_ew_60"] = R.rolling(60).corr(ew_ret)
# A11 avg close location in day range 20d
cloc = (C - L) / (H - L)
factors["range_pos_20"] = cloc.rolling(20).mean()
# A12 distance from 60d high
factors["maxdd_60"] = C / C.rolling(60).max() - 1.0
# A13 zscore 60d
factors["zscore_60"] = (C - ma60) / sd60
# A14 gain/loss ratio 20d (relative strength variant)
gu = R.where(R > 0, R, np.nan); gl = R.where(R < 0, -R, np.nan)
factors["gain_loss_20"] = gu.rolling(20).mean() / gl.rolling(20).mean()
# A15 momentum conditioned on VIX rising regime
vix_ma60 = vix.rolling(60).mean()
vix_rise = (vix > vix.shift(5)).astype(float).reindex(C.index)
factors["mom20_x_vixrise"] = mom(20, 1) * vix_rise
factors["mom20_x_vixfall"] = mom(20, 1) * (1 - vix_rise)

# ---------------- IC machinery ----------------
def ic_series_vec(X, F):
    valid = X.notna() & F.notna()
    n = valid.sum(axis=1)
    keep = n >= 8
    Xr = X.rank(axis=1).where(valid)
    Fr = F.rank(axis=1).where(valid)
    dX = Xr.sub(Xr.mean(axis=1), axis=0)
    dF = Fr.sub(Fr.mean(axis=1), axis=0)
    num = (dX * dF).sum(axis=1)
    den = np.sqrt((dX ** 2).sum(axis=1) * (dF ** 2).sum(axis=1))
    ic = (num / den).where(den > 0)
    return ic[keep & ic.notna()]

def stats(ic_ser):
    if len(ic_ser) == 0:
        return dict(n=0, ic=float('nan'), icir=float('nan'), hit=float('nan'))
    return dict(n=len(ic_ser), ic=float(ic_ser.mean()),
                icir=float(ic_ser.mean() / ic_ser.std()) if len(ic_ser) > 1 else float('nan'),
                hit=float((ic_ser > 0).mean()))

def turnover_ser(F):
    s = F.rank(axis=1)
    chg = s.diff().abs().sum(axis=1) / s.notna().sum(axis=1)
    return float(chg.mean())

fwd = {h: C.shift(-h) / C - 1.0 for h in [1, 2, 3, 5, 10]}

print(f"{'factor':26s} {'ic1':>7s} {'icir1':>7s} {'hit':>5s} {'n':>5s} {'cov':>5s} {'turn':>6s} | {'365d':>8s} {'icir':>7s} | {'120d':>8s} {'icir':>7s} | gate")
out = {}
for name, f in factors.items():
    row = {'h': {}}
    ic1 = ic_series_vec(f, fwd[1])
    row['h'][1] = stats(ic1)
    row['h'][2] = stats(ic_series_vec(f, fwd[2]))
    row['h'][3] = stats(ic_series_vec(f, fwd[3]))
    row['h'][5] = stats(ic_series_vec(f, fwd[5]))
    row['h'][10] = stats(ic_series_vec(f, fwd[10]))
    row['turn'] = turnover_ser(f)
    row['cov'] = float(f.notna().mean(axis=0).mean())
    out[name] = row
    if len(ic1):
        last365 = ic1[ic1.index >= ic1.index.max() - pd.Timedelta(days=365)]
        last120 = ic1[ic1.index >= ic1.index.max() - pd.Timedelta(days=120)]
        ic365 = last365.mean() if len(last365) else np.nan
        icir365 = last365.mean() / last365.std() if len(last365) > 1 else np.nan
        ic120 = last120.mean() if len(last120) else np.nan
        icir120 = last120.mean() / last120.std() if len(last120) > 1 else np.nan
        passed = abs(row['h'][1]['ic']) >= gate_ic and abs(row['h'][1]['icir']) >= gate_icir
        print(f"{name:26s} {row['h'][1]['ic']:7.4f} {row['h'][1]['icir']:7.3f} {row['h'][1]['hit']:5.2f} {row['h'][1]['n']:5d} {row['cov']:5.2f} {row['turn']:6.3f} | {ic365:8.4f} {icir365:7.3f} | {ic120:8.4f} {icir120:7.3f} | {'PASS' if passed else ''}")
    else:
        print(f"{name:26s} NO DATA")

with open('scripts/miner3_20300524_screenA_results.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print("saved scripts/miner3_20300524_screenA_results.json")
