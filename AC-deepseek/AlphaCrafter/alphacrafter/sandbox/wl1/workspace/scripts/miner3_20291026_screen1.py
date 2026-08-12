"""miner3 2029-10-26: revalidate effective library + screen novel candidates (batch 1)."""
import pandas as pd, numpy as np, json

panel = pd.read_pickle('scripts/panel_cache.pkl')
C = panel['close']; O = panel['open']; H = panel['high']; L = panel['low']
R = panel['ret']; V = panel['vol']; M = panel['macro']
gate_ic, gate_icir = 0.0070, 0.0840

def mom(nd, skip=1):
    return C.shift(skip) / C.shift(skip + nd) - 1.0

def vol(nd):
    return R.rolling(nd).std()

factors = {}
# ---- existing effective library (re-validation) ----
factors["mom_120d_skip5"] = mom(120, 5)
factors["vol_of_vol20x60"] = vol(20).rolling(60).std()
vix = M['VIX']; vix_ret = vix.pct_change()
cov60_vix = R.rolling(60).cov(vix_ret)
var60_vix = vix_ret.rolling(60).var()
beta_vix60 = cov60_vix / var60_vix
factors["vix_beta_cond_60x20"] = -beta_vix60 * (vix / vix.shift(20) - 1.0).clip(-0.5, 0.5)
# miner2 family rough replicas
factors["nclv_1d"] = -(C.shift(1) / C - 1.0) * (L.shift(1) / C.shift(1) - 1.0).clip(-0.1, 0.1)  # approx
factors["rev_2d"] = -(C.shift(2) / C - 1.0)

# ---- novel batch 1: trend quality / overextension ----
ma20 = C.rolling(20).mean(); ma60 = C.rolling(60).mean(); sd20 = R.rolling(20).std()
factors["zscore20"] = (C - ma20) / sd20
factors["ma_slope20x60"] = ma20 / ma60 - 1.0
ema12 = C.ewm(span=12, adjust=False).mean(); ema26 = C.ewm(span=26, adjust=False).mean()
factors["macd_hist"] = (ema12 - ema26) / C
factors["mom60_voladj"] = mom(60, 5) / vol(60)
factors["mom20_voladj"] = mom(20, 1) / vol(20)
factors["dd_40d"] = C / C.rolling(40).max() - 1.0
factors["dd_10d"] = C / C.rolling(10).max() - 1.0
# trend consistency: fraction of last 20 days with close > MA20
above20 = (C > ma20).astype(float)
factors["trend_consistency20"] = above20.rolling(20).mean()
# ---- risk asymmetry ----
d = R.where(R < 0, 0.0)
down_vol20 = np.sqrt((d ** 2).rolling(20).mean())
factors["downside_ratio20"] = down_vol20 / vol(20)
factors["vol_spike5x20"] = vol(5) / vol(20) - 1.0
factors["vol_spike10x60"] = vol(10) / vol(60) - 1.0
# ---- macro-conditional ----
vix_ma60 = vix.rolling(60).mean()
vix_hi = (vix > vix_ma60).astype(float)
factors["mom20_x_vixhi"] = mom(20, 1) * vix_hi.reindex(C.index)
factors["mom20_x_vixlo"] = mom(20, 1) * (1 - vix_hi.reindex(C.index))
dxy = M['DXY']; dxy_ret = dxy.pct_change()
cov60_dxy = R.rolling(60).cov(dxy_ret); var60_dxy = dxy_ret.rolling(60).var()
factors["dxy_beta_60_x_dxyret20"] = (cov60_dxy / var60_dxy) * (dxy / dxy.shift(20) - 1.0).reindex(C.index)

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

print(f"{'factor':26s} {'ic1':>7s} {'icir1':>7s} {'hit':>5s} {'n':>5s} {'turn':>6s} | {'365d ic':>8s} {'icir':>7s} | {'120d ic':>8s} {'icir':>7s} | gate")
out = {}
fwd1 = C.shift(-1) / C - 1.0
for name, f in factors.items():
    row = {}
    ic1 = ic_series_vec(f, fwd1)
    row['h1'] = stats(ic1)
    row['turn'] = turnover_ser(f)
    out[name] = row
    if len(ic1):
        last365 = ic1[ic1.index >= ic1.index.max() - pd.Timedelta(days=365)]
        last120 = ic1[ic1.index >= ic1.index.max() - pd.Timedelta(days=120)]
        ic365 = last365.mean() if len(last365) else np.nan
        icir365 = last365.mean() / last365.std() if len(last365) > 1 else np.nan
        ic120 = last120.mean() if len(last120) else np.nan
        icir120 = last120.mean() / last120.std() if len(last120) > 1 else np.nan
        passed = abs(row['h1']['ic']) >= gate_ic and abs(row['h1']['icir']) >= gate_icir
        print(f"{name:26s} {row['h1']['ic']:7.4f} {row['h1']['icir']:7.3f} {row['h1']['hit']:5.2f} {row['h1']['n']:5d} {row['turn']:6.3f} | {ic365:8.4f} {icir365:7.3f} | {ic120:8.4f} {icir120:7.3f} | {'PASS' if passed else ''}")
    else:
        print(f"{name:26s} NO DATA")

with open('scripts/miner3_20291026_screen1_results.json', 'w') as f:
    json.dump(out, f, indent=1, default=str)
print("saved scripts/miner3_20291026_screen1_results.json")
