"""miner3 2031-02-28: coarse scan of candidate factor families on 15-asset cross-section.
Computes rank IC / ICIR at horizons 1,5,10 over full sample and regime splits,
plus max_abs_library_correlation vs currently kept library factors (recomputed)."""
import pandas as pd, numpy as np, json, glob

panel = pd.read_pickle('scripts/panel_cache_20310228.pkl')
close = panel['close']; open_px = panel['open']; high = panel['high']; low = panel['low']
vol = panel['vol']; ret = panel['ret']; macro = panel['macro']
WATCH = list(close.columns)
n = len(close)

def rank_ic(factor, fwd_h, min_n=8):
    """Spearman IC between factor at t and forward return t..t+h. Returns Series of daily IC."""
    fwd = close.pct_change(fwd_h).shift(-fwd_h)  # return from t to t+h
    dates, ics = [], []
    fv = factor.values; rv = fwd.values
    for i in range(fwd_h, n - fwd_h):
        frow = fv[i]; rrow = rv[i]
        m = ~(np.isnan(frow) | np.isnan(rrow))
        if m.sum() < min_n:
            continue
        from scipy.stats import spearmanr
        rho, _ = spearmanr(frow[m], rrow[m])
        if not np.isnan(rho):
            dates.append(close.index[i]); ics.append(rho)
    s = pd.Series(ics, index=dates)
    return s

def summarize(name, f, direction=1):
    out = {'factor': name}
    for h in (1, 5, 10):
        ic = rank_ic(f, h)
        ic = ic * direction
        out[f'ic{h}'] = round(ic.mean(), 4)
        out[f'icir{h}'] = round(ic.mean() / ic.std(), 4) if ic.std() > 0 else np.nan
        out[f'hit{h}'] = round((ic > 0).mean(), 3)
        out[f'nd{h}'] = len(ic)
        # regime splits for h=10
        if h == 10:
            for lab, a, b in [('20-22', '2020-01-01', '2022-12-31'), ('23-25', '2023-01-01', '2025-12-31'), ('26-31', '2026-01-01', '2031-12-31')]:
                sub = ic[(ic.index >= a) & (ic.index <= b)]
                out[f'ic10_{lab}'] = round(sub.mean(), 4)
                out[f'icir10_{lab}'] = round(sub.mean() / sub.std(), 4) if len(sub) > 30 and sub.std() > 0 else np.nan
    cov = f.notna().mean().mean()
    out['coverage'] = round(cov, 3)
    return out

c = close
r = ret
# ---- candidate factors ----
F = {}
# F1: Kaufman efficiency ratio 20d
num = (c - c.shift(20)).abs()
den = r.abs().rolling(20).sum()
F['eff_ratio_20d'] = num / den
# F2: efficiency ratio 60d
F['eff_ratio_60d'] = (c - c.shift(60)).abs() / r.abs().rolling(60).sum()
# F3: risk-adjusted momentum 60d / std20
F['ram_60x20'] = (c / c.shift(60) - 1) / r.rolling(20).std()
# F4: downside-deviation-scaled momentum 20d
dd = r.rolling(20).apply(lambda x: np.sqrt((x[x < 0] ** 2).mean()), raw=True)
F['sortino_mom_20x20'] = (c / c.shift(20) - 1) / dd
# F5: return asymmetry 20d
pos = r.where(r > 0).rolling(20).mean()
neg = r.where(r < 0).rolling(20).mean()
F['asym_20d'] = pos / neg.abs()
# F6: lag-1 autocorrelation of returns (20d window)
F['ac1_20d'] = r.rolling(20).apply(lambda x: pd.Series(x).autocorr(1) if len(x) > 4 else np.nan, raw=False)
# F7: close location value 20d (overbought/oversold), direction -
F['clv20'] = ((c - low) / (high - low)).rolling(20).mean()
# F8: range-based vol 10d
F['range_vol_10d'] = ((high - low) / c).rolling(10).mean()
# F9: drawdown depth 60d (distance from 60d max), direction +
F['dd60'] = c / c.rolling(60).max()
# F10: gap ratio 5d (overnight vs intraday move)
gap = (open_px - c.shift(1)).abs()
iday = (c - open_px).abs()
F['gap_ratio_5d'] = gap.rolling(5).mean() / (iday.rolling(5).mean() + 1e-12)
# F11: DXY beta 60d (direction -: low-USD-beta / USD-sensitive assets)
dm = macro['DXY'].reindex(c.index).ffill()
dret = dm.pct_change()
F['dxy_beta_60d'] = r.rolling(60).cov(dret) / dret.rolling(60).var()
# F12: vol-of-vol via range (alternative to library vol_of_vol)
F['vov_range_20x60'] = ((high - low) / c).rolling(20).std().rolling(60).std()

print(f"{'factor':<22}{'ic1':>7}{'icir1':>8}{'ic5':>7}{'icir5':>8}{'ic10':>7}{'icir10':>8}{'hit10':>7}{'cov':>6}  ic10_2020-22  ic10_2023-25  ic10_2026-31")
for name, f in F.items():
    s = summarize(name, f, direction=1)
    print(f"{name:<22}{s['ic1']:>7.4f}{s['icir1']:>8.3f}{s['ic5']:>7.4f}{s['icir5']:>8.3f}{s['ic10']:>7.4f}{s['icir10']:>8.3f}{s['hit10']:>7.3f}{s['coverage']:>6.2f}  {s['ic10_20-22']:>9.4f}  {s['ic10_23-25']:>9.4f}  {s['ic10_26-31']:>9.4f}")

# ---- library correlation for best candidates (compute library factor signals) ----
lib = {}
lib['mom_120d_skip5'] = c.shift(5) / c.shift(125) - 1.0
lib['vol_of_vol20x60'] = r.rolling(20).std().rolling(60).std()
vix = macro['VIX'].reindex(c.index).ffill()
vixr = vix.pct_change()
lib['vix_beta_cond_60x20'] = -(r.rolling(60).cov(vixr) / vixr.rolling(60).var()) * (vix / vix.shift(20) - 1.0)
lib['nclv_1d'] = -(c - low.rolling(1).min()) / (high.rolling(1).max() - low.rolling(1).min())
lib['rev_2d'] = -(np.log(c) - np.log(c.shift(2)))
lib['rev_1d'] = -(np.log(c) - np.log(c.shift(1)))
lib['rev_3d'] = -(np.log(c) - np.log(c.shift(3)))
lib['rev_5d'] = -(np.log(c) - np.log(c.shift(5)))
lib['rev_1d_vs'] = -(np.log(c) - np.log(c.shift(1))) / r.rolling(20).std()
lib['id_rev_1d'] = -(c / open_px - 1)
lib['nbody_1d'] = -(c - open_px) / (high - low)
for k in (2, 3, 5):
    lib[f'nclv_{k}d'] = -(c - low.rolling(k).min()) / (high.rolling(k).max() - low.rolling(k).min())
lib['mom_10d_skip5'] = np.log(c / c.shift(10)) - np.log(c / c.shift(5))

print("\n--- max_abs_library_correlation (Pearson on overlapping valid pairs) ---")
for name in ['eff_ratio_20d', 'eff_ratio_60d', 'ram_60x20', 'sortino_mom_20x20', 'asym_20d', 'ac1_20d', 'clv20', 'dd60', 'gap_ratio_5d', 'dxy_beta_60d']:
    mx = 0; mxk = ''
    for k, lf in lib.items():
        a = F[name].stack(); b = lf.stack()
        df = pd.concat([a, b], axis=1, join='inner').dropna()
        if len(df) < 500:
            continue
        rho = df.iloc[:, 0].corr(df.iloc[:, 1])
        if abs(rho) > mx:
            mx = abs(rho); mxk = k
    print(f"{name:<22} max_abs_corr={mx:.3f} vs {mxk}")
