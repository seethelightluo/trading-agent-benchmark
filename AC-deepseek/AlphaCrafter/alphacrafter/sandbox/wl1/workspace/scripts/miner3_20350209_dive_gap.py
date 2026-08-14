"""miner_3 2035-02-09: deep-dive on gap_mom_10d (suspicious hit ratio) + fixed rho audit."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PANEL = 'scripts/panel_cache_20350209.pkl'
with open(PANEL, 'rb') as f:
    panel = pd.read_pickle(f)

px = panel['close']; ret = panel['ret']
hi = panel['high']; lo = panel['low']; op = panel['open']; vol = panel['vol']

lib = {}
for n in [1, 2, 3, 5]:
    lib[f'rev_{n}d'] = -(np.log(px) - np.log(px.shift(n)))
    rmax = px.rolling(n).max(); rmin = px.rolling(n).min()
    lib[f'nclv_{n}d'] = -(px - rmin) / (rmax - rmin)
lib['id_rev_1d'] = -(px / px.shift(1) - 1.0)
lib['nbody_1d'] = -((px - op) / (hi - lo))
lib['rev_1d_vs'] = -(np.log(px) - np.log(px.shift(1))) / ret.rolling(20).std()
lib['mom_120d_skip5'] = px.shift(5) / px.shift(125) - 1.0
lib['vol_of_vol20x60'] = ret.rolling(20).std().rolling(60).std()

# --- gap factor daily IC series (h=1) ---
gap = op / px.shift(1) - 1.0
cand = gap.rolling(10).sum()
fwd = px.pct_change(1).shift(-1)
ics = {}
for dt in cand.index:
    f = cand.loc[dt]; r = fwd.loc[dt]
    m = f.notna() & r.notna()
    if int(m.sum()) >= 8:
        rho, _ = spearmanr(f[m], r[m])
        ics[dt] = rho
s = pd.Series(ics)
print("gap_mom_10d h1 IC series stats:")
print("  n:", len(s), "mean:", round(s.mean(), 4), "median:", round(s.median(), 4),
      "std:", round(s.std(), 4), "skew:", round(s.skew(), 3))
print("  pctiles:", {p: round(s.quantile(p), 4) for p in [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]})
print("  hit(mean sign):", round((np.sign(s) == np.sign(s.mean())).mean(), 4))
print("  hit(median sign):", round((np.sign(s) == np.sign(s.median())).mean(), 4))
print("  |IC|>0.2 count:", int((s.abs() > 0.2).sum()), "|IC|>0.3:", int((s.abs() > 0.3).sum()))
print("\n  top 8 positive IC dates:")
print(s.sort_values(ascending=False).head(8).round(3).to_string())
print("  bottom 5 IC dates:")
print(s.sort_values().head(5).round(3).to_string())

# winsorized mean
s2 = s.clip(s.quantile(0.01), s.quantile(0.99))
print("\n  winsorized(1-99) mean:", round(s2.mean(), 4), "ICIR:", round(s2.mean()/s2.std(), 3))

# --- where do the big positive ICs come from? check assets on those dates ---
print("\n  gap values & next-day returns on the top-3 IC dates:")
for dt in s.sort_values(ascending=False).index[:3]:
    f = cand.loc[dt]; r = fwd.loc[dt]
    m = f.notna() & r.notna()
    g = f[m].sort_values(ascending=False)
    print("  ", str(dt.date()), "top gap:", g.head(3).round(4).to_dict(),
          "| fwd ret of those:", r[m][g.index[:3]].round(4).to_dict())

# --- also test alternative: gap ratio 1d / 5d, and close-to-close equivalent ---
def ic_series(fac, h=1):
    fwd2 = px.pct_change(h).shift(-h)
    out = {}
    for dt in fac.index:
        f = fac.loc[dt]; r = fwd2.loc[dt]
        m = f.notna() & r.notna()
        if int(m.sum()) >= 8:
            rho, _ = spearmanr(f[m], r[m])
            out[dt] = rho
    return pd.Series(out)

for name, fac in [
    ('gap_1d', gap),
    ('gap_5d', gap.rolling(5).sum()),
    ('gap_10d', gap.rolling(10).sum()),
    ('gap_20d', gap.rolling(20).sum()),
]:
    s_ = ic_series(fac)
    print(f"\n{name}: mean={s_.mean():+.4f} med={s_.median():+.4f} icir={s_.mean()/s_.std():+.3f} hit={((np.sign(s_)==np.sign(s_.mean())).mean()):.3f}")

# --- fixed rho audit (pairwise complete) ---
def rho_lib_fixed(fac):
    alld = fac.index.intersection(lib['rev_1d'].index)
    rho_max, anchor = 0.0, None
    for k, v in lib.items():
        vv = v.loc[alld]
        mm = fac.loc[alld].notna() & vv.notna()
        if int(mm.sum().sum()) < 200:
            continue
        a = fac.loc[alld][mm].values.flatten()
        b = vv[mm].values.flatten()
        ok = ~(np.isnan(a) | np.isnan(b))
        if ok.sum() < 200:
            continue
        rho = np.corrcoef(a[ok], b[ok])[0, 1]
        if abs(rho) > abs(rho_max):
            rho_max, anchor = rho, k
    return float(rho_max), anchor

for name, fac in [('gap_10d', cand), ('gap_20d', gap.rolling(20).sum()), ('amihud_20d', (ret.abs()/vol.replace(0,np.nan)).rolling(20).mean())]:
    rm, anc = rho_lib_fixed(fac)
    print(f"  rho audit {name}: max={rm:.3f} anchor={anc}")
