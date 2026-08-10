"""Deep validation of pos_freq (win-rate) family for persistence to factors/."""
import sys, time, numpy as np, pandas as pd
from scipy import stats as st
sys.path.insert(0, "scripts")
from miner1_common import load_close, ic_analysis, coverage, turnover, decay_analysis

t0 = time.time()
closes = load_close()
idx = None
for s in closes:
    idx = closes[s].index if idx is None else idx.intersection(closes[s].index)
idx = idx[(idx >= pd.Timestamp("2021-01-01"))]
SYMS = list(closes.keys())
CP = pd.DataFrame({s: closes[s]["close"].reindex(idx).astype(float) for s in closes})
LRET = np.log(CP / CP.shift(1))
fwd1 = CP.shift(-1) / CP - 1.0


def make_pf(n, skip=0):
    return (LRET > 0).rolling(n).mean().shift(skip)


def per_year(panel, label):
    print(f"\n--- {label} year-by-year IC (1d fwd) ---")
    ok = True
    for yr in range(2021, 2027):
        m = (idx >= pd.Timestamp(f"{yr}-01-01")) & (idx <= pd.Timestamp(f"{yr}-12-31"))
        if m.sum() < 50:
            continue
        r = ic_analysis(panel.loc[idx[m]], closes, fwd_days=1)
        if r["n_dates"]:
            print(f"  {yr}: IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} hit={r['hit']:.3f} n={r['n_dates']}")
            if abs(r["ic"]) < 0.007 or abs(r["icir"]) < 0.084:
                ok = False
    return ok


def leave_one_out(panel, label):
    print(f"\n--- {label} leave-one-out IC1 (asset exclusion) ---")
    base = ic_analysis(panel, closes, fwd_days=1)
    print(f"  all     : IC={base['ic']:+.4f} ICIR={base['icir']:+.3f} n={base['n_dates']}")
    for s in SYMS:
        sub = panel.drop(columns=[s])
        r = ic_analysis(sub, closes, fwd_days=1)
        print(f"  -{s:8s}: IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} n={r['n_dates']}")


def vix_split(panel, label):
    vix = pd.read_csv("../persistent/index_data/VIX.csv")
    vix["date"] = pd.to_datetime(vix["date"])
    v = vix.set_index("date")["close"].reindex(idx)
    med = v.median()
    print(f"\n--- {label} VIX regime split (1d) ---")
    for lab, m in [("lowVIX", v < med), ("highVIX", v >= med)]:
        r = ic_analysis(panel.loc[idx[m]], closes, fwd_days=1)
        print(f"  {lab:8s}: IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} hit={r['hit']:.3f} n={r['n_dates']}")


for n, skip, tag in [(250, 0, "pos_freq_250d"), (250, 5, "pos_freq_250d_skip5"), (400, 0, "pos_freq_400d")]:
    panel = make_pf(n, skip).reindex(idx)
    cov = float(panel.notna().sum().sum()) / (len(idx) * len(SYMS))
    to = turnover(panel)
    ic1 = ic_analysis(panel, closes, fwd_days=1)
    dec = decay_analysis(panel, closes, horizons=(1, 2, 3, 5, 10, 20, 30))
    print(f"\n===== {tag}: cov={cov:.3f} to={to:.3f} | IC1={ic1['ic']:+.4f} ICIR1={ic1['icir']:+.3f} "
          f"hit={ic1['hit']:.3f} n_dates={ic1['n_dates']} | decay={ {k: round(v,4) for k,v in dec.items()} }")
    per_year(panel, tag)
    leave_one_out(panel, tag)
    vix_split(panel, tag)

# correlation among family members (will inform which single factor to persist)
print("\n--- pairwise pooled correlation among pos_freq variants ---")
variants = {"pf250": make_pf(250, 0), "pf250s5": make_pf(250, 5), "pf300": make_pf(300, 0), "pf400": make_pf(400, 0)}
for a in variants:
    for b in variants:
        if a < b:
            x = variants[a].stack().dropna()
            y = variants[b].stack().dropna()
            common = x.index.intersection(y.index)
            r = st.pearsonr(x.loc[common], y.loc[common])[0]
            print(f"  corr({a},{b}) = {r:+.4f} n={len(common)}")

# correlation with plain 250d momentum (library-adjacent family audit)
mom250 = LRET.rolling(250).sum()
pf250 = make_pf(250, 0)
x = pf250.stack().dropna(); y = mom250.stack().dropna()
common = x.index.intersection(y.index)
print(f"\npooled corr(pf250, mom_250d) = {st.pearsonr(x.loc[common], y.loc[common])[0]:+.4f} n={len(common)}")

print(f"\n[elapsed {time.time()-t0:.1f}s]")
