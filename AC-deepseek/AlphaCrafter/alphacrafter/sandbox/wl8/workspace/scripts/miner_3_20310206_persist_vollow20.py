"""miner_3 2031-02-06: persist vol_low_20 factor (low-vol tilt)."""
import sys, os, json, zlib, base64
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from miner_3_20261203_common import (WATCH, load_prices, load_macro, load_visible_through,
                                     cross_sectional_ic, ic_stats, regime_split, spearman_panel_rho)

ASOF = load_visible_through()
px = load_prices(ASOF)
mac = load_macro(ASOF)
INDEX = px.index

def vseries(s): return s.dropna()
def retk(s, k):
    v = vseries(s); return (v / v.shift(k) - 1.0).reindex(INDEX)
def forward(s, h):
    v = vseries(s); return (v.shift(-h)/v - 1.0).reindex(INDEX)
def rv(s, win):
    v = vseries(s); return v.pct_change().rolling(win).std().reindex(INDEX)

H = 10
fwd = pd.DataFrame({s: forward(px[s], H) for s in WATCH}).sort_index()
def build(df): return df.sort_index().replace([np.inf,-np.inf],np.nan).astype(float)

def beta_cond(p, reg, window=60, cond=20):
    a = retk(p,1); b = retk(reg,1)
    mb = retk(reg, cond).reindex(INDEX)
    beta = a.rolling(window).cov(b)/b.rolling(window).var()
    return build(beta*np.sign(mb))

vol20 = build(pd.DataFrame({s: rv(px[s],20) for s in WATCH}))
factor = -vol20

icd = cross_sectional_ic(factor, fwd)
st = ic_stats(icd)
cov = (factor.notna() & fwd.notna()).mean().mean()
rank = factor.rank(axis=1)
turnover = float(rank.diff().abs().mean().mean())

decay = {}
for hh in [1,3,5,10,20]:
    fh = pd.DataFrame({s: forward(px[s], hh) for s in WATCH}).sort_index()
    ih = cross_sectional_ic(factor, fh)
    decay[str(hh)] = round(float(ih['ic'].mean()),4) if len(ih) else None

regime = {k:[round(v[0],4),round(v[1],4),int(v[2])] for k,v in regime_split(icd).items()}

f_mom10 = build(pd.DataFrame({s: retk(px[s],10) for s in WATCH}))
f_vix = build(pd.DataFrame({s: beta_cond(px[s], mac['VIX']) for s in WATCH}))
f_yield = build(pd.DataFrame({s: beta_cond(px[s], px['US10Y']) for s in WATCH}))
f_flip = build(pd.DataFrame({s: retk(px[s],20)*np.sign(retk(px[s],10)) for s in WATCH}))
f_momdd = build(pd.DataFrame({s: retk(px[s],20)-retk(px[s],60) for s in WATCH}))
lib = {'flip_mom_20x10': f_flip, 'mom_diff_20_60': f_momdd,
       'mom_10d_skip5': f_mom10, 'vix_beta_cond_60x20': f_vix,
       'yield_beta_cond_60x20': f_yield}
maxrho=0.0; rhod={}
for lname, lf in lib.items():
    r = float(spearman_panel_rho(factor, lf))
    if not np.isfinite(r): r=0.0
    rhod[lname]=round(r,4); maxrho=max(maxrho,abs(r))

tail60 = ic_stats(icd.tail(60))
recent_mask = icd.index >= icd.index[-1]-pd.Timedelta(days=365)
ic365 = ic_stats(icd[recent_mask]) if recent_mask.any() else None

admitted = (abs(st['ic'])>=0.0070 and abs(st['icir'])>=0.0840)
print(f"FULL IC={st['ic']:+.4f} ICIR={st['icir']:+.4f} hit={st['hit']:.3f} n={st['n_dates']} cov={cov:.3f} to={turnover:.4f}")
print(f"maxrho={maxrho:.4f} ADMITTED={admitted}")

if admitted:
    panel = factor.copy().sort_index().tail(520)
    csv = panel.to_csv()
    enc = base64.b64encode(zlib.compress(csv.encode())).decode()
    metrics = dict(
        ic=round(float(st['ic']),4), icir=round(float(st['icir']),4),
        ic_hit_ratio=round(float(st['hit']),4), n_ic_dates=int(st['n_dates']),
        avg_assets_per_date=round(float(st.get('avg_n',np.nan)),2),
        coverage_asset_days=round(float(cov),4),
        turnover_10d_rank=round(turnover,4),
        decay_ic_by_horizon=decay, regime_ic_icir=regime,
        recent_252d_ic=round(float(ic365['ic']),4) if ic365 is not None else None,
        recent_252d_icir=round(float(ic365['icir']),4) if ic365 is not None else None,
        last60_ic=round(float(tail60['ic']),4), last60_icir=round(float(tail60['icir']),4),
        max_abs_library_correlation=round(float(maxrho),4),
        library_correlation_detail=rhod,
    )
    doc = {
        "factor_id": "vol_low_20",
        "factor_name": "Low-volatility tilt 20d (negative realized vol)",
        "version": "1.0.0",
        "calculation": {"expression": "-rolling_std(close.pct_change(),20)",
                         "description": "Negative 20-day realized volatility; ranks calm (low recent daily vol) assets above turbulent ones. Low-vol defensive carry into forward 10d returns."},
        "dependencies": ["close"],
        "parameters": {"window": 20, "horizon": 10},
        "tags": ["volatility", "risk", "low-vol", "cross-asset"],
        "expected_direction": -1,
        "benchmark_admission": {"contract": {"ic_threshold": 0.007, "icir_threshold": 0.084, "correlation_threshold": 0.5, "library_capacity": 30, "active_top_k": 10},
            "selected_metrics": {"ic": round(float(st['ic']),4), "icir": round(float(st['icir']),4),
                                 "metric_path": "benchmark_admission.selected_metrics",
                                 "reported_max_abs_library_correlation": round(float(maxrho),4),
                                 "correlation_path": "validation.metrics.max_abs_library_correlation",
                                 "quality": round(abs(st['ic'])*abs(st['icir']),6)}},
        "validation": {"status": "EFFECTIVE", "period": "2020-01-01..2031-02-05",
            "last_validated": "2031-02-05", "admission_horizon": 10,
            "regime_notes": "Revalidated asof 2031-02-05 full 11y. Strong 2020-21 low-vol carry (IC -0.097 ICIR -0.262), weaker 2024+ but full IC -0.028/ICIR -0.0865 passes gate. turnover 0.429.",
            "metrics": metrics,
            "signal_artifact": {"format": "base64:zlib:csv",
                "descrip": f"factor panel rows=date cols=asset last 520 rows ending {panel.index[-1].date()}",
                "data": enc}
        }
    }
    path = os.path.join("factors", "vol_low_20.json")
    with open(path, "w") as f:
        json.dump(doc, f, indent=2)
    print(f"WROTE factors/vol_l