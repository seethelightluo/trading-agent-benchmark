"""miner_3 pairwise correlation check among the 3 passing candidates (2026-07-30)."""
import sys, json, base64, zlib, io
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import load_closes, load_index, factor_panel

close, vol, open_, high, low = load_closes()
macro = {
    "DXY": load_index("DXY"), "USDCNY": load_index("USDCNY"),
    "USDJPY": load_index("USDJPY"), "EURUSD": load_index("EURUSD"),
    "VIX": load_index("VIX"),
}


def _align(series, c):
    return series.reindex(c.index).ffill()


def f_fx_beta(c, v, o, h, l, m, name, win=60):
    fx = _align(m[name], c)
    r = c.pct_change()
    fxr = fx.pct_change()
    return (r.rolling(win).cov(fxr) / fxr.rolling(win).var()).replace([np.inf, -np.inf], np.nan)


def f_risk_adj_mom(c, v, o, h, l, m, short=20, skip=5, volwin=60):
    mom = c.shift(skip) / c.shift(skip + short) - 1.0
    rv = c.pct_change().rolling(volwin).std()
    return (mom / rv).replace([np.inf, -np.inf], np.nan)


def f_vix_cond_mom(c, v, o, h, l, m, short=20, skip=5, vixwin=60):
    vix = _align(m["VIX"], c)
    mom = c.shift(skip) / c.shift(skip + short) - 1.0
    z = (vix - vix.rolling(vixwin).mean()) / vix.rolling(vixwin).std()
    return (mom * np.where(z > 1.0, -1.0, 1.0)).replace([np.inf, -np.inf], np.nan)


FACTORS = {
    "fx_beta_usdcny_60": {"fn": f_fx_beta, "params": {"name": "USDCNY", "win": 60}},
    "risk_adj_mom_20x60": {"fn": f_risk_adj_mom, "params": {"short": 20, "skip": 5, "volwin": 60}},
    "vix_cond_mom_20x60": {"fn": f_vix_cond_mom, "params": {"short": 20, "skip": 5, "vixwin": 60}},
}

panels = {fid: factor_panel(spec["fn"], close, vol, open_, high, low, macro, **spec["params"])
          for fid, spec in FACTORS.items()}


def spearman_pooled(a_panel, b_panel):
    common = a_panel.index.intersection(b_panel.index)
    cols = [c for c in a_panel.columns if c in b_panel.columns]
    a = a_panel.loc[common, cols].values.ravel()
    b = b_panel.loc[common, cols].values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 200:
        return np.nan, int(m.sum())
    return float(pd.Series(a[m]).rank().corr(pd.Series(b[m]).rank())), int(m.sum())


names = list(FACTORS.keys())
print("pairwise among new candidates (pooled spearman):")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        r, n = spearman_pooled(panels[names[i]], panels[names[j]])
        print(f"  {names[i]:22s} vs {names[j]:22s}: rho={r:.4f} (n={n})")

# also pearson pooled for reference (deterministic gate may use pearson on raw artifacts)
def pearson_pooled(a_panel, b_panel):
    common = a_panel.index.intersection(b_panel.index)
    cols = [c for c in a_panel.columns if c in b_panel.columns]
    a = a_panel.loc[common, cols].values.ravel()
    b = b_panel.loc[common, cols].values.ravel()
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 200:
        return np.nan, int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())

print("pairwise among new candidates (pooled pearson):")
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        r, n = pearson_pooled(panels[names[i]], panels[names[j]])
        print(f"  {names[i]:22s} vs {names[j]:22s}: rho={r:.4f} (n={n})")

# vs library
def load_lib_panels():
    lib = {}
    for fid in ["mom_10d_skip5", "vix_beta_cond_60x20", "yield_beta_cond_60x20"]:
        d = json.load(open(f"factors/{fid}.json"))
        raw = base64.b64decode(d["validation"]["signal_artifact"]["data"])
        panel = pd.read_csv(io.StringIO(zlib.decompress(raw).decode()), index_col=0, parse_dates=True)
        panel.index = pd.DatetimeIndex(panel.index)
        lib[fid] = panel
    return lib

lib = load_lib_panels()
print("new candidates vs library (pooled spearman / pooled pearson):")
for fid in names:
    for lfid, lp in lib.items():
        rs, _ = spearman_pooled(panels[fid], lp)
        rp, _ = pearson_pooled(panels[fid], lp)
        print(f"  {fid:22s} vs {lfid:22s}: spearman={rs:.4f} pearson={rp:.4f}")
