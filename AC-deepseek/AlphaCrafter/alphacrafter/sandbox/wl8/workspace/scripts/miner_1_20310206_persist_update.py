"""miner_1 2031-02-06: refresh flip_mom_20x10 persistence after active re-validation.
Adds signal_artifact (zlib+winsor-zscore panel through visible_through) and updates
validation status/metrics/period/last_validated + regime/family notes.
Keeps admission_block untouched so benchmark gate provenance remains intact.
Only handles factors/ files; does NOT run backtest/step or touch date.json/account.json.
"""
import json, os, glob, base64, zlib, datetime
import numpy as np, pandas as pd
os.chdir(os.path.dirname(os.path.abspath(__file__)) if False else ".")
sys.path_local = "scripts"
import sys; sys.path.insert(0, sys.path_local)

from miner_3_20261203_common import WATCH, load_prices, load_macro, load_visible_through, \
    cross_sectional_ic, ic_stats, regime_split, spearman_panel_rho, zscore_series

ASOF = load_visible_through()
H = 10
px = load_prices(ASOF)
fwd = px.shift(-H) / px - 1.0

def retk(s, k):
    v = s.replace([np.inf, -np.inf], np.nan)
    return (v / v.shift(k) - 1.0)

flip = pd.DataFrame({
    s: (retk(px[s], 20) * np.sign(retk(px[s], 10)))
    for s in WATCH
}).sort_index().replace([np.inf, -np.inf], np.nan)

icd = cross_sectional_ic(flip, fwd)
st = ic_stats(icd)
ic252 = ic_stats(icd[icd.index >= icd.index[-1] - pd.Timedelta(days=365)])
ic180 = ic_stats(icd[icd.index >= icd.index[-1] - pd.Timedelta(days=180)])
ic60 = ic_stats(icd.tail(60))
regs = regime_split(icd)
cov = (flip.notna() & fwd.notna()).mean().mean()
# asset-day coverage
ad = float(flip.notna().mean().mean())
to = float(flip.rank(axis=1).diff().abs().mean(axis=1).mean())

# library correlation via persisted signal artifacts
sigs = {}
for fn in glob.glob("factors/*.json"):
    if fn.endswith(".bak") or os.path.basename(fn) in ("factor_ensemble.json",):
        continue
    try:
        d = json.load(open(fn))
    except Exception:
        continue
    art = (d.get("validation") or {}).get("signal_artifact") or {}
    raw = art.get("data")
    if not raw:
        continue
    try:
        csv = zlib.decompress(base64.b64decode(raw)).decode("utf-8")
        sigs[d["factor_id"]] = pd.read_csv(pd.io.common.StringIO(csv), index_col=0)
    except Exception:
        continue

lib_corr_detail = {}
for fid, sig in sigs.items():
    lib_corr_detail[fid] = spearman_panel_rho(flip, sig)
maxabs = max((abs(v) for v in lib_corr_detail.values() if pd.notna(v)), default=float("nan"))

# decay by horizon
decay = {}
for h in (1, 2, 3, 5, 10, 20):
    fh = px.shift(-h) / px - 1.0
    ics_h = cross_sectional_ic(flip, fh)
    decay[str(h)] = round(float(ics_h["ic"].mean()), 4) if len(ics_h) else float("nan")

# leave-one-asset-out min IC/ICIR
lo_min = [np.nan, np.nan]
lo_res = []
for s in WATCH:
    sub = flip.drop(columns=[s])
    sub_fwd = fwd.drop(columns=[s])
    try:
        ics_lo = cross_sectional_ic(sub, sub_fwd)
        st_lo = ic_stats(ics_lo)
        lo_res.append((s, st_lo['ic'], st_lo['icir'], int(st_lo['n_dates'])))
    except Exception:
        pass
if lo_res:
    lo_min = [min(x[1] for x in lo_res), min(x[2] for x in lo_res)]

def b64(df):
    csv = df.to_csv()
    return base64.b64encode(zlib.compress(csv.encode("utf-8"))).decode("ascii")

ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
MAX_IC = 0.0070; MAX_ICIR = 0.0840
effective = abs(st['ic']) >= MAX_IC and abs(st['icir']) >= MAX_ICIR

print(f"ASOF={ASOF} n_dates_full={st['n_dates']} ic={st['ic']:.4f} icir={st['icir']:.4f} hit={st['hit']:.3f}")
print(f"365d ic={ic252['ic']:.4f} icir={ic252['icir']:.4f} n={ic252['n_dates']}")
print(f"180d ic={ic180['ic']:.4f} icir={ic180['icir']:.4f} n={ic180['n_dates']}")
print(f"60d  ic={ic60['ic']:.4f} icir={ic60['icir']:.4f} n={ic60['n_dates']}")
print(f"GATE: |IC|={abs(st['ic']):.4f} (>={MAX_IC}) |ICIR|={abs(st['icir']):.4f} (>={MAX_ICIR}) -> effective={effective}")
print(f"coverage_asset_days={ad:.3f} coverage_ic_dates={cov:.3f} turnover={to:.3f}")
print(f"regimes={regs}")
print(f"decay={decay}")
print(f"leave_one_out_lo_min={lo_min} detail_sample={lo_res[:5]}")
print(f"max_abs_library_correlation={maxabs:.4f}")
print(f"lib_corr_detail={ {k: round(v,4) for k,v in lib_corr_detail.items() if pd.notna(v)} }")