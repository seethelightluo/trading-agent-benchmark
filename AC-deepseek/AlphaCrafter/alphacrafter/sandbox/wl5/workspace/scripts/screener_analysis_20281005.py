"""Screener cycle 2028-10-05: factor quality-IC-tilt + redundancy check."""
import json, os, zlib, base64, io
import pandas as pd
import numpy as np

FACTOR_DIR = "factors"
FACTOR_IDS = [
    "trend_r2_30_signed", "semi_down_ratio_20", "mom_120d_skip5", "dxy_beta_60",
    "vol_of_vol20x60", "mom_10d_skip5", "time_under_water_120", "vix_beta_cond_60x20",
    "tail_ratio_20", "kurt_20", "WTI_BETA_60",
]

def load_factor(fid):
    with open(os.path.join(FACTOR_DIR, fid + ".json")) as f:
        return json.load(f)

rows = []
for fid in FACTOR_IDS:
    d = load_factor(fid)
    v = d["validation"]
    m = v.get("metrics", {})
    ic = m.get("ic")
    icir = m.get("icir")
    if ic is None or icir is None:
        continue
    q = abs(ic) * abs(icir)
    rows.append({
        "factor_id": fid,
        "name": d.get("factor_name"),
        "expected_dir": d.get("expected_direction"),
        "ic": ic, "icir": icir, "q": q,
        "hit": m.get("ic_hit_ratio"),
        "n_ic": m.get("n_ic_dates"),
        "turnover_10d": m.get("turnover_10d_rank"),
        "max_abs_lib_corr": m.get("max_abs_library_correlation"),
        "regime_notes": (v.get("regime_notes") or "")[:160],
    })

df = pd.DataFrame(rows).sort_values("q", ascending=False)
print("=== Quality tilt (q = |IC|*|ICIR|) ===")
print(df[["factor_id", "expected_dir", "ic", "icir", "q", "hit", "turnover_10d", "max_abs_lib_corr"]].to_string(index=False))

# ---- decode signal panels and compute pairwise spearman (last 400 dates) ----
def decode_panel(fid):
    d = load_factor(fid)
    art = d["validation"]["signal_artifact"]
    raw = art["signal"]
    if isinstance(raw, list):
        raw = raw[0] if raw else None
    data = zlib.decompress(base64.b64decode(raw)).decode()
    return pd.read_csv(io.StringIO(data), index_col=0)

panels = {}
for fid in FACTOR_IDS:
    try:
        p = decode_panel(fid)
        panels[fid] = p
        print(f"decoded {fid}: shape={p.shape}")
    except Exception as e:
        print(f"decode failed {fid}: {e}")

# align on common index tail
common_idx = None
for fid, p in panels.items():
    idx = p.index
    if common_idx is None:
        common_idx = idx
    else:
        common_idx = common_idx.intersection(idx)
common_idx = common_idx[-400:]
print("\ncommon dates (tail 400):", len(common_idx))

corr = pd.DataFrame(index=FACTOR_IDS, columns=FACTOR_IDS, dtype=float)
for a in FACTOR_IDS:
    if a not in panels: continue
    for b in FACTOR_IDS:
        if b not in panels: continue
        sa = panels[a].loc[common_idx].stack()
        sb = panels[b].loc[common_idx].stack()
        mask = sa.notna() & sb.notna()
        if mask.sum() > 30:
            corr.loc[a, b] = sa[mask].rank().corr(sb[mask].rank())
print("\n=== Pairwise Spearman (tail 400d) ===")
print(corr.round(2).to_string())

# cluster high-corr pairs (>0.7)
print("\n=== Pairs with |rho| > 0.7 ===")
for a in FACTOR_IDS:
    for b in FACTOR_IDS:
        if a < b and pd.notna(corr.loc[a, b]) and abs(corr.loc[a, b]) > 0.7:
            print(f"{a} <-> {b}: {corr.loc[a,b]:.2f}")
