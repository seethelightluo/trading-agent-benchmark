"""Miner2 deep validation of the 6 new screen passers + correlation check vs library."""
import sys, time, json, os, pickle, base64, gzip, zlib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import miner2_fast as F

SYMBOLS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
           "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
cache = pickle.load(open("scripts/panel_cache.pkl", "rb"))
CP, OP = cache["close"], cache["open"]
HP, LP, V = cache["high"], cache["low"], cache["vol"]
RET, MAC = cache["ret"], cache["macro"]
idx = CP.index
fwd = {h: CP.shift(-h) / CP - 1.0 for h in (1, 2, 3, 5, 10, 20)}
N_CELLS = len(idx) * len(SYMBOLS)
LOG = np.log(CP / CP.shift(1))
MP = lambda w: max(5, w // 2)
v20 = RET.rolling(20, min_periods=MP(20)).std()

# build all 6 panels
rg = (HP - LP).replace(0, np.nan)
pos = (CP - LP) / rg
gap = OP / CP.shift(1) - 1.0
intra = CP / OP - 1.0

panels = {
    "range_pos_5d": pos.rolling(5, min_periods=3).mean(),
    "intra_1d": intra,
    "gap_5d": gap.rolling(5, min_periods=3).mean(),
    "intra_5d": intra.rolling(5, min_periods=3).mean(),
    "rev_vol_1d": -(CP / CP.shift(1) - 1.0) / (v20 + 1e-12),
    "rev_vol_5d": -(CP / CP.shift(5) - 1.0) / (v20 + 1e-12),
}
for k in panels:
    panels[k] = panels[k].reindex(idx)


def pd_rank(values):
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sv = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sv[end] == sv[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def pairwise_rho(A, B):
    n = min(A.shape[0], B.shape[0])
    A, B = A[-n:], B[-n:]
    vals = []
    for a, b in zip(A, B):
        fin = np.isfinite(a) & np.isfinite(b)
        if fin.sum() < 2:
            continue
        ra, rb = pd_rank(a[fin]), pd_rank(b[fin])
        if np.std(ra) <= 1e-12 or np.std(rb) <= 1e-12:
            continue
        vals.append(abs(float(np.corrcoef(ra, rb)[0, 1])))
    return float(np.mean(vals)) if vals else np.nan


def decode_gzipb64(payload):
    raw = base64.b64decode(payload["data"])
    try:
        raw = gzip.decompress(raw)
    except Exception:
        raw = zlib.decompress(raw)
    arr = np.frombuffer(raw, dtype="<f4").reshape(payload["n_dates"], payload["n_symbols"])
    start = pd.Timestamp(payload["date_start"]); end = pd.Timestamp(payload["date_end"])
    mask = (idx >= start) & (idx <= end)
    dates = idx[mask]
    n = min(len(dates), arr.shape[0])
    return pd.DataFrame(arr[-n:], index=dates[-n:], columns=payload.get("symbols", SYMBOLS))


def decode_b64zlibcsv(payload):
    raw = zlib.decompress(base64.b64decode(payload["data"]))
    txt = raw.decode("utf-8", errors="replace")
    rows = []
    for line in txt.splitlines():
        if not line.strip() or line.startswith("date,"):
            continue
        parts = line.split(",")
        rows.append((parts[0], [float(v) if v not in ("", "NA") else np.nan for v in parts[1:]]))
    df = pd.DataFrame([r[1] for r in rows], index=[r[0] for r in rows], columns=payload.get("columns", SYMBOLS))
    df.index = pd.to_datetime(df.index)
    return df


def load_lib(path):
    d = json.load(open(path))
    sa = d.get("signal_artifact")
    v = d.get("validation") or {}
    sa = sa or v.get("signal_artifact") or (v.get("metrics") or {}).get("signal_artifact")
    if isinstance(sa, str):
        p = os.path.join("factors", sa)
        if p.endswith(".npy") and os.path.exists(p):
            return pd.DataFrame(np.load(p), index=idx, columns=SYMBOLS)
        return None
    if isinstance(sa, dict):
        fmt = str(sa.get("format", ""))
        if "gzip" in fmt and "n_dates" in sa:
            try:
                return decode_gzipb64(sa)
            except Exception:
                pass
        if "base64:zlib:csv" in fmt or ("data" in sa and "n_dates" not in sa):
            try:
                return decode_b64zlibcsv(sa)
            except Exception:
                pass
    return None


# ---- deep validation per candidate ----
results = {}
for name, panel in panels.items():
    print(f"\n{'='*70}\n{name}")
    # by-year
    print("  [by-year]")
    for yr in range(2020, 2027):
        m = (idx.year == yr)
        if m.sum() < 50:
            continue
        r = F.fast_ic(panel[m], fwd[1].reindex(idx)[m])
        if r["n_dates"] >= 20:
            print(f"    {yr}: n={r['n_dates']:4d} IC1={r['ic']:+.4f} ICIR1={r['icir']:+.3f}")
    # groups
    print("  [groups]")
    groups = {
        "all": SYMBOLS,
        "no_crypto": [s for s in SYMBOLS if s not in ("BTC", "ETH")],
        "no_rates": [s for s in SYMBOLS if s not in ("US10Y", "CN10Y")],
        "no_china": [s for s in SYMBOLS if s not in ("000300.SH", "000688.SH")],
        "equities": ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX"],
        "crypto": ["BTC", "ETH"],
    }
    for gname, syms in groups.items():
        r = F.fast_ic(panel[syms], fwd[1].reindex(idx)[syms])
        print(f"    {gname:10s} n={r['n_dates']:4d} IC1={r['ic']:+.4f} ICIR1={r['icir']:+.3f}")
    # min names
    print("  [min_names 8/10/12/15]")
    for mn in (8, 10, 12, 15):
        r = F.fast_ic(panel, fwd[1], min_names=mn)
        print(f"    min_names={mn}: n={r['n_dates']:4d} IC1={r['ic']:+.4f} ICIR1={r['icir']:+.3f}")
    # decay
    print("  [decay]")
    decay = {}
    for h in (1, 2, 3, 5, 10, 20):
        r = F.fast_ic(panel, fwd[h])
        decay[h] = r["ic"]
        print(f"    h={h:2d}: IC={r['ic']:+.4f} ICIR={r['icir']:+.3f}")
    results[name] = {"by_year": {}, "decay": decay}

# ---- library correlation matrix ----
print(f"\n{'='*70}\n[library correlation vs 6 candidates + sign_streak]")
libs = {}
for f in sorted(os.listdir("factors")):
    if not f.endswith(".json") or f.endswith(".bak") or ".2026" in f:
        continue
    libdf = load_lib(os.path.join("factors", f))
    if libdf is None or len(libdf) < 100:
        continue
    libs[f] = libdf
print("library members with artifacts:")
for k in libs:
    print(f"  {k:44s} {libs[k].shape}")

# add sign_streak
streak_panel = None
# rebuild sign_streak
def sign_streak(ret):
    cols = {}
    for s in ret.columns:
        r = ret[s].values
        out = np.full(len(r), np.nan)
        prev = 0.0
        for i in range(len(r)):
            if not np.isfinite(r[i]):
                prev = 0.0
                continue
            sg = 1.0 if r[i] > 0 else (-1.0 if r[i] < 0 else 0.0)
            if sg == 0:
                prev = 0.0
                out[i] = 0.0
            elif np.sign(prev) == sg or prev == 0.0:
                prev = prev + sg if np.sign(prev) == sg else sg
                out[i] = prev
            else:
                prev = sg
                out[i] = sg
        cols[s] = out
    return pd.DataFrame(cols, index=ret.index)


panels["sign_streak"] = sign_streak(RET)
cand_names = list(panels.keys())
rho_mat = pd.DataFrame(np.nan, index=cand_names, columns=list(libs.keys()))
for cn in cand_names:
    for ln, ldf in libs.items():
        rho_mat.loc[cn, ln] = pairwise_rho(panels[cn].values, ldf.values)
print("\nrho matrix (candidate rows x library cols):")
print(rho_mat.round(3).to_string())
rho_mat.round(3).to_csv("scripts/miner2_lib_rho_matrix.csv")
print("\nmax abs rho per candidate:")
print(rho_mat.max(axis=1).round(3).to_string())

# candidate-candidate correlation
print("\n[candidate-candidate rho]")
cc = pd.DataFrame(np.nan, index=cand_names, columns=cand_names)
for a in cand_names:
    for b in cand_names:
        cc.loc[a, b] = pairwise_rho(panels[a].values, panels[b].values)
print(cc.round(3).to_string())

json.dump(results, open("scripts/miner2_deep_val_6.json", "w"), indent=1, default=str)
print("\ndone")
