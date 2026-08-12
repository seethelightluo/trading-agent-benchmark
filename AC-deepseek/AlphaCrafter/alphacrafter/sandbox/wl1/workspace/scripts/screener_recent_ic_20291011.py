"""Screener: compute recent rank IC (through visible date 2029-10-11) for all
active factors using stored signal panels, restricted to date <= visible_through.
No lookahead: only uses rows up to 2029-10-11; forward returns use closes <= visible.
"""
import json, glob, zlib, base64, io, csv
import numpy as np
from scipy.stats import spearmanr

VISIBLE = "2029-10-11"

ASSETS = ["000300.SH","000688.SH","BTC","CN10Y","COPPER","ETH","HSI","N225",
          "NDX","SOX","SPX","SX5E","US10Y","WTI","XAU"]
closes = {}
for a in ASSETS:
    p = f"../persistent/stock_data/{a}.csv"
    d = {}
    for r in csv.DictReader(open(p)):
        if r["date"] <= VISIBLE:
            d[r["date"]] = float(r["close"])
    closes[a] = d

dates = sorted(set.intersection(*[set(closes[a]) for a in ASSETS]))
di = {dt: i for i, dt in enumerate(dates)}
C = np.full((len(dates), len(ASSETS)), np.nan)
for i, dt in enumerate(dates):
    for j, a in enumerate(ASSETS):
        C[i, j] = closes[a][dt]

def fwd(h):
    F = np.full_like(C, np.nan)
    F[:-h] = C[h:] / C[:-h] - 1.0
    return F

F1, F5, F10 = fwd(1), fwd(5), fwd(10)

def decode_panel(fpath):
    d = json.load(open(fpath))
    sa = d["validation"]["signal_artifact"]
    raw = base64.b64decode(sa["data"])
    txt = zlib.decompress(raw).decode()
    rows = list(csv.reader(io.StringIO(txt)))
    header = rows[0]
    cols = header[1:] if header[0].lower() in ("date", "index", "") else header
    colmap = {}
    for k, c in enumerate(cols):
        c2 = c.strip()
        if c2 in ASSETS:
            colmap[c2] = k
    mat = {}
    for r in rows[1:]:
        if len(r) < 2:
            continue
        dt = r[0].strip()
        if dt > VISIBLE:
            continue
        try:
            mat[dt] = [float(r[colmap[a] + 1]) if colmap[a] + 1 < len(r) and r[colmap[a] + 1] not in ("", "nan", "NaN") else np.nan for a in ASSETS]
        except Exception:
            continue
    return mat, colmap

def rank_ic(sig_dt, F, i0, n):
    ics = []
    for t in range(i0 - n, i0):
        dt = dates[t]
        if dt not in sig_dt:
            continue
        s = np.array(sig_dt[dt], dtype=float)
        f = F[t]
        m = ~(np.isnan(s) | np.isnan(f))
        if m.sum() < 6:
            continue
        rho, _ = spearmanr(s[m], f[m])
        if not np.isnan(rho):
            ics.append(rho)
    ics = np.array(ics)
    if len(ics) == 0:
        return None, None, 0
    return ics.mean(), (ics.mean() / ics.std() if ics.std() > 0 else 0.0), len(ics)

files = [f for f in glob.glob("factors/*.json") if not f.endswith(".bak")
         and "/evicted/" not in f and "/quarantine/" not in f and "/rejected/" not in f]
i_end = len(dates)

print(f"common dates through {VISIBLE}: {len(dates)}")
print(f"{'factor':42s} {'ic1_60':>8s} {'icir1_60':>9s} {'ic1_20':>8s} {'icir1_20':>9s} {'ic5_60':>8s} {'ic10_60':>8s} {'ic10_20':>8s}")
results = {}
for f in sorted(files):
    fid = f.split("/")[-1].replace(".json", "")
    try:
        sig, colmap = decode_panel(f)
    except Exception as e:
        print(f"{fid:42s} DECODE_ERR {e}")
        continue
    ic1_60, ir1_60, n1 = rank_ic(sig, F1, i_end, 60)
    ic1_20, ir1_20, n2 = rank_ic(sig, F1, i_end, 20)
    ic5_60, _, _ = rank_ic(sig, F5, i_end, 60)
    ic10_60, _, _ = rank_ic(sig, F10, i_end, 60)
    ic10_20, _, _ = rank_ic(sig, F10, i_end, 20)
    results[fid] = dict(ic1_60=ic1_60, icir1_60=ir1_60, ic1_20=ic1_20, icir1_20=ir1_20,
                        ic5_60=ic5_60, ic10_60=ic10_60, ic10_20=ic10_20, n=n1)
    def fmt(x):
        return "    nan" if x is None else f"{x:8.4f}"
    print(f"{fid:42s} {fmt(ic1_60):>8s} {fmt(ir1_60):>9s} {fmt(ic1_20):>8s} {fmt(ir1_20):>9s} {fmt(ic5_60):>8s} {fmt(ic10_60):>8s} {fmt(ic10_20):>8s}")

json.dump(results, open("screener_recent_ic_20291011.json", "w"), indent=1)
print("saved screener_recent_ic_20291011.json")
