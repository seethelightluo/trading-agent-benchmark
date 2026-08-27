"""miner_1 revalidation of all effective factors as of visible end 2035-10-10.
Recomputes IC/ICIR of each stored factor signal vs 10d forward cross-sectional returns."""
import json, os, base64, zlib, io, csv
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from importlib import reload

VIS_END = pd.Timestamp("2035-10-10")
P = Path("../persistent")
SD = P / "stock_data"
ID = P / "index_data"
ASSETS = ["000300.SH","000688.SH","SPX","HSI","N225","SX5E","SOX","NDX",
          "XAU","COPPER","WTI","BTC","ETH","US10Y","CN10Y"]

def load(a):
    df = pd.read_csv(SD/(a+".csv"), parse_dates=["date"]).sort_values("date")
    df.columns = [str(c).lower() for c in df.columns]
    df = df[df["date"] <= VIS_END].set_index("date")
    return df["close"].astype(float)

closes = {a: load(a) for a in ASSETS}
al = set()
for c in closes.values(): al.update(c.index)
al = pd.DatetimeIndex(sorted(al))
cp = pd.DataFrame({a: closes[a].reindex(al) for a in ASSETS})
fwd10 = cp.shift(-10)/cp - 1.0
print(f"Panel: {cp.shape[0]} dates x {cp.shape[1]} assets, {cp.index[0].date()}..{cp.index[-1].date()}")

def sig(fname):
    d = json.load(open("factors/"+fname))
    sa = d.get("validation", {}).get("signal_artifact", {})
    if not sa: return None
    rows = list(csv.reader(io.StringIO(zlib.decompress(base64.b64decode(sa["data"])).decode())))
    dt = [r[0] for r in rows[1:]]
    M = np.array([[float(x) if x!="" else np.nan for x in r[1:]] for r in rows[1:]])
    return pd.DataFrame(M, index=pd.DatetimeIndex(dt), columns=rows[0][1:])

def ic_table(sig_panel, fwd, start=None, mv=8, min_dates=20):
    idx = sig_panel.index if start is None else sig_panel.index[sig_panel.index >= pd.Timestamp(start)]
    ics = []
    for t in idx:
        if t not in fwd.index: continue
        f = np.asarray(sig_panel.loc[t]); r = np.asarray(fwd.loc[t])
        v = ~(np.isnan(f)|np.isnan(r))
        if v.sum() >= mv:
            if np.std(f[v])>0 and np.std(r[v])>0:
                rho,_ = spearmanr(f[v], r[v])
                if not np.isnan(rho): ics.append(rho)
    ia = np.array(ics)
    if len(ia) < min_dates:
        return dict(ic=0.0, icir=0.0, n=len(ia), hit=0.0)
    ic = float(ia.mean()); s = float(ia.std(ddof=1))
    return dict(ic=ic, icir=float(ic/s if s>1e-10 else 0.0), n=len(ia), hit=float((ia>0).mean()))

# full sample and recency samples
print("\n=== REVALIDATION of stored factor signals (10d fwd) ===")
factor_files = sorted(os.listdir("factors"))
factor_files = [f for f in factor_files if f.endswith(".json") and f != "factor_ensemble.json"]
for f in factor_files:
    sp = sig(f)
    if sp is None:
        print(f"{f:32s} no signal artifact"); continue
    # restrict to visible end
    sp = sp[sp.index <= VIS_END]
    full = ic_table(sp, fwd10)
    rec = ic_table(sp, fwd10, start="2032-01-01")
    rec2 = ic_table(sp, fwd10, start="2034-01-01")
    print(f"{f:32s} FULL ic={full['ic']:+.4f} icir={full['icir']:+.4f} n={full['n']:5d} | "
          f"R2032 ic={rec['ic']:+.4f} icir={rec['icir']:+.4f} n={rec['n']:4d} | "
          f"R2034 ic={rec2['ic']:+.4f} icir={rec2['icir']:+.4f} n={rec2['n']:4d}")