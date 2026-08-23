"""Validate pass-candidates at lower lib-correlation by tighter orthogonality.
L2_sharpe_20 / range120 pass gate but carry high lib correlation to mom/days_since_high.
Strip the raw-momentum / rank-compress components -> test residual signal.
"""
import sys, json, base64, io, zlib
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from pathlib import Path
from miner3_20260730_harness import (load_closes, to_frame, forward_returns, rank_ic,
                                     library_correlation, VISIBLE_END, VALID_START)

FACTOR_DIR = Path("factors")

def lib_frame(fname):
    d = json.load(open(FACTOR_DIR / fname))
    art = d["validation"]["signal_artifact"]["data"]
    csv = zlib.decompress(base64.b64decode(art)).decode()
    return pd.read_csv(io.StringIO(csv), index_col=0, parse_dates=True)

if __name__ == "__main__":
    closes = load_closes()
    cands = {}
    for a, s in closes.items():
        c = s[(s.index >= VALID_START) & (s.index <= VISIBLE_END)]
        r = c.pct_change()
        cands.setdefault("sharpe20", {})[a] = (c.shift(5)/c.shift(25)-1.0)/(r.rolling(20).std()*(252**0.5)+1e-6)
        lo = c.rolling(120).min(); hi = c.rolling(120).max()
        cands.setdefault("range120", {})[a] = (c-lo)/(hi-lo+1e-9)

    mom = lib_frame("mom_10d_skip5.json")
    print("candidate orthogonality vs mom_10d_skip5 (and vs days_since_high_60):")
    base_frames = {"sharpe20": mom}
    for label, vals in cands.items():
        frame = to_frame(closes, vals)
        common = frame.index.intersection(mom.index)
        a = frame.loc[common].rank(axis=1, pct=True)
        b = mom.loc[common, frame.columns].rank(axis=1, pct=True)
        resid = a - b
        rets = forward_returns(closes, 10)
        ret_frame = pd.DataFrame({s2: rets[s2].reindex(common) for s2 in frame.columns})
        ic = rank_ic(resid, ret_frame)
        icm = float(ic.mean()); icir = icm/float(ic.std(ddof=1)) if len(ic)>2 else float('nan')
        max_r,_ = library_correlation(resid)
        print(f"{label:12s} RAW-panel residualized IC={icm:+.4f} ICIR={icir:+.4f} "
              f"max_lib={max_r:.3f} PASS={'PASS' if abs(icm)>=0.007 and abs(icir)>=0.084 else 'FAIL'}")