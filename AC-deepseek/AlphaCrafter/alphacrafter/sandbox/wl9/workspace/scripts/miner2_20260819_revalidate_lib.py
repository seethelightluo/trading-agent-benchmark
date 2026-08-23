"""Re-validate all currently EFFECTIVE factors and report drift (2026-08-19)."""
from __future__ import annotations

import base64
import io
import json
import sys
import zlib
from pathlib import Path

import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import (  # noqa: E402
    FACTOR_DIR,
    decay_profile,
    load_closes,
    rank_ic,
    to_frame,
    turnover_rank10,
)
import numpy as np

LAST_VALIDATED = "2026-08-19"


def main():
    closes = load_closes()
    print("assets:", len(closes))
    print("\n=== ROUTINE RE-VALIDATION OF EFFECTIVE FACTORS ===")
    for f in sorted(FACTOR_DIR.glob("*.json")):
        if f.name == "factor_ensemble.json":
            continue
        d = json.load(open(f))
        if d.get("validation", {}).get("status") != "EFFECTIVE":
            print(f"  SKIP {f.name}: not EFFECTIVE")
            continue
        fid = d["factor_id"]
        art = d["validation"].get("signal_artifact", {})
        if not art or "data" not in art:
            print(f"  SKIP {fid}: no artifact")
            continue
        try:
            csv = zlib.decompress(base64.b64decode(art["data"])).decode()
            frame = pd.read_csv(io.StringIO(csv), index_col=0, parse_dates=True)
        except Exception as e:
            print(f"  SKIP {fid}: artifact decode fail {e}")
            continue
        rets = {a: closes[a].reindex(frame.index) for a in frame.columns}
        # forward returns must use closes, rebuild
        frame2 = to_frame(closes, {a: frame[a] for a in frame.columns})
        ret_frame = pd.DataFrame({a: rets[a].reindex(frame2.index) for a in frame2.columns})
        horizon = int(d["validation"].get("admission_horizon", 10))
        # recompute via ranking using closes-based forward returns
        ic = rank_ic(frame2, ret_frame)
        ic_mean = float(ic.mean()) if len(ic) else float("nan")
        ic_std = float(ic.std(ddof=1)) if len(ic) > 2 else float("nan")
        icir = ic_mean / ic_std if ic_std and np.isfinite(ic_std) else float("nan")
        to = turnover_rank10(frame2)
        print(f"  {fid}: IC={ic_mean:.4f} ICIR={icir:.4f} n_ic={len(ic)} "
              f"to10={to:.3f} stored_IC={d['validation']['metrics']['ic']} "
              f"stored_ICIR={d['validation']['metrics']['icir']}")


if __name__ == "__main__":
    main()