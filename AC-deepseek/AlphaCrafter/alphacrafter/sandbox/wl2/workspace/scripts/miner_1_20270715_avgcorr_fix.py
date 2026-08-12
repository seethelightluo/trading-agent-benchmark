"""Clean patch: replace buggy avg_corr_60 with vectorized version, rerun screen."""
import sys, re
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd

src = open("scripts/miner_1_20270701_screen_candidates.py").read()

new_fn = '''def avg_corr_60(df, w=60):
    """Mean pairwise corr of this asset's returns vs all other assets (60d).
    Vectorized on the union calendar; windows aligned to df.index."""
    cols = {}
    for s, d in series.items():
        cols[s] = d["ret"]
    m = pd.concat(cols, axis=1)
    m.columns = list(cols.keys())
    self_name = None
    for s, d in series.items():
        if d is df:
            self_name = s
            break
    if self_name is None:
        raise ValueError("df not found in series")
    others = [c for c in m.columns if c != self_name]
    self_ret = m[self_name].reindex(df.index)
    acc = pd.DataFrame(index=df.index)
    for o in others:
        pair = pd.concat([self_ret, m[o].reindex(df.index)], axis=1)
        pair.columns = ["a", "b"]
        acc[o] = pair["a"].rolling(w, min_periods=30).corr(pair["b"])
    out = acc.mean(axis=1, skipna=True)
    out[acc.notna().sum(axis=1) < 5] = np.nan
    return out


'''

new_src = re.sub(r"def avg_corr_60\(df, w=60\):.*?\n\n\n", new_fn, src, count=1, flags=re.S)
assert new_src != src, "replacement failed"
# bump output filename so results are distinguishable
new_src = new_src.replace("miner_1_20270701_screen_results.json",
                          "miner_1_20270715_screen_results.json")
open("scripts/miner_1_20270715_screen_run.py", "w").write(new_src)
print("patched -> scripts/miner_1_20270715_screen_run.py")
