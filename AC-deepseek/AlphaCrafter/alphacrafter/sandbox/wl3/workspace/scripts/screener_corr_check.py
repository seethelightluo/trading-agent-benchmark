import json, numpy as np, pandas as pd

selected = ["down_beta_60","cn10y_beta_60","spx_beta_60","vol_adj_mom_20_60",
            "dxy_beta_cond_60x20","hs300_beta_60","intraday_ret_skew_20",
            "vol_of_vol20x60","dd_duration_120_resid","vix_beta_cond_60x20"]
sig = {}
for fid in selected:
    d = json.load(open(f'factors/{fid}.json'))
    arr = np.load(f'factors/{fid}_signal.npy')
    sig[fid] = arr
    grid = d.get('signal_artifact_grid', {})
    print(fid, 'shape', arr.shape, 'grid', grid.get('start','?'), grid.get('end','?'))

# use last 300 rows of each signal (all end at or near 2026-07-15)
tails = {fid: sig[fid][-300:, :] for fid in selected}

def cs_rank(a):
    return pd.DataFrame(a).rank(axis=1).values

df = pd.DataFrame({fid: cs_rank(tails[fid]).mean(axis=1) for fid in selected})
corr = df.corr().round(3)
print("\nPearson corr of mean cross-sectional rank (last 300 rows):")
print(corr.to_string())
mx = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
if len(mx):
    print("\nMax pairwise corr:", mx.idxmax(), round(mx.max(), 3))
