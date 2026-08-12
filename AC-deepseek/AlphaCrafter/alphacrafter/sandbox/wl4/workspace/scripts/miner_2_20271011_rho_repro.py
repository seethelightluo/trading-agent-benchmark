"""miner_2 2027-10-11: reproduce the post-Miner gate's pairwise rho methodology (optimized).

Gate evicted vol_ratio_20_60 (rho=0.7098) and volume_z_20 (rho=0.7328) for
correlation conflict vs vol_price_corr_20. My pooled-Pearson audit reported
0.119/0.025. Test several rho definitions on the SAME decoded artifacts:
  A) pooled Spearman / Pearson over all (date, asset) cells
  B) mean |cross-sectional Spearman| per date
  C) pooled Spearman / Pearson on per-date cross-sectional rank
"""
import json, base64, zlib
import numpy as np
from scipy.stats import spearmanr, rankdata


def decode_artifact(sa):
    if sa.get("format") == "base64:zlib:csv":
        comp = base64.b64decode(sa["data"])
        csvb = zlib.decompress(comp).decode()
        lines = csvb.split("\n")
        cols = lines[0].split(",")
        panel = {}
        for ln in lines[1:]:
            if not ln.strip():
                continue
            parts = ln.split(",")
            d = parts[0]
            for c, v in zip(cols[1:], parts[1:]):
                if v != "":
                    panel[(d, c)] = float(v)
        return panel
    if sa.get("format") == "panel_json_v1":
        panel = {}
        dates, assets = sa["dates"], sa["assets"]
        for a, arr in sa["values"].items():
            for d, v in zip(dates, arr):
                if v is not None:
                    panel[(d, a)] = float(v)
        return panel
    raise ValueError(sa.get("format"))


def load(name, folder="factors"):
    d = json.load(open(f"{folder}/{name}.json"))
    return decode_artifact(d["validation"]["signal_artifact"])


def to_by_date(panel):
    out = {}
    for (d, a), v in panel.items():
        out.setdefault(d, {})[a] = v
    return out


def rank_per_date(panel):
    by_date = to_by_date(panel)
    out = {}
    for d, m in by_date.items():
        if len(m) < 5:
            continue
        assets = sorted(m)
        vals = [m[a] for a in assets]
        r = rankdata(vals) / len(vals)
        for a, rr in zip(assets, r):
            out[(d, a)] = rr
    return out


cand = {n: load(n, "factors/evicted") for n in ["vol_ratio_20_60", "volume_z_20"]}
lib = {n: load(n) for n in ["vol_price_corr_20", "dn_mkt_beta_60d", "rate_beta_cn10y_60d"]}

cand_by_date = {n: to_by_date(p) for n, p in cand.items()}
cand_rank = {n: rank_per_date(p) for n, p in cand.items()}
lib_by_date = {n: to_by_date(p) for n, p in lib.items()}
lib_rank = {n: rank_per_date(p) for n, p in lib.items()}

for cname in cand:
    cp = cand[cname]
    cr = cand_rank[cname]
    for lname in lib:
        lp = lib[lname]
        keys = sorted(set(cp) & set(lp))
        va = np.array([cp[k] for k in keys])
        vb = np.array([lp[k] for k in keys])
        rho_pooled_s, _ = spearmanr(va, vb)
        rho_pooled_p = np.corrcoef(va, vb)[0, 1]
        # B: per-date cross-sectional spearman
        rhos = []
        for d in sorted(set(cand_by_date[cname]) & set(lib_by_date[lname])):
            cm = cand_by_date[cname][d]
            lm = lib_by_date[lname][d]
            common = sorted(set(cm) & set(lm))
            if len(common) < 5:
                continue
            r = spearmanr([cm[a] for a in common], [lm[a] for a in common]).correlation
            if np.isfinite(r):
                rhos.append(r)
        mean_abs_xs = float(np.mean(np.abs(rhos))) if rhos else np.nan
        # C: rank per date then pooled
        lr = lib_rank[lname]
        kr = sorted(set(cr) & set(lr))
        vra = np.array([cr[k] for k in kr])
        vrb = np.array([lr[k] for k in kr])
        rho_rank_s, _ = spearmanr(vra, vrb)
        rho_rank_p = np.corrcoef(vra, vrb)[0, 1]
        print(
            f"{cname:18s} vs {lname:22s} "
            f"A_spear={abs(rho_pooled_s):.4f} A_pear={abs(rho_pooled_p):.4f} "
            f"B_meanAbsXsec={mean_abs_xs:.4f} "
            f"C_rankSpear={abs(rho_rank_s):.4f} C_rankPear={abs(rho_rank_p):.4f} "
            f"overlap={len(keys)}"
        )
print()
print("Gate claims: vol_ratio_20_60 vs vol_price_corr_20 rho=0.7098; volume_z_20 rho=0.7328")
