import json, sys

d = json.load(open('scripts/miner2_reval_20310509.json'))
order = ['rev_1d','rev_2d','rev_3d','rev_5d','nclv_1d','nclv_2d','nclv_3d','nclv_5d','id_rev_1d','nbody_1d','rev_1d_vs','mom_120d_skip5','vol_of_vol20x60','vix_beta_cond_60x20']
hdr = f"{'factor':<16}{'full1':>10}{'rec1':>10}{'r2_1':>10}{'r2_5':>10}{'r2_10':>10}{'r2_ic10':>10}{'q10':>10}"
print(hdr)
print('-'*len(hdr))
for k in order:
    v = d.get(k, {})
    def g(sec, h, m):
        try: return v[sec][str(h)][m]
        except Exception: return float('nan')
    full1_ic, full1_icir = g('full',1,'ic'), g('full',1,'icir')
    rec1_ic, rec1_icir = g('recent',1,'ic'), g('recent',1,'icir')
    r2_1_ic, r2_1_icir = g('recent2',1,'ic'), g('recent2',1,'icir')
    r2_5_ic, r2_5_icir = g('recent2',5,'ic'), g('recent2',5,'icir')
    r2_10_ic, r2_10_icir = g('recent2',10,'ic'), g('recent2',10,'icir')
    q10 = abs(r2_10_ic)*abs(r2_10_icir) if r2_10_ic==r2_10_ic and r2_10_icir==r2_10_icir else float('nan')
    print(f"{k:<16}{full1_ic:>10.4f}{rec1_ic:>10.4f}{r2_1_ic:>10.4f}{r2_5_ic:>10.4f}{r2_10_ic:>10.4f}{abs(r2_10_icir):>10.4f}{q10:>10.5f}")
