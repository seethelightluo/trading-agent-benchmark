import json
d = json.load(open('scripts/miner2_20280915_reval_results.json'))
# print all remaining keys
for k in list(d.keys()):
    if k not in ['mom_120d_skip5','vol_of_vol20x60','nclv_1d','nclv_2d','nclv_3d','nclv_5d','rev_1d','rev_2d']:
        v = d[k]
        print(k, "ic=%.4f icir=%.4f n=%d maxlib=%.3f cov=%.3f" % (v['ic'], v['icir'], v['n'], v.get('maxlib',0), v.get('coverage',0)))