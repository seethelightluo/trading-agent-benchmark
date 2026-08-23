import json, glob
files = sorted(glob.glob('factors/*.json'))
for f in files:
    d = json.load(open(f))
    fid = d.get('factor_id')
    v = d.get('validation', {})
    m = v.get('metrics', {})
    # probe structure of one factor file
print(open('factors/mom_10_vixreg.json').read()[:3000])