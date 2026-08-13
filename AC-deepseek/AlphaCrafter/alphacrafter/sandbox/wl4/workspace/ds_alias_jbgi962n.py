import json
for f in ['dn_mkt_beta_60d','rate_beta_cn10y_60d']:
    d=json.load(open('factors/'+f+'.json'))
    print('====',f,'====')
    print('calc:', json.dumps(d.get('calculation',{}))[:800])
    print('params:', json.dumps(d.get('parameters',{}))[:400])
    print('deps:', d.get('dependencies'))
    print()