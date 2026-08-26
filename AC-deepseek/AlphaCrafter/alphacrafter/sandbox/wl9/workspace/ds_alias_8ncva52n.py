import json
# read a couple persisted factor validation definitions to mirror format
for f in ['ac1_120d.json','skew_20d.json','vix_roc_20d.json']:
    d=json.load(open('factors/'+f))
    print('==',f)
    v=d.get('validation',{})
    print('status',v.get('status'),'last',v.get('last_validated'),'metrics',json.dumps(v.get('metrics',{}))[:500])