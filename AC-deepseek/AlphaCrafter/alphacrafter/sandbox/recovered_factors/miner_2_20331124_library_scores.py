import glob,json,os
fs=[]
for x in glob.glob('factors/*.json'):
 d=json.load(open(x))
 if d.get('validation',{}).get('status')=='EFFECTIVE':
  m=d['validation'].get('metrics',{})
  q=abs(m.get('ic',0)*m.get('icir',0))
  fs.append((q,os.path.basename(x),m.get('ic'),m.get('icir')))
print('effective',len(fs))
for row in sorted(fs): print(*row)
