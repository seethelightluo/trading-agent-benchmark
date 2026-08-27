import json,base64,zlib,csv,io
import numpy as np
ids=['beta_VIX_60','kaufman_eff_20d','mom_120d_skip5','bb_width_20d','cny_beta_60','vol_z_20d','ac1_120d','mom_10d_skip5','dxy_corr_change_20_60','skew_20d']
sigs={}
for i in ids:
    d=json.load(open(f'factors/{i}.json'))
    data=d['validation']['signal_artifact']['data']
    if data.startswith('base64:zlib:'):
        raw=zlib.decompress(base64.b64decode(data.split('base64:zlib:',1)[1])).decode()
    else:
        raw=base64.b64decode(data).decode()
    rd=csv.reader(io.StringIO(raw)); header=next(rd)
    ci=header.index('SPX') if 'SPX' in header else 8
    col=[]
    for row in rd:
        try: col.append(float(row[ci]))
        except: col.append(np.nan)
    sigs[i]=np.array(col)
names=list(sigs.keys()); n=len(names)
mx=0; pair=None
for a in range(n):
    for b in range(a+1,n):
        x=sigs[names[a]];y=sigs[names[b]]
        m=~(np.isnan(x)|np.isnan(y))
        if m.sum()>100:
            c=np.corrcoef(x[m],y[m])[0,1]
            if abs(c)>mx and abs(c)<0.99: mx=abs(c);pair=(names[a],names[b],float(c))
print('max |corr| among selected:',round(mx,3),pair)