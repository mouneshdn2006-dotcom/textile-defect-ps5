import json, sys, numpy as np
NAMES = ['broken_yarn_needle','good','hole_burst','jute','lycra_run','stain_oil']
def load(split, detfile=None):
    rows = json.load(open(f'/home/claude/sc_{split}_a.json')) + json.load(open(f'/home/claude/sc_{split}_b.json'))
    cc = {}; [cc.update(json.load(open(f'/home/claude/scc_{split}_{h}.json'))) for h in 'ab']
    dd = {}
    if detfile: [dd.update(json.load(open(f'/home/claude/{detfile}_{split}_{h}.json'))) for h in 'ab']
    out = []
    for path, dom, lab, s, p3 in rows:
        p3f, p4, p4f = cc[path]
        out.append(dict(dom=dom, lab=lab, det=dd.get(path, s),
            probs={'cls3': p3, 'cls3+flip': list((np.array(p3)+np.array(p3f))/2), 'cls4': p4, 'cls4+flip': list((np.array(p4)+np.array(p4f))/2)}))
    return out
def flag(r, cd, th, v):
    if r['det'] > cd: return True
    p = r['probs'][v]; t = int(np.argmax(p)); return t != 1 and p[t] >= th[t]
def metrics(data, cd, th, v):
    g = {}
    for r in data:
        k = (r['dom'], 'good' if r['lab'] == 'good' else 'defect'); a = g.setdefault(k, [0, 0]); a[0] += flag(r, cd, th, v); a[1] += 1
    rt = {k: x[0]/x[1] for k, x in g.items()}
    m = dict(zju_recall=rt[('zju','defect')], zju_FA=rt[('zju','good')], cls_recall=rt[('cls','defect')], cls_FA=rt[('cls','good')], pattern_FA=rt[('pattern','good')], plaid_FA=rt.get(('plaid','good')))
    return np.mean([m['zju_recall'], 1-m['zju_FA'], m['cls_recall'], 1-m['cls_FA'], 1-m['pattern_FA']]), m
def perclass(data, cd, th, v):
    return {NAMES[c][:6]: round(100*np.mean([flag(r, cd, th, v) for r in data if r['dom']=='cls' and r['lab']==NAMES[c]])) for c in [0,2,3,4,5]}
def tune(data, v):
    best = None
    for cd in [0.05,0.08,0.1,0.12,0.15,0.2,0.25,0.3]:
        for t in np.arange(0.3, 0.96, 0.05):
            s, _ = metrics(data, cd, [t]*6, v)
            if best is None or s > best[0]: best = (s, cd, [t]*6)
    s, cd, th = best
    for _ in range(2):
        for c in [0,2,3,4,5]:
            for t in np.arange(0.25, 0.97, 0.05):
                th2 = list(th); th2[c] = float(t); s2, _ = metrics(data, cd, th2, v)
                if s2 > s + 1e-4: s, th = s2, th2
    return cd, th
detfile = sys.argv[1] if len(sys.argv) > 1 else None
tu, te = load('tune', detfile), load('test', detfile)
for v in ['cls3', 'cls3+flip', 'cls4', 'cls4+flip']:
    cd, th = tune(tu, v); s, m = metrics(te, cd, th, v)
    print(f"{detfile or 'det_v2'} {v:10s} det={cd} th={[round(x,2) for x in th]} TEST {s:.4f} " + " ".join(f"{k}={100*x:.1f}" for k, x in m.items() if x is not None) + f" {perclass(te, cd, th, v)}")
