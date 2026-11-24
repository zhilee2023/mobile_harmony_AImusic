"""Conservative repeated-node hints. These are not validated reusable/fused subgraphs."""
import hashlib,json,re

def mine_repeated_nodes(report):
    n=report['native']
    if n['status']!='completed' or not n['instrumented']:raise ValueError('Need completed operator diagnostic')
    groups={};forwards={'prefill':len(n['rounds']),'decode':sum(r['decodeForwards'] for r in n['rounds'])}
    for row in n['operators']['nodes']:
        role=re.sub(r'/(layers|blocks)\.\d+',lambda m:'/'+m.group(1)+'.{layer}',row['name'])
        metadata=dict(model=n['modelId'],mnn=n['mnnArchiveSha256'],phase=row['phase'],role=role,operator=row['type'],
                      inputShapes=row['inputShapes'],outputShapes=row['outputShapes'],backendKeys=sorted(row['backendObservations']))
        key=hashlib.sha256(json.dumps(metadata,sort_keys=True).encode()).hexdigest()
        if key not in groups:groups[key]=dict(id=key,**metadata,sites=[],observedCalls=0,totalMs=0.)
        g=groups[key];g['sites'].append(row['name']);g['observedCalls']+=row['calls'];g['totalMs']+=1000*row['seconds']
    result=[]
    for g in groups.values():
        if len(set(g['sites']))<2:continue
        f=forwards[g['phase']];g['observedForwards']=f;g['callsPerForward']=g['observedCalls']/f
        g['reuseProven']=False;g['dependencyLegalityProven']=False
        g['needsKvBucket']=g['operator']=='Attention'
        g['unverifiedFields']=['operator subtype/attributes','layout','dtype/weight quantization ABI','ISA compatibility','dependency edges']
        result.append(g)
    return sorted(result,key=lambda g:g['totalMs'],reverse=True)
