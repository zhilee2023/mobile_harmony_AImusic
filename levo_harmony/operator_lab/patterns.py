"""Finite-catalog shared-kernel master: development paid once, runtime frequency-weighted.

All activation variables and linkage rows exist up front. Thus known-catalog
placement pricing is genuine column-only pricing. Unknown new patterns are a
different row/column generation problem; no certificate is claimed for those.
"""
from dataclasses import dataclass,replace
import math,re,time


@dataclass(frozen=True)
class Kernel:
    id: str
    development_cost: float
    compatibility: str
    native: bool=False
    developed: bool=False

    @property
    def incremental_cost(self):return 0. if self.native or self.developed else self.development_cost


@dataclass(frozen=True)
class Placement:
    id: str
    kernel: str
    covers: tuple[int,...]
    frequency: int
    latency_ms: float
    compatibility: str
    cost_unit: str='per_call'
    counted_frequency: int | None=None
    extra_total_ms: float=0.
    logical_traffic_total_bytes: int=0
    workspace_bytes: int=0
    evidence: str='measured'
    relative_l2: float | None=0.

    @property
    def runtime_cost(self):
        if self.cost_unit=='per_call':
            if self.counted_frequency is not None:raise ValueError('Per-call latency already carries a count')
            return self.frequency*self.latency_ms+self.extra_total_ms
        if self.cost_unit=='workload_total' and self.counted_frequency==self.frequency:return self.latency_ms+self.extra_total_ms
        raise ValueError('Explicit consistent timing unit required')


class Catalog:
    def __init__(self,nodes,kernels,placements,*,budget_unit='candidate_build_attempts'):
        self.nodes=nodes;self.kernels={k.id:k for k in kernels};self.placements={p.id:p for p in placements};self.budget_unit=budget_unit
        if type(nodes) is not int or not 1<=nodes<=256 or len(self.kernels)!=len(kernels) or len(self.placements)!=len(placements):raise ValueError('Invalid catalog cardinalities')
        for k in kernels:
            if not re.fullmatch('[A-Za-z0-9_-]{1,80}',k.id) or not k.compatibility or not math.isfinite(k.development_cost) or k.development_cost<0:raise ValueError('Invalid kernel implementation')
            if type(k.native) is not bool or type(k.developed) is not bool:raise ValueError('Native/developed flags must be explicit booleans')
        node_frequency={}
        for p in placements:
            if not re.fullmatch('[A-Za-z0-9_-]{1,80}',p.id) or p.kernel not in self.kernels:raise ValueError('Invalid placement identity')
            if not p.covers or len(set(p.covers))!=len(p.covers) or any(type(i) is not int or not 0<=i<nodes for i in p.covers):raise ValueError('Invalid primitive coverage')
            if p.compatibility!=self.kernels[p.kernel].compatibility:raise ValueError('Incompatible pattern/shape/layout/dtype/ISA contract')
            if type(p.frequency) is not int or p.frequency<=0:raise ValueError('Positive runtime frequency required')
            for node in p.covers:
                if node in node_frequency and node_frequency[node]!=p.frequency:raise ValueError('Split invocation/context classes before covering a partial-frequency occurrence')
                node_frequency[node]=p.frequency
            if any(not math.isfinite(v) or v<0 for v in [p.latency_ms,p.extra_total_ms,p.logical_traffic_total_bytes,p.workspace_bytes]):raise ValueError('Invalid cost/resource measurement')
            if p.evidence not in {'measured','prediction','failed'}:raise ValueError('Unknown evidence')
            if p.relative_l2 is not None and (not math.isfinite(p.relative_l2) or p.relative_l2<0):raise ValueError('Invalid error')
            if not math.isfinite(p.runtime_cost):raise ValueError('Weighted runtime cost overflow')


def eligible(catalog,*,allow_predictions=False,epsilon=1e-3,workspace_budget=None):
    return [p for p in catalog.placements.values() if (p.evidence=='measured' and p.relative_l2 is not None and p.relative_l2<=epsilon or allow_predictions and p.evidence=='prediction')
            and (workspace_budget is None or p.workspace_bytes<=workspace_budget)]


def solve_master(catalog,budget,*,active=None,integer=False,spent=0,traffic_budget=None,workspace_budget=None,allow_predictions=False,epsilon=1e-3,seconds=30):
    import cplex
    if any(not math.isfinite(v) or v<0 for v in [budget,spent]) or spent>budget:raise ValueError('Invalid development budget ledger')
    if not math.isfinite(epsilon) or epsilon<0 or workspace_budget is not None and workspace_budget<0:raise ValueError('Invalid error/workspace bound')
    placements=eligible(catalog,allow_predictions=allow_predictions,workspace_budget=workspace_budget,epsilon=epsilon)
    active=set(catalog.placements) if active is None else set(active)
    if not active<=catalog.placements.keys():raise ValueError('Unknown active placement')
    columns=[p for p in placements if p.id in active];kernels=list(catalog.kernels.values())
    model=cplex.Cplex()
    for stream in ['log','results','warning','error']:getattr(model,'set_'+stream+'_stream')(None)
    model.parameters.threads.set(1);model.parameters.timelimit.set(seconds)
    if integer:model.parameters.mip.tolerances.mipgap.set(0);model.parameters.mip.tolerances.absmipgap.set(0)
    ny=len(kernels);nx=len(columns);art=0 if integer else catalog.nodes
    y={k.id:i for i,k in enumerate(kernels)};x={p.id:ny+i for i,p in enumerate(columns)}
    names=['develop_'+k.id for k in kernels]+['use_'+p.id for p in columns]+['art_'+str(i) for i in range(art)]
    real_costs=[0.]*ny+[p.runtime_cost for p in columns]
    model.variables.add(names=names,obj=real_costs if integer else [0.]*(ny+nx)+[1.]*art,
                        lb=[0.]*len(names),ub=[1.]*(ny+nx)+[cplex.infinity]*art,types='B'*len(names) if integer else '')
    rows=[];rhs=[];senses=[]
    for i in range(catalog.nodes):
        ind=[x[p.id] for p in columns if i in p.covers]
        if art:ind.append(ny+nx+i)
        rows.append(cplex.SparsePair(ind=ind,val=[1.]*len(ind)));rhs.append(1.);senses.append('E')
    # Include links for inactive columns as -y_k<=0. Their duals are available to pricing.
    link={}
    for p in placements:
        link[p.id]=len(rows);ind=[y[p.kernel]];val=[-1.]
        if p.id in x:ind.append(x[p.id]);val.append(1.)
        rows.append(cplex.SparsePair(ind=ind,val=val));rhs.append(0.);senses.append('L')
    budget_row=len(rows);rows.append(cplex.SparsePair(ind=list(range(ny)),val=[k.incremental_cost for k in kernels]));rhs.append(budget-spent);senses.append('L')
    traffic_row=None
    if traffic_budget is not None:
        if not math.isfinite(traffic_budget) or traffic_budget<0:raise ValueError('Invalid logical traffic budget')
        traffic_row=len(rows);rows.append(cplex.SparsePair(ind=[x[p.id] for p in columns],val=[p.logical_traffic_total_bytes for p in columns]));rhs.append(traffic_budget);senses.append('L')
    model.linear_constraints.add(lin_expr=rows,rhs=rhs,senses=''.join(senses));model.solve();phase='latency'
    if not integer and model.solution.is_primal_feasible():
        if model.solution.get_status()!=1:raise RuntimeError('Uncertified Phase-I optimum')
        if sum(model.solution.get_values()[ny+nx:])>1e-7:phase='feasibility'
        else:
            model.variables.set_upper_bounds([(ny+nx+i,0.) for i in range(art)]);model.objective.set_linear(list(enumerate(real_costs+[0.]*art)));model.solve()
    if not model.solution.is_primal_feasible():return dict(feasible=False,optimal=False,objectiveMs=None,selected=[],developed=[])
    optimal=model.solution.get_status()==(101 if integer else 1)
    if not integer and not optimal:raise RuntimeError('LP prices require optimal solve')
    values=model.solution.get_values();mass=sum(values[ny+nx:]);feasible=mass<=1e-7
    result=dict(feasible=feasible,optimal=optimal,objectiveMs=sum(values[x[p.id]]*p.runtime_cost for p in columns) if feasible else None,
        selected=[p.id for p in columns if values[x[p.id]]>1e-7],developed=[k.id for k in kernels if values[y[k.id]]>1e-7],
        usedKernels=sorted({p.kernel for p in columns if values[x[p.id]]>1e-7}),
        developmentCost=sum(values[y[k.id]]*k.incremental_cost for k in kernels),budgetUnit=catalog.budget_unit,spentBudget=spent,
        artificialMass=mass,pricingPhase=phase,performanceProven=False,
        evidenceScope='Finite catalog additive runtime-cost model; callback/predicted costs and LP optimum are not actual whole-graph speedup')
    if not integer:
        dual=model.solution.get_dual_values();result.update(coverDuals=dual[:catalog.nodes],linkDuals={p.id:dual[link[p.id]] for p in placements},budgetDual=dual[budget_row],trafficDual=dual[traffic_row] if traffic_row is not None else 0.)
    return result


def reduced_cost(placement,lp):
    cost=0. if lp['pricingPhase']=='feasibility' else placement.runtime_cost
    return cost-sum(lp['coverDuals'][i] for i in placement.covers)-lp['linkDuals'][placement.id]-lp['trafficDual']*placement.logical_traffic_total_bytes


def exact_catalog_cg(catalog,budget,**kwargs):
    """All metadata, activation rows and actual costs are already known; not free Agent discovery."""
    active={p.id for p in catalog.placements.values() if catalog.kernels[p.kernel].native};history=[]
    for _ in range(len(catalog.placements)+1):
        lp=solve_master(catalog,budget,active=active,**kwargs)
        if 'coverDuals' not in lp:raise RuntimeError('No certified LP prices')
        unseen=[(reduced_cost(p,lp),p.id) for p in eligible(catalog,allow_predictions=kwargs.get('allow_predictions',False),workspace_budget=kwargs.get('workspace_budget'),epsilon=kwargs.get('epsilon',1e-3)) if p.id not in active]
        best=min(unseen) if unseen else None
        history.append(dict(activeColumns=len(active),phase=lp['pricingPhase'],objectiveMs=lp['objectiveMs'],minimumReducedCost=best[0] if best else None))
        if best is None or best[0]>=-1e-7:break
        active.add(best[1])
    return dict(lp=lp,restrictedInteger=solve_master(catalog,budget,active=active,integer=True,**kwargs),
                fullInteger=solve_master(catalog,budget,integer=True,**kwargs),history=history,
                scope='Known finite catalog, up-front y variables/link rows, numerical LP pricing certificate only',restrictedIntegerGlobalClaimed=False)


def frequency_greedy(catalog,budget,*,spent=0,**kwargs):
    """Frequency-weighted marginal-gain greedy, exact redeployment within developed kernels.

    This stronger greedy is explicitly labelled; solver time must be charged in
    comparisons. Paid development is irreversible, even if later not deployed.
    """
    tick=time.perf_counter();developed={k.id for k in catalog.kernels.values() if k.native or k.developed};paid=spent;trace=[]
    def evaluate(extra=None):
        ks=[replace(k,developed=k.id in developed) for k in catalog.kernels.values()]
        current=Catalog(catalog.nodes,ks,list(catalog.placements.values()),budget_unit=catalog.budget_unit)
        active={p.id for p in current.placements.values() if p.kernel in developed or p.kernel==extra}
        result=solve_master(current,budget,active=active,integer=True,spent=paid,**kwargs)
        if result['feasible'] and not result['optimal']:raise RuntimeError('Greedy redeployment comparison requires an optimal finite-catalog solve')
        return result
    base=evaluate()
    if not base['feasible']:raise ValueError('Greedy needs a feasible native/developed baseline')
    while True:
        choices=[]
        for k in catalog.kernels.values():
            if k.id in developed or paid+k.incremental_cost>budget:continue
            trial=evaluate(k.id)
            if not trial['feasible']:continue
            gain=base['objectiveMs']-trial['objectiveMs']
            if gain>1e-7:choices.append((gain/k.incremental_cost if k.incremental_cost else math.inf,k.id,gain,trial))
        if not choices:break
        _,kid,gain,_=max(choices,key=lambda c:(c[0],c[1]));paid+=catalog.kernels[kid].incremental_cost;developed.add(kid);base=evaluate()
        trace.append(dict(kernel=kid,gainMs=gain,paidDevelopmentBudget=paid,objectiveMs=base['objectiveMs']))
    return dict(solution=base,trace=trace,paidBudget=paid,seconds=time.perf_counter()-tick,
                scope='Offline marginal-gain greedy with global redeployment; frequency weighted, development paid once; not Agent token billing')
