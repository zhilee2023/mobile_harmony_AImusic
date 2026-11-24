import importlib.util,unittest
from dataclasses import replace
from levo_harmony.operator_lab.workload import Exposure,Timing,aggregate_cost,estimate_variant
from levo_harmony.operator_lab.pool import Pool,Spec,Outcome
from levo_harmony.operator_lab.patterns import Kernel,Placement,Catalog,solve_master,exact_catalog_cg,frequency_greedy
from levo_harmony.operator_lab.mining import mine_repeated_nodes

class WorkloadTests(unittest.TestCase):
    def test_repeated_shape_hints_are_not_reuse_or_dependency_proof(self):
        node=dict(type='Convolution',phase='prefill',inputShapes=[[513,1024]],outputShapes=[[513,3072]],calls=1,seconds=.001,backendObservations={'backend0/coreBytes4':1})
        report=dict(native=dict(status='completed',instrumented=True,modelId='test',mnnArchiveSha256='m',rounds=[{'decodeForwards':2}],
                    operators={'nodes':[dict(node,name='/layers.0/mlp/up_proj/Linear'),dict(node,name='/layers.1/mlp/up_proj/Linear')]}))
        groups=mine_repeated_nodes(report);self.assertEqual(len(groups),1);self.assertEqual(groups[0]['callsPerForward'],2)
        self.assertFalse(groups[0]['reuseProven']);self.assertFalse(groups[0]['dependencyLegalityProven'])
        report['native']['operators']['nodes'][1]['inputShapes']=[[1,1024]]
        self.assertEqual(mine_repeated_nodes(report),[])
    def test_prefill_decode_and_repetitions_are_counted_once(self):
        e={'prefill':Exposure(28,1),'decode':Exposure(28,64)}
        r=aggregate_cost(e,{'prefill':Timing(1),'decode':Timing(.01)},extra_overhead_ms=0)
        self.assertAlmostEqual(r['totalCostMs'],28+17.92)
        self.assertEqual(r['contexts']['decode']['occurrences'],1792)
    def test_preweighted_totals_are_not_multiplied_again(self):
        r=aggregate_cost({'d':Exposure(28,64)},{'d':Timing(17.92,unit='workload_total',counted_occurrences=1792)},extra_overhead_ms=0)
        self.assertAlmostEqual(r['totalCostMs'],17.92)
        with self.assertRaises(ValueError):aggregate_cost({'d':Exposure(28,64)},{'d':Timing(17.92,unit='workload_total',counted_occurrences=28)})
    def test_missing_phase_and_unknown_overhead_are_not_silently_zero(self):
        with self.assertRaises(ValueError):aggregate_cost({'p':Exposure(28,1),'d':Exposure(28,64)},{'p':Timing(1)})
        r=aggregate_cost({'p':Exposure(28,1)},{'p':Timing(1)})
        self.assertIsNone(r['totalCostMs']);self.assertFalse(r['overheadsKnown'])
    def test_unchanged_decode_ignores_noisy_isolated_gain(self):
        r=aggregate_cost({'p':Exposure(28,1),'d':Exposure(28,64)},{'p':Timing(1),'d':Timing(.01)},extra_overhead_ms=0)
        estimate=estimate_variant(r,{'p':.9,'d':.5},unchanged_phases={'d'})
        self.assertAlmostEqual(estimate['savingBeforeUnmeasuredOverheadsMs'],2.8)
        self.assertIsNone(estimate['estimatedNetSavingMs']);self.assertFalse(estimate['eligibleForDeploymentFromEstimate'])
    def test_cost_domains_cannot_be_mixed_in_master_or_predictor(self):
        with self.assertRaises(ValueError):Pool(1,[Spec('n',0,1,baseline=True)],{'n':Outcome('passed',10,source='synthetic_test')},'s','synthetic_test',cost_domain='template_workload')
        with self.assertRaises(ValueError):Pool(1,[Spec('n',0,1,baseline=True,predicted_ms=2)],{'n':Outcome('passed',10,source='synthetic_test',cost_domain='template_workload')},'s','synthetic_test',cost_domain='template_workload')
    def test_partial_frequency_coverage_requires_context_split(self):
        with self.assertRaises(ValueError):Catalog(1,[Kernel('k',0,'x',native=True)],[Placement('p','k',(0,),10,1,'x'),Placement('q','k',(0,),5,.1,'x')])

@unittest.skipUnless(importlib.util.find_spec('cplex'),'Optional CPLEX required')
class PatternMasterTests(unittest.TestCase):
    def shared(self):
        ks=[Kernel('native',0,'fp32',native=True),Kernel('shared',1,'fp32')]
        ps=[Placement('n'+str(i),'native',(i,),10,1,'fp32') for i in range(3)]
        ps += [Placement('s'+str(i),'shared',(i,),10,.5,'fp32') for i in range(3)]
        return Catalog(3,ks,ps)
    def test_development_paid_once_for_three_placements(self):
        r=solve_master(self.shared(),1,integer=True)
        self.assertAlmostEqual(r['objectiveMs'],15);self.assertEqual(r['selected'],['s0','s1','s2']);self.assertAlmostEqual(r['developmentCost'],1)
    def test_failed_development_spend_and_sunk_kernels_are_distinct(self):
        c=self.shared();r=solve_master(c,1,spent=1,integer=True)
        self.assertAlmostEqual(r['objectiveMs'],30)
        c=Catalog(c.nodes,[replace(k,developed=True) for k in c.kernels.values()],list(c.placements.values()))
        self.assertAlmostEqual(solve_master(c,1,spent=1,integer=True)['objectiveMs'],15)
    def test_known_catalog_linkage_pricing_matches_full_lp(self):
        c=self.shared();cg=exact_catalog_cg(c,1);full=solve_master(c,1)
        self.assertAlmostEqual(cg['lp']['objectiveMs'],full['objectiveMs']);self.assertAlmostEqual(full['objectiveMs'],15)
        self.assertFalse(cg['restrictedIntegerGlobalClaimed'])
    def test_shared_development_lp_relaxation_is_not_an_integer_certificate(self):
        c=self.shared();c=Catalog(c.nodes,[replace(k,development_cost=2) if k.id=='shared' else k for k in c.kernels.values()],list(c.placements.values()))
        lp=solve_master(c,1);mip=solve_master(c,1,integer=True)
        self.assertAlmostEqual(lp['objectiveMs'],22.5);self.assertAlmostEqual(mip['objectiveMs'],30);self.assertLess(lp['budgetDual'],0)
    def test_phase_one_pricing_can_recover_missing_native_coverage(self):
        c=Catalog(2,[Kernel('n',0,'x',native=True),Kernel('k',1,'x')],
                  [Placement('a','n',(0,),1,1,'x'),Placement('b','k',(1,),1,1,'x')])
        result=exact_catalog_cg(c,1)
        self.assertEqual(result['history'][0]['phase'],'feasibility');self.assertTrue(result['lp']['feasible']);self.assertAlmostEqual(result['lp']['objectiveMs'],2)
    def test_frequency_greedy_prefers_total_savings_over_single_call(self):
        freq=[100,10,2,400];native=[1,4,10,1];saving=[.5,3,8,.2]
        ks=[Kernel('native',0,'x',native=True)]+[Kernel('k'+str(i),1,'x') for i in range(4)]
        ps=[Placement('n'+str(i),'native',(i,),freq[i],native[i],'x') for i in range(4)]
        ps += [Placement('p'+str(i),'k'+str(i),(i,),freq[i],native[i]-saving[i],'x') for i in range(4)]
        r=frequency_greedy(Catalog(4,ks,ps),1)
        self.assertEqual(r['trace'][0]['kernel'],'k3');self.assertAlmostEqual(r['trace'][0]['gainMs'],80)
    def test_overlap_requires_global_combination(self):
        ks=[Kernel('native',0,'x',native=True)]+[Kernel(k,1,'x') for k in ['A','B','C']]
        ps=[Placement('n'+str(i),'native',(i,),100,1,'x') for i in range(4)]
        ps += [Placement('a','A',(0,1),100,1.5,'x'),Placement('b','B',(1,2),100,1.2,'x'),Placement('c','C',(2,3),100,1.4,'x')]
        c=Catalog(4,ks,ps);full=solve_master(c,2,integer=True);greedy=frequency_greedy(c,2)
        self.assertEqual(set(full['selected']),{'a','c'});self.assertAlmostEqual(full['objectiveMs'],290)
        self.assertAlmostEqual(greedy['solution']['objectiveMs'],320)
    def test_predictions_and_bad_numerics_cannot_be_deployed_as_measurements(self):
        c=self.shared();ps=[replace(p,evidence='prediction',relative_l2=None) if p.kernel=='shared' else p for p in c.placements.values()]
        c=Catalog(3,list(c.kernels.values()),ps)
        self.assertAlmostEqual(solve_master(c,1,integer=True)['objectiveMs'],30)
        self.assertAlmostEqual(solve_master(c,1,integer=True,allow_predictions=True)['objectiveMs'],15)
        self.assertFalse(solve_master(c,1,integer=True,allow_predictions=True)['performanceProven'])
    def test_incompatible_reuse_is_rejected(self):
        with self.assertRaises(ValueError):Catalog(1,[Kernel('k',1,'FP16')],[Placement('p','k',(0,),10,1,'FP32')])

if __name__=='__main__':unittest.main()
