import importlib.util
import tempfile
import unittest
import sys
from pathlib import Path
from levo_harmony.operator_lab.pool import Pool,Spec,Outcome,ObservationDatabase
from levo_harmony.operator_lab.solver import solve,shortest_path,exact_pool_cg
from levo_harmony.operator_lab.search import discover,roofline_ms
from levo_harmony.operator_lab.pipeline import Budget,execute_candidate
from levo_harmony.operator_lab.runner import DeviceLock,ensure_idle
from levo_harmony.operator_lab.reporting import operator_tables,baseline_summary,roofline_calibration
from levo_harmony.operator_lab.claude import decode_output,review,invoke
from unittest.mock import patch

def example():
    specs=[Spec('n'+str(i),i,i+1,traffic_bytes=2,baseline=True) for i in range(4)]
    specs += [Spec('a',0,2,traffic_bytes=3,predicted_ms=1),Spec('b',0,2,traffic_bytes=3,predicted_ms=2),Spec('c',2,4,traffic_bytes=3,predicted_ms=3)]
    values={s.id:Outcome('passed',5,0,'synthetic_test') for s in specs[:4]}
    values.update(a=Outcome('passed',1,0,'synthetic_test'),b=Outcome('passed',.9,0,'synthetic_test'),c=Outcome('passed',4,0,'synthetic_test'))
    return Pool(4,specs,values,'unit-test-only','synthetic_test')

class PoolTests(unittest.TestCase):
    def test_agent_failure_does_not_report_zero_token_cost(self):
        with tempfile.TemporaryDirectory() as t:
            r=execute_candidate({'id':'agent-failure','signature':'s'},
                {'agent':[sys.executable,'-c','raise SystemExit(2)'],'compile':['unused'],'correctness':['unused'],'device':['unused']},Path(t)/'trial',Budget())
            self.assertEqual(r['status'],'agent_failed');self.assertIsNone(r['budget']['agentTokens'])
            self.assertEqual(r['budget']['compilationAttempts'],0)

    def test_timeout_stops_owned_cli_process_tree(self):
        import subprocess,os
        from unittest.mock import MagicMock
        process=MagicMock();process.pid=12345;process.communicate.side_effect=[subprocess.TimeoutExpired('cli',1),(b'',b'')]
        with patch('levo_harmony.operator_lab.claude.subprocess.Popen',return_value=process),patch('levo_harmony.operator_lab.claude.subprocess.run') as stop:
            with self.assertRaisesRegex(RuntimeError,'usage unknown'):invoke(['test-cli'],b'task',timeout=1,cwd='.')
            if os.name=='nt':self.assertEqual(stop.call_args.args[0],['taskkill.exe','/PID','12345','/T','/F'])
            else:process.kill.assert_called_once()

    def test_claude_windows_output_can_be_utf8_or_gb18030(self):
        for encoding in ['utf-8','gb18030']:
            self.assertEqual(decode_output('算子基准'.encode(encoding)),'算子基准')

    def test_claude_delegation_is_read_only_and_records_actual_model(self):
        import json
        from types import SimpleNamespace
        payload=dict(subtype='success',is_error=False,usage={'input_tokens':10},modelUsage={'test-model':{}},result='{}')
        with tempfile.TemporaryDirectory() as t,patch('levo_harmony.operator_lab.claude.invoke',return_value=SimpleNamespace(returncode=0,stdout=json.dumps(payload).encode())) as call:
            r=review({'operator':'Convolution'},Path(t)/'review.json',executable='test-claude')
            argv=call.call_args.args[0]
            self.assertIn('--tools=',argv);self.assertIn('--safe-mode',argv);self.assertIn('--strict-mcp-config',argv)
            self.assertEqual(list(r['cliResponse']['modelUsage']),['test-model'])
            with self.assertRaises(ValueError):review({},Path(t)/'review.json',executable='test-claude')

    def test_callback_decode_cost_is_normalized_by_forwards(self):
        node=dict(name='layer',type='Convolution',calls=2,seconds=.006,inputShapes=[[1,4]],outputShapes=[[1,4]],logicalInputBytes=16,logicalOutputBytes=16,backendObservations={'backend0/coreBytes4':2})
        r=dict(controlledAppTimingAccepted=True,native=dict(status='completed',instrumented=True,rounds=[{'decodeForwards':2}],operators={'nodes':[dict(node,phase='decode')]}))
        t=operator_tables(r);self.assertEqual(t['types'][0]['sumPerForwardMs'],3);self.assertEqual(t['nodes'][0]['meanCallMs'],3)
        self.assertIn('not measured DRAM',t['scope'])
        r['controlledAppTimingAccepted']=False
        with self.assertRaises(ValueError):operator_tables(r)

    def test_callback_run_cannot_become_uninstrumented_baseline(self):
        with self.assertRaises(ValueError):baseline_summary(dict(controlledAppTimingAccepted=True,native=dict(status='completed',instrumented=True)))

    def test_roofline_calibration_retains_evidence_limits(self):
        row=dict(operator='MatMul',attainedFlopsPerSecond=1e9,flops=200,logicalBytes=100,modelShape='small',phaseShape='decode',warmMedianMs=.1)
        r=dict(screenAccepted=True,native=dict(success=True,rows=[row],threads=4,precision='high',bandwidth={'nominalBytesPerSecond':1e8}))
        c=roofline_calibration(r);self.assertEqual(c['points'][0]['arithmeticIntensity'],2)
        self.assertFalse(c['memoryBoundClassificationClaimed']);self.assertFalse(c['strictLowerBoundClaimed'])
        r['screenAccepted']=False
        with self.assertRaises(ValueError):roofline_calibration(r)

    def test_unknown_running_app_state_cannot_be_force_stopped(self):
        class HdcStub:
            def __init__(self,state):self.state=state
            def shell(self,command):return '123' if command.startswith('pidof ') else self.state
        for state in ['null','{}','{"busy":true}']:
            with self.assertRaises(RuntimeError):ensure_idle(HdcStub(state),'example.app','/data/example')
        ensure_idle(HdcStub('{"busy":false}'),'example.app','/data/example')

    def test_nonfinite_numerical_error_cannot_enter_pool(self):
        for error in [float('nan'),float('inf'),-1]:
            with self.assertRaises(ValueError):Outcome('passed',1,error,'synthetic_test').validate()

    def test_device_campaigns_cannot_overlap(self):
        with tempfile.TemporaryDirectory() as t:
            path=Path(t)/'device.lock'
            with DeviceLock(path):
                with self.assertRaises(RuntimeError):
                    with DeviceLock(path):pass
            self.assertFalse(path.exists())

    def test_compile_failure_is_counted_and_later_stages_do_not_run(self):
        with tempfile.TemporaryDirectory() as t:
            budget=Budget(compile_attempts=1)
            result=execute_candidate({'id':'bad','signature':'s'},
                {'compile':[sys.executable,'-c','raise SystemExit(3)'],'correctness':['unused'],'device':['unused']},Path(t)/'trial',budget)
            self.assertEqual(result['status'],'compile_failed');self.assertEqual(budget.compile_attempts,1)
            self.assertEqual(budget.device_profiles,0);self.assertIsNone(result['latencyMs']);self.assertFalse(result['deployable'])

    def test_exhausted_compile_budget_never_launches_compiler(self):
        with tempfile.TemporaryDirectory() as t:
            result=execute_candidate({'id':'no-budget','signature':'s'},
                {'compile':[sys.executable,'-c','raise AssertionError()'],'correctness':['unused'],'device':['unused']},Path(t)/'trial',Budget(compile_attempts=0))
            self.assertEqual(result['status'],'budget_exhausted');self.assertEqual(result['stages'],[])

    def test_correct_kernel_without_graph_measurement_is_not_deployable_e2e_result(self):
        import json
        with tempfile.TemporaryDirectory() as t:
            payloads={'compile':{'signature':'s'},'correctness':{'signature':'s','passed':True,'relativeL2':0},'device':{'signature':'s','passed':True,'medianMs':1}}
            commands={stage:[sys.executable,'-c','from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2])','{result}',json.dumps(payload)] for stage,payload in payloads.items()}
            r=execute_candidate({'id':'module-only','signature':'s'},commands,Path(t)/'trial',Budget())
            self.assertEqual(r['status'],'passed');self.assertEqual(r['latencyMs'],1)
            self.assertFalse(r['deployable']);self.assertIsNone(r['wholeGraphLatencyMs'])

    def test_failed_device_controls_cannot_become_valid_latency(self):
        import json
        with tempfile.TemporaryDirectory() as t:
            payloads={'compile':{'signature':'s'},'correctness':{'signature':'s','passed':True,'relativeL2':0},'device':{'signature':'s','passed':False,'medianMs':1}}
            commands={stage:[sys.executable,'-c','from pathlib import Path; import sys; Path(sys.argv[1]).write_text(sys.argv[2])','{result}',json.dumps(payload)] for stage,payload in payloads.items()}
            r=execute_candidate({'id':'failed-device','signature':'s'},commands,Path(t)/'trial',Budget())
            self.assertEqual(r['status'],'device_failed');self.assertIsNone(r['latencyMs']);self.assertFalse(r['deployable'])

    def test_failed_candidates_have_no_latency_and_consume_queries(self):
        p=example();outcomes=p.full_outcomes_for_evaluation_only();outcomes['a']=Outcome('compile_failed',source='synthetic_test')
        p=Pool(p.nodes,list(p.specs.values()),outcomes,p.signature,p.scope);oracle=p.oracle()
        self.assertEqual(oracle.query('a').status,'compile_failed');self.assertEqual(len(oracle.calls),1)
        with self.assertRaises(ValueError):oracle.query('a')
        with self.assertRaises(ValueError):Outcome('compile_failed',1,source='synthetic_test').validate()

    def test_database_partitions_device_threads_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as t:
            db=ObservationDatabase(Path(t)/'pool.db')
            base=dict(device='test',modelSha256='m',mnnSha256='r',precision='high',threads=4,phase='decode',inputSha256='i')
            a=db.signature(base);b=db.signature(dict(base,threads=6));self.assertNotEqual(a,b)
            db.put(a,'node',{'ms':3});db.put(b,'node',{'ms':1})
            with self.assertRaises(ValueError):db.put(a,'node',{'ms':1})
            db.close()

    def test_roofline_does_not_invent_hardware_bandwidth(self):
        with self.assertRaises(ValueError):roofline_ms(Spec('x',0,1),None)
        for bandwidth in [-1,float('nan'),float('inf')]:
            with self.assertRaises(ValueError):roofline_ms(Spec('x',0,1),dict(flops_per_second=1e9,bytes_per_second=bandwidth))

    def test_serial_dp_finds_best_complete_nonoverlapping_cover(self):
        p=example();result=shortest_path(p,p.full_outcomes_for_evaluation_only())
        self.assertAlmostEqual(result.objective_ms,4.9);self.assertEqual(result.selected,['b','c'])
        with self.assertRaises(ValueError):shortest_path(p,p.seeds(),traffic_budget=4)

@unittest.skipUnless(importlib.util.find_spec('cplex'),'Optional CPLEX required for mathematical baselines')
class CplexTests(unittest.TestCase):
    def test_true_phase_one_does_not_trade_feasibility_for_latency(self):
        p=Pool(1,[Spec('n',0,1,traffic_bytes=1000000,baseline=True),Spec('c',0,1,traffic_bytes=999999)],
               {'n':Outcome('passed',1,0,'synthetic_test'),'c':Outcome('passed',1000,0,'synthetic_test')},'phase1-counterexample','synthetic_test')
        first=solve(p,p.seeds(),traffic_budget=999999)
        self.assertFalse(first.feasible);self.assertEqual(first.pricing_phase,'feasibility')
        full=solve(p,p.full_outcomes_for_evaluation_only(),traffic_budget=999999)
        self.assertTrue(full.feasible);self.assertAlmostEqual(full.objective_ms,1000);self.assertEqual(full.pricing_phase,'latency')
        cg=exact_pool_cg(p,traffic_budget=999999)
        self.assertAlmostEqual(cg['lpLowerBoundMs'],1000);self.assertEqual(cg['history'][0]['pricingPhase'],'feasibility')

    def test_traffic_dual_sign_and_fractional_integer_gap(self):
        p=Pool(1,[Spec('n',0,1,traffic_bytes=10,baseline=True),Spec('c',0,1,traffic_bytes=20)],
               {'n':Outcome('passed',1,0,'synthetic_test'),'c':Outcome('passed',.5,0,'synthetic_test')},'traffic-dual','synthetic_test')
        values=p.full_outcomes_for_evaluation_only();lp=solve(p,values,traffic_budget=15);mip=solve(p,values,integer=True,traffic_budget=15)
        self.assertAlmostEqual(lp.objective_ms,.75);self.assertAlmostEqual(mip.objective_ms,1);self.assertLess(lp.traffic_dual,0)
        for spec in p.specs.values():self.assertAlmostEqual(values[spec.id].latency_ms-sum(lp.duals[spec.start:spec.end])-lp.traffic_dual*spec.traffic_bytes,0)

    def test_full_pool_integer_oracle_matches_serial_dp(self):
        p=example();v=p.full_outcomes_for_evaluation_only();self.assertAlmostEqual(solve(p,v,integer=True).objective_ms,shortest_path(p,v).objective_ms)

    def test_exact_cg_certifies_lp_not_restricted_integer_optimality(self):
        r=exact_pool_cg(example());self.assertAlmostEqual(r['lpLowerBoundMs'],4.9)
        self.assertTrue(r['lpPricingComplete']);self.assertFalse(r['restrictedIntegerGloballyOptimalClaimed'])

    def test_updating_duals_changes_next_query_compared_to_frozen_prices(self):
        dynamic=discover(example(),'dual',2);frozen=discover(example(),'frozen_dual',2)
        self.assertEqual([r['candidate'] for r in dynamic['trace']],['a','c'])
        self.assertEqual([r['candidate'] for r in frozen['trace']],['a','b'])
        self.assertIsNone(dynamic['globalLowerBoundMs'])

    def test_scratch_cap_filters_candidates_instead_of_summing_as_fake_sram(self):
        p=example();specs=[Spec(**dict(vars(s),scratch_bytes=10 if s.id=='b' else 0)) for s in p.specs.values()]
        p=Pool(4,specs,p.full_outcomes_for_evaluation_only(),p.signature,p.scope)
        self.assertAlmostEqual(solve(p,p.full_outcomes_for_evaluation_only(),integer=True,scratch_budget=5).objective_ms,5)

    def test_phase_one_artificial_columns_are_not_reported_as_real_plan(self):
        p=example();lp=solve(p,p.seeds(),traffic_budget=6)
        self.assertFalse(lp.feasible);self.assertGreater(lp.artificial_mass,0);self.assertIsNone(lp.objective_ms)
        result=solve(p,p.full_outcomes_for_evaluation_only(),integer=True,traffic_budget=6)
        self.assertTrue(result.feasible);self.assertAlmostEqual(result.objective_ms,4.9)

    def test_failure_still_counts_towards_hidden_query_budget(self):
        p=example();values=p.full_outcomes_for_evaluation_only();values['a']=Outcome('correctness_failed',source='synthetic_test')
        p=Pool(4,list(p.specs.values()),values,p.signature,p.scope)
        r=discover(p,'dual',1);self.assertEqual(r['queryCount'],1)
        self.assertEqual(r['trace'][0]['outcome'],'correctness_failed');self.assertEqual(r['trace'][0]['bestModelledMs'],20)

if __name__=='__main__':unittest.main()
