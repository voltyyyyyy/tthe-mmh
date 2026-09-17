import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

from meta_memory import SQLiteStore
from experiments.tau3_mmh.guidelines import GuidelineMemory, Guideline
from experiments.tau3_mmh.proposer import Proposal
from experiments.tau3_mmh.full_study import Study, make_schedule, inspect_record, task_key


class FullStudyTests(unittest.TestCase):
    def test_schedule_coverage_and_disjoint_validation(self):
        counts = {'airline': (30, 10), 'retail': (68, 23), 'telecom': (68, 23),
                  'banking_knowledge': (59, 19)}
        splits = {'search': {}, 'validation': {}}
        for domain, (n, v) in counts.items():
            splits['search'][domain] = list(range(n))
            splits['validation'][domain] = list(range(n, n+v))
        schedule = make_schedule(splits)
        search = [t for s in schedule for t in s['tasks']]
        validation = {t for s in schedule for t in s['validation']}
        self.assertEqual(len(search), 225)
        self.assertEqual(len(set(search)), 225)
        self.assertEqual(len(validation), 75)
        self.assertFalse(set(search) & validation)
        self.assertEqual(schedule[0]['subset'], schedule[3]['subset'])
        self.assertEqual(task_key('telecom:x[PERSONA:Hard]'), 'telecom_x[PERSONA:Hard]')

    def test_trace_transport_error_but_not_harness_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'events.jsonl'
            record = {'reward': 0, 'num_turns': 2, 'wall_time_s': 10}
            def events(error):
                path.write_text(json.dumps({'type':'agent_error', 'error':error})+'\n'+
                                json.dumps({'type':'grading', 'reward':0})+'\n')
            events('ValueError: invalid model-generated arguments')
            self.assertIsNone(inspect_record(record, tmp)[0])
            events('ConnectError: Connection refused')
            self.assertIn('ConnectError', inspect_record(record, tmp)[0])

    def test_same_file_snapshot_does_not_hang(self):
        with tempfile.TemporaryDirectory() as tmp:
            memory = GuidelineMemory(store=SQLiteStore(Path(tmp) / 'memory.sqlite'))
            memory.stage_add(Guideline('r', 'booking changes', 'look up the booking'), 1)
            memory.snapshot_to(Path(tmp) / 'memory.sqlite')
            self.assertIsNotNone(memory.store.get_rule('r'))
            memory.store.close()

    def test_real_driver_adapter_promotes_after_repeated_paired_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            study = Study.__new__(Study)
            study.out = Path(tmp)
            (study.out / 'mmh').mkdir()
            memory = GuidelineMemory(store=SQLiteStore())
            study.memories = {'mmh':memory}
            proposal = Proposal(Guideline('r1', 'booking changes', 'look up booking'),
                                'hypothesis', 'benefit', 'risk', ('airline_0',), .8)
            study.proposer = SimpleNamespace(propose=lambda **kw: [proposal] if kw['round_id']==1 else [])
            study.failure_evidence = lambda *args: []
            def evaluate(arm, label, tasks, split, rules):
                self.assertEqual(split, 'validation')
                self.assertTrue(all('heldout' in t for t in tasks))
                return [{'reward':1. if rules else 0.} for _ in tasks]
            study.evaluate = evaluate
            for i in range(1,5):
                memory.advance(i)
                study.adapt('mmh', {'round':i,'domain':'airline',
                                   'validation':['airline:heldout'+str(i%2)],
                                   'subset':str(i%2)}, [])
            self.assertEqual(len(memory.stable()), 1)
            self.assertEqual(memory.stable()[0].successful_lifespan, 3)
            self.assertEqual(len(memory.stable()[0].independent_subsets), 2)
            memory.store.close()


if __name__ == '__main__':
    unittest.main()
