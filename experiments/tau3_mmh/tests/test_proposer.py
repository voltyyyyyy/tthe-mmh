import json
import unittest

from experiments.tau3_mmh.proposer import LLMProposer, _LLM_SYSTEM, _normalize_add_target


class ProposerValidationTests(unittest.TestCase):
    def _proposer(self):
        proposer = object.__new__(LLMProposer)
        proposer.last_rejections = []
        return proposer

    def _payload(self, target='new'):
        return {
            'thinking': 'test',
            'operation': 'Add',
            'target_rule_id': target,
            'new_rule': {
                'phi': 'the task concerns account state',
                'psi': 'look up account state before answering',
                'omega': 'pending',
                'confidence': 0.8,
                'lifespan': 0,
            },
            'split_details': [],
            'judge_confidence': 0.8,
        }

    def test_prompt_uses_literal_new_sentinel(self):
        self.assertIn('"target_rule_id": "new"', _LLM_SYSTEM)
        self.assertNotIn('<new>', _LLM_SYSTEM)

    def test_add_target_alias_is_normalized(self):
        proposer = self._proposer()
        proposals = proposer._to_proposals(json.dumps(self._payload('new_rule_1')), 3, [])
        self.assertEqual(1, len(proposals))
        self.assertEqual('look up account state before answering', proposals[0].guideline.psi)
        self.assertEqual([], proposer.last_rejections)

    def test_normalizer_does_not_mutate_input(self):
        payload = self._payload('rule_04')
        normalized = _normalize_add_target(payload)
        self.assertEqual('rule_04', payload['target_rule_id'])
        self.assertEqual('new', normalized['target_rule_id'])

    def test_non_add_is_not_normalized(self):
        payload = self._payload('rule_04')
        payload['operation'] = 'Refine'
        normalized = _normalize_add_target(payload)
        self.assertEqual('rule_04', normalized['target_rule_id'])

    def test_invalid_rule_shape_records_rejection(self):
        payload = self._payload('new_rule_1')
        del payload['new_rule']['psi']
        proposer = self._proposer()
        self.assertEqual([], proposer._to_proposals(json.dumps(payload), 4, []))
        self.assertEqual('validation_error', proposer.last_rejections[0]['reason'])

    def test_model_none_records_rejection(self):
        proposer = self._proposer()
        self.assertEqual([], proposer._to_proposals('{"operation": "None"}', 4, []))
        self.assertEqual('model_none', proposer.last_rejections[0]['reason'])


if __name__ == '__main__':
    unittest.main()