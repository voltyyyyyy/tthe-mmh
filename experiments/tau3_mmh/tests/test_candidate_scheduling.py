import unittest

from meta_memory import PatchStatus, Rule, RuleTier
from experiments.tau3_mmh.full_study import candidate_rules


class _Store:
    def __init__(self, rules):
        self.rules = rules

    def list_rules(self):
        return list(self.rules)


class _Memory:
    def __init__(self, rules):
        self.store = _Store(rules)


def _rule(rule_id, *, status, created, last_measured=None, tier=RuleTier.VOLATILE):
    provenance = {}
    if last_measured is not None:
        provenance['last_measured'] = last_measured
    return Rule(
        rule_id=rule_id,
        phi=f'condition {rule_id}',
        psi=f'action {rule_id}',
        status=status,
        tier=tier,
        created_round=created,
        provenance=provenance,
    )


class CandidateSchedulingTests(unittest.TestCase):
    def test_all_pending_rules_are_always_included(self):
        rules = [
            _rule('p1', status=PatchStatus.PENDING, created=1),
            _rule('p2', status=PatchStatus.PENDING, created=2),
            _rule('m1', status=PatchStatus.VALIDATED, created=3, last_measured=3),
        ]
        selected = candidate_rules(_Memory(rules), round_id=10, max_maturing=2)
        self.assertEqual(['p1', 'p2', 'm1'], [r.rule_id for r in selected])

    def test_maturing_rules_are_selected_by_oldest_evidence(self):
        rules = [
            _rule('p1', status=PatchStatus.PENDING, created=1),
            _rule('m_old', status=PatchStatus.VALIDATED, created=2, last_measured=4),
            _rule('m_mid', status=PatchStatus.VALIDATED, created=3, last_measured=5),
            _rule('m_new', status=PatchStatus.VALIDATED, created=4, last_measured=6),
        ]
        selected = candidate_rules(_Memory(rules), round_id=10, max_maturing=2)
        self.assertEqual(['p1', 'm_old', 'm_mid'], [r.rule_id for r in selected])

    def test_new_pending_rule_does_not_displace_maturing_rules(self):
        rules = [
            _rule('old', status=PatchStatus.VALIDATED, created=1, last_measured=2),
            _rule('new', status=PatchStatus.PENDING, created=9),
        ]
        selected = candidate_rules(_Memory(rules), round_id=10, max_maturing=2)
        self.assertEqual(['new', 'old'], [r.rule_id for r in selected])

    def test_stable_rules_are_not_maturing_candidates(self):
        rules = [
            _rule('s', status=PatchStatus.VALIDATED, created=1, last_measured=1,
                  tier=RuleTier.STABLE),
            _rule('m', status=PatchStatus.VALIDATED, created=2, last_measured=2),
        ]
        selected = candidate_rules(_Memory(rules), round_id=10, max_maturing=2)
        self.assertEqual(['m'], [r.rule_id for r in selected])


if __name__ == '__main__':
    unittest.main()