import itertools
import math
import random

import pytest

from timetable.metrics import (constraint_dominates, coverage, crowding_distance,
    dominates, hypervolume, non_dominated_sort, nondominated_indices, paired_comparison, spacing)


def test_sort_constraints_duplicates_and_crowding():
    vals = [(1, 2, 3), (2, 1, 3), (3, 3, 3), (1, 2, 3)]
    assert set(non_dominated_sort(vals)[0]) == {0, 1, 3}
    assert nondominated_indices(vals) == [0, 1]
    assert non_dominated_sort(vals, [1, 0, 0, 2]) == [[1], [2], [0], [3]]
    assert constraint_dominates((999,), (0,), 0, 1)
    assert not constraint_dominates((0,), (999,), 2, 2)
    d = crowding_distance([(1, 1), (1, 1), (1, 1)], [0, 1, 2])
    assert d == {0: 0, 1: 0, 2: 0}
    d = crowding_distance([(0, 2), (1, 1), (2, 0)], [0, 1, 2])
    assert math.isinf(d[0]) and math.isinf(d[2]) and d[1] == 2


def test_hv_known_union_and_invalid_input():
    assert hypervolume([], (1, 1, 1)) == 0
    assert hypervolume([(0.2, 0.3, 0.4)], (1, 1, 1)) == pytest.approx(0.8 * 0.7 * 0.6)
    points = [(0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    assert hypervolume(points, (1, 1, 1)) == pytest.approx(0.5)
    assert hypervolume(points + points + [(0.8, 0.8, 0.8)], (1, 1, 1)) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        hypervolume([(1.2, 0, 0)], (1, 1, 1))
    with pytest.raises(ValueError):
        hypervolume([(float('nan'), 0, 0)])


def test_hv_matches_independent_inclusion_exclusion():
    rng = random.Random(8)
    for n in range(1, 7):
        points = [tuple(rng.random() for _ in range(3)) for _ in range(n)]
        total = 0
        for k in range(1, n + 1):
            for subset in itertools.combinations(points, k):
                total += (-1) ** (k + 1) * math.prod(1 - max(p[j] for p in subset) for j in range(3))
        assert hypervolume(points, (1, 1, 1)) == pytest.approx(total)


def test_sort_matches_bruteforce():
    rng = random.Random(99)
    vals = [tuple(rng.randrange(10) for _ in range(3)) for _ in range(50)]
    front = [i for i, v in enumerate(vals) if not any(dominates(w, v) for w in vals)]
    assert set(non_dominated_sort(vals)[0]) == set(front)


def test_secondary_indicators_and_paired_sign_test():
    assert spacing([]) is None
    assert spacing([(1, 1), (1, 1)]) is None
    assert spacing([(1, 2), (2, 1)]) == 0
    assert coverage([(1, 1)], [(2, 2), (0, 0)]) == 0.5
    assert coverage([], []) is None
    result = paired_comparison([2] * 6, [1] * 6)
    assert result['sign_test_p_two_sided'] == 0.03125
    assert result['paired_superiority'] == 1
    assert paired_comparison([1], [1])['sign_test_p_two_sided'] == 1
